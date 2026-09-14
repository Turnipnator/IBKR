"""
Tests for the 2026-09-14 per-cycle committed-cash tally in the settled-cash gate.

The motivating case, 2026-09-14 13:01:53–54: a CMOD top-up (BUY 26 @ ~$36.38)
consumed ~£700 of settled cash; 0.2 s later the IJPN entry read AvailableFunds
from ib_insync's cached account summary, which still said £1,804.49 (IBKR's
real figure was £1,102.65), so `_affordable_quantity` let 74 shares through
untrimmed and IBKR rejected the order (Error 201, needed £1,486.99). The gate's
own "cash needed" estimate matched IBKR's to within £5 — only the input was
stale.

After: every BUY accepted in a cycle is charged to `_cash_committed_base`
(same cost estimate the gate uses), a rejected BUY is released again, the
tally is reset when `run_analysis` starts a new cycle, and the gate nets the
tally off the cached read.

These drive the REAL `execute_opportunity`, `top_up_position` and
`_affordable_quantity` on a DecisionEngine built via `__new__`; the order
manager and connection are stubs. The account summary stub is deliberately
constant — that IS the stale-cache condition under test.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.engine import DecisionEngine, TradeDecision
from src.orders import OrderResult


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

FX = {"USD": 0.7425, "GBP": 1.0, "EUR": 0.8567}   # the 2026-09-14 13:01 rates


def _filled_trade():
    return SimpleNamespace(orderStatus=SimpleNamespace(status="Filled"), log=[])


def _rejected_trade(order_id=5822):
    msg = (f"Error 201, reqId {order_id}: Order rejected - reason:Available "
           f"settled cash converted to base: 1102.65 GBP Cash needed for this "
           f"order and other pending orders: <br>1486.99 GBP")
    return SimpleNamespace(
        orderStatus=SimpleNamespace(status="Cancelled"),
        log=[SimpleNamespace(status="Cancelled", message=msg, errorCode=201)],
    )


def _engine(available=1804.49, trade=None, dry_run=False, fx_raises=False):
    eng = DecisionEngine.__new__(DecisionEngine)
    eng.dry_run = dry_run
    eng.config = SimpleNamespace(
        settled_cash_buffer=0.06,
        min_partial_entry_pct=0.50,
        min_partial_topup_pct=0.50,
        atr_stop_multiplier=3.0,
        topup_min_stop_buffer_atr=1.0,
    )
    summary = {
        "AvailableFunds": {"value": str(available), "currency": "GBP"},
        "NetLiquidation": {"value": "4723.42", "currency": "GBP"},
    }

    def rates():
        if fx_raises:
            raise RuntimeError("FX feed down")
        return dict(FX)

    eng.connection = SimpleNamespace(
        ib=SimpleNamespace(sleep=lambda s: None),
        get_account_summary=lambda: summary,
        get_fx_rates=rates,
    )
    om = MagicMock()
    om.place_market_order = MagicMock(
        return_value=OrderResult(success=True, order_id=5818,
                                 trade=trade or _filled_trade())
    )
    om.place_trailing_stop_order = MagicMock(return_value=OrderResult(success=True))
    om.protective_stops_for = MagicMock(return_value=[])   # no ratchet to respect
    om.replace_trailing_stop = MagicMock(return_value=OrderResult(success=True))
    eng.order_manager = om
    return eng, om


def _cmod_topup():
    """CMOD at 13:01:53: target 52 vs 26 held, $36.38, ATR 0.36 -> stop 35.29."""
    return SimpleNamespace(
        symbol="CMOD", position_size=52, current_price=36.38, signal_score=1.0,
        atr_value=0.362375, stop_loss_price=35.29, decision=TradeDecision.BUY,
    )


def _ijpn_entry():
    """IJPN at 13:01:53: 74 shares @ 1,890p (GBX), ATR 24.95p -> stop 1,815.15p."""
    return SimpleNamespace(
        symbol="IJPN", position_size=74, current_price=1890.0, signal_score=0.50,
        atr_value=24.95, stop_loss_price=1815.15, decision=TradeDecision.BUY,
    )


def _placed_qty(om, call=-1):
    return om.place_market_order.call_args_list[call].kwargs["quantity"]


CMOD_COST = 26 * 36.38 * FX["USD"] * 1.06      # £744.34
IJPN_UNIT = 1890.0 * 0.01 * 1.06               # £20.03 per share


# --------------------------------------------------------------------------
# 1. the replay
# --------------------------------------------------------------------------

class TestReplay20260914:

    def test_ijpn_is_trimmed_after_the_cmod_topup_on_a_stale_read(self, caplog):
        eng, om = _engine(available=1804.49)     # cache never moves

        with caplog.at_level(logging.INFO):
            res = eng.top_up_position(_cmod_topup(), held_qty=26)
            assert res.success and _placed_qty(om) == 26
            assert eng._cash_committed() == pytest.approx(CMOD_COST, abs=0.01)

            res = eng.execute_opportunity(_ijpn_entry())

        assert res.success
        expected = int((1804.49 - CMOD_COST) / IJPN_UNIT)     # 52
        assert expected == 52
        assert _placed_qty(om) == expected
        # and the stop covers exactly what was bought
        stop_kwargs = om.place_trailing_stop_order.call_args.kwargs
        assert stop_kwargs["symbol"] == "IJPN" and stop_kwargs["quantity"] == expected
        assert "already committed this cycle" in caplog.text
        assert "trimming entry to 52" in caplog.text

    def test_control_without_the_tally_the_old_bug_reproduces(self):
        """Same stale read, tally reset -> 74 shares go out untrimmed (the bug)."""
        eng, om = _engine(available=1804.49)
        eng.top_up_position(_cmod_topup(), held_qty=26)
        eng.reset_cash_committed()

        eng.execute_opportunity(_ijpn_entry())

        assert _placed_qty(om) == 74

    def test_two_entries_charge_cumulatively(self):
        """Third order in a cycle sees both earlier commits."""
        eng, om = _engine(available=1804.49)
        eng.top_up_position(_cmod_topup(), held_qty=26)
        eng.execute_opportunity(_ijpn_entry())           # 52 sh -> ~£1,041.6
        after_two = eng._cash_committed()
        assert after_two == pytest.approx(CMOD_COST + 52 * IJPN_UNIT, abs=0.01)

        # A further entry now sees ~£18 left: below the 50% entry floor -> skipped.
        res = eng.execute_opportunity(SimpleNamespace(
            symbol="CRUD", position_size=30, current_price=17.59, signal_score=0.93,
            atr_value=0.40, stop_loss_price=16.38, decision=TradeDecision.BUY,
        ))
        assert res.success is False
        assert "Insufficient settled cash" in res.message
        assert om.place_market_order.call_count == 2


# --------------------------------------------------------------------------
# 2. lifecycle: reset, release, never negative
# --------------------------------------------------------------------------

class TestTallyLifecycle:

    def test_reset_starts_a_new_cycle_clean(self):
        eng, om = _engine()
        eng.top_up_position(_cmod_topup(), held_qty=26)
        assert eng._cash_committed() > 0
        eng.reset_cash_committed()
        assert eng._cash_committed() == 0.0
        eng.execute_opportunity(_ijpn_entry())
        assert _placed_qty(om) == 74          # full read available again

    def test_rejected_entry_releases_its_charge(self, caplog):
        eng, om = _engine(available=1804.49, trade=_rejected_trade())
        with caplog.at_level(logging.WARNING):
            res = eng.execute_opportunity(_ijpn_entry())
        assert res.success is False and res.message.startswith("Rejected:")
        assert eng._cash_committed() == 0.0
        om.place_trailing_stop_order.assert_not_called()

    def test_rejected_topup_releases_its_charge(self):
        eng, om = _engine(available=1804.49, trade=_rejected_trade(5818))
        res = eng.top_up_position(_cmod_topup(), held_qty=26)
        assert res.success is False and res.message.startswith("Rejected:")
        assert eng._cash_committed() == 0.0
        om.replace_trailing_stop.assert_not_called()

    def test_release_never_goes_negative_and_missing_attr_reads_zero(self):
        eng, _ = _engine()
        assert not hasattr(eng, "_cash_committed_base")   # built via __new__
        assert eng._cash_committed() == 0.0
        eng._release_cash("CMOD", 26, 36.38)
        assert eng._cash_committed() == 0.0

    def test_unfilled_but_accepted_buy_stays_charged(self):
        """IBKR reserves cash for a resting order too ('and other pending
        orders' in the Error 201 text) — an accepted-but-unfilled BUY must
        keep its charge for the rest of the cycle."""
        resting = SimpleNamespace(orderStatus=SimpleNamespace(status="PreSubmitted"), log=[])
        eng, om = _engine(available=1804.49, trade=resting)
        res = eng.execute_opportunity(_ijpn_entry())
        assert res.success                      # accepted, stop deferred
        assert eng._cash_committed() == pytest.approx(74 * IJPN_UNIT, abs=0.01)


# --------------------------------------------------------------------------
# 3. fail-open and dry-run
# --------------------------------------------------------------------------

class TestFailOpen:

    def test_cost_estimate_failure_does_not_block_the_order(self, caplog):
        eng, om = _engine(available=1804.49, fx_raises=True)
        # _affordable_quantity itself also calls get_fx_rates; make the gate
        # bypass so we isolate the commit path.
        eng._get_settled_cash_base = lambda: None
        with caplog.at_level(logging.WARNING):
            res = eng.execute_opportunity(_ijpn_entry())
        assert res.success
        assert _placed_qty(om) == 74
        assert eng._cash_committed() == 0.0
        assert "could not estimate committed cash" in caplog.text

    def test_unreadable_available_funds_still_places_full_size(self):
        eng, om = _engine(available=1804.49)
        eng.top_up_position(_cmod_topup(), held_qty=26)      # charge exists
        eng._get_settled_cash_base = lambda: None            # read fails
        eng.execute_opportunity(_ijpn_entry())
        assert _placed_qty(om) == 74      # old behaviour: IBKR decides

    def test_dry_run_charges_nothing(self):
        eng, om = _engine(dry_run=True)
        res = eng.execute_opportunity(_ijpn_entry())
        assert res.success and "[DRY RUN]" in res.message
        om.place_market_order.assert_not_called()
        assert eng._cash_committed() == 0.0

    def test_sell_orders_are_never_charged(self):
        eng, om = _engine()
        sell = SimpleNamespace(
            symbol="CMOD", position_size=26, current_price=36.38, signal_score=-1.0,
            atr_value=0.36, stop_loss_price=None, decision=TradeDecision.SELL,
        )
        eng.execute_opportunity(sell)
        assert eng._cash_committed() == 0.0
