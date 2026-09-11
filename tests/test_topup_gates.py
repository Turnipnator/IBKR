"""
Tests for the two top-up gates added 2026-09-11.

1. Stop-buffer gate (`DecisionEngine._topup_stop_buffer_ok`, consulted by
   `top_up_position`): never buy more of a name sitting within
   `topup_min_stop_buffer_atr` ATRs of the stop it will carry after the swap.
   A falling price RAISES the share target, so a position drifting down toward
   its own ratcheted stop is exactly when it crosses the 30% drift line — and
   because the swap keeps the ratchet, the added shares carry almost no
   distance to the stop: a stop-out turns the whole top-up into commission.

2. Cash-trimmed top-up floor (`_affordable_quantity`, `min_partial_topup_pct`):
   a top-up that settled cash trims below 50% of the wanted delta is skipped
   instead of buying a handful of shares for the $4 minimum commission.

Both drive the REAL engine methods on a DecisionEngine built via `__new__`;
only the collaborators (order manager, connection) are stubbed. Fixture
numbers are the live cases that motivated each gate: AIGA on 2026-09-11
(126 held, 7.05 ratchet, ATR 0.11) and the 2026-08-31 CMOD top-up (13 wanted,
£101.55 settled, trimmed to 3).
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config import TradingConfig
from src.engine import DecisionEngine
from src.orders import OrderResult

UNSET_DOUBLE = 1.7976931348623157e308  # IBKR "not set" sentinel


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _stop(trigger, qty=126, symbol="AIGA"):
    return SimpleNamespace(
        order=SimpleNamespace(
            orderId=5700, orderType="TRAIL", action="SELL",
            totalQuantity=qty, auxPrice=0.33, trailStopPrice=trigger,
        ),
        contract=SimpleNamespace(symbol=symbol),
        orderStatus=SimpleNamespace(status="PreSubmitted", filled=0),
    )


def _engine(stops=(), stops_raise=False, cfg=None, dry_run=False):
    om = MagicMock()
    filled = SimpleNamespace(orderStatus=SimpleNamespace(status="Filled"), log=[])
    om.place_market_order = MagicMock(
        return_value=OrderResult(success=True, order_id=900, trade=filled)
    )
    if stops_raise:
        om.protective_stops_for = MagicMock(side_effect=RuntimeError("IBKR down"))
    else:
        om.protective_stops_for = MagicMock(return_value=list(stops))
    om.replace_trailing_stop = MagicMock(return_value=OrderResult(success=True))

    eng = DecisionEngine.__new__(DecisionEngine)
    eng.dry_run = dry_run
    eng.config = cfg if cfg is not None else SimpleNamespace(
        atr_stop_multiplier=3.0, topup_min_stop_buffer_atr=1.0
    )
    eng.order_manager = om
    eng.connection = SimpleNamespace(ib=SimpleNamespace(sleep=lambda s: None))
    # settled cash is not under test here: pass the delta straight through
    eng._affordable_quantity = MagicMock(
        side_effect=lambda sym, qty, price, *, is_new_entry: qty
    )
    return eng, om


def _aiga(price=7.09, atr=0.11, target=181):
    """AIGA at the 2026-09-11 rebalance if it dipped to 7.09: target 181 vs
    126 held (69.6% → past the 30% drift line), fresh 3xATR level 6.76."""
    return SimpleNamespace(
        symbol="AIGA", position_size=target, current_price=price,
        signal_score=0.93, atr_value=atr, stop_loss_price=round(price - 3 * atr, 4),
    )


# --------------------------------------------------------------------------
# 1. stop-buffer gate
# --------------------------------------------------------------------------

class TestStopBufferGate:

    def test_aiga_within_one_atr_of_its_ratchet_is_skipped(self, caplog):
        """The motivating case: 7.09 is 0.04 (0.36xATR) above the 7.05 ratchet."""
        eng, om = _engine(stops=[_stop(7.05)])
        with caplog.at_level(logging.INFO):
            res = eng.top_up_position(_aiga(), held_qty=126)
        assert res.success is False
        assert res.message.startswith("Skipped:")
        assert "0.36xATR" in res.message and "7.050" in res.message
        om.place_market_order.assert_not_called()
        om.replace_trailing_stop.assert_not_called()
        assert "top-up skipped (floor 1xATR)" in caplog.text
        assert "pure commission" in caplog.text

    def test_gate_runs_before_the_settled_cash_read(self):
        """A skip must cost nothing — no AvailableFunds round-trip."""
        eng, om = _engine(stops=[_stop(7.05)])
        eng.top_up_position(_aiga(), held_qty=126)
        eng._affordable_quantity.assert_not_called()

    def test_ratchet_far_below_price_proceeds_on_the_fresh_level(self):
        """Ratchet 6.70 < fresh 6.76: the swap would carry 6.76, buffer 3xATR."""
        eng, om = _engine(stops=[_stop(6.70)])
        res = eng.top_up_position(_aiga(), held_qty=126)
        assert res.success is True and res.filled_quantity == 55
        assert om.place_market_order.call_args.kwargs["quantity"] == 55
        kw = om.replace_trailing_stop.call_args.kwargs
        assert kw["quantity"] == 181
        assert kw["initial_stop_price"] == pytest.approx(6.76)

    def test_carried_stop_is_the_higher_of_ratchet_and_fresh_level(self):
        """Ratchet 6.95 > fresh 6.76, and 7.09-6.95 = 0.14 > 1xATR: proceeds.
        Ratchet 7.00 → 0.09 < 0.11: skipped. The gate must use the ratchet
        when it is the binding stop, not the fresh level."""
        eng, _ = _engine(stops=[_stop(6.95)])
        assert eng.top_up_position(_aiga(), held_qty=126).success is True
        eng, _ = _engine(stops=[_stop(7.00)])
        assert eng.top_up_position(_aiga(), held_qty=126).success is False

    def test_exactly_one_atr_passes_despite_float_noise(self):
        """7.20 - 7.10 is 0.09999999999999964 in binary; must still count as 1xATR."""
        opp = SimpleNamespace(symbol="AIGA", position_size=181, current_price=7.20,
                              signal_score=0.9, atr_value=0.10, stop_loss_price=6.90)
        eng, om = _engine(stops=[_stop(7.10)])
        assert eng.top_up_position(opp, held_qty=126).success is True
        eng, om = _engine(stops=[_stop(7.101)])
        assert eng.top_up_position(opp, held_qty=126).success is False

    def test_multiplier_is_read_from_config(self):
        """Buffer 0.15 = 1.5xATR: passes at the 1.0 default, skipped at 2.0."""
        opp = SimpleNamespace(symbol="AIGA", position_size=181, current_price=7.20,
                              signal_score=0.9, atr_value=0.10, stop_loss_price=6.90)
        eng, _ = _engine(stops=[_stop(7.05)])
        assert eng.top_up_position(opp, held_qty=126).success is True
        cfg = SimpleNamespace(atr_stop_multiplier=3.0, topup_min_stop_buffer_atr=2.0)
        eng, _ = _engine(stops=[_stop(7.05)], cfg=cfg)
        res = eng.top_up_position(opp, held_qty=126)
        assert res.success is False and "floor 2xATR" in res.message

    def test_stale_price_at_or_below_the_stop_is_skipped(self):
        """Price 7.00 under a 7.05 ratchet: the stop should already have fired.
        Never add to that."""
        eng, om = _engine(stops=[_stop(7.05)])
        res = eng.top_up_position(_aiga(price=7.00), held_qty=126)
        assert res.success is False
        om.place_market_order.assert_not_called()

    def test_naked_position_falls_back_to_the_fresh_level(self):
        """No working stop (a BUY that filled after the wait window): the only
        level is price-3xATR, buffer 3xATR → proceeds; reconcile covers it."""
        eng, om = _engine(stops=[])
        res = eng.top_up_position(_aiga(), held_qty=126)
        assert res.success is True
        om.place_market_order.assert_called_once()

    def test_unset_double_sentinel_is_not_a_ratchet(self):
        eng, om = _engine(stops=[_stop(UNSET_DOUBLE)])
        assert eng.top_up_position(_aiga(), held_qty=126).success is True

    def test_only_sell_stops_are_consulted(self):
        eng, om = _engine(stops=[_stop(7.05)])
        eng.top_up_position(_aiga(), held_qty=126)
        om.protective_stops_for.assert_called_once_with("AIGA", "SELL")

    def test_stop_lookup_error_fails_open(self, caplog):
        eng, om = _engine(stops_raise=True)
        with caplog.at_level(logging.WARNING):
            res = eng.top_up_position(_aiga(), held_qty=126)
        assert res.success is True
        om.place_market_order.assert_called_once()
        assert "top-up gate not applied" in caplog.text

    def test_zero_atr_fails_open(self, caplog):
        eng, om = _engine(stops=[_stop(7.05)])
        with caplog.at_level(logging.WARNING):
            res = eng.top_up_position(_aiga(atr=0.0), held_qty=126)
        assert res.success is True                 # the BUY went through...
        om.replace_trailing_stop.assert_not_called()  # ...and the ATR=0 swap-skip still applies
        assert "top-up gate not applied" in caplog.text

    def test_zero_or_missing_config_disables_the_gate(self):
        cfg = SimpleNamespace(atr_stop_multiplier=3.0, topup_min_stop_buffer_atr=0.0)
        eng, om = _engine(stops=[_stop(7.05)], cfg=cfg)
        assert eng.top_up_position(_aiga(), held_qty=126).success is True
        om.protective_stops_for.assert_not_called()
        cfg = SimpleNamespace(atr_stop_multiplier=3.0)   # attribute absent
        eng, om = _engine(stops=[_stop(7.05)], cfg=cfg)
        assert eng.top_up_position(_aiga(), held_qty=126).success is True

    def test_live_config_default_is_one_atr(self):
        assert TradingConfig().topup_min_stop_buffer_atr == 1.0

    def test_dry_run_never_touches_the_book(self):
        eng, om = _engine(stops=[_stop(7.05)], dry_run=True)
        res = eng.top_up_position(_aiga(), held_qty=126)
        assert res.success is True
        om.protective_stops_for.assert_not_called()
        om.place_market_order.assert_not_called()

    def test_no_top_up_needed_short_circuits_before_the_gate(self):
        eng, om = _engine(stops=[_stop(7.05)])
        res = eng.top_up_position(_aiga(target=120), held_qty=126)
        assert res.success is False and "no top-up needed" in res.message
        om.protective_stops_for.assert_not_called()


# --------------------------------------------------------------------------
# 2. cash-trimmed top-up floor
# --------------------------------------------------------------------------

FX_USD = 0.7403          # BASE per USD at the 2026-09-10 rebalance
CMOD_PX = 33.76          # the 2026-08-31 top-up print
CMOD_UNIT = CMOD_PX * FX_USD * 1.06   # £26.49 per share incl. the 6% buffer


def _cash_engine(settled, topup_floor=0.50):
    eng = DecisionEngine.__new__(DecisionEngine)
    cfg = TradingConfig()
    cfg.min_partial_topup_pct = topup_floor
    eng.config = cfg
    eng.connection = SimpleNamespace(
        get_fx_rates=lambda: {"USD": FX_USD, "GBP": 1.0},
        get_account_summary=lambda: {
            "AvailableFunds": {"value": str(settled), "currency": "GBP"},
            "NetLiquidation": {"value": "4700", "currency": "GBP"},
        },
    )
    return eng


class TestTopUpFloor:

    def test_0831_cmod_top_up_trimmed_to_3_of_13_is_now_skipped(self, caplog):
        """Replays 2026-08-31: £101.55 covered 3 of the 13 wanted (23%) and the
        bot bought 3 shares for a $4 commission. Under the floor it skips."""
        eng = _cash_engine(101.55)
        assert int(101.55 / CMOD_UNIT) == 3          # fixture sanity
        with caplog.at_level(logging.INFO):
            assert eng._affordable_quantity("CMOD", 13, CMOD_PX, is_new_entry=False) == 0
        assert "covers only 3/13 shares (23%, below 50% top-up floor)" in caplog.text

    def test_top_up_at_or_above_the_floor_is_trimmed_not_skipped(self, caplog):
        eng = _cash_engine(250.0)                    # 9 of 13 = 69%
        with caplog.at_level(logging.INFO):
            assert eng._affordable_quantity("CMOD", 13, CMOD_PX, is_new_entry=False) == 9
        assert "trimming top-up to 9" in caplog.text

    def test_exactly_at_the_floor_passes(self):
        eng = _cash_engine(3 * CMOD_UNIT + 0.5)      # 3 of 6 = 50%
        assert eng._affordable_quantity("CMOD", 6, CMOD_PX, is_new_entry=False) == 3

    def test_full_cover_is_untouched(self):
        eng = _cash_engine(1148.16)                  # today's AvailableFunds
        assert eng._affordable_quantity("CMOD", 13, CMOD_PX, is_new_entry=False) == 13

    def test_zero_floor_restores_the_old_behaviour(self):
        eng = _cash_engine(101.55, topup_floor=0.0)
        assert eng._affordable_quantity("CMOD", 13, CMOD_PX, is_new_entry=False) == 3

    def test_entry_floor_is_unchanged(self, caplog):
        eng = _cash_engine(101.55)
        with caplog.at_level(logging.INFO):
            assert eng._affordable_quantity("CMOD", 13, CMOD_PX, is_new_entry=True) == 0
        assert "below 50% entry floor" in caplog.text

    def test_zero_cover_still_skips_regardless_of_floor(self):
        eng = _cash_engine(10.0, topup_floor=0.0)
        assert eng._affordable_quantity("CMOD", 13, CMOD_PX, is_new_entry=False) == 0

    def test_live_config_default_is_half(self):
        assert TradingConfig().min_partial_topup_pct == 0.50
