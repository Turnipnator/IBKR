"""
Tests for the 2026-08-28 hardening of `OrderManager.replace_trailing_stop`.

Before: cancel the working stop(s), confirm the book is clear, place the
replacement — and if the replacement failed, log NAKED and return, leaving the
position uncovered until the post-rebalance sweep (30-120s later).
After: on replacement failure the cancelled stop(s) are re-placed immediately
with their original remaining quantity, trail amount and ratcheted trigger.

These drive the REAL `replace_trailing_stop` on an OrderManager built via
`__new__`; only its collaborators (connection/ib/db) and the two IBKR-touching
methods it calls (`protective_stops_for`, `cancel_order`,
`place_trailing_stop_order`) are stubbed. The fixture numbers are EIMU's live
stop on 2026-08-28: 110 shares, trail $0.39, trigger 6.883.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.orders import OrderManager, OrderAction, OrderResult

UNSET_DOUBLE = 1.7976931348623157e308  # IBKR "not set" sentinel


def _stop(order_id=501, qty=110, filled=0, trail=0.39, trigger=6.883):
    order = SimpleNamespace(
        orderId=order_id, orderType="TRAIL", action="SELL",
        totalQuantity=qty, auxPrice=trail, trailStopPrice=trigger,
    )
    return SimpleNamespace(
        order=order,
        contract=SimpleNamespace(symbol="EIMU"),
        remaining=lambda: qty - filled,
    )


def _om(existing, place_results, cancel_confirms=True):
    om = OrderManager.__new__(OrderManager)
    om.connection = SimpleNamespace(
        ensure_connected=lambda: True,
        ib=SimpleNamespace(sleep=lambda s: None),
    )
    om.db = MagicMock()
    om._pending_orders = {}
    om._order_callbacks = []
    if cancel_confirms:
        # first call: the working stops; every later call: book clear
        seq = [list(existing)] + [[]] * 45
    else:
        seq = [list(existing)] * 46
    om.protective_stops_for = MagicMock(side_effect=seq)
    om.cancel_order = MagicMock(return_value=True)
    om.place_trailing_stop_order = MagicMock(side_effect=list(place_results))
    return om


def _replace(om, quantity=141, trail=0.33, trigger=6.90):
    return om.replace_trailing_stop(
        symbol="EIMU", action=OrderAction.SELL, quantity=quantity,
        trail_amount=trail, initial_stop_price=trigger,
        reason="Top-up: re-cover full position of 141",
    )


class TestRestoreOnFailure:

    def test_restores_old_cover_when_replacement_fails(self):
        om = _om([_stop()], [
            OrderResult(success=False, message="Error 201"),
            OrderResult(success=True, order_id=777),
        ])
        res = _replace(om)

        assert res.success is False
        assert "previous cover restored" in res.message
        assert om.cancel_order.call_args_list[0].args == (501,)
        assert om.place_trailing_stop_order.call_count == 2
        first, second = om.place_trailing_stop_order.call_args_list
        # the attempted replacement covers the full (topped-up) position
        assert first.kwargs["quantity"] == 141
        assert first.kwargs["trail_amount"] == 0.33
        # the restore puts back exactly what was cancelled
        assert second.kwargs["symbol"] == "EIMU"
        assert second.kwargs["action"] is OrderAction.SELL
        assert second.kwargs["quantity"] == 110
        assert second.kwargs["trail_amount"] == 0.39
        assert second.kwargs["initial_stop_price"] == 6.883
        assert "Restore after failed replacement" in second.kwargs["reason"]

    def test_partially_filled_stop_restores_remaining_only(self):
        om = _om([_stop(qty=110, filled=10)], [
            OrderResult(success=False, message="rejected"),
            OrderResult(success=True, order_id=778),
        ])
        _replace(om)
        assert om.place_trailing_stop_order.call_args_list[1].kwargs["quantity"] == 100

    def test_unset_trigger_sentinel_is_not_sent_back(self):
        om = _om([_stop(trigger=UNSET_DOUBLE)], [
            OrderResult(success=False, message="rejected"),
            OrderResult(success=True, order_id=779),
        ])
        _replace(om)
        assert om.place_trailing_stop_order.call_args_list[1].kwargs["initial_stop_price"] is None

    def test_trade_without_remaining_falls_back_to_total_quantity(self):
        stop = _stop()
        del stop.remaining
        om = _om([stop], [
            OrderResult(success=False, message="rejected"),
            OrderResult(success=True, order_id=780),
        ])
        _replace(om)
        assert om.place_trailing_stop_order.call_args_list[1].kwargs["quantity"] == 110

    def test_multiple_cancelled_stops_are_all_restored(self):
        om = _om([_stop(order_id=501, qty=60), _stop(order_id=502, qty=50)], [
            OrderResult(success=False, message="rejected"),
            OrderResult(success=True, order_id=781),
            OrderResult(success=True, order_id=782),
        ])
        res = _replace(om)
        assert om.place_trailing_stop_order.call_count == 3
        qtys = [c.kwargs["quantity"] for c in om.place_trailing_stop_order.call_args_list[1:]]
        assert sorted(qtys) == [50, 60]
        assert "previous cover restored" in res.message

    def test_restore_failure_still_returns_failure_and_logs_naked(self, caplog):
        om = _om([_stop()], [
            OrderResult(success=False, message="Error 201"),
            OrderResult(success=False, message="gateway down"),
        ])
        with caplog.at_level("ERROR"):
            res = _replace(om)
        assert res.success is False
        assert "restored" not in res.message           # message left as the raw failure
        assert res.message == "Error 201"
        assert any("NAKED" in r.message and "0/1 restored" in r.message for r in caplog.records)

    def test_restore_exception_is_contained(self):
        bad = _stop()
        bad.remaining = MagicMock(side_effect=RuntimeError("boom"))
        om = _om([bad], [OrderResult(success=False, message="rejected")])
        res = _replace(om)                              # must not raise
        assert res.success is False
        assert om.place_trailing_stop_order.call_count == 1


class TestRatchetIsKept:
    """2026-08-28: a top-up swap used to re-arm at price - kxATR regardless of
    where the old stop had ratcheted to. The old trigger must win when higher."""

    def _placed_trigger(self, om):
        return om.place_trailing_stop_order.call_args_list[0].kwargs["initial_stop_price"]

    def test_fresh_level_below_old_ratchet_is_lifted_to_the_ratchet(self, caplog):
        om = _om([_stop(trigger=7.10)], [OrderResult(success=True, order_id=800)])
        with caplog.at_level("INFO"):
            res = _replace(om, trigger=6.90)
        assert res.success is True
        assert self._placed_trigger(om) == 7.10
        assert any("keeping ratcheted trigger 7.1000" in r.message for r in caplog.records)

    def test_fresh_level_above_old_ratchet_is_used_as_is(self):
        om = _om([_stop(trigger=6.883)], [OrderResult(success=True, order_id=801)])
        _replace(om, trigger=6.90)
        assert self._placed_trigger(om) == 6.90

    def test_no_fresh_level_falls_back_to_the_ratchet(self):
        om = _om([_stop(trigger=7.10)], [OrderResult(success=True, order_id=802)])
        _replace(om, trigger=None)
        assert self._placed_trigger(om) == 7.10

    def test_unset_sentinel_and_zero_are_ignored(self):
        om = _om([_stop(trigger=UNSET_DOUBLE), _stop(order_id=502, trigger=0.0)],
                 [OrderResult(success=True, order_id=803)])
        _replace(om, trigger=6.90)
        assert self._placed_trigger(om) == 6.90

    def test_highest_of_several_old_stops_wins_for_a_sell(self):
        om = _om([_stop(order_id=501, trigger=7.00), _stop(order_id=502, trigger=7.20)],
                 [OrderResult(success=True, order_id=804)])
        _replace(om, trigger=6.90)
        assert self._placed_trigger(om) == 7.20

    def test_buy_stop_short_cover_keeps_the_lowest(self):
        stop = _stop(trigger=6.50); stop.order.action = "BUY"
        om = _om([stop], [OrderResult(success=True, order_id=805)])
        om.replace_trailing_stop(symbol="EIMU", action=OrderAction.BUY, quantity=10,
                                 trail_amount=0.3, initial_stop_price=6.80, reason="t")
        assert self._placed_trigger(om) == 6.50

    def test_restore_after_failure_still_uses_each_old_stops_own_trigger(self):
        om = _om([_stop(trigger=7.10)], [
            OrderResult(success=False, message="rejected"),
            OrderResult(success=True, order_id=806),
        ])
        _replace(om, trigger=6.90)
        first, second = om.place_trailing_stop_order.call_args_list
        assert first.kwargs["initial_stop_price"] == 7.10      # lifted replacement
        assert second.kwargs["initial_stop_price"] == 7.10     # restore of the original

    def test_through_the_real_top_up_position(self):
        """DecisionEngine.top_up_position -> real replace_trailing_stop: the
        engine passes its fresh price-3xATR level (6.90) and the working stop's
        7.10 ratchet must be what lands on the book, covering the full position."""
        from src.engine import DecisionEngine
        om = _om([_stop(qty=110, trigger=7.10)], [OrderResult(success=True, order_id=807)])
        filled_trade = SimpleNamespace(orderStatus=SimpleNamespace(status="Filled"), log=[])
        om.place_market_order = MagicMock(return_value=OrderResult(success=True, order_id=700, trade=filled_trade))
        eng = DecisionEngine.__new__(DecisionEngine)
        eng.dry_run = False
        eng.config = SimpleNamespace(atr_stop_multiplier=3.0)
        eng.order_manager = om
        eng.connection = om.connection
        eng._affordable_quantity = lambda sym, qty, price, *, is_new_entry: qty
        opp = SimpleNamespace(symbol="EIMU", position_size=141, current_price=7.40,
                              signal_score=0.9, atr_value=0.1667, stop_loss_price=6.90)
        res = eng.top_up_position(opp, held_qty=110)
        assert res.success is True and res.filled_quantity == 31
        om.place_market_order.assert_called_once()
        assert om.place_market_order.call_args.kwargs["quantity"] == 31
        kw = om.place_trailing_stop_order.call_args_list[0].kwargs
        assert kw["quantity"] == 141
        assert kw["initial_stop_price"] == 7.10
        assert kw["trail_amount"] == pytest.approx(0.5001)


class TestUnchangedPaths:

    def test_success_path_places_once_and_never_restores(self):
        om = _om([_stop()], [OrderResult(success=True, order_id=790)])
        res = _replace(om)
        assert res.success is True
        assert om.place_trailing_stop_order.call_count == 1

    def test_unconfirmed_cancel_aborts_before_placing_anything(self):
        om = _om([_stop()], [OrderResult(success=True, order_id=791)], cancel_confirms=False)
        res = _replace(om)
        assert res.success is False
        assert "cancellation not confirmed" in res.message
        om.place_trailing_stop_order.assert_not_called()

    def test_cancel_happens_before_any_placement(self):
        om = _om([_stop()], [
            OrderResult(success=False, message="rejected"),
            OrderResult(success=True, order_id=792),
        ])
        tracker = MagicMock()
        tracker.attach_mock(om.cancel_order, "cancel")
        tracker.attach_mock(om.place_trailing_stop_order, "place")
        _replace(om)
        names = [c[0] for c in tracker.mock_calls]
        assert names.index("cancel") < names.index("place")

    def test_no_existing_stop_failure_is_reported_as_naked_without_restore(self, caplog):
        om = _om([], [OrderResult(success=False, message="rejected")])
        with caplog.at_level("ERROR"):
            res = _replace(om)
        assert res.success is False
        assert om.place_trailing_stop_order.call_count == 1
        assert any("NAKED" in r.message for r in caplog.records)

    def test_not_connected_short_circuits(self):
        om = _om([_stop()], [])
        om.connection = SimpleNamespace(ensure_connected=lambda: False, ib=None)
        res = _replace(om)
        assert res.success is False
        om.cancel_order.assert_not_called()
