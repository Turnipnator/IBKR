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
