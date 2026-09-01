"""
Tests for the 2026-08-28 isolation of the live execution loop
(`TradingBot._execute_live` / `_execute_one` / `_post_rebalance_stop_sweep`).

Before: one unexpected exception inside the per-opportunity loop propagated out
of `run_once`, skipping every later name AND the +30s/+90s protective-stop
sweep — so a BUY that had already filled could sit naked until the next risk
check. After: each name is isolated, the failure is logged + alerted, and the
sweep is forced whenever an exception fired (the order state is unknown).

These drive the REAL methods on a TradingBot built via `__new__`; the engine,
connection and reconcile are stubs. `ib.sleep` is a no-op so the 30/90s waits
don't actually happen.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.bot import TradingBot
from src.orders import OrderResult


def _opp(sym, size=10):
    return SimpleNamespace(
        symbol=sym, position_size=size, current_price=1.0,
        signal_score=1.0, reasons=[], decision=SimpleNamespace(value="BUY"),
    )


def _bot(held=None, execute=None, top_up=None, open_orders=None, notifier=None):
    bot = TradingBot.__new__(TradingBot)
    bot.dry_run = False
    bot.notifier = notifier
    bot.connection = SimpleNamespace(ib=SimpleNamespace(sleep=lambda s: None))
    pm = SimpleNamespace(get_positions=lambda: [
        SimpleNamespace(symbol=s, quantity=q) for s, q in (held or {}).items()
    ])
    om = SimpleNamespace(get_open_orders=lambda: list(open_orders or []))
    bot.engine = SimpleNamespace(
        position_manager=pm,
        order_manager=om,
        execute_opportunity=execute or MagicMock(
            return_value=OrderResult(success=True, order_id=1)),
        top_up_position=top_up or MagicMock(
            return_value=OrderResult(success=True, order_id=2, filled_quantity=5)),
    )
    bot._reconcile_protective_stops = MagicMock(return_value=0)
    return bot


def _results():
    return {"trades_executed": 0}


class TestIsolation:

    def test_exception_on_one_name_does_not_abort_the_rest(self):
        execute = MagicMock(side_effect=[RuntimeError("boom"),
                                         OrderResult(success=True, order_id=9)])
        bot = _bot(execute=execute)
        results = _results()
        bot._execute_live([_opp("CMOD"), _opp("CSPX")], results)

        assert execute.call_count == 2
        assert results["trades_executed"] == 1
        assert bot._reconcile_protective_stops.call_count == 2   # +30s and +90s

    def test_exception_forces_sweep_even_when_nothing_was_counted(self):
        bot = _bot(execute=MagicMock(side_effect=RuntimeError("boom")))
        results = _results()
        bot._execute_live([_opp("CMOD")], results)
        assert results["trades_executed"] == 0
        assert bot._reconcile_protective_stops.call_count == 2

    def test_top_up_exception_is_isolated_too(self):
        top_up = MagicMock(side_effect=KeyError("held"))
        execute = MagicMock(return_value=OrderResult(success=True, order_id=3))
        bot = _bot(held={"AIGS": 2}, top_up=top_up, execute=execute)
        results = _results()
        bot._execute_live([_opp("AIGS", size=10), _opp("CSPX")], results)

        top_up.assert_called_once()
        execute.assert_called_once()
        assert results["trades_executed"] == 1
        assert bot._reconcile_protective_stops.call_count == 2

    def test_notifier_is_alerted_once_per_failed_name(self):
        notifier = MagicMock(enabled=True)
        bot = _bot(execute=MagicMock(side_effect=RuntimeError("boom")), notifier=notifier)
        bot._execute_live([_opp("CMOD")], _results())
        notifier.notify_error.assert_called_once()
        assert notifier.notify_error.call_args.args[1] == "Live execution"
        assert "CMOD" in notifier.notify_error.call_args.args[0]

    def test_notifier_failure_does_not_mask_the_loop(self):
        notifier = MagicMock(enabled=True)
        notifier.notify_error.side_effect = ConnectionError("telegram down")
        execute = MagicMock(side_effect=[RuntimeError("boom"),
                                         OrderResult(success=True, order_id=9)])
        bot = _bot(execute=execute, notifier=notifier)
        results = _results()
        bot._execute_live([_opp("CMOD"), _opp("CSPX")], results)
        assert execute.call_count == 2
        assert results["trades_executed"] == 1


class TestBehaviourPreserved:
    """The refactor moved code; these pin the pre-existing semantics."""

    def test_no_trades_no_sweep(self):
        bot = _bot(execute=MagicMock(return_value=OrderResult(success=False, message="rejected")))
        results = _results()
        bot._execute_live([_opp("CMOD")], results)
        assert results["trades_executed"] == 0
        bot._reconcile_protective_stops.assert_not_called()

    def test_successful_entry_counts_and_sweeps_twice(self):
        bot = _bot()
        results = _results()
        bot._execute_live([_opp("CMOD")], results)
        assert results["trades_executed"] == 1
        assert bot._reconcile_protective_stops.call_count == 2

    def test_held_within_drift_threshold_is_skipped_not_topped_up(self):
        top_up = MagicMock()
        bot = _bot(held={"AIGS": 9}, top_up=top_up)      # 9 >= 10 * 0.7
        results = _results()
        bot._execute_live([_opp("AIGS", size=10)], results)
        top_up.assert_not_called()
        assert results["trades_executed"] == 0
        bot._reconcile_protective_stops.assert_not_called()

    def test_held_below_drift_threshold_is_topped_up(self):
        top_up = MagicMock(return_value=OrderResult(success=True, order_id=2, filled_quantity=8))
        bot = _bot(held={"AIGS": 2}, top_up=top_up)      # 2 < 10 * 0.7
        results = _results()
        bot._execute_live([_opp("AIGS", size=10)], results)
        top_up.assert_called_once()
        assert top_up.call_args.args[1] == 2
        assert results["trades_executed"] == 1

    def test_accepted_but_unfilled_top_up_still_counts_for_the_sweep(self):
        top_up = MagicMock(return_value=OrderResult(success=True, order_id=2, filled_quantity=0))
        bot = _bot(held={"AIGS": 2}, top_up=top_up)
        results = _results()
        bot._execute_live([_opp("AIGS", size=10)], results)
        assert results["trades_executed"] == 1
        assert bot._reconcile_protective_stops.call_count == 2

    def test_unfilled_top_up_logs_the_actual_placed_quantity(self, caplog):
        """Replays 2026-09-01: the top-up BUY was trimmed to settled cash
        (3 of the 13-share shortfall) but the unfilled line claimed "(+13)".
        It must report the order actually resting on the book — the
        quantity on the Trade — not the untrimmed target-held arithmetic."""
        trade = SimpleNamespace(order=SimpleNamespace(totalQuantity=3.0))
        top_up = MagicMock(return_value=OrderResult(
            success=True, order_id=2, filled_quantity=0, trade=trade))
        bot = _bot(held={"CMOD": 23}, top_up=top_up)
        with caplog.at_level(logging.INFO, logger="src.bot"):
            bot._execute_live([_opp("CMOD", size=36)], _results())
        assert "Top-up BUY placed for CMOD (+3) but not yet filled" in caplog.text
        assert "(+13)" not in caplog.text

    def test_unfilled_top_up_without_trade_falls_back_to_shortfall(self, caplog):
        """No Trade attached (mock/edge path): fall back to target-held
        rather than crash — the old behaviour, now only the last resort."""
        top_up = MagicMock(return_value=OrderResult(
            success=True, order_id=2, filled_quantity=0))
        bot = _bot(held={"AIGS": 2}, top_up=top_up)
        with caplog.at_level(logging.INFO, logger="src.bot"):
            bot._execute_live([_opp("AIGS", size=10)], _results())
        assert "Top-up BUY placed for AIGS (+8) but not yet filled" in caplog.text

    def test_pending_mkt_entry_is_deduped(self):
        execute = MagicMock()
        pending = [SimpleNamespace(contract=SimpleNamespace(symbol="CSPX"),
                                   order=SimpleNamespace(orderType="MKT"))]
        bot = _bot(execute=execute, open_orders=pending)
        bot._execute_live([_opp("CSPX")], _results())
        execute.assert_not_called()

    def test_open_orders_failure_does_not_block_entries(self):
        execute = MagicMock(return_value=OrderResult(success=True, order_id=4))
        bot = _bot(execute=execute)
        bot.engine.order_manager = SimpleNamespace(
            get_open_orders=MagicMock(side_effect=RuntimeError("ib down")))
        results = _results()
        bot._execute_live([_opp("CSPX")], results)
        execute.assert_called_once()
        assert results["trades_executed"] == 1

    def test_sweep_survives_reconcile_error_and_stops_after_it(self):
        bot = _bot()
        bot._reconcile_protective_stops = MagicMock(side_effect=RuntimeError("reconcile broke"))
        bot._execute_live([_opp("CMOD")], _results())      # must not raise
        assert bot._reconcile_protective_stops.call_count == 1  # break after first failure

    def test_empty_opportunities_do_nothing(self):
        bot = _bot()
        results = _results()
        bot._execute_live([], results)
        assert results["trades_executed"] == 0
        bot._reconcile_protective_stops.assert_not_called()
