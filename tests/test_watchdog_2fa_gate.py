"""
Tests for the loop watchdog's 2FA gate (2026-09-07 follow-up).

That morning the Sunday-night gateway restart sat on an unapproved 2FA push
until 07:44. The mid-login guard (86ae98a + 778e9ca) kept restart_gateway()
from killing the push, but `_check_watchdog` had no such gate: at 07:40:38 it
hit 6 consecutive probe failures and `sys.exit(1)`-ed a perfectly healthy bot
into a 29-restart crash loop against a gateway that could not come up until a
human approved. The watchdog now consults the same `login_in_progress()` guard
and holds off while a 2FA/login is in flight — and still fires (fail-open) if
the guard is missing, broken, or reports nothing.

These drive the REAL `_check_watchdog` on a TradingBot built via `__new__`;
data_health / gateway_monitor are stubs.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.bot import TradingBot
from src.config import trading_config

# Mirrors _check_watchdog's arithmetic at the default 5-min probe interval
# (watchdog_timeout_min=30 -> 6 consecutive failures).
THRESHOLD = max(3, trading_config.watchdog_timeout_min // 5)


def _bot(failures, gateway_monitor=None, with_monitor=True):
    bot = TradingBot.__new__(TradingBot)
    bot.notifier = None
    health = SimpleNamespace(
        consecutive_failures=failures,
        probe_interval_sec=300,
        time_since_last_success=lambda: None,
    )
    if with_monitor:
        health.gateway_monitor = gateway_monitor
    bot.data_health = health
    return bot


class TestWatchdog2FAGate:

    def test_2fa_wait_holds_off_the_self_restart(self, caplog):
        """Replays 2026-09-07 07:40:38: 6 failures with the dialog open must
        NOT sys.exit — the gateway is waiting for a human, not wedged."""
        monitor = SimpleNamespace(login_in_progress=MagicMock(
            return_value="2FA push outstanding (dialog open)"))
        bot = _bot(THRESHOLD, gateway_monitor=monitor)
        with caplog.at_level(logging.WARNING, logger="src.bot"):
            bot._check_watchdog()      # must not raise SystemExit
        monitor.login_in_progress.assert_called_once()
        assert "holding off self-restart" in caplog.text
        assert "2FA push outstanding (dialog open)" in caplog.text

    def test_watchdog_still_fires_when_no_login_in_progress(self):
        """Guard says the gateway is quiet -> genuinely wedged -> old
        behaviour: sys.exit(1) so Docker recreates the container."""
        monitor = SimpleNamespace(login_in_progress=MagicMock(return_value=None))
        bot = _bot(THRESHOLD, gateway_monitor=monitor)
        with pytest.raises(SystemExit):
            bot._check_watchdog()

    def test_watchdog_fails_open_without_a_guard(self):
        """No gateway_monitor on the checker (stale mock / old wiring):
        the watchdog must still fire rather than silently disarm."""
        bot = _bot(THRESHOLD, with_monitor=False)
        with pytest.raises(SystemExit):
            bot._check_watchdog()

    def test_watchdog_fails_open_when_guard_raises(self, caplog):
        """A broken guard (Docker API down, etc.) must neither leak the
        exception into the main loop nor disarm the watchdog."""
        monitor = SimpleNamespace(login_in_progress=MagicMock(
            side_effect=RuntimeError("docker socket gone")))
        bot = _bot(THRESHOLD, gateway_monitor=monitor)
        with caplog.at_level(logging.WARNING, logger="src.bot"):
            with pytest.raises(SystemExit):
                bot._check_watchdog()
        assert "fail-open" in caplog.text

    def test_below_threshold_never_consults_the_guard(self):
        monitor = SimpleNamespace(login_in_progress=MagicMock())
        bot = _bot(THRESHOLD - 1, gateway_monitor=monitor)
        bot._check_watchdog()
        monitor.login_in_progress.assert_not_called()

    def test_zero_failures_short_circuits(self):
        monitor = SimpleNamespace(login_in_progress=MagicMock())
        bot = _bot(0, gateway_monitor=monitor)
        bot._check_watchdog()
        monitor.login_in_progress.assert_not_called()
