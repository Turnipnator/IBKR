"""
Tests for DataHealthChecker - probes market data farm and self-heals.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.data_health_checker import (
    DataHealthChecker,
    FAILURE_THRESHOLD,
    PROBE_SYMBOL,
)


def _mk_connection(connected: bool = True):
    conn = MagicMock()
    conn.ib = MagicMock()
    conn.ib.isConnected.return_value = connected
    return conn


def _mk_bars(n: int = 2):
    return [MagicMock() for _ in range(n)]


def _mk_gateway(login_reason=None):
    """Gateway monitor mock. login_reason=None means "not mid-login/2FA".

    A bare MagicMock() returns a truthy mock from login_in_progress(), which
    trips the mid-login guard (86ae98a) and skips the restart path entirely.
    """
    gw = MagicMock()
    gw.login_in_progress.return_value = login_reason
    return gw


class TestProbe:
    def test_probe_returns_true_on_success(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = _mk_bars(2)

        checker = DataHealthChecker(conn, MagicMock())
        assert checker.probe() is True
        assert checker.consecutive_failures == 0

    def test_probe_returns_false_on_empty_bars(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = []

        checker = DataHealthChecker(conn, MagicMock())
        assert checker.probe() is False
        assert checker.consecutive_failures == 1

    def test_probe_returns_false_on_exception(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.side_effect = TimeoutError("timeout")

        checker = DataHealthChecker(conn, MagicMock())
        assert checker.probe() is False
        assert checker.consecutive_failures == 1

    def test_probe_counts_as_failure_when_disconnected(self):
        conn = _mk_connection(connected=False)

        checker = DataHealthChecker(conn, MagicMock())
        assert checker.probe() is False
        assert checker.consecutive_failures == 1
        # Shouldn't try to fetch data when disconnected
        conn.ib.reqHistoricalData.assert_not_called()

    def test_probe_resets_counter_on_success(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = _mk_bars(2)

        checker = DataHealthChecker(conn, MagicMock())
        checker._consecutive_failures = 5  # pretend history of failures

        assert checker.probe() is True
        assert checker.consecutive_failures == 0

    def test_probe_uses_spy_symbol(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = _mk_bars(2)

        checker = DataHealthChecker(conn, MagicMock())
        checker.probe()

        contract = conn.ib.reqHistoricalData.call_args[0][0]
        assert contract.symbol == PROBE_SYMBOL


class TestShouldProbe:
    def test_should_probe_initially(self):
        checker = DataHealthChecker(MagicMock(), MagicMock())
        assert checker.should_probe() is True

    def test_should_not_probe_immediately_after_probe(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = _mk_bars(2)
        checker = DataHealthChecker(conn, MagicMock(), probe_interval_sec=300)
        checker.probe()
        assert checker.should_probe() is False

    def test_should_probe_after_interval_elapsed(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = _mk_bars(2)
        checker = DataHealthChecker(conn, MagicMock(), probe_interval_sec=300)
        checker.probe()
        # Pretend the probe happened 6 minutes ago
        checker._last_probe_time = datetime.now() - timedelta(seconds=400)
        assert checker.should_probe() is True


class TestCheckAndHeal:
    def test_healthy_no_recovery(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = _mk_bars(2)
        gw = _mk_gateway()

        checker = DataHealthChecker(conn, gw)
        result = checker.check_and_heal()

        assert result is True
        gw.restart_gateway.assert_not_called()
        conn.disconnect.assert_not_called()

    def test_below_threshold_no_recovery(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = []  # empty = failure
        gw = _mk_gateway()

        checker = DataHealthChecker(conn, gw)
        # First failure only - not over threshold
        result = checker.check_and_heal()

        assert result is False
        assert checker.consecutive_failures == 1
        gw.restart_gateway.assert_not_called()

    def test_at_threshold_triggers_recovery(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = []  # every probe fails
        gw = _mk_gateway()
        gw.restart_gateway.return_value = True

        # After restart, connection reconnects successfully
        conn.ensure_connected.return_value = True

        checker = DataHealthChecker(conn, gw)
        # Fail up to threshold
        for _ in range(FAILURE_THRESHOLD - 1):
            checker.check_and_heal()
        assert gw.restart_gateway.call_count == 0

        # This call crosses the threshold
        result = checker.check_and_heal()

        gw.restart_gateway.assert_called_once()
        conn.disconnect.assert_called_once()
        conn.ensure_connected.assert_called_once()
        assert checker.consecutive_failures == 0  # reset after successful heal
        assert result is True

    def test_recovery_aborts_when_gateway_restart_fails(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = []
        gw = _mk_gateway()
        gw.restart_gateway.return_value = False  # hit daily limit, etc

        checker = DataHealthChecker(conn, gw)
        for _ in range(FAILURE_THRESHOLD):
            result = checker.check_and_heal()

        gw.restart_gateway.assert_called_once()
        # Shouldn't try to disconnect/reconnect if gateway restart failed
        conn.disconnect.assert_not_called()
        assert result is False

    def test_recovery_notifies_on_failure(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = []
        gw = _mk_gateway()
        gw.restart_gateway.return_value = True
        conn.ensure_connected.return_value = True

        notifier = MagicMock()
        notifier.enabled = True

        checker = DataHealthChecker(conn, gw, notifier=notifier)
        for _ in range(FAILURE_THRESHOLD):
            checker.check_and_heal()

        notifier.notify_error.assert_called_once()
        assert "data farm" in notifier.notify_error.call_args[0][0].lower()

    def test_recovery_notifies_on_successful_heal(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = []
        gw = _mk_gateway()
        gw.restart_gateway.return_value = True
        conn.ensure_connected.return_value = True

        notifier = MagicMock()
        notifier.enabled = True

        checker = DataHealthChecker(conn, gw, notifier=notifier)
        for _ in range(FAILURE_THRESHOLD):
            checker.check_and_heal()

        # Should have sent a "recovery successful" message via send_sync
        notifier.send_sync.assert_called_once()
        assert "recovery" in notifier.send_sync.call_args[0][0].lower()

    def test_recovery_failed_reconnect(self):
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = []
        gw = _mk_gateway()
        gw.restart_gateway.return_value = True
        conn.ensure_connected.return_value = False  # reconnect failed

        checker = DataHealthChecker(conn, gw)
        for _ in range(FAILURE_THRESHOLD):
            result = checker.check_and_heal()

        gw.restart_gateway.assert_called_once()
        conn.disconnect.assert_called_once()
        assert result is False
        # Counter NOT reset since we didn't heal
        assert checker.consecutive_failures == FAILURE_THRESHOLD

    def test_no_restart_while_gateway_mid_login(self):
        """Mid-login/2FA guard: at threshold the checker must NOT restart the
        gateway (that would kill the outstanding push — the 2026-08-17
        incident) and must NOT send the misleading "restarting to recover"
        alert; it nags via the monitor's rate-limited 2FA path instead."""
        conn = _mk_connection()
        conn.ib.reqHistoricalData.return_value = []
        gw = _mk_gateway(login_reason="2FA push outstanding (dialog open)")

        notifier = MagicMock()
        notifier.enabled = True

        checker = DataHealthChecker(conn, gw, notifier=notifier)
        for _ in range(FAILURE_THRESHOLD):
            result = checker.check_and_heal()

        assert result is False
        gw.restart_gateway.assert_not_called()
        conn.disconnect.assert_not_called()
        notifier.notify_error.assert_not_called()
        gw._notify_awaiting_2fa.assert_called_once_with(
            "2FA push outstanding (dialog open)"
        )
        # Failures keep accumulating so recovery fires once login clears
        assert checker.consecutive_failures == FAILURE_THRESHOLD
