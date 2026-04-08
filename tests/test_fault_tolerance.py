"""
Tests for fault tolerance features:
- Gateway auto-restart on connection failure
- Immediate Telegram alerting
- Disconnect state in Telegram command responses
"""

import http.client
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.gateway_monitor import GatewayMonitor, MAX_RESTARTS_PER_DAY, DOCKER_SOCKET
from src.connection import ConnectionManager
from src.telegram_bot import TelegramNotifier


# ============================================================
# GatewayMonitor tests
# ============================================================


class TestGatewayMonitor:
    """Tests for the GatewayMonitor class."""

    def _make_monitor(self, notifier=None):
        return GatewayMonitor(notifier=notifier)

    @patch("src.gateway_monitor.UnixHTTPConnection")
    @patch("src.gateway_monitor.time.sleep")
    def test_restart_gateway_success(self, mock_sleep, mock_conn_cls):
        """Successful restart returns True and increments counter."""
        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.read.return_value = b""

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response
        mock_conn_cls.return_value = mock_conn

        monitor = self._make_monitor()
        result = monitor.restart_gateway()

        assert result is True
        assert monitor.restarts_today == 1
        mock_conn.request.assert_called_once_with(
            "POST", f"/v1.24/containers/ib-gateway/restart?t=30"
        )
        mock_sleep.assert_called_once()

    @patch("src.gateway_monitor.UnixHTTPConnection")
    @patch("src.gateway_monitor.time.sleep")
    def test_restart_gateway_api_error(self, mock_sleep, mock_conn_cls):
        """Non-204 response returns False."""
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.read.return_value = b"no such container"

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response
        mock_conn_cls.return_value = mock_conn

        monitor = self._make_monitor()
        result = monitor.restart_gateway()

        assert result is False
        assert monitor.restarts_today == 0

    @patch("src.gateway_monitor.UnixHTTPConnection")
    @patch("src.gateway_monitor.time.sleep")
    def test_restart_respects_daily_limit(self, mock_sleep, mock_conn_cls):
        """Restart is skipped after MAX_RESTARTS_PER_DAY."""
        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.read.return_value = b""

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response
        mock_conn_cls.return_value = mock_conn

        monitor = self._make_monitor()

        # Exhaust the daily limit
        for _ in range(MAX_RESTARTS_PER_DAY):
            assert monitor.restart_gateway() is True

        # Next one should be blocked
        result = monitor.restart_gateway()
        assert result is False
        assert monitor.restarts_today == MAX_RESTARTS_PER_DAY

    @patch("src.gateway_monitor.UnixHTTPConnection")
    def test_restart_docker_socket_missing(self, mock_conn_cls):
        """FileNotFoundError (no socket) returns False and notifies."""
        mock_conn_cls.side_effect = FileNotFoundError("No such file")

        notifier = MagicMock()
        notifier.enabled = True
        monitor = self._make_monitor(notifier=notifier)

        result = monitor.restart_gateway()

        assert result is False
        notifier.notify_error.assert_called_once()
        assert "Docker socket" in notifier.notify_error.call_args[0][0]

    @patch("src.gateway_monitor.UnixHTTPConnection")
    @patch("src.gateway_monitor.time.sleep")
    def test_restart_sends_telegram_notification(self, mock_sleep, mock_conn_cls):
        """Successful restart sends a Telegram alert."""
        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.read.return_value = b""

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response
        mock_conn_cls.return_value = mock_conn

        notifier = MagicMock()
        notifier.enabled = True
        monitor = self._make_monitor(notifier=notifier)

        monitor.restart_gateway()

        # Should have called send_sync for the "restarting..." message
        notifier.send_sync.assert_called_once()
        assert "Restarting" in notifier.send_sync.call_args[0][0]

    @patch("src.gateway_monitor.UnixHTTPConnection")
    @patch("src.gateway_monitor.time.sleep")
    def test_daily_counter_resets_on_new_day(self, mock_sleep, mock_conn_cls):
        """Counter resets when date changes."""
        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.read.return_value = b""

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_response
        mock_conn_cls.return_value = mock_conn

        monitor = self._make_monitor()
        monitor.restart_gateway()
        assert monitor.restarts_today == 1

        # Simulate date change
        monitor._restart_date = date(2020, 1, 1)
        assert monitor.restarts_today == 0


# ============================================================
# ConnectionManager reconnect-failed callback tests
# ============================================================


class TestConnectionReconnectFailed:
    """Tests for on_reconnect_failed callback in ConnectionManager."""

    def _make_cm(self):
        cm = ConnectionManager.__new__(ConnectionManager)
        cm.ib = MagicMock()
        cm.config = MagicMock()
        cm.config.host = "127.0.0.1"
        cm.config.port = 4002
        cm.config.client_id = 1
        cm.config.timeout = 1
        cm.config.readonly = False
        cm._connected = False
        cm._reconnect_attempts = 0
        cm._max_reconnect_attempts = 2  # fewer for fast tests
        cm._reconnect_delay = 0.01  # near-instant
        cm._on_connect_callbacks = []
        cm._on_disconnect_callbacks = []
        cm._on_reconnect_failed_callbacks = []
        cm.ib.disconnectedEvent = MagicMock()
        return cm

    @patch("src.connection.time.sleep")
    def test_reconnect_failure_fires_callback(self, mock_sleep):
        """on_reconnect_failed callbacks fire when reconnect exhausts retries."""
        cm = self._make_cm()
        cm.ib.isConnected.return_value = False
        cm.ib.connect.side_effect = TimeoutError("test")

        callback = MagicMock()
        cm.on_reconnect_failed(callback)

        result = cm.reconnect()

        assert result is False
        callback.assert_called_once()

    @patch("src.connection.time.sleep")
    def test_reconnect_success_does_not_fire_callback(self, mock_sleep):
        """Callbacks don't fire on successful reconnect."""
        cm = self._make_cm()
        cm.ib.isConnected.return_value = False
        # First attempt fails, second succeeds
        cm.ib.connect.side_effect = [TimeoutError("test"), None]

        # After connect() succeeds, isConnected returns True
        def connect_side_effect(*args, **kwargs):
            if cm.ib.connect.call_count == 2:
                cm._connected = True
                return None
            raise TimeoutError("test")

        cm.ib.connect.side_effect = connect_side_effect
        cm.ib.client = MagicMock()
        cm.ib.client.serverVersion.return_value = 178

        callback = MagicMock()
        cm.on_reconnect_failed(callback)

        result = cm.reconnect()

        assert result is True
        callback.assert_not_called()

    @patch("src.connection.time.sleep")
    def test_ensure_connected_retries_after_gateway_restart(self, mock_sleep):
        """ensure_connected tries a second round after callbacks fire."""
        cm = self._make_cm()
        cm.ib.isConnected.return_value = False

        # Mock reconnect to fail first, then succeed
        reconnect_results = [False, True]

        def patched_reconnect():
            if reconnect_results:
                result = reconnect_results.pop(0)
                if not result:
                    # Fire callbacks like the real code does
                    for cb in cm._on_reconnect_failed_callbacks:
                        cb()
                return result
            return False

        cm.reconnect = patched_reconnect

        restart_called = MagicMock()
        cm.on_reconnect_failed(restart_called)

        result = cm.ensure_connected()

        assert result is True
        restart_called.assert_called_once()


# ============================================================
# Telegram disconnect warning tests
# ============================================================


class TestTelegramDisconnectWarning:
    """Tests for disconnect state shown in Telegram command responses."""

    def _make_notifier(self):
        config = MagicMock()
        config.enabled = True
        config.bot_token = "test"
        config.chat_id = "123"
        return TelegramNotifier(config=config)

    def _make_db(self, trades=None):
        db = MagicMock()
        db.get_open_paper_trades.return_value = trades or [
            {
                "id": 1,
                "symbol": "GLD",
                "action": "BUY",
                "quantity": 100,
                "entry_price": 430.0,
                "stop_loss": 400.0,
                "take_profit": None,
                "best_price": 435.0,
            }
        ]
        return db

    def test_positions_shows_disconnect_warning_when_no_prices(self):
        """When price_fetcher returns empty dict, show disconnect banner."""
        notifier = self._make_notifier()
        db = self._make_db()

        # price_fetcher returns empty dict (disconnected)
        response = notifier.process_command("/positions", db, lambda syms: {})

        assert "disconnected" in response.lower()
        assert "GLD" in response

    def test_positions_shows_disconnect_warning_on_exception(self):
        """When price_fetcher raises, show disconnect banner."""
        notifier = self._make_notifier()
        db = self._make_db()

        def failing_fetcher(syms):
            raise ConnectionError("not connected")

        response = notifier.process_command("/positions", db, failing_fetcher)

        assert "disconnected" in response.lower()

    def test_positions_no_warning_when_prices_available(self):
        """When prices are fetched successfully, no disconnect banner."""
        notifier = self._make_notifier()
        db = self._make_db()

        def good_fetcher(syms):
            return {"GLD": 435.0}

        response = notifier.process_command("/positions", db, good_fetcher)

        assert "disconnected" not in response.lower()
        assert "GLD" in response

    def test_positions_no_warning_when_no_fetcher(self):
        """When no price_fetcher is provided, no disconnect banner (off-hours)."""
        notifier = self._make_notifier()
        db = self._make_db()

        response = notifier.process_command("/positions", db, None)

        assert "disconnected" not in response.lower()


# ============================================================
# Bot alert threshold tests
# ============================================================


class TestBotAlertThreshold:
    """Tests for immediate alerting on connection failure."""

    @patch("src.bot.GatewayMonitor")
    @patch("src.bot.get_notifier")
    @patch("src.bot.DecisionEngine")
    @patch("src.bot.ConnectionManager")
    @patch("src.bot.Database")
    def test_alert_on_first_failure(
        self, mock_db, mock_cm, mock_engine, mock_notifier_fn, mock_gw
    ):
        """Bot alerts on the very first connection failure, not the third."""
        from src.bot import TradingBot

        notifier = MagicMock()
        notifier.enabled = True
        mock_notifier_fn.return_value = notifier

        mock_conn = MagicMock()
        mock_conn.ensure_connected.return_value = False
        mock_cm.return_value = mock_conn

        bot = TradingBot(dry_run=True, enable_telegram=True)
        bot.notifier = notifier

        # First failure should trigger alert
        bot.run_once()

        assert bot._consecutive_failures == 1
        notifier.notify_error.assert_called_once()
        assert "Connection failed" in notifier.notify_error.call_args[0][0]
