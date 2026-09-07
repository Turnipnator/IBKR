"""
Mid-login guard, state-based 2FA check (fix for the 2026-09-07 blind spot).

The guard added in 86ae98a asked "any IBC login marker in the last 300 s?".
Once IBC opens the 2FA dialog it logs nothing until the dialog times out
(4-15 min observed overnight 2026-09-06/07), so after five quiet minutes the
guard saw an empty window and let the gateway restart — killing the push the
user was about to approve. Twice that morning (07:18:47, 07:34:27).

These tests replay the real IBC log lines from that morning against the real
GatewayMonitor, with the Docker API mocked by request path AND by the `since`
parameter, so the 300 s "recent" window and the 30 min "lookback" window can
be answered differently.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.gateway_monitor import (
    GatewayMonitor,
    LOGIN_ACTIVITY_WINDOW,
    TWOFA_DIALOG_LOOKBACK,
)

OLD_START = "2026-01-01T00:00:00.000000000Z"  # long-running container

# --- real ib-gateway log excerpts, 2026-09-07 ------------------------------

# 07:13:42-07:13:43: the login that was still waiting when the 07:18:47
# restart went through. After the last line IBC wrote NOTHING for 5+ min.
DIALOG_OPEN_0713 = b"""
2026-09-07 07:13:42:199 IBC: Login attempt: 1
2026-09-07 07:13:42:238 IBC: Click button: Log In
2026-09-07 07:13:42:857 IBC: detected frame entitled: Authenticating...; event=Opened
2026-09-07 07:13:43:129 IBC: detected dialog entitled: Second Factor Authentication; event=Opened
2026-09-07 07:13:43:130 IBC: Second Factor Authentication initiated
2026-09-07 07:13:43:130 IBC: detected dialog entitled: Second Factor Authentication; event=Activated
2026-09-07 07:13:43:131 IBC: detected dialog entitled: Second Factor Authentication; event=Focused
"""

# 07:28:42-07:28:47: same shape, the push killed by the 07:34:27 restart.
DIALOG_OPEN_0728 = b"""
2026-09-07 07:28:42:801 IBC: Login attempt: 1
2026-09-07 07:28:47:412 IBC: detected dialog entitled: Second Factor Authentication; event=Opened
2026-09-07 07:28:47:413 IBC: Second Factor Authentication initiated
"""

# 00:08:02 -> 00:12:10 overnight: dialog timed out (Closed), then IBC went
# on to the next attempt. If the log then goes quiet for 5 min, the gateway
# really is stuck and a restart is the right call.
DIALOG_TIMED_OUT = b"""
2026-09-07 00:08:02:784 IBC: detected dialog entitled: Second Factor Authentication; event=Opened
2026-09-07 00:08:02:784 IBC: Second Factor Authentication initiated
2026-09-07 00:12:10:260 IBC: detected dialog entitled: Second Factor Authentication; event=Closed
"""

# 07:44:27 -> 07:44:36: the user approved; login completed.
DIALOG_APPROVED = b"""
2026-09-07 07:44:27:535 IBC: detected dialog entitled: Second Factor Authentication; event=Opened
2026-09-07 07:44:27:536 IBC: Second Factor Authentication initiated
2026-09-07 07:44:32:335 IBC: detected dialog entitled: Second Factor Authentication; event=Closed
2026-09-07 07:44:36:497 IBC: Login has completed
"""

# 07:39:27 -> 07:40:07: IBC flickers the dialog open/closed within 300 ms
# (IBKR refusing to push), gives up with exit 1111, and restarts itself.
FLICKER_THEN_IBC_RESTART = b"""
2026-09-07 07:39:27:522 IBC: detected dialog entitled: Second Factor Authentication; event=Opened
2026-09-07 07:39:27:523 IBC: Second Factor Authentication initiated
2026-09-07 07:39:27:830 IBC: detected dialog entitled: Second Factor Authentication; event=Closed
2026-09-07 07:39:27:830 IBC: If login has not completed, IBC will exit in 60 seconds
2026-09-07 07:40:04:078 IBC: Exiting with exit code=1111
IBC returned exit status 87
2026-09-07 07:40:04:750 IBC: Starting session: will exit if login dialog is not displayed within 60 seconds
2026-09-07 07:40:07:721 IBC: Login attempt: 1
"""

# Dialog opened, then IBC exited/restarted WITHOUT ever logging Closed.
OPENED_THEN_IBC_RESTART_NO_CLOSE = b"""
2026-09-07 06:14:50:100 IBC: detected dialog entitled: Second Factor Authentication; event=Opened
2026-09-07 06:15:51:000 IBC: Exiting with exit code=1111
2026-09-07 06:15:52:000 IBC: Starting session: will exit if login dialog is not displayed within 60 seconds
"""

# Several overnight cycles; the LAST event is an Opened with nothing after.
OVERNIGHT_CYCLES_LAST_OPEN = DIALOG_TIMED_OUT + b"""
2026-09-07 00:12:15:358 IBC: Login attempt: 3
2026-09-07 00:12:30:199 IBC: detected dialog entitled: Second Factor Authentication; event=Opened
2026-09-07 00:12:30:199 IBC: Second Factor Authentication initiated
2026-09-07 00:27:16:559 IBC: detected dialog entitled: Second Factor Authentication; event=Closed
2026-09-07 00:27:21:664 IBC: Login attempt: 4
2026-09-07 00:28:20:339 IBC: detected dialog entitled: Second Factor Authentication; event=Opened
2026-09-07 00:28:20:339 IBC: Second Factor Authentication initiated
"""


def _resp(status: int, body: bytes = b""):
    r = MagicMock()
    r.status = status
    r.read.return_value = body
    return r


def _wire(mock_conn, *, recent=b"", lookback=b"", lookback_status=200,
          restart_status=204, started=OLD_START):
    """Route the mocked Docker API by path and, for /logs, by `since`.

    A `since` within LOGIN_ACTIVITY_WINDOW (+ slack) of now is the 300 s
    "recent" fetch; anything older is the TWOFA_DIALOG_LOOKBACK fetch.
    """
    def getresponse():
        _method, path = mock_conn.request.call_args[0]
        if path.endswith("/json"):
            return _resp(200, b'{"State": {"StartedAt": "%s"}}' % started.encode())
        if "/logs?" in path:
            since = int(path.rsplit("since=", 1)[1])
            age = int(time.time()) - since
            if age <= LOGIN_ACTIVITY_WINDOW + 30:
                return _resp(200, recent)
            assert age >= TWOFA_DIALOG_LOOKBACK - 30, f"unexpected since age {age}"
            return _resp(lookback_status, lookback)
        return _resp(restart_status)

    mock_conn.getresponse.side_effect = getresponse


def _log_fetches(mock_conn):
    return [c for c in mock_conn.request.call_args_list if "/logs?" in c[0][1]]


def _posts(mock_conn):
    return [c for c in mock_conn.request.call_args_list if c[0][0] == "POST"]


# ============================================================
# Pure state helper
# ============================================================


class TestTwofaDialogOpen:
    def test_empty_log_is_not_open(self):
        assert GatewayMonitor._twofa_dialog_open("") is False

    def test_opened_with_nothing_after_is_open(self):
        assert GatewayMonitor._twofa_dialog_open(DIALOG_OPEN_0713.decode()) is True

    def test_opened_then_closed_is_not_open(self):
        assert GatewayMonitor._twofa_dialog_open(DIALOG_TIMED_OUT.decode()) is False

    def test_opened_then_login_completed_is_not_open(self):
        assert GatewayMonitor._twofa_dialog_open(DIALOG_APPROVED.decode()) is False

    def test_opened_then_ibc_exit_without_close_is_not_open(self):
        assert GatewayMonitor._twofa_dialog_open(
            OPENED_THEN_IBC_RESTART_NO_CLOSE.decode()) is False

    def test_flicker_is_not_open(self):
        assert GatewayMonitor._twofa_dialog_open(
            FLICKER_THEN_IBC_RESTART.decode()) is False

    def test_last_of_several_cycles_open(self):
        assert GatewayMonitor._twofa_dialog_open(
            OVERNIGHT_CYCLES_LAST_OPEN.decode()) is True

    def test_frame_headers_and_noise_do_not_matter(self):
        # Docker multiplexes 8-byte binary frame headers; substring logic
        # must survive them (decoded with errors='replace' upstream).
        noisy = "\x01\x00\x00\x00\x00\x00\x00\x50" + DIALOG_OPEN_0713.decode()
        assert GatewayMonitor._twofa_dialog_open(noisy) is True


# ============================================================
# login_in_progress() / restart_gateway() through the mocked Docker API
# ============================================================


class TestGuardStateBased:

    @patch("src.gateway_monitor.UnixHTTPConnection")
    @patch("src.gateway_monitor.time.sleep")
    def test_replay_0718_open_dialog_outside_recent_window_blocks_restart(
            self, mock_sleep, mock_conn_cls):
        """2026-09-07 07:18:47: recent 300 s empty, dialog opened 07:13:43.

        Old guard: empty recent window -> None -> restart (killed the push).
        New guard: lookback shows the dialog still open -> skip, cap intact.
        """
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=b"", lookback=DIALOG_OPEN_0713)

        notifier = MagicMock()
        notifier.enabled = True
        monitor = GatewayMonitor(notifier=notifier)

        assert monitor.restart_gateway() is False
        assert monitor.restarts_today == 0
        assert _posts(mock_conn) == []
        notifier.notify_error.assert_called_once()
        msg = notifier.notify_error.call_args[0][0]
        assert "approve" in msg.lower()
        assert "still open" in msg

    @patch("src.gateway_monitor.UnixHTTPConnection")
    def test_replay_0734_second_leak(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=b"", lookback=DIALOG_OPEN_0728)

        reason = GatewayMonitor().login_in_progress()

        assert reason is not None
        assert "2FA push outstanding" in reason
        assert len(_log_fetches(mock_conn)) == 2  # recent, then lookback

    @patch("src.gateway_monitor.UnixHTTPConnection")
    @patch("src.gateway_monitor.time.sleep")
    def test_dialog_timed_out_then_silence_allows_restart(
            self, mock_sleep, mock_conn_cls):
        """Closed dialog + 5 quiet minutes = wedged gateway; restart is right."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=b"", lookback=DIALOG_TIMED_OUT)

        monitor = GatewayMonitor()
        assert monitor.login_in_progress() is None
        assert monitor.restart_gateway() is True
        assert monitor.restarts_today == 1
        assert len(_posts(mock_conn)) == 1

    @patch("src.gateway_monitor.UnixHTTPConnection")
    def test_login_completed_after_dialog_allows_restart(self, mock_conn_cls):
        """Approved + logged in, but API dead for 5 min -> not a 2FA wait."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=b"", lookback=DIALOG_APPROVED)

        assert GatewayMonitor().login_in_progress() is None

    @patch("src.gateway_monitor.UnixHTTPConnection")
    def test_ibc_restart_without_close_clears_dialog_state(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=b"", lookback=OPENED_THEN_IBC_RESTART_NO_CLOSE)

        assert GatewayMonitor().login_in_progress() is None

    @patch("src.gateway_monitor.UnixHTTPConnection")
    def test_overnight_cycles_last_event_open_blocks(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=b"", lookback=OVERNIGHT_CYCLES_LAST_OPEN)

        reason = GatewayMonitor().login_in_progress()
        assert reason and "still open" in reason

    @patch("src.gateway_monitor.UnixHTTPConnection")
    def test_flicker_in_recent_window_is_login_in_progress(self, mock_conn_cls):
        """Open/close within 300 ms is not an open dialog, but IBC is clearly
        alive and cycling — the recency check must still hold the restart."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=FLICKER_THEN_IBC_RESTART, lookback=b"")

        reason = GatewayMonitor().login_in_progress()
        assert reason is not None
        assert reason.startswith("login in progress")
        assert len(_log_fetches(mock_conn)) == 1  # lookback not needed

    @patch("src.gateway_monitor.UnixHTTPConnection")
    def test_open_dialog_in_recent_window_short_circuits(self, mock_conn_cls):
        """Unchanged 86ae98a behaviour: an Opened line in the recent window
        is reported as 'dialog open' without a second Docker call."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=DIALOG_OPEN_0713, lookback=b"")

        assert GatewayMonitor().login_in_progress() == "2FA push outstanding (dialog open)"
        assert len(_log_fetches(mock_conn)) == 1

    @patch("src.gateway_monitor.UnixHTTPConnection")
    def test_plain_marker_in_recent_window_unchanged(self, mock_conn_cls):
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=b"IBC: Login attempt: 3", lookback=b"")

        reason = GatewayMonitor().login_in_progress()
        assert reason == f"login in progress ('Login attempt' in last {LOGIN_ACTIVITY_WINDOW}s)"
        assert len(_log_fetches(mock_conn)) == 1

    @patch("src.gateway_monitor.UnixHTTPConnection")
    @patch("src.gateway_monitor.time.sleep")
    def test_quiet_log_both_windows_allows_restart(self, mock_sleep, mock_conn_cls):
        """The weekend-freeze case: logged in for days, data farm dead."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=b"", lookback=b"")

        monitor = GatewayMonitor()
        assert monitor.login_in_progress() is None
        assert monitor.restart_gateway() is True
        assert len(_posts(mock_conn)) == 1

    @patch("src.gateway_monitor.UnixHTTPConnection")
    @patch("src.gateway_monitor.time.sleep")
    def test_lookback_fetch_error_fails_open(self, mock_sleep, mock_conn_cls):
        """A Docker error on the lookback must not block the old restart path."""
        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn
        _wire(mock_conn, recent=b"", lookback=b"boom", lookback_status=500)

        monitor = GatewayMonitor()
        assert monitor.login_in_progress() is None
        assert monitor.restart_gateway() is True
