"""
Gateway Monitor - restarts ib-gateway container when connection is unrecoverable.

Uses the Docker Engine API over Unix socket (no external dependencies).
Requires /var/run/docker.sock mounted into the trading-bot container.
"""

import http.client
import logging
import socket
import time
from datetime import datetime, date, timezone

logger = logging.getLogger(__name__)

DOCKER_SOCKET = "/var/run/docker.sock"
GATEWAY_CONTAINER = "ib-gateway"
MAX_RESTARTS_PER_DAY = 3
POST_RESTART_WAIT = 60  # seconds to wait after restart before reconnecting

# --- Mid-login guard -------------------------------------------------------
# IBC (inside ib-gateway) drives the login itself and, when the 2FA push isn't
# approved within ~215s, re-logs-in and pushes AGAIN — every few minutes, for
# as long as it takes. A gateway that is mid-login is therefore alive and
# self-healing; restarting it only invalidates the push the user is trying to
# approve. That is exactly what happened on Monday 2026-08-17: the Sunday-night
# auto-restart needed 2FA, the data probe restarted the gateway at 07:02, 07:06
# and 07:09 (the whole daily cap), each restart killed the outstanding push,
# and the user "kept failing" approvals that were already stale. Restart is for
# a WEDGED gateway (logged in, data farm dead — the weekend-freeze case), not
# for one that's asking for 2FA. So: if the gateway's own log shows login-flow
# activity within LOGIN_ACTIVITY_WINDOW seconds, or the container only just
# started, skip the restart, don't count it against the daily cap, and tell the
# user to approve the push instead (rate-limited so a 5-min probe loop doesn't
# spam Telegram).
LOGIN_ACTIVITY_WINDOW = 300      # seconds of gateway log to inspect
FRESH_START_GRACE = 180          # container younger than this = still booting/logging in
TWOFA_NOTIFY_INTERVAL = 900      # at most one "approve 2FA" nag per 15 min
LOGIN_MARKERS = (
    "Second Factor Authentication",           # dialog Opened/Closed, "initiated"
    "Re-login after second factor",           # IBC re-login after push timeout
    "Login attempt",                          # IBC "Login attempt: N"
    "Click button: Log In",
    "Authenticating",                         # "Authenticating..." frame
    "Login has completed",                    # logged in, API not yet up
    "Starting application",
)


class UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP connection over a Unix domain socket."""

    def __init__(self, socket_path: str, timeout: int = 30):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


class GatewayMonitor:
    """
    Monitors IB Gateway health and restarts the container when needed.

    Tracks daily restart count to prevent restart loops.
    Sends Telegram alerts on restart attempts.
    """

    def __init__(self, notifier=None):
        self.notifier = notifier
        self._restart_count = 0
        self._restart_date: date | None = None
        self._last_2fa_notify: float = 0.0

    # ---- mid-login guard --------------------------------------------------

    def _gateway_recent_log(self, window_s: int) -> str:
        """Last `window_s` seconds of the gateway container's stdout/stderr.

        Docker multiplexes the stream with 8-byte binary frame headers when the
        container has no TTY; we only substring-match IBC markers, so decoding
        with errors='replace' and leaving the headers in place is fine.
        """
        since = int(time.time()) - window_s
        status, body = self._docker_api(
            "GET",
            f"/containers/{GATEWAY_CONTAINER}/logs"
            f"?stdout=true&stderr=true&since={since}",
        )
        if status != 200:
            raise RuntimeError(f"docker logs HTTP {status}: {body[:200]}")
        return body

    def _gateway_uptime_s(self) -> float | None:
        status, body = self._docker_api(
            "GET", f"/containers/{GATEWAY_CONTAINER}/json"
        )
        if status != 200:
            return None
        import json
        started = json.loads(body).get("State", {}).get("StartedAt", "")
        # e.g. "2026-08-17T07:09:00.295998609Z" — trim ns to µs for fromisoformat
        if not started or started.startswith("0001-"):
            return None
        head, _, frac = started.rstrip("Z").partition(".")
        iso = f"{head}.{frac[:6]}" if frac else head
        started_dt = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - started_dt).total_seconds()

    def login_in_progress(self) -> str | None:
        """Return a reason string if the gateway is booting or mid-login/2FA,
        else None. Never raises — on any Docker/parse error returns None so
        the caller falls back to the plain restart path (fail-open: the guard
        must never make things *worse* than the old behaviour)."""
        try:
            uptime = self._gateway_uptime_s()
            if uptime is not None and uptime < FRESH_START_GRACE:
                return f"gateway container started {uptime:.0f}s ago (still booting)"
            log = self._gateway_recent_log(LOGIN_ACTIVITY_WINDOW)
            hits = [m for m in LOGIN_MARKERS if m in log]
            if hits:
                # Be specific when we can: an OPEN 2FA dialog is the case that
                # matters most to the user.
                if "Second Factor Authentication; event=Opened" in log:
                    return "2FA push outstanding (dialog open)"
                return f"login in progress ({hits[0]!r} in last {LOGIN_ACTIVITY_WINDOW}s)"
            return None
        except Exception as e:
            logger.debug(f"login_in_progress check failed (fail-open): {e}")
            return None

    def _notify_awaiting_2fa(self, reason: str) -> None:
        """Rate-limited Telegram nag: approve the push instead of us restarting."""
        now = time.time()
        if now - self._last_2fa_notify < TWOFA_NOTIFY_INTERVAL:
            return
        self._last_2fa_notify = now
        self._notify(
            f"Gateway restart SKIPPED — {reason}.\n\n"
            f"The gateway is alive and mid-login; restarting it would only kill "
            f"the outstanding 2FA push. Please approve the login on IBKR Mobile "
            f"(a fresh push is re-sent every ~3.5 min until you do).",
            critical=True,
        )

    @property
    def restarts_today(self) -> int:
        if self._restart_date != date.today():
            self._restart_count = 0
            self._restart_date = date.today()
        return self._restart_count

    def _docker_api(self, method: str, path: str) -> tuple[int, str]:
        """Make a request to the Docker Engine API via Unix socket."""
        conn = UnixHTTPConnection(DOCKER_SOCKET)
        try:
            conn.request(method, path)
            response = conn.getresponse()
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
        finally:
            conn.close()

    def restart_gateway(self) -> bool:
        """
        Restart the ib-gateway container.

        Returns True if restart was initiated, False if skipped or failed.
        """
        # Reset daily counter if new day
        if self._restart_date != date.today():
            self._restart_count = 0
            self._restart_date = date.today()

        # Guard first — even at the daily cap the right message is "approve
        # the push", not "manual intervention required".
        reason = self.login_in_progress()
        if reason:
            logger.warning(f"Gateway restart skipped — {reason}; awaiting 2FA/login")
            self._notify_awaiting_2fa(reason)
            return False

        if self._restart_count >= MAX_RESTARTS_PER_DAY:
            msg = (
                f"Gateway restart skipped - already restarted "
                f"{self._restart_count} times today (max {MAX_RESTARTS_PER_DAY}). "
                f"Manual intervention required."
            )
            logger.error(msg)
            self._notify(msg, critical=True)
            return False

        logger.warning(f"Restarting {GATEWAY_CONTAINER} container...")
        self._notify(
            f"Restarting {GATEWAY_CONTAINER} container "
            f"(attempt {self._restart_count + 1}/{MAX_RESTARTS_PER_DAY} today)..."
        )

        try:
            # Docker API: POST /containers/{id}/restart?t=30  (30s grace period)
            # No version prefix = use server default (avoids version-mismatch errors)
            status, body = self._docker_api(
                "POST",
                f"/containers/{GATEWAY_CONTAINER}/restart?t=30",
            )

            if status == 204:
                self._restart_count += 1
                logger.info(
                    f"Gateway restart successful. "
                    f"Waiting {POST_RESTART_WAIT}s for it to become healthy..."
                )
                time.sleep(POST_RESTART_WAIT)
                return True
            else:
                msg = f"Gateway restart failed: HTTP {status} - {body}"
                logger.error(msg)
                self._notify(msg, critical=True)
                return False

        except FileNotFoundError:
            msg = (
                "Docker socket not found at /var/run/docker.sock. "
                "Mount it in docker-compose.yml to enable auto-restart."
            )
            logger.error(msg)
            self._notify(msg, critical=True)
            return False

        except Exception as e:
            msg = f"Gateway restart error: {e}"
            logger.error(msg)
            self._notify(msg, critical=True)
            return False

    def _notify(self, message: str, critical: bool = False):
        """Send a Telegram notification."""
        if self.notifier and self.notifier.enabled:
            if critical:
                self.notifier.notify_error(message, "Gateway Monitor")
            else:
                emoji = "\u26A0\uFE0F"
                self.notifier.send_sync(
                    f"{emoji} <b>Gateway Monitor</b>\n\n{message}\n\n"
                    f"<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
                )
