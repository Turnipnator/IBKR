"""
Gateway Monitor - restarts ib-gateway container when connection is unrecoverable.

Uses the Docker Engine API over Unix socket (no external dependencies).
Requires /var/run/docker.sock mounted into the trading-bot container.
"""

import http.client
import logging
import socket
import time
from datetime import datetime, date

logger = logging.getLogger(__name__)

DOCKER_SOCKET = "/var/run/docker.sock"
GATEWAY_CONTAINER = "ib-gateway"
MAX_RESTARTS_PER_DAY = 3
POST_RESTART_WAIT = 60  # seconds to wait after restart before reconnecting


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
