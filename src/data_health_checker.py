"""
Data-farm Health Checker.

Probes IBKR's market data farm using a known-liquid UCITS symbol (CSPX, the
LSE-listed S&P 500 UCITS — the analog of SPY for our PRIIPs-compliant universe).
When the gateway's TCP port is up but the data farm connections have died
(IBKR Error 162 timeouts), a normal TCP healthcheck doesn't catch it. This
probe does — and triggers gateway restart + force-reconnect when data stops
flowing.
"""

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
from datetime import datetime
from typing import Optional

from .contracts import resolve_contract

logger = logging.getLogger(__name__)

PROBE_SYMBOL = "CSPX"
PROBE_TIMEOUT_SEC = 15
FAILURE_THRESHOLD = 2  # consecutive failures before triggering recovery


class DataHealthChecker:
    """
    Periodically probes market data. Heals via gateway restart on failure.

    Usage:
        checker = DataHealthChecker(connection, gateway_monitor, notifier)
        # In scheduled loop:
        if checker.should_probe():
            checker.check_and_heal()
    """

    def __init__(
        self,
        connection,
        gateway_monitor,
        notifier=None,
        probe_interval_sec: int = 300,  # 5 minutes
    ):
        self.connection = connection
        self.gateway_monitor = gateway_monitor
        self.notifier = notifier
        self.probe_interval_sec = probe_interval_sec
        self._consecutive_failures = 0
        self._last_probe_time: Optional[datetime] = None

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def should_probe(self) -> bool:
        """True if enough time has passed since the last probe."""
        if self._last_probe_time is None:
            return True
        elapsed = (datetime.now() - self._last_probe_time).total_seconds()
        return elapsed >= self.probe_interval_sec

    def probe(self) -> bool:
        """
        Fetch the probe symbol's last 2 daily bars. Returns True if data flows.

        Failures (including "not connected") count toward the threshold,
        since a stale connection presents as a data-farm failure too.
        """
        self._last_probe_time = datetime.now()

        if not self.connection.ib.isConnected():
            logger.warning("Data probe: not connected to IBKR")
            self._consecutive_failures += 1
            return False

        try:
            contract = resolve_contract(PROBE_SYMBOL)
            bars = self.connection.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                timeout=PROBE_TIMEOUT_SEC,
            )

            if bars and len(bars) > 0:
                last_close = getattr(bars[-1], "close", None)
                if self._consecutive_failures > 0:
                    logger.info(
                        f"Data probe recovered after "
                        f"{self._consecutive_failures} failures "
                        f"({PROBE_SYMBOL} close ${last_close})"
                    )
                else:
                    logger.info(f"Data probe OK ({PROBE_SYMBOL} close ${last_close})")
                self._consecutive_failures = 0
                return True

            logger.warning(f"Data probe for {PROBE_SYMBOL} returned no bars")
            self._consecutive_failures += 1
            return False

        except Exception as e:
            logger.warning(f"Data probe failed: {e}")
            self._consecutive_failures += 1
            return False

    def check_and_heal(self) -> bool:
        """
        Run a probe. If consecutive failures >= threshold, restart
        gateway and force bot reconnect.

        Returns True if healthy (or if recovery was attempted and
        succeeded).
        """
        if self.probe():
            return True

        logger.warning(
            f"Data probe failed "
            f"({self._consecutive_failures}/{FAILURE_THRESHOLD})"
        )

        if self._consecutive_failures < FAILURE_THRESHOLD:
            return False

        # Data farm is dead - trigger recovery
        msg = (
            f"Market data probe failed {self._consecutive_failures}x. "
            f"Gateway TCP is up but data farm is unresponsive. "
            f"Restarting gateway to recover..."
        )
        logger.error(msg)
        if self.notifier and self.notifier.enabled:
            self.notifier.notify_error(msg, "Data Health")

        if not self.gateway_monitor.restart_gateway():
            # Restart failed (daily limit hit, docker error, etc)
            # GatewayMonitor has already notified.
            return False

        # Force the bot's stale ib_insync connection to drop so the
        # next ensure_connected() actually reconnects to the fresh gateway.
        logger.info("Forcing bot reconnect to fresh gateway...")
        try:
            self.connection.disconnect()
        except Exception as e:
            logger.warning(f"Error during force disconnect: {e}")

        if self.connection.ensure_connected():
            self._consecutive_failures = 0
            logger.info("Bot reconnected successfully after gateway restart")
            if self.notifier and self.notifier.enabled:
                self.notifier.send_sync(
                    "\u2705 <b>Data Health</b>\n\nRecovery successful - "
                    "gateway restarted and bot reconnected.\n\n"
                    f"<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
                )
            return True

        logger.error("Bot failed to reconnect after gateway restart")
        return False
