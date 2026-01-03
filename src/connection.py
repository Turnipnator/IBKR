"""
IBKR Connection Manager with auto-reconnect capability.
"""

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
import time
from typing import Optional, Callable
from ib_insync import IB, util

from .config import ibkr_config, IBKRConfig

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages IBKR connection with automatic reconnection.

    Usage:
        cm = ConnectionManager()
        cm.connect()

        # Use cm.ib for API calls
        accounts = cm.ib.managedAccounts()

        # Disconnect when done
        cm.disconnect()
    """

    def __init__(self, config: Optional[IBKRConfig] = None):
        self.config = config or ibkr_config
        self.ib = IB()
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds
        self._on_connect_callbacks: list[Callable] = []
        self._on_disconnect_callbacks: list[Callable] = []

        # Set up disconnect handler
        self.ib.disconnectedEvent += self._on_disconnect

    def connect(self) -> bool:
        """
        Connect to IBKR TWS/Gateway.

        Returns:
            True if connection successful, False otherwise.
        """
        if self._connected and self.ib.isConnected():
            logger.info("Already connected to IBKR")
            return True

        try:
            logger.info(
                f"Connecting to IBKR at {self.config.host}:{self.config.port} "
                f"(clientId={self.config.client_id})"
            )
            self.ib.connect(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly,
            )
            self._connected = True
            self._reconnect_attempts = 0

            logger.info(f"Connected to IBKR (Server v{self.ib.client.serverVersion()})")

            # Fire connect callbacks
            for callback in self._on_connect_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in connect callback: {e}")

            return True

        except ConnectionRefusedError:
            logger.error(
                "Connection refused. Ensure TWS/Gateway is running "
                "with API connections enabled."
            )
            return False

        except TimeoutError:
            logger.error(
                f"Connection timed out after {self.config.timeout}s. "
                "Check TWS/Gateway is running and port is correct."
            )
            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib.isConnected():
            logger.info("Disconnecting from IBKR")
            self.ib.disconnect()
        self._connected = False

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to IBKR with exponential backoff.

        Returns:
            True if reconnection successful, False if max attempts reached.
        """
        while self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))

            logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/"
                f"{self._max_reconnect_attempts} in {delay}s..."
            )
            time.sleep(delay)

            if self.connect():
                return True

        logger.error(
            f"Failed to reconnect after {self._max_reconnect_attempts} attempts"
        )
        return False

    def ensure_connected(self) -> bool:
        """
        Ensure connection is active, reconnecting if necessary.

        Returns:
            True if connected (or reconnected), False otherwise.
        """
        if self.ib.isConnected():
            return True

        logger.warning("Connection lost, attempting to reconnect...")
        self._reconnect_attempts = 0
        return self.reconnect()

    def _on_disconnect(self):
        """Handle disconnect event."""
        self._connected = False
        logger.warning("Disconnected from IBKR")

        for callback in self._on_disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in disconnect callback: {e}")

    def on_connect(self, callback: Callable):
        """Register a callback for connection events."""
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable):
        """Register a callback for disconnection events."""
        self._on_disconnect_callbacks.append(callback)

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self.ib.isConnected()

    def get_accounts(self) -> list[str]:
        """Get list of managed accounts."""
        if not self.ensure_connected():
            return []
        return self.ib.managedAccounts()

    def get_account_summary(self) -> dict:
        """Get account summary as a dictionary."""
        if not self.ensure_connected():
            return {}

        summary = {}
        for item in self.ib.accountSummary():
            summary[item.tag] = {
                "value": item.value,
                "currency": item.currency,
            }
        return summary


# Singleton instance for convenience
_connection_manager: Optional[ConnectionManager] = None


def get_connection() -> ConnectionManager:
    """Get or create the global connection manager."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager
