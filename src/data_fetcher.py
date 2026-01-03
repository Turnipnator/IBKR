"""
Historical data fetcher for IBKR.
Fetches OHLCV data for symbols and stores in database.
"""

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from ib_insync import Stock, util

from .connection import ConnectionManager, get_connection
from .config import trading_config

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches historical market data from IBKR.

    Usage:
        fetcher = DataFetcher()
        df = fetcher.get_historical_data("AAPL", duration="1 M", bar_size="1 day")
    """

    # Valid bar sizes for IBKR API
    VALID_BAR_SIZES = [
        "1 secs", "5 secs", "10 secs", "15 secs", "30 secs",
        "1 min", "2 mins", "3 mins", "5 mins", "10 mins", "15 mins", "20 mins", "30 mins",
        "1 hour", "2 hours", "3 hours", "4 hours", "8 hours",
        "1 day", "1 week", "1 month",
    ]

    # Valid duration strings
    VALID_DURATIONS = [
        "60 S", "120 S", "1800 S",  # Seconds
        "1 D", "2 D", "5 D", "10 D",  # Days
        "1 W", "2 W", "3 W", "4 W",  # Weeks
        "1 M", "2 M", "3 M", "6 M",  # Months
        "1 Y", "2 Y",  # Years
    ]

    def __init__(self, connection: Optional[ConnectionManager] = None):
        self.connection = connection or get_connection()

    def get_historical_data(
        self,
        symbol: str,
        duration: str = "1 M",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            duration: How far back to fetch (e.g., "1 M", "1 Y", "30 D")
            bar_size: Bar size (e.g., "1 day", "1 hour", "5 mins")
            what_to_show: Data type - TRADES, MIDPOINT, BID, ASK
            use_rth: Use regular trading hours only
            exchange: Exchange (SMART for best routing)
            currency: Currency (USD, GBP, etc.)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, average, barCount
            Returns None if fetch fails.
        """
        if not self.connection.ensure_connected():
            logger.error("Cannot fetch data: not connected to IBKR")
            return None

        try:
            # Create contract
            contract = Stock(symbol, exchange, currency)

            # Qualify the contract (fills in conId and other details)
            qualified = self.connection.ib.qualifyContracts(contract)
            if not qualified:
                logger.error(f"Could not qualify contract for {symbol}")
                return None

            contract = qualified[0]
            logger.info(f"Fetching {duration} of {bar_size} bars for {symbol}")

            # Request historical data
            bars = self.connection.ib.reqHistoricalData(
                contract,
                endDateTime="",  # Empty = now
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )

            if not bars:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Convert to DataFrame
            df = util.df(bars)

            # Add symbol column
            df["symbol"] = symbol

            # Rename date column for consistency
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def get_multiple_symbols(
        self,
        symbols: list[str],
        duration: str = "1 M",
        bar_size: str = "1 day",
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of stock tickers
            duration: How far back to fetch
            bar_size: Bar size
            **kwargs: Additional arguments passed to get_historical_data

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}

        for symbol in symbols:
            df = self.get_historical_data(
                symbol, duration=duration, bar_size=bar_size, **kwargs
            )
            if df is not None:
                results[symbol] = df

            # Small delay to avoid rate limiting
            self.connection.ib.sleep(0.5)

        logger.info(f"Fetched data for {len(results)}/{len(symbols)} symbols")
        return results

    def get_all_universe(
        self,
        duration: str = "1 M",
        bar_size: str = "1 day",
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical data for all symbols in the trading universe.

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        all_symbols = []
        for sector_symbols in trading_config.symbols.values():
            all_symbols.extend(sector_symbols)

        return self.get_multiple_symbols(
            all_symbols, duration=duration, bar_size=bar_size, **kwargs
        )

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest closing price for a symbol.
        Uses historical data to avoid market data subscription requirements.

        Returns:
            Latest close price or None if unavailable.
        """
        df = self.get_historical_data(symbol, duration="2 D", bar_size="1 day")
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """
        Get latest closing prices for multiple symbols.

        Returns:
            Dictionary mapping symbol to price.
        """
        prices = {}
        for symbol in symbols:
            price = self.get_latest_price(symbol)
            if price is not None:
                prices[symbol] = price
            self.connection.ib.sleep(0.3)
        return prices
