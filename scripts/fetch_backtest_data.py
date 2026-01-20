#!/usr/bin/env python3
"""
Fetch 5-minute historical data from IBKR for backtesting.

Usage:
    python scripts/fetch_backtest_data.py              # Fetch 30 days (default)
    python scripts/fetch_backtest_data.py --days 60    # Fetch 60 days
    python scripts/fetch_backtest_data.py --symbols AAPL NVDA  # Specific symbols only

Run this script when connected to IBKR to populate the database with
5-minute bars needed for backtesting.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connection import ConnectionManager
from src.data_fetcher import DataFetcher
from src.database import Database
from src.config import trading_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default symbols: trading universe + SPY for market filter
DEFAULT_SYMBOLS = [
    "SPY",  # Required for market filter
    "GLD", "SLV",  # Precious metals
    "NVDA", "AMD", "GOOGL", "MSFT",  # AI
    "AAPL", "TSLA", "META", "AMZN",  # Tech
]


def fetch_data(days: int = 30, symbols: list[str] = None):
    """
    Fetch 5-minute historical data from IBKR.

    Args:
        days: Number of days of history to fetch
        symbols: List of symbols (uses DEFAULT_SYMBOLS if None)
    """
    symbols = symbols or DEFAULT_SYMBOLS

    logger.info(f"Fetching {days} days of 5-min data for {len(symbols)} symbols")

    # Connect to IBKR
    conn = ConnectionManager()
    if not conn.connect():
        logger.error("Failed to connect to IBKR. Is TWS/Gateway running?")
        return False

    try:
        fetcher = DataFetcher(conn)
        db = Database()

        success_count = 0

        for symbol in symbols:
            logger.info(f"Fetching {symbol}...")

            df = fetcher.get_historical_data(
                symbol,
                duration=f"{days} D",
                bar_size="5 mins",
                use_rth=True,  # Regular trading hours only
            )

            if df is not None and not df.empty:
                db.save_ohlcv(df, symbol)
                logger.info(f"  {symbol}: Saved {len(df)} bars")
                success_count += 1
            else:
                logger.warning(f"  {symbol}: No data returned")

            # Rate limiting
            conn.ib.sleep(1)

        logger.info(f"Done! Fetched data for {success_count}/{len(symbols)} symbols")
        return success_count > 0

    finally:
        conn.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch 5-minute IBKR data for backtesting"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Days of history to fetch (default: 30)"
    )
    parser.add_argument(
        "--symbols", nargs="+",
        help="Specific symbols to fetch (default: all trading symbols + SPY)"
    )

    args = parser.parse_args()

    success = fetch_data(days=args.days, symbols=args.symbols)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
