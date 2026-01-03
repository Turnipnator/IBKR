#!/usr/bin/env python3
"""
Test script for the data layer.
Fetches historical data and stores it in the database.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.connection import ConnectionManager
from src.data_fetcher import DataFetcher
from src.database import Database
from src.config import trading_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_data_layer():
    """Test the complete data layer: connect, fetch, store, retrieve."""

    print("=" * 60)
    print("DATA LAYER TEST")
    print("=" * 60)

    # Initialize components
    cm = ConnectionManager()
    db = Database()

    # Step 1: Connect
    print("\n[1] Connecting to IBKR...")
    if not cm.connect():
        print("FAILED: Could not connect to IBKR")
        return False

    print(f"    Connected. Accounts: {cm.get_accounts()}")

    # Step 2: Initialize data fetcher
    print("\n[2] Initializing data fetcher...")
    fetcher = DataFetcher(cm)

    # Step 3: Fetch data for a test symbol
    test_symbol = "AAPL"
    print(f"\n[3] Fetching 1 month of daily data for {test_symbol}...")

    df = fetcher.get_historical_data(
        test_symbol,
        duration="1 M",
        bar_size="1 day"
    )

    if df is None or df.empty:
        print(f"    WARNING: No data returned for {test_symbol}")
        print("    (This may be due to market data subscription)")
    else:
        print(f"    Fetched {len(df)} bars")
        print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"    Latest close: ${df['close'].iloc[-1]:.2f}")

        # Step 4: Save to database
        print(f"\n[4] Saving data to database...")
        db.save_ohlcv(df, test_symbol)
        print(f"    Saved {len(df)} records")

        # Step 5: Retrieve from database
        print(f"\n[5] Retrieving data from database...")
        stored_df = db.load_ohlcv(test_symbol, days=30)
        print(f"    Retrieved {len(stored_df)} records")

        if not stored_df.empty:
            print(f"    Latest stored close: ${stored_df['close'].iloc[-1]:.2f}")

    # Step 6: Test fetching multiple symbols
    print("\n[6] Fetching data for multiple symbols...")
    test_symbols = ["NVDA", "MSFT"]
    results = fetcher.get_multiple_symbols(
        test_symbols,
        duration="2 W",
        bar_size="1 day"
    )

    for symbol, data in results.items():
        print(f"    {symbol}: {len(data)} bars, latest ${data['close'].iloc[-1]:.2f}")
        db.save_ohlcv(data, symbol)

    # Step 7: Show stored symbols
    print("\n[7] Symbols in database:")
    for symbol in db.get_symbols_with_data():
        start, end = db.get_data_range(symbol)
        print(f"    {symbol}: {start} to {end}")

    # Cleanup
    print("\n[8] Disconnecting...")
    cm.disconnect()

    print("\n" + "=" * 60)
    print("DATA LAYER TEST COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_data_layer()
    sys.exit(0 if success else 1)
