#!/usr/bin/env python3
"""
Test script for technical analysis indicators.
Fetches data and generates signals for our trading universe.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.connection import ConnectionManager
from src.data_fetcher import DataFetcher
from src.database import Database
from src.indicators import TechnicalAnalyzer, sma, rsi, macd
from src.config import trading_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_indicators():
    """Test technical indicators with real market data."""

    print("=" * 60)
    print("TECHNICAL ANALYSIS TEST")
    print("=" * 60)

    cm = ConnectionManager()
    db = Database()

    # Connect
    print("\n[1] Connecting to IBKR...")
    if not cm.connect():
        print("FAILED: Could not connect")
        return False

    fetcher = DataFetcher(cm)

    # Test symbols - one from each sector
    test_symbols = ["AAPL", "NVDA", "GLD"]

    print(f"\n[2] Fetching 1 year of data for analysis...")

    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"  ANALYZING: {symbol}")
        print(f"{'='*60}")

        # Fetch enough data for 200-day SMA
        df = fetcher.get_historical_data(
            symbol,
            duration="1 Y",
            bar_size="1 day"
        )

        if df is None or df.empty:
            print(f"  WARNING: No data for {symbol}")
            continue

        print(f"  Data points: {len(df)}")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

        # Save to database
        db.save_ohlcv(df, symbol)

        # Run technical analysis
        analyzer = TechnicalAnalyzer(
            df,
            sma_fast=trading_config.sma_fast,
            sma_slow=trading_config.sma_slow,
            rsi_period=trading_config.rsi_period,
            rsi_overbought=trading_config.rsi_overbought,
            rsi_oversold=trading_config.rsi_oversold,
        )

        analyzed_df = analyzer.calculate_all()

        # Show latest indicators
        print(f"\n  Latest Indicators:")
        indicators = analyzer.get_latest_indicators()

        close = indicators.get('close')
        sma50 = indicators.get('sma_50')
        sma200 = indicators.get('sma_200')
        rsi_val = indicators.get('rsi')
        macd_val = indicators.get('macd')
        macd_sig = indicators.get('macd_signal')
        trend = indicators.get('trend')

        print(f"    Close:    ${close:.2f}" if close else "    Close:    N/A")
        print(f"    SMA 50:   ${sma50:.2f}" if sma50 else "    SMA 50:   N/A")
        print(f"    SMA 200:  ${sma200:.2f}" if sma200 else "    SMA 200:  N/A")
        print(f"    RSI:      {rsi_val:.1f}" if rsi_val else "    RSI:      N/A")
        print(f"    MACD:     {macd_val:.4f}" if macd_val else "    MACD:     N/A")
        print(f"    Signal:   {macd_sig:.4f}" if macd_sig else "    Signal:   N/A")
        print(f"    Trend:    {trend}" if trend else "    Trend:    N/A")

        # Generate trading signal
        signal = analyzer.generate_signal(symbol)

        print(f"\n  Trading Signal:")
        print(f"    Action:   {signal.action}")
        print(f"    Strength: {signal.strength:.1%}")
        print(f"    Reasons:")
        for reason in signal.reasons:
            print(f"      - {reason}")

        cm.ib.sleep(0.5)

    # Disconnect
    print(f"\n{'='*60}")
    print("\n[3] Disconnecting...")
    cm.disconnect()

    print("\n" + "=" * 60)
    print("TECHNICAL ANALYSIS TEST COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_indicators()
    sys.exit(0 if success else 1)
