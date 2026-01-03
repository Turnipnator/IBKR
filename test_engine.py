#!/usr/bin/env python3
"""
Test script for the Decision Engine.
Runs a full analysis cycle in dry-run mode.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.bot import TradingBot, setup_logging

# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_engine():
    """Test the decision engine with a full analysis run."""

    print("=" * 60)
    print("DECISION ENGINE TEST (Dry Run)")
    print("=" * 60)

    # Create bot in dry-run mode
    bot = TradingBot(dry_run=True)

    print("\n[1] Connecting to IBKR...")
    if not bot.connect():
        print("FAILED: Could not connect")
        return False

    print("\n[2] Running full analysis cycle...")
    print("    This will analyze all symbols in the trading universe.")
    print("    No trades will be executed (dry run mode).\n")

    try:
        results = bot.run_once()

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Symbols Analyzed: {results['symbols_analyzed']}")
        print(f"Opportunities Found: {results['opportunities']}")
        print(f"Trades Executed: {results['trades_executed']}")

        if results['opportunities_detail']:
            print("\nOpportunities Detail:")
            for opp in results['opportunities_detail']:
                print(f"\n  {opp['symbol']}:")
                print(f"    Decision: {opp['decision']}")
                print(f"    Price: ${opp['price']:.2f}")
                print(f"    Size: {opp['size']} shares")
                print(f"    Confidence: {opp['strength']:.0%}")
                print(f"    Reasons: {', '.join(opp['reasons'][:2])}")

    except Exception as e:
        logger.error(f"Error during test: {e}")
        raise
    finally:
        print("\n[3] Disconnecting...")
        bot.disconnect()

    print("\n" + "=" * 60)
    print("DECISION ENGINE TEST COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_engine()
    sys.exit(0 if success else 1)
