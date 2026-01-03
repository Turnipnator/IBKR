#!/usr/bin/env python3
"""
Test script for Telegram integration.
Tests notification sending capabilities.

Before running, set environment variables:
    export TELEGRAM_BOT_TOKEN="your_bot_token"
    export TELEGRAM_CHAT_ID="your_chat_id"
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.telegram_bot import TelegramNotifier
from src.config import telegram_config


def test_telegram():
    """Test Telegram notification functionality."""

    print("=" * 60)
    print("TELEGRAM INTEGRATION TEST")
    print("=" * 60)

    # Check configuration
    print("\n[1] Checking configuration...")
    print(f"    Bot Token: {'*' * 10 + telegram_config.bot_token[-5:] if telegram_config.bot_token else 'NOT SET'}")
    print(f"    Chat ID: {telegram_config.chat_id or 'NOT SET'}")
    print(f"    Enabled: {telegram_config.enabled}")

    if not telegram_config.enabled:
        print("\n" + "=" * 60)
        print("TELEGRAM NOT CONFIGURED")
        print("=" * 60)
        print("""
To configure Telegram:

1. Create a bot via @BotFather on Telegram:
   - Message @BotFather
   - Send /newbot
   - Follow prompts to create bot
   - Copy the API token

2. Get your Chat ID:
   - Message @userinfobot on Telegram
   - It will reply with your user ID

3. Set environment variables:
   export TELEGRAM_BOT_TOKEN="your_token_here"
   export TELEGRAM_CHAT_ID="your_chat_id_here"

4. Run this test again
""")
        return False

    # Test notification
    print("\n[2] Creating notifier...")
    notifier = TelegramNotifier()

    print("\n[3] Sending test message...")
    success = notifier.send_sync(
        "\U0001F916 <b>Test Message</b>\n\n"
        "If you see this, Telegram integration is working!\n\n"
        "<code>IBKR Trading Bot</code>"
    )

    if success:
        print("    SUCCESS: Message sent!")
    else:
        print("    FAILED: Could not send message")
        return False

    # Test different notification types
    print("\n[4] Testing notification templates...")

    # Trade opportunity
    print("    Sending trade opportunity notification...")
    notifier.notify_trade_opportunity(
        symbol="AAPL",
        action="BUY",
        quantity=100,
        price=175.50,
        confidence=0.75,
        stop_loss=157.95,
        take_profit=219.38,
        reasons=["Bullish trend (SMA50 > SMA200)", "RSI oversold (28.5)"],
    )

    # Analysis complete
    print("    Sending analysis complete notification...")
    notifier.notify_analysis_complete(
        symbols_analyzed=12,
        opportunities=3,
        trades_executed=0,
        dry_run=True,
    )

    # Bot started
    print("    Sending bot started notification...")
    notifier.notify_bot_started("TEST MODE")

    print("\n" + "=" * 60)
    print("TELEGRAM TEST COMPLETE")
    print("=" * 60)
    print("\nCheck your Telegram for the test messages!")

    return True


if __name__ == "__main__":
    success = test_telegram()
    sys.exit(0 if success else 1)
