"""
Main Trading Bot - Entry point for automated trading.
"""

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
import time
import signal
import sys
from datetime import datetime, time as dtime
from typing import Optional

from .connection import ConnectionManager
from .engine import DecisionEngine
from .config import ibkr_config, telegram_config
from .telegram_bot import TelegramNotifier, get_notifier

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot that runs the decision engine on a schedule.

    Usage:
        bot = TradingBot(dry_run=True)
        bot.run_once()      # Single analysis
        bot.run_scheduled() # Continuous scheduled runs
    """

    # US Market hours (Eastern Time)
    MARKET_OPEN = dtime(9, 30)
    MARKET_CLOSE = dtime(16, 0)

    def __init__(
        self,
        dry_run: bool = True,
        run_interval_minutes: int = 60,
        enable_telegram: bool = True,
    ):
        """
        Initialize the trading bot.

        Args:
            dry_run: If True, analyze but don't execute trades
            run_interval_minutes: Minutes between analysis runs
            enable_telegram: Enable Telegram notifications
        """
        self.dry_run = dry_run
        self.run_interval = run_interval_minutes * 60  # Convert to seconds
        self.running = False

        self.connection = ConnectionManager()
        self.engine = DecisionEngine(
            connection=self.connection,
            dry_run=dry_run,
        )

        # Telegram notifications
        self.notifier = get_notifier() if enable_telegram else None

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, stopping bot...")
        self.running = False

    def _is_market_hours(self) -> bool:
        """Check if currently within US market hours."""
        now = datetime.now()

        # Skip weekends
        if now.weekday() >= 5:
            return False

        current_time = now.time()
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE

    def connect(self) -> bool:
        """Establish connection to IBKR."""
        logger.info("Connecting to IBKR...")
        return self.connection.connect()

    def disconnect(self):
        """Disconnect from IBKR."""
        logger.info("Disconnecting from IBKR...")
        self.connection.disconnect()

    def run_once(self) -> dict:
        """
        Run a single analysis/trading cycle.

        Returns:
            Dict with results summary
        """
        if not self.connection.ensure_connected():
            logger.error("Failed to connect to IBKR")
            return {"success": False, "error": "Connection failed"}

        logger.info("=" * 50)
        logger.info(f"TRADING BOT RUN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}")
        logger.info("=" * 50)

        # Run analysis
        opportunities = self.engine.run_analysis()

        results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "symbols_analyzed": self.engine.state.symbols_analyzed,
            "opportunities": len(opportunities),
            "trades_executed": 0,
            "opportunities_detail": [],
        }

        # Log opportunities
        for opp in opportunities:
            detail = {
                "symbol": opp.symbol,
                "decision": opp.decision.value,
                "price": opp.current_price,
                "size": opp.position_size,
                "strength": opp.signal.strength,
                "reasons": opp.reasons,
                "stop_loss": opp.stop_loss_price,
                "take_profit": opp.take_profit_price,
            }
            results["opportunities_detail"].append(detail)

            logger.info(
                f"Opportunity: {opp.decision.value} {opp.position_size} {opp.symbol} "
                f"@ ${opp.current_price:.2f} ({opp.signal.strength:.0%})"
            )
            if opp.stop_loss_price:
                logger.info(f"  Stop Loss: ${opp.stop_loss_price:.2f}")
            if opp.take_profit_price:
                logger.info(f"  Take Profit: ${opp.take_profit_price:.2f}")

            # Send Telegram notification for opportunity
            if self.notifier and self.notifier.enabled:
                self.notifier.notify_trade_opportunity(
                    symbol=opp.symbol,
                    action=opp.decision.value,
                    quantity=opp.position_size,
                    price=opp.current_price,
                    confidence=opp.signal.strength,
                    stop_loss=opp.stop_loss_price,
                    take_profit=opp.take_profit_price,
                    reasons=opp.reasons,
                )

        # Execute if not dry run
        if not self.dry_run and opportunities:
            logger.info("Executing trades...")
            order_results = []
            for opp in opportunities:
                result = self.engine.execute_opportunity(opp)
                order_results.append(result)
                if result.success:
                    results["trades_executed"] += 1
                    # Notify trade execution
                    if self.notifier and self.notifier.enabled:
                        self.notifier.notify_trade_executed(
                            symbol=opp.symbol,
                            action=opp.decision.value,
                            quantity=opp.position_size,
                            price=result.fill_price or opp.current_price,
                            reason=", ".join(opp.reasons[:2]),
                        )

        # Send analysis complete notification
        if self.notifier and self.notifier.enabled:
            self.notifier.notify_analysis_complete(
                symbols_analyzed=results["symbols_analyzed"],
                opportunities=results["opportunities"],
                trades_executed=results["trades_executed"],
                dry_run=self.dry_run,
            )

        # Status report
        logger.info("\n" + self.engine.get_status_report())

        return results

    def run_scheduled(self):
        """
        Run the bot on a schedule until stopped.
        """
        self.running = True

        if not self.connect():
            logger.error("Failed to connect, exiting")
            if self.notifier and self.notifier.enabled:
                self.notifier.notify_error("Failed to connect to IBKR", "Bot startup")
            return

        mode = "DRY RUN" if self.dry_run else "LIVE TRADING"
        logger.info(f"Bot started - running every {self.run_interval // 60} minutes")
        logger.info(f"Mode: {mode}")
        logger.info("Press Ctrl+C to stop")

        # Send startup notification
        if self.notifier and self.notifier.enabled:
            self.notifier.notify_bot_started(mode)

        try:
            while self.running:
                # Check market hours (optional - can trade outside hours for some)
                if not self._is_market_hours():
                    logger.info("Outside market hours, waiting...")
                    time.sleep(60)  # Check every minute
                    continue

                # Run analysis
                self.run_once()

                # Wait for next run
                if self.running:
                    logger.info(f"Next run in {self.run_interval // 60} minutes...")
                    for _ in range(self.run_interval):
                        if not self.running:
                            break
                        time.sleep(1)

        except Exception as e:
            logger.error(f"Bot error: {e}")
            if self.notifier and self.notifier.enabled:
                self.notifier.notify_error(str(e), "Bot runtime")
        finally:
            self.disconnect()
            logger.info("Bot stopped")
            if self.notifier and self.notifier.enabled:
                self.notifier.notify_bot_stopped("Scheduled shutdown")


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """Configure logging for the bot."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # Reduce noise from ib_insync
    logging.getLogger("ib_insync").setLevel(logging.WARNING)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="IBKR Trading Bot")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (actually execute trades)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't schedule)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Minutes between runs (default: 60)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/trading.log",
        help="Log file path",
    )

    args = parser.parse_args()

    setup_logging(log_file=args.log_file)

    dry_run = not args.live

    if args.live:
        logger.warning("=" * 50)
        logger.warning("LIVE TRADING MODE - REAL ORDERS WILL BE PLACED")
        logger.warning("=" * 50)
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            logger.info("Cancelled")
            return

    bot = TradingBot(dry_run=dry_run, run_interval_minutes=args.interval)

    if args.once:
        bot.connect()
        bot.run_once()
        bot.disconnect()
    else:
        bot.run_scheduled()


if __name__ == "__main__":
    main()
