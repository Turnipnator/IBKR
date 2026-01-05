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
from zoneinfo import ZoneInfo

from .connection import ConnectionManager
from .engine import DecisionEngine
from .database import Database
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
    US_EASTERN = ZoneInfo("America/New_York")

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

        # Database for paper trade tracking
        self.db = Database()

        # Telegram notifications
        self.notifier = get_notifier() if enable_telegram else None

        # Track daily summary sent status
        self._last_summary_date: Optional[str] = None

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, stopping bot...")
        self.running = False

    def _is_market_hours(self) -> bool:
        """Check if currently within US market hours (Eastern Time)."""
        # Get current time in US Eastern, regardless of server timezone
        now_et = datetime.now(self.US_EASTERN)

        # Skip weekends
        if now_et.weekday() >= 5:
            return False

        current_time = now_et.time()
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE

    def _check_paper_trades(self) -> int:
        """
        Check open paper trades against current prices.
        Close any that hit stop loss or take profit.

        Returns:
            Number of paper trades closed
        """
        open_trades = self.db.get_open_paper_trades()
        if not open_trades:
            return 0

        closed_count = 0
        logger.info(f"Checking {len(open_trades)} open paper trades...")

        for trade in open_trades:
            symbol = trade['symbol']
            trade_id = trade['id']
            entry_price = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']

            # Get current price
            try:
                df = self.engine.fetcher.get_historical_data(
                    symbol, duration="1 D", bar_size="1 min"
                )
                if df is None or df.empty:
                    continue

                current_price = float(df['close'].iloc[-1])
            except Exception as e:
                logger.warning(f"Could not get price for {symbol}: {e}")
                continue

            # Check stop loss (for BUY trades, price drops below SL)
            if stop_loss and current_price <= stop_loss:
                result = self.db.close_paper_trade(trade_id, current_price, "CLOSED_SL")
                closed_count += 1
                logger.info(f"Paper trade #{trade_id} {symbol} hit STOP LOSS @ ${current_price:.2f}")

                # SET COOLDOWN after stop loss (anti-churning from Binance strategy)
                cooldown_mins = getattr(self.engine.config, 'cooldown_minutes', 20)
                self.db.set_symbol_cooldown(symbol, cooldown_mins, "stop_loss")
                logger.info(f"  {symbol} in cooldown for {cooldown_mins} minutes")

                if self.notifier and self.notifier.enabled:
                    self.notifier.notify_paper_trade_closed(
                        trade_id=trade_id,
                        symbol=symbol,
                        action=trade['action'],
                        quantity=trade['quantity'],
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl_amount=result['pnl_amount'],
                        pnl_percent=result['pnl_percent'],
                        exit_reason="CLOSED_SL",
                    )

            # Check take profit (for BUY trades, price rises above TP)
            elif take_profit and current_price >= take_profit:
                result = self.db.close_paper_trade(trade_id, current_price, "CLOSED_TP")
                closed_count += 1
                logger.info(f"Paper trade #{trade_id} {symbol} hit TAKE PROFIT @ ${current_price:.2f}")

                if self.notifier and self.notifier.enabled:
                    self.notifier.notify_paper_trade_closed(
                        trade_id=trade_id,
                        symbol=symbol,
                        action=trade['action'],
                        quantity=trade['quantity'],
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl_amount=result['pnl_amount'],
                        pnl_percent=result['pnl_percent'],
                        exit_reason="CLOSED_TP",
                    )

            self.connection.ib.sleep(0.5)  # Rate limiting

        return closed_count

    def _send_daily_summary(self):
        """Send daily summary at market close if not already sent."""
        today = datetime.now(self.US_EASTERN).strftime('%Y-%m-%d')

        # Only send once per day
        if self._last_summary_date == today:
            return

        # Get stats from database
        stats = self.db.get_paper_trade_stats()

        if stats['total_trades'] == 0:
            return  # No trades to report

        # Calculate today's stats (would need to enhance DB for this)
        # For now, send overall stats
        if self.notifier and self.notifier.enabled:
            self.notifier.notify_daily_summary(
                date=today,
                trades_opened=stats['open_trades'],
                trades_closed=stats['closed_trades'],
                winning_trades=stats['winning_trades'],
                losing_trades=stats['losing_trades'],
                day_pnl=stats['total_pnl'],  # Would be day-specific ideally
                total_pnl=stats['total_pnl'],
                win_rate=stats['win_rate'],
            )
            self._last_summary_date = today
            logger.info("Daily summary sent")

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

        # Check existing paper trades first (in dry run mode)
        if self.dry_run:
            closed_count = self._check_paper_trades()
            if closed_count > 0:
                logger.info(f"Closed {closed_count} paper trades")

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

            # In dry run mode, save as paper trade for tracking
            if self.dry_run:
                # Only open paper trade if we don't already have one for this symbol
                if not self.db.has_open_paper_trade(opp.symbol):
                    trade_id = self.db.save_paper_trade(
                        symbol=opp.symbol,
                        action=opp.decision.value,
                        quantity=opp.position_size,
                        entry_price=opp.current_price,
                        stop_loss=opp.stop_loss_price,
                        take_profit=opp.take_profit_price,
                        reasons=opp.reasons,
                    )

                    # Increment daily trade count for this symbol (anti-churning)
                    self.db.increment_daily_trade_count(opp.symbol)

                    # Send paper trade opened notification
                    if self.notifier and self.notifier.enabled:
                        self.notifier.notify_paper_trade_opened(
                            trade_id=trade_id,
                            symbol=opp.symbol,
                            action=opp.decision.value,
                            quantity=opp.position_size,
                            entry_price=opp.current_price,
                            stop_loss=opp.stop_loss_price,
                            take_profit=opp.take_profit_price,
                        )
                else:
                    logger.info(f"  Skipping {opp.symbol} - already have open paper trade")
                    # Still send opportunity notification
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
            else:
                # In live mode, just send opportunity notification
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
                now_et = datetime.now(self.US_EASTERN)

                # Check if near market close (3:55 PM ET) - send daily summary
                if now_et.hour == 15 and now_et.minute >= 55:
                    self._send_daily_summary()

                # Check market hours
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
