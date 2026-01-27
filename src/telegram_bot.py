"""
Telegram bot integration for trade notifications and commands.
"""

import asyncio
import logging
import urllib.request
import urllib.parse
import json
from typing import Optional
from datetime import datetime

from .config import telegram_config, TelegramConfig

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends notifications to Telegram.
    Uses simple HTTP requests for maximum compatibility.

    Usage:
        notifier = TelegramNotifier()
        notifier.send_sync("Hello!")
    """

    TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, config: Optional[TelegramConfig] = None):
        self.config = config or telegram_config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def send_sync(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """Send a message synchronously using HTTP."""
        if not self.enabled:
            logger.debug("Telegram not configured, skipping notification")
            return False

        try:
            url = self.TELEGRAM_API.format(token=self.config.bot_token)

            data = {
                "chat_id": self.config.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            }

            payload = json.dumps(data).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                if result.get("ok"):
                    return True
                else:
                    logger.error(f"Telegram API error: {result}")
                    return False

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    # ==================== Notification Templates ====================

    def notify_trade_executed(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        reason: str = "",
    ) -> bool:
        """Send notification for executed trade."""
        emoji = "\U0001F7E2" if action == "BUY" else "\U0001F534"  # Green/Red circle

        message = f"""
{emoji} <b>Trade Executed</b>

<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Quantity:</b> {quantity:,} shares
<b>Price:</b> ${price:,.2f}
<b>Value:</b> ${quantity * price:,.2f}

<i>{reason}</i>

<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_sync(message.strip())

    def notify_trade_opportunity(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        confidence: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reasons: list[str] = None,
    ) -> bool:
        """Send notification for trade opportunity (dry run)."""
        emoji = "\U0001F4CA"  # Chart emoji

        sl_text = f"${stop_loss:,.2f}" if stop_loss else "N/A"
        tp_text = f"${take_profit:,.2f}" if take_profit else "N/A"
        reasons_text = "\n".join(f"  - {r}" for r in (reasons or [])[:3])

        message = f"""
{emoji} <b>Trade Opportunity</b>

<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Size:</b> {quantity:,} shares @ ${price:,.2f}
<b>Value:</b> ${quantity * price:,.2f}
<b>Confidence:</b> {confidence:.0%}

<b>Stop Loss:</b> {sl_text}
<b>Take Profit:</b> {tp_text}

<b>Reasons:</b>
{reasons_text}

<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_sync(message.strip())

    def notify_analysis_complete(
        self,
        symbols_analyzed: int,
        opportunities: int,
        trades_executed: int,
        dry_run: bool = True,
    ) -> bool:
        """Send notification when analysis completes."""
        mode = "DRY RUN" if dry_run else "LIVE"
        emoji = "\U00002705" if trades_executed > 0 or dry_run else "\U0001F6AB"

        message = f"""
{emoji} <b>Analysis Complete</b> [{mode}]

<b>Symbols Analyzed:</b> {symbols_analyzed}
<b>Opportunities Found:</b> {opportunities}
<b>Trades Executed:</b> {trades_executed}

<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_sync(message.strip())

    def notify_status_report(
        self,
        net_liquidation: float,
        buying_power: float,
        unrealized_pnl: float,
        positions: list[dict],
        open_orders: int,
    ) -> bool:
        """Send portfolio status report."""
        emoji = "\U0001F4B0"  # Money bag

        pnl_emoji = "\U0001F7E2" if unrealized_pnl >= 0 else "\U0001F534"

        positions_text = ""
        if positions:
            for pos in positions[:5]:  # Limit to 5
                positions_text += f"\n  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_cost']:.2f}"
        else:
            positions_text = "\n  No open positions"

        message = f"""
{emoji} <b>Portfolio Status</b>

<b>Net Liquidation:</b> ${net_liquidation:,.2f}
<b>Buying Power:</b> ${buying_power:,.2f}
{pnl_emoji} <b>Unrealized P&L:</b> ${unrealized_pnl:,.2f}

<b>Positions:</b>{positions_text}

<b>Open Orders:</b> {open_orders}

<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_sync(message.strip())

    def notify_error(self, error_message: str, context: str = "") -> bool:
        """Send error notification."""
        emoji = "\U0001F6A8"  # Alert

        message = f"""
{emoji} <b>Error Alert</b>

<b>Context:</b> {context or 'Unknown'}
<b>Error:</b> {error_message}

<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_sync(message.strip())

    def notify_bot_started(self, mode: str = "DRY RUN") -> bool:
        """Send notification when bot starts."""
        emoji = "\U0001F680"  # Rocket

        message = f"""
{emoji} <b>Trading Bot Started</b>

<b>Mode:</b> {mode}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Bot is now monitoring the market.
"""
        return self.send_sync(message.strip())

    def notify_bot_stopped(self, reason: str = "User requested") -> bool:
        """Send notification when bot stops."""
        emoji = "\U0001F6D1"  # Stop sign

        message = f"""
{emoji} <b>Trading Bot Stopped</b>

<b>Reason:</b> {reason}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_sync(message.strip())

    def notify_market_blocked(
        self,
        opportunities: list,
        market_reason: str,
        symbols_analyzed: int,
    ) -> bool:
        """Send notification when opportunities found but blocked by market condition."""
        emoji = "\u26A0\uFE0F"  # Warning

        # Build opportunity list
        opp_lines = []
        for opp in opportunities[:5]:  # Limit to 5
            opp_lines.append(
                f"  \u2022 {opp.symbol} @ ${opp.current_price:.2f} ({opp.signal.strength:.0%})"
            )

        opp_text = "\n".join(opp_lines) if opp_lines else "  None"
        more_text = f"\n  ... and {len(opportunities) - 5} more" if len(opportunities) > 5 else ""

        message = f"""
{emoji} <b>Analysis Complete - Market Weak</b>

<b>Market:</b> {market_reason}

<b>Symbols Analyzed:</b> {symbols_analyzed}
<b>Opportunities Found:</b> {len(opportunities)}

<b>Would have traded:</b>
{opp_text}{more_text}

<i>No trades opened due to weak market conditions.</i>

<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_sync(message.strip())

    # ==================== Paper Trade Notifications ====================

    def notify_paper_trade_opened(
        self,
        trade_id: int,
        symbol: str,
        action: str,
        quantity: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Send notification when a paper trade is opened."""
        emoji = "\U0001F4DD"  # Memo/paper

        sl_text = f"${stop_loss:,.2f}" if stop_loss else "N/A"
        tp_text = f"${take_profit:,.2f}" if take_profit else "N/A"
        value = quantity * entry_price

        message = f"""
{emoji} <b>Paper Trade Opened</b> #{trade_id}

<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Size:</b> {quantity:,} shares @ ${entry_price:,.2f}
<b>Value:</b> ${value:,.2f}

<b>Stop Loss:</b> {sl_text}
<b>Take Profit:</b> {tp_text}

<i>Tracking this trade to measure performance...</i>

<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_sync(message.strip())

    def notify_paper_trade_closed(
        self,
        trade_id: int,
        symbol: str,
        action: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        pnl_amount: float,
        pnl_percent: float,
        exit_reason: str,  # CLOSED_TP, CLOSED_SL, CLOSED_MANUAL
    ) -> bool:
        """Send notification when a paper trade is closed."""
        # Determine emoji and result text based on outcome
        if pnl_amount > 0:
            emoji = "\U0001F4B0"  # Money bag (profit)
            result = "PROFIT"
        elif pnl_amount < 0:
            emoji = "\U0001F4C9"  # Chart down (loss)
            result = "LOSS"
        else:
            emoji = "\U0001F7F0"  # Grey equals
            result = "BREAK EVEN"

        # Exit reason text
        reason_text = {
            "CLOSED_TP": "Take Profit Hit",
            "CLOSED_SL": "Stop Loss Hit",
            "CLOSED_MANUAL": "Manual Close",
        }.get(exit_reason, exit_reason)

        pnl_sign = "+" if pnl_amount >= 0 else ""

        message = f"""
{emoji} <b>Paper Trade Closed</b> #{trade_id} - {result}

<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Size:</b> {quantity:,} shares

<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f}
<b>Reason:</b> {reason_text}

<b>P&L:</b> {pnl_sign}${pnl_amount:,.2f} ({pnl_sign}{pnl_percent:.1f}%)

<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_sync(message.strip())

    def notify_paper_trade_stats(
        self,
        total_trades: int,
        open_trades: int,
        closed_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        win_rate: float,
    ) -> bool:
        """Send paper trading statistics summary."""
        emoji = "\U0001F4CA"  # Chart

        pnl_emoji = "\U0001F7E2" if total_pnl >= 0 else "\U0001F534"
        pnl_sign = "+" if total_pnl >= 0 else ""

        message = f"""
{emoji} <b>Paper Trading Stats</b>

<b>Total Trades:</b> {total_trades}
<b>Open:</b> {open_trades}
<b>Closed:</b> {closed_trades}

<b>Winners:</b> {winning_trades}
<b>Losers:</b> {losing_trades}
<b>Win Rate:</b> {win_rate:.1f}%

{pnl_emoji} <b>Total P&L:</b> {pnl_sign}${total_pnl:,.2f}

<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>
"""
        return self.send_sync(message.strip())

    def notify_daily_summary(
        self,
        date: str,
        trades_opened: int,
        trades_closed: int,
        winning_trades: int,
        losing_trades: int,
        day_pnl: float,
        total_pnl: float,
        win_rate: float,
        best_trade: Optional[dict] = None,
        worst_trade: Optional[dict] = None,
    ) -> bool:
        """Send end-of-day trading summary."""
        emoji = "\U0001F4C5"  # Calendar

        day_pnl_emoji = "\U0001F7E2" if day_pnl >= 0 else "\U0001F534"
        total_pnl_emoji = "\U0001F7E2" if total_pnl >= 0 else "\U0001F534"
        day_sign = "+" if day_pnl >= 0 else ""
        total_sign = "+" if total_pnl >= 0 else ""

        # Performance indicator
        if win_rate >= 60:
            perf_emoji = "\U0001F525"  # Fire - great
            perf_text = "Excellent!"
        elif win_rate >= 50:
            perf_emoji = "\U00002705"  # Check - good
            perf_text = "Good"
        elif win_rate >= 40:
            perf_emoji = "\U0001F7E1"  # Yellow - okay
            perf_text = "Needs improvement"
        else:
            perf_emoji = "\U0001F6A8"  # Alert - poor
            perf_text = "Review strategy"

        best_text = ""
        if best_trade:
            best_text = f"\n<b>Best:</b> {best_trade['symbol']} +${best_trade['pnl']:.2f}"

        worst_text = ""
        if worst_trade:
            worst_text = f"\n<b>Worst:</b> {worst_trade['symbol']} -${abs(worst_trade['pnl']):.2f}"

        message = f"""
{emoji} <b>Daily Summary - {date}</b>

<b>Today's Activity:</b>
  Opened: {trades_opened} trades
  Closed: {trades_closed} trades
  Won: {winning_trades} | Lost: {losing_trades}

{day_pnl_emoji} <b>Today's P&L:</b> {day_sign}${day_pnl:,.2f}
{total_pnl_emoji} <b>Total P&L:</b> {total_sign}${total_pnl:,.2f}

<b>Win Rate:</b> {win_rate:.1f}% {perf_emoji} {perf_text}
{best_text}{worst_text}

<i>Strategy: Scalping (TP 1.5% / SL 0.75%)</i>
"""
        return self.send_sync(message.strip())

    def notify_running_pnl(
        self,
        open_trades: int,
        unrealized_pnl: float,
        realized_pnl: float,
        total_pnl: float,
    ) -> bool:
        """Send running P&L update (can be called periodically)."""
        emoji = "\U0001F4B5"  # Dollar

        total_emoji = "\U0001F7E2" if total_pnl >= 0 else "\U0001F534"
        total_sign = "+" if total_pnl >= 0 else ""
        realized_sign = "+" if realized_pnl >= 0 else ""
        unrealized_sign = "+" if unrealized_pnl >= 0 else ""

        message = f"""
{emoji} <b>P&L Update</b>

<b>Open Trades:</b> {open_trades}
<b>Unrealized:</b> {unrealized_sign}${unrealized_pnl:,.2f}
<b>Realized:</b> {realized_sign}${realized_pnl:,.2f}

{total_emoji} <b>Total:</b> {total_sign}${total_pnl:,.2f}

<code>{datetime.now().strftime('%H:%M:%S')}</code>
"""
        return self.send_sync(message.strip())


    # ==================== Command Handling ====================

    def get_updates(self, offset: int = 0) -> list[dict]:
        """
        Fetch new messages/commands from Telegram.

        Args:
            offset: Update ID to start from (use last_update_id + 1)

        Returns:
            List of update objects
        """
        if not self.enabled:
            return []

        try:
            url = f"https://api.telegram.org/bot{self.config.bot_token}/getUpdates"
            params = {"offset": offset, "timeout": 1}

            req_url = f"{url}?offset={offset}&timeout=1"
            req = urllib.request.Request(req_url, method="GET")

            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                if result.get("ok"):
                    return result.get("result", [])
                return []

        except Exception as e:
            logger.debug(f"Error fetching Telegram updates: {e}")
            return []

    def process_command(self, text: str, db=None, price_fetcher=None) -> Optional[str]:
        """
        Process a command and return the response.

        Args:
            text: The command text (e.g., "/status")
            db: Database instance for fetching trade data
            price_fetcher: Optional callable that takes list of symbols and returns dict of prices

        Returns:
            Response message or None if not a command
        """
        if not text.startswith("/"):
            return None

        command = text.split()[0].lower()

        if command in ["/status", "/positions", "/pos"]:
            return self._handle_positions_command(db, price_fetcher)
        elif command in ["/stats", "/performance"]:
            return self._handle_stats_command(db)
        elif command in ["/help", "/start"]:
            return self._handle_help_command()

        return None

    def _handle_positions_command(self, db, price_fetcher=None) -> str:
        """Handle /positions command - show open paper trades with P&L."""
        if db is None:
            return "\u26A0\uFE0F Cannot fetch positions - no database connection"

        try:
            trades = db.get_open_paper_trades()

            if not trades:
                return "\U0001F4CB <b>Open Positions</b>\n\nNo open paper trades."

            # Fetch current prices if price_fetcher available
            current_prices = {}
            if price_fetcher:
                symbols = [t['symbol'] for t in trades]
                try:
                    current_prices = price_fetcher(symbols)
                except Exception as e:
                    logger.warning(f"Could not fetch current prices: {e}")

            lines = [f"\U0001F4CB <b>Open Positions</b> ({len(trades)})\n"]
            total_entry_value = 0
            total_current_value = 0
            total_pnl = 0

            for trade in trades:
                symbol = trade['symbol']
                qty = trade['quantity']
                entry = trade['entry_price']
                sl = trade.get('stop_loss')
                tp = trade.get('take_profit')

                entry_value = qty * entry
                total_entry_value += entry_value

                # Calculate P&L if we have current price
                current = current_prices.get(symbol)
                if current:
                    pnl_amount = (current - entry) * qty
                    pnl_pct = ((current - entry) / entry) * 100
                    total_pnl += pnl_amount
                    current_value = qty * current
                    total_current_value += current_value

                    # Emoji based on P&L
                    if pnl_pct >= 1.0:
                        emoji = "\U0001F7E2"  # Green circle
                    elif pnl_pct >= 0:
                        emoji = "\U0001F7E1"  # Yellow circle
                    elif pnl_pct > -1.5:
                        emoji = "\U0001F7E0"  # Orange circle
                    else:
                        emoji = "\U0001F534"  # Red circle

                    pnl_sign = "+" if pnl_pct >= 0 else ""

                    # Distance to SL and TP from current price
                    to_sl = ((current - sl) / current * 100) if sl else 0
                    to_tp = ((tp - current) / current * 100) if tp else 0

                    lines.append(
                        f"\n<b>{symbol}</b> {emoji} {pnl_sign}{pnl_pct:.2f}%\n"
                        f"  ${entry:.2f} \u2192 ${current:.2f}\n"
                        f"  SL ${sl:.2f} ({to_sl:.1f}% away) | TP ${tp:.2f} ({to_tp:.1f}% away)"
                    )
                else:
                    # No current price - show entry and targets
                    total_current_value += entry_value
                    lines.append(
                        f"\n<b>{symbol}</b> (no live price)\n"
                        f"  Entry: ${entry:.2f}\n"
                        f"  SL: ${sl:.2f} | TP: ${tp:.2f}"
                    )

            # Summary
            if current_prices:
                total_pnl_pct = (total_pnl / total_entry_value * 100) if total_entry_value else 0
                pnl_emoji = "\U0001F7E2" if total_pnl >= 0 else "\U0001F534"
                pnl_sign = "+" if total_pnl >= 0 else ""
                lines.append(f"\n\n{pnl_emoji} <b>Total P&L:</b> {pnl_sign}{total_pnl_pct:.2f}% (${pnl_sign}{total_pnl:.2f})")

            lines.append(f"\n<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return f"\u26A0\uFE0F Error fetching positions: {e}"

    def _handle_stats_command(self, db) -> str:
        """Handle /stats command - show paper trade statistics."""
        if db is None:
            return "\u26A0\uFE0F Cannot fetch stats - no database connection"

        try:
            stats = db.get_paper_trade_stats()

            pnl = stats['total_pnl']
            pnl_emoji = "\U0001F7E2" if pnl >= 0 else "\U0001F534"
            pnl_sign = "+" if pnl >= 0 else ""

            # Calculate portfolio balance
            initial_capital = 100_000.0
            balance = initial_capital + pnl
            return_pct = (pnl / initial_capital) * 100

            win_rate = stats['win_rate']
            if win_rate >= 60:
                perf_emoji = "\U0001F525"
            elif win_rate >= 50:
                perf_emoji = "\u2705"
            else:
                perf_emoji = "\U0001F7E1"

            return f"""\U0001F4CA <b>Paper Trading Stats</b>

\U0001F4B0 <b>Balance:</b> ${balance:,.2f}
{pnl_emoji} <b>P&L:</b> {pnl_sign}${pnl:,.2f} ({pnl_sign}{return_pct:.2f}%)

<b>Total Trades:</b> {stats['total_trades']}
<b>Open:</b> {stats['open_trades']} | <b>Closed:</b> {stats['closed_trades']}

<b>Winners:</b> {stats['winning_trades']} | <b>Losers:</b> {stats['losing_trades']}
<b>Win Rate:</b> {win_rate:.1f}% {perf_emoji}

<code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"""

        except Exception as e:
            logger.error(f"Error fetching stats: {e}")
            return f"\u26A0\uFE0F Error fetching stats: {e}"

    def _handle_help_command(self) -> str:
        """Handle /help command."""
        return """\U0001F916 <b>IBKR Trading Bot Commands</b>

<b>/positions</b> - Show open paper trades
<b>/stats</b> - Show trading statistics
<b>/help</b> - Show this help message

The bot will automatically notify you of:
\u2022 New trade opportunities
\u2022 Paper trades opened/closed
\u2022 Daily summaries
\u2022 Connection issues"""


# Singleton notifier instance
_notifier: Optional[TelegramNotifier] = None
_last_update_id: int = 0


def get_notifier() -> TelegramNotifier:
    """Get or create the global notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def check_telegram_commands(db=None, price_fetcher=None) -> None:
    """
    Check for and process any pending Telegram commands.
    Call this periodically from the main bot loop.

    Args:
        db: Database instance for fetching trade data
        price_fetcher: Optional callable that takes list of symbols and returns dict of prices
    """
    global _last_update_id

    notifier = get_notifier()
    if not notifier.enabled:
        return

    try:
        updates = notifier.get_updates(offset=_last_update_id)

        for update in updates:
            _last_update_id = update.get("update_id", 0) + 1

            message = update.get("message", {})
            text = message.get("text", "")
            chat_id = message.get("chat", {}).get("id")

            # Only respond to messages from the configured chat
            if str(chat_id) != notifier.config.chat_id:
                continue

            response = notifier.process_command(text, db, price_fetcher)
            if response:
                notifier.send_sync(response)
                logger.info(f"Responded to Telegram command: {text}")

    except Exception as e:
        logger.debug(f"Error checking Telegram commands: {e}")
