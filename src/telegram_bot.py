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


# Singleton notifier instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the global notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
