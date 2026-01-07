"""
Configuration settings for the IBKR Trading Bot.
Uses environment variables with sensible defaults for development.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, use environment variables directly


@dataclass
class IBKRConfig:
    """IBKR connection configuration."""
    host: str = "127.0.0.1"
    port: int = 7497  # 7497=TWS Paper, 7496=TWS Live, 4002=Gateway Paper, 4001=Gateway Live
    client_id: int = 1
    timeout: int = 10
    readonly: bool = False

    @classmethod
    def from_env(cls) -> "IBKRConfig":
        return cls(
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            port=int(os.getenv("IBKR_PORT", "7497")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
            timeout=int(os.getenv("IBKR_TIMEOUT", "10")),
            readonly=os.getenv("IBKR_READONLY", "false").lower() == "true",
        )


@dataclass
class TradingConfig:
    """Trading parameters - configured for MOMENTUM SCALPING strategy.

    Based on winning strategy from Binance Bot (13/13 wins):
    - Wider stop loss to avoid noise (3% vs 0.75%)
    - Quick take profit (1.5%)
    - High quality entries only (60%+ signal strength)
    - BULLISH trend requirement
    - Volume confirmation (1.5x average)
    - Cooldown after losses (20 min)
    """
    # Asset universe - focus on liquid, volatile stocks
    symbols: dict = field(default_factory=lambda: {
        "precious_metals": ["GLD", "SLV"],
        "ai": ["NVDA", "AMD", "GOOGL", "MSFT"],
        "tech": ["AAPL", "TSLA", "META", "AMZN"],
    })

    # Risk management - MOMENTUM settings (wider stops, quick profits)
    max_position_pct: float = 0.10  # Max 10% of portfolio per position
    stop_loss_pct: float = 0.03     # 3% stop loss (wider to avoid noise!)
    take_profit_pct: float = 0.015  # 1.5% take profit (lock in gains quickly)
    max_sector_pct: float = 0.40    # Max 40% in any sector

    # Technical parameters - EMA-based for momentum
    ema_fast: int = 9              # Fast EMA
    ema_slow: int = 21             # Slow EMA
    ema_trend: int = 50            # Trend EMA (for direction filter)
    rsi_period: int = 7            # Shorter RSI for faster signals
    rsi_overbought: int = 70
    rsi_oversold: int = 30

    # Entry filters (from Binance winning strategy)
    min_signal_strength: float = 0.60   # Only enter on 60%+ signals (was 50%)
    volume_multiplier: float = 1.1      # Require 1.1x average volume (stocks less volatile than crypto)
    require_bullish_trend: bool = True  # Only trade BULLISH trends

    # Anti-churning protections
    cooldown_minutes: int = 20          # Cooldown after stop loss hit
    max_trades_per_symbol_day: int = 3  # Max trades per symbol per day
    max_daily_loss: float = 5000.0      # Stop trading if daily loss exceeds this

    # Scalping-specific settings
    bar_size: str = "5 mins"       # 5-minute candles for scalping
    data_duration: str = "2 D"     # 2 days of data (enough for 5-min bars)
    min_volume: int = 100000       # Minimum volume filter

    # Legacy support (kept for compatibility)
    sma_fast: int = 9
    sma_slow: int = 21


@dataclass
class DataConfig:
    """Data storage configuration."""
    db_path: str = "data/trading.db"
    log_path: str = "logs/trading.log"

    @classmethod
    def from_env(cls) -> "DataConfig":
        return cls(
            db_path=os.getenv("DB_PATH", "data/trading.db"),
            log_path=os.getenv("LOG_PATH", "logs/trading.log"),
        )


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        return cls(
            bot_token=token,
            chat_id=chat_id,
            enabled=bool(token and chat_id),
        )


# Global config instances
ibkr_config = IBKRConfig.from_env()
trading_config = TradingConfig()
data_config = DataConfig.from_env()
telegram_config = TelegramConfig.from_env()
