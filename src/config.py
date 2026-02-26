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

    Optimized via backtesting (Jan 2026):
    - 1:1 TP/SL ratio (2%/2%) for balanced risk/reward
    - 50% signal strength (2 of 4 indicators) for more trades
    - Max 2 positions for focused capital deployment
    - BULLISH trend requirement (SPY + stock)
    - Volume confirmation (1.0x average)
    - Cooldown after losses (20 min)

    Backtest results: +1.26% monthly, 71% win rate, Sharpe 0.85
    """
    # Asset universe - focus on liquid tech/AI stocks (precious metals removed: poor edge, high gap risk)
    symbols: dict = field(default_factory=lambda: {
        "ai": ["NVDA", "AMD", "GOOGL", "MSFT"],
        "tech": ["TSLA", "META", "AMZN"],
    })

    # Risk management - OPTIMIZED 1:1 ratio
    max_position_pct: float = 0.10  # Max 10% of portfolio per position
    stop_loss_pct: float = 0.02     # 2% stop loss (1:1 with TP)
    take_profit_pct: float = 0.02   # 2% take profit (1:1 with SL)
    max_sector_pct: float = 0.40    # Max 40% in any sector

    # Technical parameters - EMA-based for momentum
    ema_fast: int = 9              # Fast EMA
    ema_slow: int = 21             # Slow EMA
    ema_trend: int = 50            # Trend EMA (for direction filter)
    rsi_period: int = 7            # Shorter RSI for faster signals
    rsi_overbought: int = 70
    rsi_oversold: int = 30

    # Entry filters - OPTIMIZED for more high-quality trades
    min_signal_strength: float = 0.70   # 70% = 3 of 4 indicators minimum
    volume_multiplier: float = 1.0      # Require at least average volume
    require_bullish_trend: bool = True  # Only trade BULLISH trends (for longs)
    use_hourly_confirmation: bool = True  # Check 1H trend aligns with 5-min entry
    enable_shorting: bool = True        # Allow SHORT positions on BEARISH trends

    # Trailing stop - lock in profits once trade moves in your favour
    use_trailing_stop: bool = True           # Enable trailing stop loss
    trailing_activation_pct: float = 0.005   # Activate after 0.5% in profit
    trailing_stop_pct: float = 0.015         # Trail 1.5% from best price

    # Anti-churning protections
    cooldown_minutes: int = 20          # Cooldown after stop loss hit
    max_trades_per_symbol_day: int = 3  # Max trades per symbol per day
    max_daily_loss: float = 50000.0     # Stop trading if daily loss exceeds this (high for paper testing)
    max_open_positions: int = 4         # Max 4 positions for diversification
    max_trades_per_sector: int = 2      # Max 2 open trades per sector (avoid concentration)

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
