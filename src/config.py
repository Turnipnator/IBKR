"""
Configuration settings for the IBKR Trading Bot.
Uses environment variables with sensible defaults for development.
"""

import json
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


# Default universe: 30 liquid ETFs across 5 uncorrelated asset classes
_DEFAULT_UNIVERSE = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ", "FXI"],
    "bond": ["TLT", "IEF", "SHY", "LQD", "HYG", "EMB", "TIP"],
    "commodity": ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "PDBC", "CPER", "DBC"],
    "fx": ["UUP", "FXE", "FXY", "FXB"],
    "alt": ["VNQ", "BITO"],
}

_WATCHLIST_PATH = Path(os.getenv("WATCHLIST_PATH", "data/watchlist.json"))


def _load_watchlist() -> dict:
    """Load symbols from watchlist.json if it exists, otherwise use defaults."""
    try:
        if _WATCHLIST_PATH.exists():
            with open(_WATCHLIST_PATH) as f:
                data = json.load(f)
            symbols = data.get("symbols", {})
            if symbols and isinstance(symbols, dict):
                # Validate structure: all values must be lists of strings
                for sector, tickers in symbols.items():
                    if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
                        raise ValueError(f"Invalid watchlist format for sector '{sector}'")
                return symbols
    except Exception as e:
        print(f"Warning: Failed to load watchlist from {_WATCHLIST_PATH}: {e}. Using defaults.")
    return dict(_DEFAULT_UNIVERSE)


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
    """Trading parameters - TREND FOLLOWING / MOMENTUM strategy.

    Based on academic research (Moskowitz, Ooi, Pedersen 2012):
    - Time-series momentum (TSMOM) across multiple lookback windows
    - Cross-sectional momentum (CSMOM) for relative strength ranking
    - Volatility-scaled (ATR-based) position sizing
    - Diversified universe of 30 ETFs across 5 asset classes
    - Daily rebalancing with ATR trailing stops
    """
    # Asset universe - loaded from watchlist.json if present, otherwise defaults
    symbols: dict = field(default_factory=lambda: _load_watchlist())

    # Signal parameters
    lookback_short: int = 21       # 1-month lookback (trading days)
    lookback_medium: int = 63      # 3-month lookback
    lookback_long: int = 252       # 12-month lookback
    tsmom_weight: float = 0.6      # Weight for time-series momentum
    csmom_weight: float = 0.4      # Weight for cross-sectional momentum
    signal_threshold: float = 0.5  # Min |signal| to trade (0.5 = strong conviction, reduces turnover at small capital)

    # Position sizing - volatility-scaled (inverse ATR)
    atr_period: int = 20           # ATR lookback for volatility
    atr_stop_multiplier: float = 3.0  # Trailing stop = 3x ATR from peak
    risk_budget: float = 0.20      # Target 20% annualised portfolio volatility
    max_position_pct: float = 0.15 # Max 15% of equity per position (sized for £10k with top-N concentration)
    max_asset_class_pct: float = 0.40  # Max 40% in any asset class
    max_gross_exposure: float = 1.0    # No leverage — cash account

    # Risk management
    min_hold_days: int = 5         # Minimum hold period to prevent whipsaws
    drawdown_reduce_pct: float = 0.10  # Reduce positions 50% at 10% drawdown
    drawdown_halt_pct: float = 0.20    # Close all + halt at 20% drawdown
    max_daily_loss: float = 300.0      # Daily loss limit (3% of £10k)

    # Shorting
    enable_shorting: bool = False   # Start long-only, add shorts later

    # Data parameters
    bar_size: str = "1 day"        # Daily bars for trend following
    data_duration: str = "1 Y"     # 1 year of history for lookbacks

    # Scheduling
    rebalance_hour: int = 15       # Rebalance at 3:30 PM ET
    rebalance_minute: int = 30
    risk_check_interval_hours: int = 4  # Check trailing stops every 4 hours

    # Max open positions — top-N by signal strength (enforced in engine._calculate_target_positions)
    max_open_positions: int = 8


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
