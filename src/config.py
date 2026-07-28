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


# ISO-4217 code -> display symbol. Used to label account-level values, which
# IBKR reports in the account base currency (GBP for this account), not USD.
_CURRENCY_SYMBOLS = {
    "USD": "$", "GBP": "£", "EUR": "€", "JPY": "¥",
    "CAD": "C$", "AUD": "A$", "CHF": "CHF ",
}


def currency_symbol(code: str | None) -> str:
    """Map an ISO-4217 currency code to its display symbol.

    Falls back to '<CODE> ' for unknown codes, and '$' when no code is given
    (so existing call sites degrade to prior behaviour rather than crashing).
    """
    if not code:
        return "$"
    return _CURRENCY_SYMBOLS.get(code.upper(), f"{code.upper()} ")


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
    # 18% x 5 slots = 90% max deployment, leaving ~10% headroom so a cash
    # account with T+2 unsettled proceeds can still fund the next entry.
    # Raised from 15% (which paired with 8 slots) as part of the 2026-07-27
    # commission-drag fix — see max_open_positions below.
    max_position_pct: float = 0.18 # Max 18% of equity per position
    max_asset_class_pct: float = 0.40  # Max 40% in any asset class
    max_gross_exposure: float = 1.0    # No leverage — cash account

    # Risk management
    min_hold_days: int = 5         # Minimum hold period to prevent whipsaws
    # Block re-entry into a symbol for N calendar days after its protective stop
    # fires. Without this the bot re-bought failing names within days: CNYA cost
    # -$59.25 over 3 round-trips and EIMU -$34.74 over 2 (plus ~$40 commission)
    # between 2026-06-26 and 2026-07-17. Only blocks NEW entries — positions
    # already held are untouched, and the freed slot backfills to the next signal.
    reentry_cooldown_days: int = 10
    # Top up a held position only once it has drifted this far below target.
    # Positions never resized before 2026-07-28 — the rebalance skipped any
    # symbol already held — so they froze at their opening size and the book
    # sat at 36% deployed against a ~73% target. Every top-up costs a fresh
    # ~$4 commission, so this deliberately ignores small drift: at 0.30 a
    # position is left alone until it is under 70% of target.
    topup_drift_threshold: float = 0.30
    drawdown_reduce_pct: float = 0.10  # Reduce positions 50% at 10% drawdown
    drawdown_halt_pct: float = 0.20    # Close all + halt at 20% drawdown
    max_daily_loss: float = 200.0      # Daily loss limit, base GBP (~4% of £5k NLV). HARDCODED — the only risk limit that does NOT auto-scale off NLV; re-bump if capital changes.

    # Shorting
    enable_shorting: bool = False   # Start long-only, add shorts later

    # Data parameters
    bar_size: str = "1 day"        # Daily bars for trend following
    data_duration: str = "1 Y"     # 1 year of history for lookbacks

    # Scheduling — Europe/London local time (LSE session is 08:00-16:30)
    rebalance_hour: int = 14       # 14:00 London = mid-LSE session
    rebalance_minute: int = 0
    risk_check_interval_hours: int = 4  # Check trailing stops every 4 hours

    # Max open positions — top-N by signal strength (enforced in engine._calculate_target_positions)
    #
    # Cut 8 -> 5 on 2026-07-27 to fix commission drag. IBKR's minimum commission
    # is $4/order, so a round-trip costs ~$8 (~£6) regardless of size. At 8 slots
    # the per-position cap was ~£380, giving only ~£6-31 of risk-to-stop per
    # trade — commission averaged ~43% of the risk unit and hit 93% on RTWO.
    # No trend-following edge survives that. Fewer, larger positions spread the
    # fixed cost over more capital: measured 43% -> 15% on the 2026-07-27 book.
    # 5 also matches reality: only 6-7 names clear signal_threshold at any time.
    max_open_positions: int = 5

    # Universe version — bump when the instrument set changes; useful for
    # cross-referencing portfolio snapshots vs. the universe in effect at the time.
    instrument_universe_version: str = "v2_ucits"

    # Watchdog: if the data probe has been continuously failing for this many
    # minutes (i.e. self-heal via gateway restart isn't working — typical cause
    # is the 3-restart daily cap hitting and then sitting silent), the bot
    # sys.exit(1)s so Docker's restart policy recreates the container.
    watchdog_timeout_min: int = 30


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
