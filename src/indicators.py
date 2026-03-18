"""
Technical indicators for trend-following and momentum analysis.
Pure Python/NumPy implementation - no TA-Lib dependency.
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass


@dataclass
class Signal:
    """Represents a trading signal."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float  # -1.0 (strong short) to +1.0 (strong long)
    reasons: list[str]
    indicators: dict


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range - measures volatility."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def total_return(close: pd.Series, lookback: int) -> float:
    """Calculate total return over a lookback window."""
    if len(close) < lookback + 1:
        return 0.0
    current = close.iloc[-1]
    past = close.iloc[-lookback - 1]
    if past == 0 or np.isnan(past) or np.isnan(current):
        return 0.0
    return (current - past) / past


def annualised_volatility(close: pd.Series, period: int = 20) -> float:
    """Calculate annualised volatility from daily returns."""
    if len(close) < period + 1:
        return 0.0
    returns = close.pct_change().dropna().tail(period)
    if len(returns) < period:
        return 0.0
    return float(returns.std() * np.sqrt(252))


# ---------------------------------------------------------------------------
# Trend-Following Analyzer
# ---------------------------------------------------------------------------

class TrendFollowingAnalyzer:
    """
    Computes trend-following and momentum signals for a single instrument.

    Signals:
    - Time-Series Momentum (TSMOM): multi-lookback blended signal
    - ATR trailing stop for risk management
    - Volatility (for position sizing by the engine)

    Usage:
        analyzer = TrendFollowingAnalyzer(df, lookbacks=[21, 63, 252])
        signal = analyzer.compute_tsmom_signal()
        vol = analyzer.compute_volatility()
        atr_val = analyzer.compute_atr()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookbacks: list[int] = None,
        atr_period: int = 20,
    ):
        """
        Args:
            df: DataFrame with columns: date, open, high, low, close, volume
            lookbacks: Lookback periods in trading days [short, medium, long]
            atr_period: Period for ATR calculation
        """
        self.df = df.copy()
        self.lookbacks = lookbacks or [21, 63, 252]
        self.atr_period = atr_period

    def compute_tsmom_signal(self) -> tuple[float, list[str]]:
        """
        Compute blended time-series momentum signal.

        Blends sign of returns across multiple lookbacks:
        - 12-month (252d): 40% weight
        - 3-month (63d):   30% weight
        - 1-month (21d):   30% weight

        Returns:
            Tuple of (signal score from -1.0 to +1.0, list of reasons)
        """
        close = self.df['close']
        # Need at least enough data for the shortest lookback
        if len(close) < min(self.lookbacks) + 2:
            return 0.0, ["Insufficient data for TSMOM"]

        weights = [0.3, 0.3, 0.4]  # short, medium, long
        signals = []
        reasons = []
        active_weight_total = 0.0

        for lookback, weight in zip(self.lookbacks, weights):
            if len(close) < lookback + 2:
                # Not enough data for this lookback — use max available
                lookback = len(close) - 2
            ret = total_return(close, lookback)
            sig = np.sign(ret)
            signals.append(sig * weight)
            active_weight_total += weight

            period_name = {21: "1M", 63: "3M", 252: "12M"}.get(lookback, f"{lookback}d")
            direction = "up" if ret > 0 else "down"
            reasons.append(f"{period_name} {direction} {ret:+.1%}")

        blended = sum(signals)
        return float(round(blended, 3)), reasons

    def compute_volatility(self, period: int = 20) -> float:
        """Compute annualised volatility from daily returns."""
        return annualised_volatility(self.df['close'], period)

    def compute_atr(self) -> float:
        """Compute current ATR value."""
        if len(self.df) < self.atr_period + 1:
            return 0.0
        atr_series = atr(
            self.df['high'], self.df['low'], self.df['close'],
            period=self.atr_period,
        )
        val = atr_series.iloc[-1]
        return float(val) if not np.isnan(val) else 0.0

    def compute_trailing_stop(self, entry_price: float, is_long: bool = True) -> float:
        """
        Compute the initial trailing stop level (3x ATR from current price).

        Args:
            entry_price: Entry price of the position
            is_long: True for long, False for short

        Returns:
            Stop price
        """
        atr_val = self.compute_atr()
        if atr_val == 0:
            # Fallback: 5% stop
            return entry_price * (0.95 if is_long else 1.05)

        if is_long:
            return round(entry_price - 3.0 * atr_val, 2)
        else:
            return round(entry_price + 3.0 * atr_val, 2)

    def get_current_price(self) -> float:
        """Get the latest closing price."""
        if self.df.empty:
            return 0.0
        return float(self.df['close'].iloc[-1])

    def get_sma(self, period: int) -> float:
        """Get current SMA value."""
        s = sma(self.df['close'], period)
        val = s.iloc[-1] if not s.empty else 0.0
        return float(val) if not np.isnan(val) else 0.0


# ---------------------------------------------------------------------------
# Cross-Sectional Momentum ranking (module-level function)
# ---------------------------------------------------------------------------

def rank_cross_sectional(
    tsmom_scores: dict[str, float],
) -> dict[str, float]:
    """
    Rank instruments by their TSMOM scores and assign CSMOM scores.

    Top 20% get +1.0, bottom 20% get -1.0, middle scaled linearly.

    Args:
        tsmom_scores: Dict of symbol -> TSMOM signal score

    Returns:
        Dict of symbol -> CSMOM score (-1.0 to +1.0)
    """
    if not tsmom_scores:
        return {}

    # Sort by TSMOM score
    sorted_symbols = sorted(tsmom_scores.items(), key=lambda x: x[1])
    n = len(sorted_symbols)

    csmom_scores = {}
    for i, (symbol, _) in enumerate(sorted_symbols):
        # Percentile rank: 0.0 (worst) to 1.0 (best)
        if n == 1:
            percentile = 0.5
        else:
            percentile = i / (n - 1)
        # Map to -1.0 to +1.0
        csmom_scores[symbol] = round(percentile * 2 - 1, 3)

    return csmom_scores


def compute_combined_signal(
    tsmom_score: float,
    csmom_score: float,
    tsmom_weight: float = 0.6,
    csmom_weight: float = 0.4,
) -> float:
    """
    Combine TSMOM and CSMOM into a final signal.

    Returns:
        Combined score from -1.0 to +1.0
    """
    return round(tsmom_score * tsmom_weight + csmom_score * csmom_weight, 3)
