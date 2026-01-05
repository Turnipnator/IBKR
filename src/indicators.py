"""
Technical indicators for trading analysis.
Pure Python/NumPy implementation - no TA-Lib dependency.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Signal:
    """Represents a trading signal."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float  # 0.0 to 1.0
    reasons: list[str]
    indicators: dict


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        series: Price series (typically close prices)
        period: Number of periods

    Returns:
        SMA series
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        series: Price series
        period: Number of periods

    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Args:
        series: Price series (typically close prices)
        period: RSI period (default 14)

    Returns:
        RSI series (0-100)
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.

    Args:
        series: Price series
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    fast_ema = ema(series, fast_period)
    slow_ema = ema(series, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Args:
        series: Price series
        period: SMA period (default 20)
        std_dev: Number of standard deviations (default 2)

    Returns:
        Tuple of (Upper band, Middle band/SMA, Lower band)
    """
    middle = sma(series, period)
    std = series.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range - measures volatility.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.rolling(window=period).mean()


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period (default 14)
        d_period: %D period (default 3)

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()

    return k, d


class TechnicalAnalyzer:
    """
    Calculates technical indicators and generates signals for a DataFrame.
    Optimized for SCALPING with fast EMA crossovers and short RSI periods.

    Usage:
        analyzer = TechnicalAnalyzer(df)
        analyzer.calculate_all()
        signal = analyzer.generate_signal("AAPL")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sma_fast: int = 9,      # Fast EMA for scalping (was 50)
        sma_slow: int = 21,     # Slow EMA for scalping (was 200)
        rsi_period: int = 7,    # Shorter RSI for scalping (was 14)
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        use_ema: bool = True,   # Use EMA instead of SMA for scalping
    ):
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns: date, open, high, low, close, volume
            sma_fast: Fast MA period (9 for scalping)
            sma_slow: Slow MA period (21 for scalping)
            rsi_period: RSI period (7 for scalping)
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
            use_ema: Use EMA instead of SMA (True for scalping)
        """
        self.df = df.copy()
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.use_ema = use_ema

    def calculate_all(self) -> pd.DataFrame:
        """Calculate all technical indicators optimized for scalping."""

        # Moving averages - use EMA for scalping (faster response)
        if self.use_ema:
            self.df[f'ema_{self.sma_fast}'] = ema(self.df['close'], self.sma_fast)
            self.df[f'ema_{self.sma_slow}'] = ema(self.df['close'], self.sma_slow)
            # Alias for compatibility
            self.df[f'sma_{self.sma_fast}'] = self.df[f'ema_{self.sma_fast}']
            self.df[f'sma_{self.sma_slow}'] = self.df[f'ema_{self.sma_slow}']
        else:
            self.df[f'sma_{self.sma_fast}'] = sma(self.df['close'], self.sma_fast)
            self.df[f'sma_{self.sma_slow}'] = sma(self.df['close'], self.sma_slow)

        # Standard EMAs for MACD
        self.df['ema_12'] = ema(self.df['close'], 12)
        self.df['ema_26'] = ema(self.df['close'], 26)

        # RSI with shorter period for scalping
        self.df['rsi'] = rsi(self.df['close'], self.rsi_period)

        # MACD - use faster settings for scalping (8, 17, 9)
        macd_line, signal_line, hist = macd(self.df['close'], fast_period=8, slow_period=17, signal_period=9)
        self.df['macd'] = macd_line
        self.df['macd_signal'] = signal_line
        self.df['macd_hist'] = hist

        # Bollinger Bands - shorter period for scalping
        upper, middle, lower = bollinger_bands(self.df['close'], period=10, std_dev=2.0)
        self.df['bb_upper'] = upper
        self.df['bb_middle'] = middle
        self.df['bb_lower'] = lower

        # ATR (if we have high/low) - shorter period for scalping
        if 'high' in self.df.columns and 'low' in self.df.columns:
            self.df['atr'] = atr(self.df['high'], self.df['low'], self.df['close'], period=7)

            # Stochastic - faster settings for scalping
            k, d = stochastic(self.df['high'], self.df['low'], self.df['close'], k_period=5, d_period=3)
            self.df['stoch_k'] = k
            self.df['stoch_d'] = d

        # Trend signals using EMA
        fast_col = f'ema_{self.sma_fast}' if self.use_ema else f'sma_{self.sma_fast}'
        slow_col = f'ema_{self.sma_slow}' if self.use_ema else f'sma_{self.sma_slow}'

        self.df['trend'] = np.where(
            self.df[fast_col] > self.df[slow_col],
            'BULLISH',
            'BEARISH'
        )

        # MACD crossover
        self.df['macd_cross'] = np.where(
            (self.df['macd'] > self.df['macd_signal']) &
            (self.df['macd'].shift(1) <= self.df['macd_signal'].shift(1)),
            'BULLISH',
            np.where(
                (self.df['macd'] < self.df['macd_signal']) &
                (self.df['macd'].shift(1) >= self.df['macd_signal'].shift(1)),
                'BEARISH',
                'NEUTRAL'
            )
        )

        # EMA crossover signal (important for scalping)
        self.df['ema_cross'] = np.where(
            (self.df[fast_col] > self.df[slow_col]) &
            (self.df[fast_col].shift(1) <= self.df[slow_col].shift(1)),
            'BULLISH',
            np.where(
                (self.df[fast_col] < self.df[slow_col]) &
                (self.df[fast_col].shift(1) >= self.df[slow_col].shift(1)),
                'BEARISH',
                'NEUTRAL'
            )
        )

        return self.df

    def get_latest_indicators(self) -> dict:
        """Get the most recent indicator values."""
        if self.df.empty:
            return {}

        latest = self.df.iloc[-1]

        return {
            'date': latest.get('date'),
            'close': latest.get('close'),
            f'sma_{self.sma_fast}': latest.get(f'sma_{self.sma_fast}'),
            f'sma_{self.sma_slow}': latest.get(f'sma_{self.sma_slow}'),
            'rsi': latest.get('rsi'),
            'macd': latest.get('macd'),
            'macd_signal': latest.get('macd_signal'),
            'macd_hist': latest.get('macd_hist'),
            'trend': latest.get('trend'),
            'bb_upper': latest.get('bb_upper'),
            'bb_lower': latest.get('bb_lower'),
            'atr': latest.get('atr'),
        }

    def generate_signal(self, symbol: str) -> Signal:
        """
        Generate a trading signal based on indicators.

        Returns:
            Signal object with action recommendation
        """
        if self.df.empty:
            return Signal(
                symbol=symbol,
                action='HOLD',
                strength=0.0,
                reasons=['No data available'],
                indicators={}
            )

        # Warn if insufficient data for slow SMA but continue with available indicators
        min_periods = min(len(self.df), self.sma_slow)

        indicators = self.get_latest_indicators()
        reasons = []
        buy_signals = 0
        sell_signals = 0
        total_signals = 0

        # 1. Trend analysis (SMA crossover)
        sma_fast_val = indicators.get(f'sma_{self.sma_fast}')
        sma_slow_val = indicators.get(f'sma_{self.sma_slow}')
        close = indicators.get('close')

        # Check for valid (non-NaN) values
        sma_fast_valid = sma_fast_val is not None and not np.isnan(sma_fast_val)
        sma_slow_valid = sma_slow_val is not None and not np.isnan(sma_slow_val)

        if sma_fast_valid and sma_slow_valid:
            total_signals += 1
            if sma_fast_val > sma_slow_val:
                buy_signals += 1
                reasons.append(f'Bullish trend (SMA{self.sma_fast} > SMA{self.sma_slow})')
            else:
                sell_signals += 1
                reasons.append(f'Bearish trend (SMA{self.sma_fast} < SMA{self.sma_slow})')
        elif sma_fast_valid:
            # Use price vs fast SMA if slow SMA unavailable
            total_signals += 1
            if close and close > sma_fast_val:
                buy_signals += 1
                reasons.append(f'Price above SMA{self.sma_fast}')
            elif close:
                sell_signals += 1
                reasons.append(f'Price below SMA{self.sma_fast}')

        # 2. RSI analysis
        rsi_val = indicators.get('rsi')
        if rsi_val is not None and not np.isnan(rsi_val):
            total_signals += 1
            if rsi_val < self.rsi_oversold:
                buy_signals += 1
                reasons.append(f'RSI oversold ({rsi_val:.1f})')
            elif rsi_val > self.rsi_overbought:
                sell_signals += 1
                reasons.append(f'RSI overbought ({rsi_val:.1f})')
            else:
                reasons.append(f'RSI neutral ({rsi_val:.1f})')

        # 3. MACD analysis
        macd_val = indicators.get('macd')
        macd_signal_val = indicators.get('macd_signal')
        macd_hist = indicators.get('macd_hist')

        if macd_val is not None and macd_signal_val is not None:
            total_signals += 1
            if macd_val > macd_signal_val and macd_hist > 0:
                buy_signals += 1
                reasons.append('MACD bullish crossover')
            elif macd_val < macd_signal_val and macd_hist < 0:
                sell_signals += 1
                reasons.append('MACD bearish crossover')

        # 4. Bollinger Band analysis
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')

        if bb_upper and bb_lower and close:
            total_signals += 1
            if close <= bb_lower:
                buy_signals += 1
                reasons.append('Price at lower Bollinger Band')
            elif close >= bb_upper:
                sell_signals += 1
                reasons.append('Price at upper Bollinger Band')

        # Determine action and strength
        if total_signals == 0:
            action = 'HOLD'
            strength = 0.0
        else:
            buy_ratio = buy_signals / total_signals
            sell_ratio = sell_signals / total_signals

            if buy_ratio >= 0.5:
                action = 'BUY'
                strength = buy_ratio
            elif sell_ratio >= 0.5:
                action = 'SELL'
                strength = sell_ratio
            else:
                action = 'HOLD'
                strength = 1.0 - max(buy_ratio, sell_ratio)

        return Signal(
            symbol=symbol,
            action=action,
            strength=strength,
            reasons=reasons,
            indicators=indicators
        )
