"""
Backtesting System - Offline strategy testing and parameter optimization.

Replicates the exact trading strategy from engine.py and bot.py to test
historical performance and optimize parameters.

Usage:
    from src.backtester import Backtester, BacktestConfig, BacktestRunner

    # Basic backtest with current settings
    bt = Backtester()
    results = bt.run('2025-06-01', '2025-12-31')
    print(results.report)

    # Test different configurations
    runner = BacktestRunner()
    configs = [
        BacktestConfig(take_profit_pct=0.01, stop_loss_pct=0.02),
        BacktestConfig(take_profit_pct=0.015, stop_loss_pct=0.03),
    ]
    results = runner.run_all(configs, '2025-06-01', '2025-12-31')
    print(runner.compare_results(results))
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from itertools import product
import numpy as np
import pandas as pd

from .database import Database
from .indicators import TechnicalAnalyzer, Signal
from .config import trading_config

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest matching current strategy parameters."""

    # Capital
    initial_capital: float = 100_000.0

    # Risk management
    take_profit_pct: float = 0.015  # 1.5%
    stop_loss_pct: float = 0.03    # 3%
    max_position_pct: float = 0.10  # 10% of portfolio per position
    max_open_positions: int = 3

    # Signal filters
    min_signal_strength: float = 0.75  # 3 of 4 indicators
    volume_multiplier: float = 1.0     # At least average volume
    min_momentum: float = 0.5          # Momentum score threshold
    require_bullish_trend: bool = True

    # Technical indicator settings
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50
    rsi_period: int = 7
    rsi_overbought: int = 70
    rsi_oversold: int = 30

    # MACD settings (used by TechnicalAnalyzer internally: 8/17/9)
    # BB settings (used by TechnicalAnalyzer internally: 10/2)

    # Anti-churning
    cooldown_bars: int = 4       # 4 bars = 20 min (5-min bars)
    max_trades_per_symbol_day: int = 3

    # Symbols to trade
    symbols: list = field(default_factory=lambda: [
        "GLD", "SLV",           # Precious metals
        "NVDA", "AMD", "GOOGL", "MSFT",  # AI
        "AAPL", "TSLA", "META", "AMZN",  # Tech
    ])

    # SPY is always needed for market filter
    spy_symbol: str = "SPY"

    def __post_init__(self):
        """Validate configuration."""
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")
        if not 0 < self.min_signal_strength <= 1:
            raise ValueError("min_signal_strength must be between 0 and 1")


@dataclass
class BacktestPosition:
    """Represents an open position during backtest."""
    symbol: str
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    entry_time: datetime
    entry_bar_idx: int


@dataclass
class BacktestTrade:
    """Completed trade record."""
    id: int
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl_amount: float
    pnl_percent: float
    exit_reason: str  # 'TP', 'SL', 'SIGNAL', 'EOD'
    holding_bars: int


@dataclass
class BacktestResult:
    """Results container from a backtest run."""
    config: BacktestConfig
    start_date: str
    end_date: str
    trades: list[BacktestTrade]
    metrics: dict
    portfolio_history: pd.DataFrame
    report: str

    def __repr__(self) -> str:
        return self.report


class Backtester:
    """
    Main backtesting engine.

    Simulates the trading strategy offline using historical data.
    Replicates entry/exit logic from engine.py and bot.py.
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        db: Optional[Database] = None,
    ):
        self.config = config or BacktestConfig()
        self.db = db or Database()

        # State during simulation
        self._positions: dict[str, BacktestPosition] = {}
        self._trades: list[BacktestTrade] = []
        self._trade_counter = 0
        self._capital = self.config.initial_capital
        self._portfolio_history: list[dict] = []

        # Cooldowns: symbol -> bar_idx when cooldown expires
        self._cooldowns: dict[str, int] = {}

        # Daily trade counts: (symbol, date) -> count
        self._daily_trades: dict[tuple[str, str], int] = {}

        # Data storage
        self._data: dict[str, pd.DataFrame] = {}
        self._spy_trend_cache: dict[int, str] = {}

        # Pre-calculated indicators (for speed)
        self._indicators: dict[str, pd.DataFrame] = {}

    def _load_data(
        self,
        start_date: str,
        end_date: str,
        fetcher=None,
    ) -> bool:
        """
        Load historical data for all symbols.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fetcher: Optional DataFetcher for fetching missing data

        Returns:
            True if data loaded successfully
        """
        all_symbols = self.config.symbols + [self.config.spy_symbol]
        symbols_loaded = 0

        for symbol in all_symbols:
            # Try loading from database first
            df = self.db.load_ohlcv(symbol, start_date=start_date, end_date=end_date)

            if df.empty or len(df) < 50:
                logger.warning(f"{symbol}: Insufficient data in database ({len(df)} bars)")

                # Try fetching if fetcher available
                if fetcher:
                    try:
                        # Calculate days needed
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                        days_needed = (end_dt - start_dt).days + 5

                        logger.info(f"{symbol}: Fetching {days_needed} days of 5-min data...")
                        fetch_df = fetcher.get_historical_data(
                            symbol,
                            duration=f"{days_needed} D",
                            bar_size="5 mins",
                        )

                        if fetch_df is not None and not fetch_df.empty:
                            self.db.save_ohlcv(fetch_df, symbol)
                            df = self.db.load_ohlcv(
                                symbol, start_date=start_date, end_date=end_date
                            )
                            logger.info(f"{symbol}: Fetched and saved {len(df)} bars")
                    except Exception as e:
                        logger.error(f"{symbol}: Failed to fetch data: {e}")

            if not df.empty and len(df) >= 50:
                self._data[symbol] = df
                symbols_loaded += 1
                logger.info(f"{symbol}: Loaded {len(df)} bars")
            else:
                logger.warning(f"{symbol}: No usable data, will skip in backtest")

        # Must have SPY data
        if self.config.spy_symbol not in self._data:
            logger.error("SPY data required for market filter - cannot run backtest")
            return False

        logger.info(f"Loaded data for {symbols_loaded}/{len(all_symbols)} symbols")
        return symbols_loaded > 0

    def _create_analyzer(
        self,
        df: pd.DataFrame,
    ) -> TechnicalAnalyzer:
        """Create a TechnicalAnalyzer with backtest config settings."""
        return TechnicalAnalyzer(
            df,
            sma_fast=self.config.ema_fast,
            sma_slow=self.config.ema_slow,
            ema_trend=self.config.ema_trend,
            rsi_period=self.config.rsi_period,
            rsi_overbought=self.config.rsi_overbought,
            rsi_oversold=self.config.rsi_oversold,
            use_ema=True,
        )

    def _precalculate_indicators(self):
        """Pre-calculate all indicators for all symbols (huge speedup)."""
        logger.info("Pre-calculating indicators for all symbols...")

        for symbol, df in self._data.items():
            analyzer = self._create_analyzer(df)
            analyzer.calculate_all()
            self._indicators[symbol] = analyzer.df.copy()

        logger.info(f"Pre-calculated indicators for {len(self._indicators)} symbols")

    def _get_spy_trend(self, bar_idx: int) -> str:
        """
        Get SPY trend at a specific bar index using pre-calculated indicators.
        """
        if bar_idx in self._spy_trend_cache:
            return self._spy_trend_cache[bar_idx]

        spy_df = self._indicators.get(self.config.spy_symbol)
        if spy_df is None or bar_idx >= len(spy_df) or bar_idx < 50:
            return 'SIDEWAYS'

        row = spy_df.iloc[bar_idx]
        close = row.get('close')
        ema_fast = row.get(f'ema_{self.config.ema_fast}')
        ema_slow = row.get(f'ema_{self.config.ema_slow}')
        ema_trend = row.get(f'ema_{self.config.ema_trend}')

        # Check for valid values
        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in [close, ema_fast, ema_slow, ema_trend]):
            trend = 'SIDEWAYS'
        elif ema_fast > ema_slow > ema_trend and close > ema_trend:
            trend = 'BULLISH'
        elif ema_fast < ema_slow < ema_trend and close < ema_trend:
            trend = 'BEARISH'
        else:
            trend = 'SIDEWAYS'

        self._spy_trend_cache[bar_idx] = trend
        return trend

    def _is_in_cooldown(self, symbol: str, bar_idx: int) -> bool:
        """Check if symbol is in cooldown at given bar index."""
        if symbol not in self._cooldowns:
            return False
        return bar_idx < self._cooldowns[symbol]

    def _set_cooldown(self, symbol: str, bar_idx: int):
        """Set cooldown for symbol starting at bar_idx."""
        self._cooldowns[symbol] = bar_idx + self.config.cooldown_bars

    def _get_daily_trade_count(self, symbol: str, date_str: str) -> int:
        """Get trade count for symbol on a specific date."""
        key = (symbol, date_str)
        return self._daily_trades.get(key, 0)

    def _increment_daily_trade_count(self, symbol: str, date_str: str):
        """Increment trade count for symbol on a specific date."""
        key = (symbol, date_str)
        self._daily_trades[key] = self._daily_trades.get(key, 0) + 1

    def _calculate_position_size(self, price: float) -> int:
        """Calculate position size based on available capital."""
        max_value = self._capital * self.config.max_position_pct
        quantity = int(max_value / price)
        return max(0, quantity)

    def _check_entry_signal(
        self,
        symbol: str,
        bar_idx: int,
        spy_trend: str,
    ) -> Optional[tuple[float, float, float, Signal]]:
        """
        Check if entry conditions are met using pre-calculated indicators.

        Replicates logic from engine.py:
        1. SPY trend must be BULLISH
        2. Stock trend must be BULLISH
        3. Volume >= multiplier * average
        4. Momentum score >= 0.5
        5. Signal strength >= min_signal_strength
        6. RSI not overbought

        Returns:
            Tuple of (price, stop_loss, take_profit, signal) or None
        """
        # 1. Check SPY market condition
        if self.config.require_bullish_trend and spy_trend != 'BULLISH':
            return None

        # Need at least 50 bars for indicators
        if bar_idx < 50:
            return None

        # Get pre-calculated indicators
        df = self._indicators.get(symbol)
        if df is None or bar_idx >= len(df):
            return None

        row = df.iloc[bar_idx]

        # Get indicator values from row
        close = row.get('close')
        ema_fast = row.get(f'ema_{self.config.ema_fast}')
        ema_slow = row.get(f'ema_{self.config.ema_slow}')
        ema_trend = row.get(f'ema_{self.config.ema_trend}')
        rsi_val = row.get('rsi')
        macd_val = row.get('macd')
        macd_signal = row.get('macd_signal')
        macd_hist = row.get('macd_hist')
        volume_ratio = row.get('volume_ratio')
        bb_upper = row.get('bb_upper')
        bb_lower = row.get('bb_lower')

        # Check for valid EMAs
        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in [close, ema_fast, ema_slow, ema_trend]):
            return None

        # 2. Check stock trend (BULLISH = ema_fast > ema_slow > ema_trend and close > ema_trend)
        if self.config.require_bullish_trend:
            if not (ema_fast > ema_slow > ema_trend and close > ema_trend):
                return None

        # 3. Check volume confirmation
        if volume_ratio is None or np.isnan(volume_ratio):
            return None
        if volume_ratio < self.config.volume_multiplier:
            return None

        # 4 & 5. Calculate signal strength from indicators
        buy_signals = 0
        total_signals = 0

        # EMA trend
        total_signals += 1
        if ema_fast > ema_slow:
            buy_signals += 1

        # RSI
        if rsi_val is not None and not np.isnan(rsi_val):
            total_signals += 1
            if rsi_val < self.config.rsi_oversold:
                buy_signals += 1

        # MACD
        if all(v is not None and not np.isnan(v) for v in [macd_val, macd_signal, macd_hist]):
            total_signals += 1
            if macd_val > macd_signal and macd_hist > 0:
                buy_signals += 1

        # Bollinger Bands
        if all(v is not None and not np.isnan(v) for v in [bb_upper, bb_lower]):
            total_signals += 1
            if close <= bb_lower:
                buy_signals += 1

        if total_signals == 0:
            return None

        strength = buy_signals / total_signals

        if strength < self.config.min_signal_strength:
            return None

        # 6. Check RSI not overbought
        if rsi_val is not None and not np.isnan(rsi_val):
            if rsi_val > self.config.rsi_overbought:
                return None

        # Entry conditions met!
        entry_price = float(close)
        stop_loss = round(entry_price * (1 - self.config.stop_loss_pct), 2)
        take_profit = round(entry_price * (1 + self.config.take_profit_pct), 2)

        # Create signal for logging
        signal = Signal(
            symbol=symbol,
            action='BUY',
            strength=strength,
            reasons=[f'Trend BULLISH', f'RSI {rsi_val:.1f}' if rsi_val else ''],
            indicators={'rsi': rsi_val, 'close': close}
        )

        return (entry_price, stop_loss, take_profit, signal)

    def _check_exit_conditions(
        self,
        position: BacktestPosition,
        bar: pd.Series,
    ) -> Optional[tuple[float, str]]:
        """
        Check if position should be closed based on bar data.

        Uses bar high/low for realistic SL/TP detection:
        - SL hit if bar.low <= stop_loss
        - TP hit if bar.high >= take_profit

        Returns:
            Tuple of (exit_price, exit_reason) or None
        """
        bar_low = float(bar['low'])
        bar_high = float(bar['high'])

        # Check stop loss first (assume it triggers before TP if both hit)
        if bar_low <= position.stop_loss:
            # Exit at stop loss price (not bar low, for realism)
            return (position.stop_loss, 'SL')

        # Check take profit
        if bar_high >= position.take_profit:
            # Exit at take profit price
            return (position.take_profit, 'TP')

        return None

    def _open_position(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        bar_time: datetime,
        bar_idx: int,
    ):
        """Open a new position."""
        quantity = self._calculate_position_size(entry_price)

        if quantity <= 0:
            return

        # Check max open positions
        if len(self._positions) >= self.config.max_open_positions:
            return

        position = BacktestPosition(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=bar_time,
            entry_bar_idx=bar_idx,
        )

        self._positions[symbol] = position

        # Reserve capital
        position_value = entry_price * quantity
        self._capital -= position_value

        # Track daily trade count
        date_str = bar_time.strftime('%Y-%m-%d')
        self._increment_daily_trade_count(symbol, date_str)

        logger.debug(
            f"OPEN: {symbol} @ ${entry_price:.2f} x {quantity} "
            f"(SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})"
        )

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        bar_time: datetime,
        bar_idx: int,
    ):
        """Close an existing position."""
        if symbol not in self._positions:
            return

        position = self._positions.pop(symbol)

        # Calculate P&L
        pnl_amount = (exit_price - position.entry_price) * position.quantity
        pnl_percent = ((exit_price - position.entry_price) / position.entry_price) * 100
        holding_bars = bar_idx - position.entry_bar_idx

        # Return capital + P&L
        self._capital += (exit_price * position.quantity)

        # Create trade record
        self._trade_counter += 1
        trade = BacktestTrade(
            id=self._trade_counter,
            symbol=symbol,
            entry_time=position.entry_time,
            exit_time=bar_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl_amount=pnl_amount,
            pnl_percent=pnl_percent,
            exit_reason=exit_reason,
            holding_bars=holding_bars,
        )
        self._trades.append(trade)

        # Set cooldown if stop loss hit
        if exit_reason == 'SL':
            self._set_cooldown(symbol, bar_idx)

        logger.debug(
            f"CLOSE: {symbol} @ ${exit_price:.2f} ({exit_reason}) "
            f"P&L: ${pnl_amount:.2f} ({pnl_percent:+.2f}%)"
        )

    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value including open positions."""
        total = self._capital

        for symbol, position in self._positions.items():
            if symbol in self._data:
                # Use latest close price
                df = self._data[symbol]
                if not df.empty:
                    current_price = float(df.iloc[-1]['close'])
                    total += current_price * position.quantity

        return total

    def run(
        self,
        start_date: str,
        end_date: str,
        fetcher=None,
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fetcher: Optional DataFetcher for fetching missing data from IBKR

        Returns:
            BacktestResult with trades, metrics, and report
        """
        logger.info(f"Starting backtest: {start_date} to {end_date}")

        # Reset state
        self._positions = {}
        self._trades = []
        self._trade_counter = 0
        self._capital = self.config.initial_capital
        self._portfolio_history = []
        self._cooldowns = {}
        self._daily_trades = {}
        self._data = {}
        self._spy_trend_cache = {}
        self._indicators = {}

        # Load data
        if not self._load_data(start_date, end_date, fetcher):
            return BacktestResult(
                config=self.config,
                start_date=start_date,
                end_date=end_date,
                trades=[],
                metrics={'error': 'Failed to load data'},
                portfolio_history=pd.DataFrame(),
                report="ERROR: Failed to load data. Ensure data is available in database.",
            )

        # Pre-calculate all indicators once (huge speedup for grid search)
        self._precalculate_indicators()

        # Get SPY data for bar iteration
        spy_df = self._data[self.config.spy_symbol]
        total_bars = len(spy_df)

        logger.info(f"Simulating {total_bars} bars across {len(self._data)} symbols")

        # Main simulation loop - iterate through each bar
        for bar_idx in range(total_bars):
            spy_bar = spy_df.iloc[bar_idx]
            bar_time = pd.to_datetime(spy_bar['date'])
            date_str = bar_time.strftime('%Y-%m-%d')

            # Get SPY trend for this bar
            spy_trend = self._get_spy_trend(bar_idx)

            # Check existing positions for exits first
            positions_to_check = list(self._positions.keys())
            for symbol in positions_to_check:
                if symbol not in self._data:
                    continue

                sym_df = self._data[symbol]
                if bar_idx >= len(sym_df):
                    continue

                bar = sym_df.iloc[bar_idx]
                position = self._positions[symbol]

                exit_result = self._check_exit_conditions(position, bar)
                if exit_result:
                    exit_price, exit_reason = exit_result
                    self._close_position(symbol, exit_price, exit_reason, bar_time, bar_idx)

            # Check for new entry signals
            for symbol in self.config.symbols:
                # Skip if already have position
                if symbol in self._positions:
                    continue

                # Skip if no data
                if symbol not in self._data:
                    continue

                # Skip if in cooldown
                if self._is_in_cooldown(symbol, bar_idx):
                    continue

                # Skip if daily trade limit reached
                if self._get_daily_trade_count(symbol, date_str) >= self.config.max_trades_per_symbol_day:
                    continue

                # Check entry conditions
                entry_result = self._check_entry_signal(symbol, bar_idx, spy_trend)

                if entry_result:
                    entry_price, stop_loss, take_profit, signal = entry_result
                    self._open_position(
                        symbol, entry_price, stop_loss, take_profit, bar_time, bar_idx
                    )

            # Record portfolio value at end of each bar
            portfolio_value = self._get_portfolio_value()
            self._portfolio_history.append({
                'date': bar_time,
                'portfolio_value': portfolio_value,
                'cash': self._capital,
                'open_positions': len(self._positions),
            })

        # Close any remaining positions at end of backtest
        final_bar = spy_df.iloc[-1]
        final_time = pd.to_datetime(final_bar['date'])

        for symbol in list(self._positions.keys()):
            if symbol in self._data:
                sym_df = self._data[symbol]
                final_price = float(sym_df.iloc[-1]['close'])
                self._close_position(symbol, final_price, 'EOD', final_time, total_bars - 1)

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Generate report
        report = self._generate_report(metrics, start_date, end_date)

        # Create portfolio history DataFrame
        portfolio_df = pd.DataFrame(self._portfolio_history)
        if not portfolio_df.empty:
            portfolio_df.set_index('date', inplace=True)

        return BacktestResult(
            config=self.config,
            start_date=start_date,
            end_date=end_date,
            trades=self._trades,
            metrics=metrics,
            portfolio_history=portfolio_df,
            report=report,
        )

    def _calculate_metrics(self) -> dict:
        """Calculate performance metrics from trades."""
        metrics = {
            'total_trades': len(self._trades),
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'total_return_pct': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'max_drawdown_amount': 0.0,
            'avg_holding_bars': 0.0,
            'trades_by_exit': {'TP': 0, 'SL': 0, 'SIGNAL': 0, 'EOD': 0},
        }

        if not self._trades:
            return metrics

        # Basic trade stats
        wins = [t for t in self._trades if t.pnl_amount > 0]
        losses = [t for t in self._trades if t.pnl_amount <= 0]

        metrics['winning_trades'] = len(wins)
        metrics['losing_trades'] = len(losses)
        metrics['win_rate'] = len(wins) / len(self._trades) if self._trades else 0

        # P&L stats
        pnls = [t.pnl_amount for t in self._trades]
        metrics['total_pnl'] = sum(pnls)
        metrics['total_return_pct'] = (metrics['total_pnl'] / self.config.initial_capital) * 100

        if wins:
            win_pnls = [t.pnl_amount for t in wins]
            metrics['avg_win'] = np.mean(win_pnls)
            metrics['largest_win'] = max(win_pnls)

        if losses:
            loss_pnls = [t.pnl_amount for t in losses]
            metrics['avg_loss'] = np.mean(loss_pnls)
            metrics['largest_loss'] = min(loss_pnls)

        # Profit factor
        gross_profit = sum(t.pnl_amount for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_amount for t in losses)) if losses else 0
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy (average P&L per trade)
        metrics['expectancy'] = np.mean(pnls) if pnls else 0

        # Holding period
        holding_bars = [t.holding_bars for t in self._trades]
        metrics['avg_holding_bars'] = np.mean(holding_bars) if holding_bars else 0

        # Exit reasons
        for trade in self._trades:
            if trade.exit_reason in metrics['trades_by_exit']:
                metrics['trades_by_exit'][trade.exit_reason] += 1

        # Calculate Sharpe and Sortino from portfolio history
        if self._portfolio_history:
            portfolio_values = [h['portfolio_value'] for h in self._portfolio_history]

            if len(portfolio_values) > 1:
                returns = pd.Series(portfolio_values).pct_change().dropna()

                if len(returns) > 0 and returns.std() > 0:
                    # Annualized Sharpe (assuming 5-min bars, 78 bars per day, 252 days)
                    bars_per_year = 78 * 252
                    excess_return = returns.mean() - (0.05 / bars_per_year)  # 5% risk-free
                    metrics['sharpe_ratio'] = (excess_return / returns.std()) * np.sqrt(bars_per_year)

                    # Sortino (downside deviation)
                    downside = returns[returns < 0]
                    if len(downside) > 0 and downside.std() > 0:
                        metrics['sortino_ratio'] = (excess_return / downside.std()) * np.sqrt(bars_per_year)

                # Max drawdown
                cummax = pd.Series(portfolio_values).cummax()
                drawdown = (pd.Series(portfolio_values) - cummax) / cummax
                metrics['max_drawdown_pct'] = float(drawdown.min()) * 100
                metrics['max_drawdown_amount'] = float((pd.Series(portfolio_values) - cummax).min())

        return metrics

    def _generate_report(self, metrics: dict, start_date: str, end_date: str) -> str:
        """Generate formatted backtest report."""
        lines = [
            "=" * 60,
            f"BACKTEST RESULTS: {start_date} to {end_date}",
            "=" * 60,
            "",
            "CONFIGURATION:",
            f"  Initial Capital: ${self.config.initial_capital:,.0f}",
            f"  Take Profit: {self.config.take_profit_pct*100:.1f}%  |  Stop Loss: {self.config.stop_loss_pct*100:.1f}%",
            f"  Min Signal Strength: {self.config.min_signal_strength*100:.0f}%",
            f"  Volume Multiplier: {self.config.volume_multiplier}x",
            f"  Max Open Positions: {self.config.max_open_positions}",
            "",
            "PERFORMANCE:",
            f"  Total Return: ${metrics['total_pnl']:,.2f} ({metrics['total_return_pct']:+.2f}%)",
            f"  Trades: {metrics['total_trades']} (Win: {metrics['win_rate']*100:.1f}%)",
            f"  Winning: {metrics['winning_trades']}  |  Losing: {metrics['losing_trades']}",
            "",
            "RISK METRICS:",
            f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}",
            f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}% (${metrics['max_drawdown_amount']:,.2f})",
            f"  Profit Factor: {metrics['profit_factor']:.2f}",
            "",
            "TRADE ANALYSIS:",
            f"  Avg Win: ${metrics['avg_win']:,.2f}  |  Avg Loss: ${metrics['avg_loss']:,.2f}",
            f"  Largest Win: ${metrics['largest_win']:,.2f}  |  Largest Loss: ${metrics['largest_loss']:,.2f}",
            f"  Expectancy: ${metrics['expectancy']:.2f} per trade",
            f"  Avg Holding: {metrics['avg_holding_bars']:.1f} bars",
            "",
            "EXIT BREAKDOWN:",
            f"  Take Profit: {metrics['trades_by_exit']['TP']}",
            f"  Stop Loss: {metrics['trades_by_exit']['SL']}",
            f"  Signal: {metrics['trades_by_exit']['SIGNAL']}",
            f"  End of Data: {metrics['trades_by_exit']['EOD']}",
            "=" * 60,
        ]

        return "\n".join(lines)

    def get_trades_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis."""
        if not self._trades:
            return pd.DataFrame()

        data = []
        for t in self._trades:
            data.append({
                'id': t.id,
                'symbol': t.symbol,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl_amount': t.pnl_amount,
                'pnl_percent': t.pnl_percent,
                'exit_reason': t.exit_reason,
                'holding_bars': t.holding_bars,
            })

        return pd.DataFrame(data)


class BacktestRunner:
    """
    Helper class for running multiple backtests with different configurations.

    Useful for parameter optimization and strategy comparison.
    """

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.results: list[BacktestResult] = []

    def run_all(
        self,
        configs: list[BacktestConfig],
        start_date: str,
        end_date: str,
        fetcher=None,
    ) -> list[BacktestResult]:
        """
        Run backtests for multiple configurations.

        Args:
            configs: List of BacktestConfig objects
            start_date: Start date
            end_date: End date
            fetcher: Optional DataFetcher

        Returns:
            List of BacktestResult objects
        """
        self.results = []

        for i, config in enumerate(configs):
            logger.info(f"Running backtest {i+1}/{len(configs)}")
            bt = Backtester(config=config, db=self.db)
            result = bt.run(start_date, end_date, fetcher)
            self.results.append(result)

        return self.results

    def grid_search(
        self,
        param_grid: dict,
        start_date: str,
        end_date: str,
        fetcher=None,
    ) -> pd.DataFrame:
        """
        Run grid search over parameter combinations.

        Args:
            param_grid: Dict mapping parameter names to lists of values
                Example: {
                    'take_profit_pct': [0.01, 0.015, 0.02],
                    'stop_loss_pct': [0.02, 0.03, 0.04],
                }
            start_date: Start date
            end_date: End date
            fetcher: Optional DataFetcher

        Returns:
            DataFrame with results for each parameter combination
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Running grid search with {len(combinations)} combinations")

        results_data = []

        for i, combo in enumerate(combinations):
            # Create config with this parameter combination
            config_dict = dict(zip(param_names, combo))

            try:
                config = BacktestConfig(**config_dict)
            except ValueError as e:
                logger.warning(f"Invalid config {config_dict}: {e}")
                continue

            logger.info(f"  [{i+1}/{len(combinations)}] {config_dict}")

            bt = Backtester(config=config, db=self.db)
            result = bt.run(start_date, end_date, fetcher)

            # Collect results
            row = {**config_dict}
            row.update({
                'total_trades': result.metrics.get('total_trades', 0),
                'win_rate': result.metrics.get('win_rate', 0),
                'total_pnl': result.metrics.get('total_pnl', 0),
                'total_return_pct': result.metrics.get('total_return_pct', 0),
                'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': result.metrics.get('max_drawdown_pct', 0),
                'profit_factor': result.metrics.get('profit_factor', 0),
            })
            results_data.append(row)

        df = pd.DataFrame(results_data)

        # Sort by Sharpe ratio
        if 'sharpe_ratio' in df.columns:
            df = df.sort_values('sharpe_ratio', ascending=False)

        return df

    def compare_results(self, results: Optional[list[BacktestResult]] = None) -> pd.DataFrame:
        """
        Compare multiple backtest results in a table.

        Args:
            results: List of results to compare (uses self.results if None)

        Returns:
            DataFrame with comparison metrics
        """
        results = results or self.results

        if not results:
            return pd.DataFrame()

        data = []
        for i, r in enumerate(results):
            row = {
                'config_id': i + 1,
                'tp_pct': r.config.take_profit_pct * 100,
                'sl_pct': r.config.stop_loss_pct * 100,
                'min_signal': r.config.min_signal_strength * 100,
                'trades': r.metrics.get('total_trades', 0),
                'win_rate': r.metrics.get('win_rate', 0) * 100,
                'total_pnl': r.metrics.get('total_pnl', 0),
                'return_pct': r.metrics.get('total_return_pct', 0),
                'sharpe': r.metrics.get('sharpe_ratio', 0),
                'max_dd': r.metrics.get('max_drawdown_pct', 0),
                'profit_factor': r.metrics.get('profit_factor', 0),
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df.sort_values('sharpe', ascending=False)


def quick_backtest(
    start_date: str = '2025-12-01',
    end_date: str = '2026-01-20',
    **config_overrides,
) -> BacktestResult:
    """
    Quick helper function to run a single backtest.

    Args:
        start_date: Start date
        end_date: End date
        **config_overrides: Override default config values

    Returns:
        BacktestResult

    Example:
        result = quick_backtest(take_profit_pct=0.02, stop_loss_pct=0.04)
        print(result.report)
    """
    config = BacktestConfig(**config_overrides)
    bt = Backtester(config=config)
    return bt.run(start_date, end_date)
