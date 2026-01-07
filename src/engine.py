"""
Decision Engine - Core trading logic.
Analyzes signals and executes trades based on strategy rules.
"""

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from .connection import ConnectionManager, get_connection
from .data_fetcher import DataFetcher
from .database import Database
from .indicators import TechnicalAnalyzer, Signal
from .orders import OrderManager, PositionManager, OrderAction, OrderResult
from .config import trading_config, TradingConfig

logger = logging.getLogger(__name__)


class TradeDecision(Enum):
    """Possible trade decisions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"  # Close existing position


@dataclass
class TradeOpportunity:
    """Represents a potential trade."""
    symbol: str
    decision: TradeDecision
    signal: Signal
    current_price: float
    position_size: int
    reasons: list[str]
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None


@dataclass
class EngineState:
    """Current state of the decision engine."""
    last_run: Optional[datetime] = None
    symbols_analyzed: int = 0
    trades_executed: int = 0
    errors: list[str] = field(default_factory=list)
    opportunities: list[TradeOpportunity] = field(default_factory=list)


class DecisionEngine:
    """
    Core decision engine for automated trading.

    Workflow:
    1. Fetch latest data for all symbols
    2. Calculate technical indicators
    3. Generate signals
    4. Apply risk management rules
    5. Execute qualifying trades

    Usage:
        engine = DecisionEngine()
        engine.run_analysis()  # Analyze only
        engine.run_trading()   # Analyze and execute
    """

    def __init__(
        self,
        connection: Optional[ConnectionManager] = None,
        config: Optional[TradingConfig] = None,
        dry_run: bool = True,  # Safety: default to not executing
    ):
        self.connection = connection or get_connection()
        self.config = config or trading_config
        self.dry_run = dry_run

        # Initialize components
        self.db = Database()
        self.fetcher = DataFetcher(self.connection)
        self.order_manager = OrderManager(self.connection, self.db)
        self.position_manager = PositionManager(self.connection, self.order_manager)

        # State
        self.state = EngineState()

    def _get_all_symbols(self) -> list[str]:
        """Get all symbols from trading universe."""
        symbols = []
        for sector_symbols in self.config.symbols.values():
            symbols.extend(sector_symbols)
        return symbols

    def _get_sector(self, symbol: str) -> Optional[str]:
        """Get the sector for a symbol."""
        for sector, symbols in self.config.symbols.items():
            if symbol in symbols:
                return sector
        return None

    def _check_sector_exposure(self, sector: str) -> float:
        """Calculate current exposure to a sector as % of portfolio."""
        portfolio = self.position_manager.get_portfolio_value()
        net_liq = portfolio.get('net_liquidation', 0)
        if net_liq <= 0:
            return 0.0

        sector_value = 0.0
        positions = self.position_manager.get_positions()
        sector_symbols = self.config.symbols.get(sector, [])

        for pos in positions:
            if pos.symbol in sector_symbols:
                sector_value += abs(pos.market_value)

        return sector_value / net_liq

    def _passes_risk_checks(self, opportunity: TradeOpportunity) -> tuple[bool, list[str]]:
        """
        Check if a trade opportunity passes risk management rules.

        Based on Binance winning strategy filters:
        - Signal strength >= 60%
        - No cooldown active
        - Under max trades per symbol per day
        - Under daily loss limit

        Returns:
            Tuple of (passes, list of failure reasons)
        """
        failures = []

        # Check 1: Signal strength threshold (use config value, default 60%)
        min_strength = getattr(self.config, 'min_signal_strength', 0.60)
        if opportunity.signal.strength < min_strength:
            failures.append(f"Signal strength too low ({opportunity.signal.strength:.0%} < {min_strength:.0%})")

        # Check 2: Cooldown check (anti-churning)
        in_cooldown, cooldown_reason = self.db.is_symbol_in_cooldown(opportunity.symbol)
        if in_cooldown:
            failures.append(f"Symbol in cooldown: {cooldown_reason}")

        # Check 3: Max trades per symbol per day
        max_trades = getattr(self.config, 'max_trades_per_symbol_day', 3)
        daily_count = self.db.get_daily_trade_count(opportunity.symbol)
        if daily_count >= max_trades:
            failures.append(f"Max daily trades reached for {opportunity.symbol} ({daily_count}/{max_trades})")

        # Check 4: Daily loss limit
        max_daily_loss = getattr(self.config, 'max_daily_loss', 5000.0)
        daily_pnl = self.db.get_daily_pnl()
        if daily_pnl < -max_daily_loss:
            failures.append(f"Daily loss limit exceeded (${daily_pnl:.2f} < -${max_daily_loss:.2f})")

        # Check 2: Position size > 0
        if opportunity.position_size <= 0:
            failures.append("Position size is zero")

        # Check 3: Max position size (% of portfolio)
        portfolio = self.position_manager.get_portfolio_value()
        net_liq = portfolio.get('net_liquidation', 0)
        if net_liq > 0:
            position_value = opportunity.position_size * opportunity.current_price
            position_pct = position_value / net_liq
            if position_pct > self.config.max_position_pct:
                failures.append(
                    f"Position too large ({position_pct:.1%} > {self.config.max_position_pct:.1%})"
                )

        # Check 4: Sector exposure limit
        sector = self._get_sector(opportunity.symbol)
        if sector:
            current_exposure = self._check_sector_exposure(sector)
            if current_exposure >= self.config.max_sector_pct:
                failures.append(
                    f"Sector {sector} exposure at limit ({current_exposure:.1%})"
                )

        # Check 5: Don't buy if we already have a position
        if opportunity.decision == TradeDecision.BUY:
            existing_qty = self.position_manager.get_position_quantity(opportunity.symbol)
            if existing_qty > 0:
                failures.append(f"Already have position ({existing_qty} shares)")

        return (len(failures) == 0, failures)

    def analyze_symbol(self, symbol: str) -> Optional[TradeOpportunity]:
        """
        Analyze a single symbol and generate trade opportunity if any.
        Uses 5-minute bars for scalping strategy.

        Returns:
            TradeOpportunity or None if no action recommended
        """
        try:
            # Fetch data using config settings (5-min bars for scalping)
            bar_size = getattr(self.config, 'bar_size', '5 mins')
            duration = getattr(self.config, 'data_duration', '2 D')

            df = self.fetcher.get_historical_data(
                symbol,
                duration=duration,
                bar_size=bar_size
            )

            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # Save to database
            self.db.save_ohlcv(df, symbol)

            # Calculate indicators with momentum scalping settings
            ema_fast = getattr(self.config, 'ema_fast', self.config.sma_fast)
            ema_slow = getattr(self.config, 'ema_slow', self.config.sma_slow)
            ema_trend = getattr(self.config, 'ema_trend', 50)

            analyzer = TechnicalAnalyzer(
                df,
                sma_fast=ema_fast,
                sma_slow=ema_slow,
                ema_trend=ema_trend,
                rsi_period=self.config.rsi_period,
                rsi_overbought=self.config.rsi_overbought,
                rsi_oversold=self.config.rsi_oversold,
                use_ema=True,  # Use EMA for scalping
            )
            analyzer.calculate_all()

            # Check BULLISH trend filter (from Binance winning strategy)
            require_bullish = getattr(self.config, 'require_bullish_trend', True)
            if require_bullish:
                trend = analyzer.detect_trend()
                if trend != 'BULLISH':
                    logger.info(f"  {symbol}: Skipped - trend is {trend} (need BULLISH)")
                    return None

            # Check volume confirmation (from Binance winning strategy)
            volume_mult = getattr(self.config, 'volume_multiplier', 1.5)
            vol_confirmed, vol_ratio = analyzer.check_volume_confirmation(volume_mult)
            if not vol_confirmed:
                logger.info(f"  {symbol}: Skipped - volume {vol_ratio:.1f}x (need {volume_mult}x)")
                return None

            # Generate signal
            signal = analyzer.generate_signal(symbol)

            # Use momentum score if available
            momentum_score = analyzer.get_momentum_score()
            if momentum_score < 0.5:
                logger.info(f"  {symbol}: Skipped - momentum {momentum_score:.0%} (need 50%+)")
                return None

            # Get current price
            current_price = float(df['close'].iloc[-1])

            # Check existing position
            existing_position = self.position_manager.get_position_quantity(symbol)

            # Determine decision
            if signal.action == 'BUY' and existing_position == 0:
                decision = TradeDecision.BUY
                position_size = self.position_manager.calculate_position_size(
                    symbol, current_price, self.config.max_position_pct
                )
                # Calculate stop-loss and take-profit
                stop_loss = round(current_price * (1 - self.config.stop_loss_pct), 2)
                take_profit = round(current_price * (1 + self.config.take_profit_pct), 2)

            elif signal.action == 'SELL' and existing_position > 0:
                decision = TradeDecision.CLOSE
                position_size = existing_position
                stop_loss = None
                take_profit = None

            elif existing_position > 0:
                # Check if we should close based on stop-loss logic
                # (In production, this would check against entry price)
                decision = TradeDecision.HOLD
                position_size = 0
                stop_loss = None
                take_profit = None

            else:
                decision = TradeDecision.HOLD
                position_size = 0
                stop_loss = None
                take_profit = None

            if decision == TradeDecision.HOLD:
                return None

            return TradeOpportunity(
                symbol=symbol,
                decision=decision,
                signal=signal,
                current_price=current_price,
                position_size=position_size,
                reasons=signal.reasons,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
            )

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            self.state.errors.append(f"{symbol}: {str(e)}")
            return None

    def run_analysis(self) -> list[TradeOpportunity]:
        """
        Run analysis on all symbols without executing trades.

        Returns:
            List of trade opportunities
        """
        if not self.connection.ensure_connected():
            logger.error("Cannot run analysis: not connected")
            return []

        logger.info("Starting market analysis...")
        self.state = EngineState(last_run=datetime.now())

        symbols = self._get_all_symbols()
        opportunities = []

        for symbol in symbols:
            logger.info(f"Analyzing {symbol}...")
            opportunity = self.analyze_symbol(symbol)

            if opportunity:
                # Check risk management
                passes, failures = self._passes_risk_checks(opportunity)

                if passes:
                    opportunities.append(opportunity)
                    logger.info(
                        f"  {symbol}: {opportunity.decision.value} "
                        f"({opportunity.signal.strength:.0%} confidence)"
                    )
                else:
                    logger.info(f"  {symbol}: Opportunity rejected - {', '.join(failures)}")
            else:
                logger.info(f"  {symbol}: HOLD (no action)")

            self.state.symbols_analyzed += 1
            self.connection.ib.sleep(0.5)  # Rate limiting

        self.state.opportunities = opportunities
        logger.info(f"Analysis complete: {len(opportunities)} opportunities found")

        return opportunities

    def execute_opportunity(self, opportunity: TradeOpportunity) -> OrderResult:
        """Execute a single trade opportunity."""

        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {opportunity.decision.value} "
                       f"{opportunity.position_size} {opportunity.symbol}")
            return OrderResult(
                success=True,
                message=f"[DRY RUN] {opportunity.decision.value} {opportunity.symbol}"
            )

        if opportunity.decision == TradeDecision.BUY:
            return self.order_manager.place_market_order(
                symbol=opportunity.symbol,
                action=OrderAction.BUY,
                quantity=opportunity.position_size,
                reason=f"Signal: {', '.join(opportunity.reasons[:2])}",
            )

        elif opportunity.decision == TradeDecision.CLOSE:
            return self.position_manager.close_position(
                symbol=opportunity.symbol,
                reason=f"Signal: {', '.join(opportunity.reasons[:2])}",
            )

        return OrderResult(success=False, message="Unknown decision type")

    def run_trading(self) -> list[OrderResult]:
        """
        Run full trading cycle: analyze and execute.

        Returns:
            List of order results
        """
        opportunities = self.run_analysis()
        results = []

        if not opportunities:
            logger.info("No trading opportunities found")
            return results

        logger.info(f"Executing {len(opportunities)} trades...")

        for opportunity in opportunities:
            result = self.execute_opportunity(opportunity)
            results.append(result)

            if result.success:
                self.state.trades_executed += 1
                logger.info(f"  {opportunity.symbol}: {result.message}")
            else:
                logger.error(f"  {opportunity.symbol}: FAILED - {result.message}")

            self.connection.ib.sleep(1)  # Pause between orders

        return results

    def get_status_report(self) -> str:
        """Generate a status report of current state."""

        lines = [
            "=" * 50,
            "TRADING BOT STATUS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            "",
        ]

        # Portfolio
        portfolio = self.position_manager.get_portfolio_value()
        lines.extend([
            "PORTFOLIO:",
            f"  Net Liquidation: ${portfolio.get('net_liquidation', 0):,.2f}",
            f"  Buying Power:    ${portfolio.get('buying_power', 0):,.2f}",
            f"  Unrealized P&L:  ${portfolio.get('unrealized_pnl', 0):,.2f}",
            "",
        ])

        # Positions
        positions = self.position_manager.get_positions()
        lines.append("POSITIONS:")
        if positions:
            for pos in positions:
                lines.append(
                    f"  {pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f} "
                    f"(P&L: ${pos.unrealized_pnl:,.2f})"
                )
        else:
            lines.append("  No open positions")
        lines.append("")

        # Open orders
        open_orders = self.order_manager.get_open_orders()
        lines.append("OPEN ORDERS:")
        if open_orders:
            for trade in open_orders:
                lines.append(
                    f"  {trade.order.action} {trade.order.totalQuantity} "
                    f"{trade.contract.symbol} ({trade.orderStatus.status})"
                )
        else:
            lines.append("  No open orders")
        lines.append("")

        # Last run stats
        if self.state.last_run:
            lines.extend([
                "LAST ANALYSIS:",
                f"  Time: {self.state.last_run.strftime('%Y-%m-%d %H:%M:%S')}",
                f"  Symbols analyzed: {self.state.symbols_analyzed}",
                f"  Opportunities: {len(self.state.opportunities)}",
                f"  Trades executed: {self.state.trades_executed}",
            ])

            if self.state.errors:
                lines.append(f"  Errors: {len(self.state.errors)}")

        lines.append("=" * 50)

        return "\n".join(lines)
