"""
Decision Engine - Trend-following / momentum strategy.
Computes TSMOM + CSMOM signals, generates rebalance orders,
and manages portfolio-level risk.
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
from .contracts import CONTRACT_REGISTRY
from .data_fetcher import DataFetcher
from .database import Database
from .indicators import (
    TrendFollowingAnalyzer, Signal,
    rank_cross_sectional, compute_combined_signal,
)
from .orders import OrderManager, PositionManager, OrderAction, OrderResult
from .config import trading_config, TradingConfig, currency_symbol

logger = logging.getLogger(__name__)


class TradeDecision(Enum):
    """Possible trade decisions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class TradeOpportunity:
    """Represents a rebalance action for one instrument."""
    symbol: str
    decision: TradeDecision
    signal: Signal
    current_price: float
    position_size: int  # target delta in shares (positive = buy, always positive here)
    reasons: list[str]
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None  # always None for trend-following
    target_weight: float = 0.0  # target portfolio weight
    signal_score: float = 0.0   # combined TSMOM + CSMOM score
    atr_value: float = 0.0      # current ATR for position sizing


@dataclass
class EngineState:
    """Current state of the decision engine."""
    last_run: Optional[datetime] = None
    symbols_analyzed: int = 0
    trades_executed: int = 0
    errors: list[str] = field(default_factory=list)
    opportunities: list[TradeOpportunity] = field(default_factory=list)
    market_ok: bool = True
    market_reason: str = ""
    # Trend-following state
    signals: dict = field(default_factory=dict)  # symbol -> combined signal
    peak_equity: float = 0.0
    current_drawdown: float = 0.0


class DecisionEngine:
    """
    Trend-following decision engine.

    Workflow:
    1. Fetch 1Y daily data for all instruments
    2. Compute TSMOM signal per instrument (multi-lookback)
    3. Compute CSMOM ranking across instruments
    4. Combine signals and apply threshold
    5. Calculate volatility-scaled target positions
    6. Generate rebalance orders (delta from current)
    7. Check portfolio-level risk (drawdown circuit breaker)

    Usage:
        engine = DecisionEngine()
        opportunities = engine.run_analysis()
    """

    def __init__(
        self,
        connection: Optional[ConnectionManager] = None,
        config: Optional[TradingConfig] = None,
        dry_run: bool = True,
    ):
        self.connection = connection or get_connection()
        self.config = config or trading_config
        self.dry_run = dry_run

        self.db = Database()
        self.fetcher = DataFetcher(self.connection)
        self.order_manager = OrderManager(self.connection, self.db)
        self.position_manager = PositionManager(self.connection, self.order_manager)

        self.state = EngineState()

    def _get_all_symbols(self) -> list[str]:
        """Get all symbols from trading universe (reloads watchlist each cycle)."""
        from .config import _load_watchlist
        self.config.symbols = _load_watchlist()
        symbols = []
        for asset_class_symbols in self.config.symbols.values():
            symbols.extend(asset_class_symbols)
        return symbols

    def _get_asset_class(self, symbol: str) -> Optional[str]:
        """Get the asset class for a symbol."""
        for asset_class, symbols in self.config.symbols.items():
            if symbol in symbols:
                return asset_class
        return None

    def _compute_all_signals(self, data: dict[str, any]) -> dict[str, dict]:
        """
        Compute TSMOM and CSMOM signals for all instruments.

        Args:
            data: Dict of symbol -> DataFrame

        Returns:
            Dict of symbol -> {tsmom, csmom, combined, reasons, price, atr, volatility}
        """
        lookbacks = [
            self.config.lookback_short,
            self.config.lookback_medium,
            self.config.lookback_long,
        ]

        # Step 1: Compute TSMOM for each instrument
        tsmom_scores = {}
        instrument_data = {}

        for symbol, df in data.items():
            analyzer = TrendFollowingAnalyzer(df, lookbacks=lookbacks, atr_period=self.config.atr_period)
            tsmom_score, reasons = analyzer.compute_tsmom_signal()
            price = analyzer.get_current_price()
            atr_val = analyzer.compute_atr()
            vol = analyzer.compute_volatility()

            tsmom_scores[symbol] = tsmom_score
            instrument_data[symbol] = {
                "tsmom": tsmom_score,
                "reasons": reasons,
                "price": price,
                "atr": atr_val,
                "volatility": vol,
            }

            logger.info(
                f"  {symbol}: TSMOM={tsmom_score:+.2f} | "
                f"${price:.2f} | ATR=${atr_val:.2f} | Vol={vol:.1%} | "
                f"{', '.join(reasons)}"
            )

        # Step 2: Compute CSMOM ranking
        csmom_scores = rank_cross_sectional(tsmom_scores)

        # Step 3: Combine signals
        results = {}
        for symbol in instrument_data:
            tsmom = instrument_data[symbol]["tsmom"]
            csmom = csmom_scores.get(symbol, 0.0)
            combined = compute_combined_signal(
                tsmom, csmom,
                self.config.tsmom_weight,
                self.config.csmom_weight,
            )

            instrument_data[symbol]["csmom"] = csmom
            instrument_data[symbol]["combined"] = combined
            results[symbol] = instrument_data[symbol]

            logger.info(
                f"  {symbol}: Combined={combined:+.2f} "
                f"(TSMOM={tsmom:+.2f} * {self.config.tsmom_weight} + "
                f"CSMOM={csmom:+.2f} * {self.config.csmom_weight})"
            )

        return results

    def _fx_to_base(self, currency: str, fx_rates: dict[str, float]) -> float:
        """Rate to convert 1 unit of `currency` into base currency.

        For the account base currency this is always 1.0. For others it's the
        IBKR ExchangeRate value (BASE per 1 CCY). Missing rates fall back to
        1.0 with a warning — better to size conservatively-in-units than to
        skip the symbol entirely.
        """
        if not currency or currency in ("BASE", ""):
            return 1.0
        rate = fx_rates.get(currency)
        if rate is None:
            logger.warning(
                f"FX rate for {currency} not available; assuming 1.0. "
                f"Sizing for {currency}-denominated symbols may be off."
            )
            return 1.0
        return rate

    def _calculate_target_positions(
        self,
        signals: dict[str, dict],
        capital: float,
    ) -> dict[str, dict]:
        """
        Calculate target position sizes using volatility-scaled sizing.

        Currency handling:
            `capital` is in the account's base currency (GBP for this account).
            Prices and ATRs come from IBKR in each contract's local currency
            (USD for most UCITS, GBP for EQQQ/VEUR/IJPN). Sizing must be done
            in the contract's local currency to avoid implicit FX errors.

            Per-symbol we convert `capital` -> `capital_local` using IBKR's
            published ExchangeRate. `target_weight` is computed in BASE so
            that asset-class and gross-exposure limits compare apples to apples.

        Args:
            signals: Dict of symbol -> signal data (from _compute_all_signals)
            capital: Deployable equity in BASE currency. Used as the sizing
                denominator (excludes accrued interest / paper inflation).

        Returns:
            Dict of symbol -> {target_shares, target_weight, direction, stop_price,
                               price, atr, currency, fx_to_base}
        """
        threshold = self.config.signal_threshold
        targets = {}

        # Filter to instruments with signals above threshold
        active_signals = {
            sym: data for sym, data in signals.items()
            if abs(data["combined"]) >= threshold
        }

        if not active_signals:
            logger.info("No signals above threshold — all flat")
            return targets

        # Rank by signal strength and cap to max_open_positions
        total_active = len(active_signals)
        sorted_signals = sorted(
            active_signals.items(),
            key=lambda x: abs(x[1]["combined"]),
            reverse=True,
        )
        active_signals = dict(sorted_signals[:self.config.max_open_positions])
        logger.info(
            f"Active signals: {total_active}, trading top {len(active_signals)}"
        )

        # FX rates: BASE per 1 CCY. Fetched once per rebalance.
        fx_rates = self.connection.get_fx_rates()
        if fx_rates:
            logger.info(
                "FX rates (BASE per 1 CCY): "
                + ", ".join(f"{k}={v:.4f}" for k, v in sorted(fx_rates.items()))
            )

        # Count active positions for risk budget distribution
        num_active = len(active_signals)
        # risk_per_position kept in BASE; converted per-symbol below
        risk_per_position_base = (capital * self.config.risk_budget) / num_active

        for symbol, data in active_signals.items():
            combined = data["combined"]
            price = data["price"]            # in symbol's local currency
            atr_val = data["atr"]            # in symbol's local currency
            vol = data["volatility"]

            if price <= 0 or atr_val <= 0:
                continue

            # Direction
            is_long = combined > 0
            if not is_long and not self.config.enable_shorting:
                continue  # Skip shorts if disabled

            # Currency-aware sizing: convert capital to symbol's local currency
            ccy = CONTRACT_REGISTRY.get(symbol, ("USD", "LSEETF"))[0]
            fx = self._fx_to_base(ccy, fx_rates)  # BASE per 1 CCY
            capital_local = capital / fx if fx > 0 else capital
            risk_per_position_local = risk_per_position_base / fx if fx > 0 else risk_per_position_base

            # Position size = risk_budget / (ATR * multiplier)
            # Equal-risk contribution per position. All units in local currency.
            local_risk = atr_val * self.config.atr_stop_multiplier
            if local_risk <= 0:
                continue

            target_value_local = risk_per_position_local / (local_risk / price)
            # Clamp to max position size (also in local currency)
            max_value_local = capital_local * self.config.max_position_pct
            target_value_local = min(target_value_local, max_value_local)

            # Round to nearest whole share, not floor. int() truncation
            # systematically under-sizes every position; at small capital that
            # bias is large on high-priced names (e.g. a $54 ETF at £2.5k buys
            # only ~4 shares, so flooring loses up to ~25% of the intended size).
            # The per-position cap is re-checked immediately below and aggregate
            # gross exposure is clamped downstream, so rounding up can't breach
            # either limit.
            target_shares = round(target_value_local / price)
            if target_shares * price > max_value_local:
                target_shares = int(max_value_local / price)  # never exceed 15% cap
            if target_shares <= 0:
                continue

            # Apply direction
            if not is_long:
                target_shares = -target_shares

            # Trailing stop: 3x ATR from current price (all in local currency)
            if is_long:
                stop_price = round(price - self.config.atr_stop_multiplier * atr_val, 2)
            else:
                stop_price = round(price + self.config.atr_stop_multiplier * atr_val, 2)

            # Target weight in BASE — comparable across currencies
            target_weight = (target_shares * price * fx) / capital

            targets[symbol] = {
                "target_shares": target_shares,
                "target_weight": target_weight,
                "direction": "LONG" if is_long else "SHORT",
                "stop_price": stop_price,
                "signal_score": combined,
                "atr": atr_val,
                "price": price,
                "currency": ccy,
                "fx_to_base": fx,
            }

        # Enforce asset class limits (uses target_weight already in BASE)
        targets = self._apply_asset_class_limits(targets, capital)

        # Enforce gross exposure limit
        targets = self._apply_gross_exposure_limit(targets, capital)

        return targets

    def _apply_asset_class_limits(
        self, targets: dict, capital: float,
    ) -> dict:
        """Reduce positions if an asset class exceeds its limit.

        target_weight is in BASE currency (set by _calculate_target_positions),
        so weight-based scaling works across mixed-currency contracts.
        """
        max_pct = self.config.max_asset_class_pct
        class_exposure = {}

        for symbol, t in targets.items():
            ac = self._get_asset_class(symbol) or "other"
            class_exposure.setdefault(ac, 0.0)
            class_exposure[ac] += abs(t["target_weight"])

        for ac, exposure in class_exposure.items():
            if exposure > max_pct:
                scale = max_pct / exposure
                for symbol, t in targets.items():
                    if (self._get_asset_class(symbol) or "other") == ac:
                        t["target_shares"] = int(t["target_shares"] * scale)
                        t["target_weight"] = (
                            t["target_shares"] * t["price"] * t["fx_to_base"]
                        ) / capital
                logger.info(f"Scaled {ac} from {exposure:.1%} to {max_pct:.1%}")

        return targets

    def _apply_gross_exposure_limit(
        self, targets: dict, capital: float,
    ) -> dict:
        """Scale all positions if gross exposure exceeds limit."""
        gross = sum(abs(t["target_weight"]) for t in targets.values())
        if gross > self.config.max_gross_exposure:
            scale = self.config.max_gross_exposure / gross
            for t in targets.values():
                t["target_shares"] = int(t["target_shares"] * scale)
                t["target_weight"] = (
                    t["target_shares"] * t["price"] * t["fx_to_base"]
                ) / capital
            logger.info(f"Scaled gross exposure from {gross:.1%} to {self.config.max_gross_exposure:.1%}")
        return targets

    def _check_portfolio_risk(self, net_liq: float) -> tuple[bool, str]:
        """
        Check portfolio-level risk (drawdown circuit breaker).

        Returns:
            Tuple of (is_ok, reason)
        """
        peak = self.db.get_peak_equity()
        if peak <= 0:
            peak = net_liq
            self.db.save_portfolio_snapshot(net_liq, 0.0, peak)

        # Update peak if new high
        if net_liq > peak:
            peak = net_liq

        drawdown = (peak - net_liq) / peak if peak > 0 else 0.0
        self.state.peak_equity = peak
        self.state.current_drawdown = drawdown

        self.db.save_portfolio_snapshot(net_liq, drawdown, peak)

        if drawdown >= self.config.drawdown_halt_pct:
            return False, f"HALT: {drawdown:.1%} drawdown from peak ${peak:,.0f} (threshold {self.config.drawdown_halt_pct:.0%})"

        if drawdown >= self.config.drawdown_reduce_pct:
            return True, f"REDUCE: {drawdown:.1%} drawdown — halving position sizes"

        return True, f"OK: {drawdown:.1%} drawdown from peak ${peak:,.0f}"

    def run_analysis(self) -> list[TradeOpportunity]:
        """
        Run full trend-following analysis on all instruments.

        Returns:
            List of TradeOpportunity objects representing rebalance actions
        """
        if not self.connection.ensure_connected():
            logger.error("Cannot run analysis: not connected")
            return []

        logger.info("Starting trend-following analysis...")
        self.state = EngineState(last_run=datetime.now())

        # Get portfolio value
        portfolio = self.position_manager.get_portfolio_value()
        net_liq = portfolio.get('net_liquidation', 0)
        sizing_capital = portfolio.get('sizing_capital') or net_liq
        if net_liq <= 0 or sizing_capital <= 0:
            logger.error("Cannot get portfolio value")
            return []

        logger.info(
            f"Portfolio: net_liq={net_liq:,.0f} sizing_capital={sizing_capital:,.0f} "
            f"(accrued={portfolio.get('accrued_cash', 0):,.0f})"
        )

        # Check portfolio-level risk (drawdown uses total wealth — NetLiq)
        risk_ok, risk_reason = self._check_portfolio_risk(net_liq)
        self.state.market_ok = risk_ok
        self.state.market_reason = risk_reason
        logger.info(f"Risk check: {risk_reason}")

        if "HALT" in risk_reason:
            # Engine flags the halt; bot.run_once checks state.market_ok and
            # flattens positions (live close_all + cancel orders, or close all
            # paper_trades). Returning [] here just blocks new entries.
            logger.warning(f"DRAWDOWN HALT — {risk_reason}. Bot will flatten.")
            return []

        # Fetch data for all instruments
        symbols = self._get_all_symbols()
        logger.info(f"\n--- Fetching {len(symbols)} instruments ---")

        data = {}
        for symbol in symbols:
            try:
                df = self.fetcher.get_historical_data(
                    symbol,
                    duration=self.config.data_duration,
                    bar_size=self.config.bar_size,
                )
                if df is not None and len(df) >= self.config.lookback_short + 5:
                    data[symbol] = df
                    self.db.save_ohlcv(df, symbol)
                else:
                    logger.warning(f"  {symbol}: insufficient data ({len(df) if df is not None else 0} bars)")
            except Exception as e:
                logger.error(f"  {symbol}: fetch error — {e}")
                self.state.errors.append(f"{symbol}: {e}")

            self.state.symbols_analyzed += 1
            self.connection.ib.sleep(0.5)

        logger.info(f"Got data for {len(data)}/{len(symbols)} instruments")

        if not data:
            logger.error("No data fetched — aborting")
            return []

        # Compute signals
        logger.info("\n--- Computing signals ---")
        signals = self._compute_all_signals(data)
        self.state.signals = {s: d["combined"] for s, d in signals.items()}

        # Save signals to DB for audit trail
        for symbol, sig_data in signals.items():
            self.db.save_instrument_signal(
                symbol=symbol,
                tsmom_score=sig_data["tsmom"],
                csmom_score=sig_data["csmom"],
                combined_score=sig_data["combined"],
                price=sig_data["price"],
                atr_value=sig_data["atr"],
                volatility=sig_data["volatility"],
            )

        # Calculate target positions (sized against deployable equity)
        logger.info("\n--- Calculating target positions ---")
        reduce_mode = "REDUCE" in risk_reason
        targets = self._calculate_target_positions(signals, sizing_capital)

        if reduce_mode:
            logger.info("REDUCE mode — halving all targets")
            for t in targets.values():
                t["target_shares"] = int(t["target_shares"] * 0.5)
                t["target_weight"] = (
                    t["target_shares"] * t["price"] * t["fx_to_base"]
                ) / sizing_capital

        # Generate opportunities (rebalance orders)
        opportunities = []
        for symbol, target in targets.items():
            target_shares = target["target_shares"]
            direction = target["direction"]
            price = target["price"]

            # For paper trading: create opportunity for new positions
            # (the bot.py handles the actual paper trade opening/closing)
            if target_shares > 0:
                decision = TradeDecision.BUY
            elif target_shares < 0:
                decision = TradeDecision.SELL
            else:
                continue

            reasons = signals[symbol]["reasons"] + [
                f"Combined signal: {target['signal_score']:+.2f}",
                f"Direction: {direction}",
                f"ATR: ${target['atr']:.2f}",
            ]

            opp = TradeOpportunity(
                symbol=symbol,
                decision=decision,
                signal=Signal(
                    symbol=symbol,
                    action=decision.value,
                    strength=abs(target["signal_score"]),
                    reasons=reasons,
                    indicators={
                        "tsmom": signals[symbol]["tsmom"],
                        "csmom": signals[symbol]["csmom"],
                        "combined": signals[symbol]["combined"],
                        "atr": target["atr"],
                        "volatility": signals[symbol]["volatility"],
                    },
                ),
                current_price=price,
                position_size=abs(target_shares),
                reasons=reasons,
                stop_loss_price=target["stop_price"],
                take_profit_price=None,
                target_weight=target["target_weight"],
                signal_score=target["signal_score"],
                atr_value=target["atr"],
            )
            opportunities.append(opp)

            logger.info(
                f"  {decision.value} {abs(target_shares)} {symbol} "
                f"@ ${price:.2f} (signal {target['signal_score']:+.2f}, "
                f"stop ${target['stop_price']:.2f})"
            )

        self.state.opportunities = opportunities
        logger.info(f"\nAnalysis complete: {len(opportunities)} rebalance actions")

        return opportunities

    def execute_opportunity(self, opportunity: TradeOpportunity) -> OrderResult:
        """Execute a single trade opportunity."""
        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would execute: {opportunity.decision.value} "
                f"{opportunity.position_size} {opportunity.symbol}"
            )
            return OrderResult(
                success=True,
                message=f"[DRY RUN] {opportunity.decision.value} {opportunity.symbol}",
            )

        action = (
            OrderAction.BUY if opportunity.decision == TradeDecision.BUY
            else OrderAction.SELL
        )
        result = self.order_manager.place_market_order(
            symbol=opportunity.symbol,
            action=action,
            quantity=opportunity.position_size,
            reason=f"Trend signal: {opportunity.signal_score:+.2f}",
        )

        # placeOrder() returns synchronously but IBKR rejections (e.g. Error 201
        # PRIIPs/KID) arrive ~50-200ms later as the order moves to Cancelled.
        # Wait briefly so the "trades executed" count reflects real fate.
        if result.success and result.trade is not None:
            terminal_bad = {"Cancelled", "ApiCancelled", "Inactive"}
            for _ in range(30):  # up to ~3s
                self.connection.ib.sleep(0.1)
                if result.trade.orderStatus.status in terminal_bad:
                    err = "; ".join(
                        f"{log.status}:{log.message[:120]}"
                        for log in result.trade.log
                        if log.message
                    ) or result.trade.orderStatus.status
                    logger.warning(
                        f"{opportunity.symbol}: order rejected post-submit — {err}"
                    )
                    result.success = False
                    result.message = f"Rejected: {err}"
                    return result
                if result.trade.orderStatus.status in ("PreSubmitted", "Submitted", "Filled"):
                    break

        # Attach native trailing stop server-side. Survives bot/gateway crashes
        # AND ratchets up automatically as price moves favourably.
        if result.success and opportunity.stop_loss_price:
            stop_action = (
                OrderAction.SELL if action == OrderAction.BUY else OrderAction.BUY
            )
            trail_amount = self.config.atr_stop_multiplier * opportunity.atr_value
            if trail_amount > 0:
                self.order_manager.place_trailing_stop_order(
                    symbol=opportunity.symbol,
                    action=stop_action,
                    quantity=opportunity.position_size,
                    trail_amount=trail_amount,
                    initial_stop_price=opportunity.stop_loss_price,
                    reason=f"Trailing stop {self.config.atr_stop_multiplier}xATR",
                )
            else:
                # ATR is zero — fall back to a fixed stop so the position is still protected
                logger.warning(
                    f"{opportunity.symbol}: ATR=0, using fixed stop instead of trailing"
                )
                self.order_manager.place_stop_order(
                    symbol=opportunity.symbol,
                    action=stop_action,
                    quantity=opportunity.position_size,
                    stop_price=opportunity.stop_loss_price,
                    reason="Fallback fixed stop (ATR=0)",
                )

        return result

    def get_status_report(self) -> str:
        """Generate a status report of current state."""
        lines = [
            "=" * 50,
            "TREND-FOLLOWING BOT STATUS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            "",
        ]

        portfolio = self.position_manager.get_portfolio_value()
        # Account-level values are reported in the account base currency (GBP
        # for this account), not USD — label them accordingly.
        base = currency_symbol(portfolio.get('currency'))
        lines.extend([
            "PORTFOLIO:",
            f"  Net Liquidation:  {base}{portfolio.get('net_liquidation', 0):,.2f}",
            f"  Sizing Capital:   {base}{portfolio.get('sizing_capital', 0):,.2f}",
            f"  Accrued Cash:     {base}{portfolio.get('accrued_cash', 0):,.2f}",
            f"  Buying Power:     {base}{portfolio.get('buying_power', 0):,.2f}",
            f"  Unrealized P&L:   {base}{portfolio.get('unrealized_pnl', 0):,.2f}",
            f"  Drawdown:         {self.state.current_drawdown:.1%}",
            "",
        ])

        positions = self.position_manager.get_positions()
        lines.append("POSITIONS:")
        if positions:
            for pos in positions:
                # Per-position price/PnL are in the instrument's local currency.
                local = currency_symbol(CONTRACT_REGISTRY.get(pos.symbol, ("USD",))[0])
                lines.append(
                    f"  {pos.symbol}: {pos.quantity} shares @ {local}{pos.avg_cost:.2f} "
                    f"(P&L: {local}{pos.unrealized_pnl:,.2f})"
                )
        else:
            lines.append("  No open positions")
        lines.append("")

        if self.state.signals:
            lines.append("SIGNALS (top 10):")
            sorted_sigs = sorted(self.state.signals.items(), key=lambda x: abs(x[1]), reverse=True)
            for sym, score in sorted_sigs[:10]:
                direction = "LONG" if score > 0 else "SHORT" if score < 0 else "FLAT"
                lines.append(f"  {sym}: {score:+.2f} ({direction})")
            lines.append("")

        if self.state.last_run:
            lines.extend([
                "LAST ANALYSIS:",
                f"  Time: {self.state.last_run.strftime('%Y-%m-%d %H:%M:%S')}",
                f"  Instruments analyzed: {self.state.symbols_analyzed}",
                f"  Rebalance actions: {len(self.state.opportunities)}",
                f"  Risk status: {self.state.market_reason}",
            ])
            if self.state.errors:
                lines.append(f"  Errors: {len(self.state.errors)}")

        lines.append("=" * 50)
        return "\n".join(lines)
