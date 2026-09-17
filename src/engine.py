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
from .contracts import CONTRACT_REGISTRY, GBX_PER_GBP
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
        self._cash_committed_base = 0.0  # see reset_cash_committed()

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
        if currency == "GBX":
            # Pence-quoted LSE line (IBKR priceMagnifier=100; see contracts.py).
            # 100 GBX = 1 GBP, so BASE per 1 GBX is a hundredth of BASE per GBP.
            return self._fx_to_base("GBP", fx_rates) / GBX_PER_GBP
        rate = fx_rates.get(currency)
        if rate is None:
            logger.warning(
                f"FX rate for {currency} not available; assuming 1.0. "
                f"Sizing for {currency}-denominated symbols may be off."
            )
            return 1.0
        return rate

    # ---- forward-test sleeve (attempt 9) ------------------------------------
    # The sleeve (research/2026-09-17_forward_test/PREREG_9) runs a separate
    # strategy inside the same account. Momentum must not size off its money,
    # spend its cash, or treat its holdings as its own.

    def _sleeve_symbols(self) -> set:
        """Symbols owned by the forward-test sleeve; empty when it is off."""
        sleeve = getattr(self, "sleeve", None)
        try:
            return sleeve.symbols if sleeve and sleeve.config.enabled else set()
        except Exception:
            return set()

    def _sleeve_claim(self) -> float:
        """Sleeve positions + cash reserve, in base currency (0 when off)."""
        sleeve = getattr(self, "sleeve", None)
        try:
            return float(sleeve.claim_on_account()) if sleeve and sleeve.config.enabled else 0.0
        except Exception as e:
            logger.warning(f"Sleeve claim unavailable; sizing left unadjusted: {e}")
            return 0.0

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
            (USD for most UCITS, GBP for VEUR, GBX/pence for EQQQ/IJPN). Sizing must be done
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

        # Filter to tradeable instruments with signals above threshold.
        # When shorting is disabled, drop short signals HERE so they never
        # compete for position slots — a strong short would otherwise rank high
        # by |signal| only to be discarded in the sizing loop, under-deploying
        # the book (e.g. 2026-05-25: shorts IDTM -1.00 / NGAS -0.96 displaced
        # the longs CNYA/IJPN).
        # Re-entry cooldown: a symbol whose protective stop fired recently is
        # dropped HERE (same reasoning as the shorts filter above) so the slot
        # backfills to the next-ranked name instead of being wasted downstream.
        # Symbols we still HOLD are never filtered — that would zero their target
        # and force a sale, which is the opposite of the intent.
        try:
            sleeve_syms = self._sleeve_symbols()
            held = {
                p.symbol for p in self.position_manager.get_positions()
                if p.quantity != 0 and p.symbol not in sleeve_syms
            }
        except Exception as e:
            logger.warning(f"Could not fetch positions for cooldown filter: {e}")
            held = set()
        cooldowns = self.db.get_active_cooldowns()
        blocked = {
            sym: until for sym, until in cooldowns.items() if sym not in held
        }
        for sym, until in sorted(blocked.items()):
            if sym in signals and abs(signals[sym]["combined"]) >= threshold:
                logger.info(
                    f"  {sym}: in re-entry cooldown until {until[:10]} "
                    f"— slot backfilled from next-ranked signal"
                )

        # Minimum-volatility filter: cash proxies (T-bill/short-bond ETFs)
        # score top-rank momentum by standing still in a down tape, then a
        # sub-1% ATR stop makes the fixed ~£6 round-trip commission exceed
        # the entire risk unit (IBTA 2026-07-29: 215% commission-to-risk).
        # Applied to HELD symbols too — unlike cooldown, the intent is "no
        # new money in untrendable names", and since opportunities are only
        # generated from targets, dropping a held symbol never forces a
        # sale: its trailing stop stays live and handles the exit.
        min_vol = self.config.min_volatility
        for sym, data in sorted(signals.items()):
            if (data["volatility"] < min_vol
                    and abs(data["combined"]) >= threshold
                    and (self.config.enable_shorting or data["combined"] > 0)
                    and sym not in blocked):
                logger.info(
                    f"  {sym}: vol {data['volatility']:.1%} below "
                    f"{min_vol:.0%} floor — untradeable cash proxy, "
                    f"slot backfilled from next-ranked signal"
                )

        active_signals = {
            sym: data for sym, data in signals.items()
            if abs(data["combined"]) >= threshold
            and (self.config.enable_shorting or data["combined"] > 0)
            and sym not in blocked
            and data["volatility"] >= min_vol
        }

        if not active_signals:
            logger.info("No signals above threshold — all flat")
            return targets

        # Rank by signal strength. We deliberately do NOT pre-slice to
        # max_open_positions: a high-ranked name can be unaffordable (e.g. a
        # ~£1.8k/share ETF that breaches the per-position cap → 0 shares at ~£5k
        # NLV). Pre-slicing would let it consume a slot and shrink the book.
        # Instead the sizing loop walks this full ranked list and fills up to
        # max_open_positions with names that clear ≥1 share, skipping the rest.
        total_active = len(active_signals)
        sorted_signals = sorted(
            active_signals.items(),
            key=lambda x: abs(x[1]["combined"]),
            reverse=True,
        )

        # FX rates: BASE per 1 CCY. Fetched once per rebalance.
        fx_rates = self.connection.get_fx_rates()
        if fx_rates:
            logger.info(
                "FX rates (BASE per 1 CCY): "
                + ", ".join(f"{k}={v:.4f}" for k, v in sorted(fx_rates.items()))
            )

        # Intended position count sets the per-position risk budget — capped at
        # the number of tradeable signals available. Sizing uses this fixed N so
        # each position carries consistent risk regardless of how many of the
        # top-ranked names end up affordable.
        num_active = min(self.config.max_open_positions, total_active)
        logger.info(
            f"Active signals: {total_active}, filling up to {num_active} positions"
        )
        # risk_per_position kept in BASE; converted per-symbol below
        risk_per_position_base = (capital * self.config.risk_budget) / num_active

        for symbol, data in sorted_signals:
            # Stop once the book is full of affordable names; everything below
            # this rank is surplus to max_open_positions.
            if len(targets) >= self.config.max_open_positions:
                break
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
                logger.info(
                    f"  {symbol}: 0 shares within "
                    f"{self.config.max_position_pct:.0%} cap @ {price:.2f} {ccy} "
                    f"— unaffordable, slot backfilled from next-ranked signal"
                )
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
        self.reset_cash_committed()

        # Get portfolio value
        portfolio = self.position_manager.get_portfolio_value()
        net_liq = portfolio.get('net_liquidation', 0)
        sizing_capital = portfolio.get('sizing_capital') or net_liq
        # Ring-fence the forward-test sleeve: its positions and cash reserve are
        # not momentum's to size off. net_liq is deliberately left whole — the
        # drawdown brakes compare it with a stored peak from before the sleeve
        # existed, so subtracting here would fake an instant drawdown and halt.
        sleeve_claim = self._sleeve_claim()
        if sleeve_claim > 0:
            sizing_capital = max(0.0, sizing_capital - sleeve_claim)
            logger.info(f"Sleeve ring-fence: {sleeve_claim:,.2f} excluded from sizing capital")
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

    def _get_settled_cash_base(self) -> Optional[float]:
        """Settled cash available for a new BUY, in account base currency.

        On a cash account IBKR's `AvailableFunds` IS settled cash (probe
        2026-08-18: AvailableFunds == SettledCash == BuyingPower == £638.05
        while TotalCashValue was £1,084 with £446 of same-day proceeds
        unsettled). It is also the figure IBKR quotes in the Error 201 text
        ("Available settled cash converted to base: 641.69 GBP"). Returns None
        if it can't be read, so callers fall back to the old behaviour (place
        the full order and let IBKR decide) rather than blocking trading.
        """
        try:
            summary = self.connection.get_account_summary()
            raw = (summary.get("AvailableFunds") or {}).get("value")
            if raw is None:
                return None
            val = float(raw)
            if val < 0:
                return None
            # The sleeve's reserve is spoken for; momentum may not spend it.
            sleeve = getattr(self, "sleeve", None)
            try:
                if sleeve and sleeve.config.enabled:
                    val = max(0.0, val - float(sleeve.reserve()))
            except Exception as e:
                logger.warning(f"Sleeve reserve unavailable; settled cash not adjusted: {e}")
            return val
        except Exception as e:  # never let a read failure block an entry
            logger.warning(f"Could not read AvailableFunds: {e}")
            return None

    # ---- per-cycle committed-cash tally ------------------------------------
    # IBKR reserves settled cash the moment a BUY is accepted, but the
    # AvailableFunds we read is ib_insync's cached account summary, which IBKR
    # pushes on a lag. 2026-09-14 13:01:53: a CMOD top-up consumed ~£700 and
    # 0.2 s later the IJPN entry still read £1,804 (IBKR's real figure was
    # £1,102.65), sized 74 shares untrimmed and was rejected Error 201 — the
    # gate's own "cash needed" matched IBKR's to within £5; only the input was
    # stale. So every BUY accepted in this cycle is charged to a tally that
    # `_affordable_quantity` subtracts from the read, a rejected BUY is
    # released again (IBKR never reserved for it), and the tally is reset when
    # a new analysis cycle starts. Deterministic; no extra broker round-trip.

    def reset_cash_committed(self) -> None:
        """Start of a rebalance cycle: nothing placed yet."""
        self._cash_committed_base = 0.0

    def _cash_committed(self) -> float:
        return float(getattr(self, "_cash_committed_base", 0.0) or 0.0)

    def _estimated_cost_base(self, symbol: str, quantity: int, price: float) -> float:
        """quantity × price × fx × (1 + settled_cash_buffer) in base currency —
        the same estimate `_affordable_quantity` uses, so charge and check agree."""
        if quantity <= 0 or price <= 0:
            return 0.0
        ccy = CONTRACT_REGISTRY.get(symbol, ("USD", "LSEETF"))[0]
        get_rates = getattr(self.connection, "get_fx_rates", None)
        rates = (get_rates() if callable(get_rates) else None) or {}
        fx = self._fx_to_base(ccy, rates)
        buffer = float(getattr(self.config, "settled_cash_buffer", 0.0) or 0.0)
        return quantity * price * fx * (1.0 + buffer)

    def _commit_cash(self, symbol: str, quantity: int, price: float) -> None:
        """A BUY was accepted by IBKR: charge its estimated cost to this cycle."""
        try:
            cost = self._estimated_cost_base(symbol, quantity, price)
        except Exception as e:  # never let bookkeeping break the order path
            logger.warning(f"{symbol}: could not estimate committed cash: {e}")
            return
        if cost <= 0:
            return
        self._cash_committed_base = self._cash_committed() + cost
        logger.info(
            f"  {symbol}: ~{cost:,.2f} base committed by this BUY "
            f"(cycle total {self._cash_committed():,.2f})"
        )

    def _release_cash(self, symbol: str, quantity: int, price: float) -> None:
        """A BUY was rejected: IBKR reserved nothing, so un-charge it."""
        try:
            cost = self._estimated_cost_base(symbol, quantity, price)
        except Exception as e:
            logger.warning(f"{symbol}: could not estimate released cash: {e}")
            return
        self._cash_committed_base = max(0.0, self._cash_committed() - cost)

    def _affordable_quantity(
        self, symbol: str, quantity: int, price: float, *, is_new_entry: bool
    ) -> int:
        """Trim a BUY to what settled cash covers. Returns the quantity to send
        (0 = skip). Never scales UP.

        Cost is estimated as ``quantity × price × fx × (1 + settled_cash_buffer)``
        in base currency; the buffer covers commission plus IBKR's market-order
        price cushion (both are what pushed the 08-17/18 EIMU orders £31/£37 over
        the line). A trimmed NEW entry smaller than ``min_partial_entry_pct`` of
        target is skipped — the $4 minimum commission makes tiny lots pointless
        and the top-up path will size it properly once proceeds settle. A
        trimmed TOP-UP smaller than ``min_partial_topup_pct`` of the wanted
        delta is skipped for the same reason: on 2026-08-31 a 13-share CMOD
        top-up trimmed to 3 shares paid the $4 minimum on ~$100 of stock. (The
        caller already gated on the 30% drift threshold, so the untrimmed
        delta is never tiny — only a cash-starved one is.)
        """
        if quantity <= 0 or price <= 0:
            return 0
        settled = self._get_settled_cash_base()
        if settled is None:
            return quantity  # can't tell — behave as before

        committed = self._cash_committed()
        if committed > 0:
            # The cached AvailableFunds may predate BUYs placed seconds ago in
            # this same cycle (IJPN 2026-09-14) — net them off first.
            effective = max(0.0, settled - committed)
            logger.info(
                f"  {symbol}: settled cash {settled:,.2f} less {committed:,.2f} "
                f"already committed this cycle = {effective:,.2f} available"
            )
            settled = effective

        ccy = CONTRACT_REGISTRY.get(symbol, ("USD", "LSEETF"))[0]
        fx = self._fx_to_base(ccy, self.connection.get_fx_rates() or {})
        unit_cost_base = price * fx * (1.0 + self.config.settled_cash_buffer)
        if unit_cost_base <= 0:
            return quantity

        affordable = int(settled / unit_cost_base)
        if affordable >= quantity:
            return quantity

        base_sym = currency_symbol(self._base_currency())
        need = quantity * unit_cost_base
        if affordable <= 0:
            logger.info(
                f"  {symbol}: settled cash {base_sym}{settled:,.2f} covers 0 of "
                f"{quantity} shares (~{base_sym}{need:,.2f} needed) — skipping "
                f"until proceeds settle"
            )
            return 0
        frac = affordable / quantity
        kind = "entry" if is_new_entry else "top-up"
        floor = (
            self.config.min_partial_entry_pct
            if is_new_entry
            else float(getattr(self.config, "min_partial_topup_pct", 0.0) or 0.0)
        )
        if floor > 0 and frac < floor:
            logger.info(
                f"  {symbol}: settled cash {base_sym}{settled:,.2f} covers only "
                f"{affordable}/{quantity} shares ({frac:.0%}, below "
                f"{floor:.0%} {kind} floor) — skipping until proceeds settle"
            )
            return 0
        logger.info(
            f"  {symbol}: settled cash {base_sym}{settled:,.2f} covers "
            f"{affordable}/{quantity} shares — trimming "
            f"{'entry' if is_new_entry else 'top-up'} to {affordable} "
            f"({frac:.0%} of target); remainder tops up once proceeds settle"
        )
        return affordable

    def _base_currency(self) -> str:
        try:
            summary = self.connection.get_account_summary()
            return (summary.get("NetLiquidation") or {}).get("currency") or ""
        except Exception:
            return ""

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
        quantity = opportunity.position_size
        if action == OrderAction.BUY:
            # Cash account: trim to settled cash instead of letting IBKR reject
            # the whole order (Error 201) and leaving the slot idle for a day.
            quantity = self._affordable_quantity(
                opportunity.symbol, quantity, opportunity.current_price,
                is_new_entry=True,
            )
            if quantity <= 0:
                return OrderResult(
                    success=False,
                    message="Insufficient settled cash — deferred until proceeds settle",
                )
        result = self.order_manager.place_market_order(
            symbol=opportunity.symbol,
            action=action,
            quantity=quantity,
            reason=f"Trend signal: {opportunity.signal_score:+.2f}",
        )
        if action == OrderAction.BUY and result.success:
            self._commit_cash(opportunity.symbol, quantity, opportunity.current_price)

        # placeOrder() returns synchronously, but the BUY isn't on the books
        # until it FILLS. Attaching the protective SELL before the fill makes
        # IBKR read it as opening a short → Error 201 on a cash account (this
        # left 4 of 8 positions naked on the 2026-05-22 live cutover). So wait
        # for a terminal outcome — Filled (good) or a rejection (bad) — before
        # attaching the stop. Breaking on Submitted/PreSubmitted was the bug:
        # those mean "accepted by IBKR", not "I now hold shares".
        filled = False
        if result.success and result.trade is not None:
            terminal_bad = {"Cancelled", "ApiCancelled", "Inactive"}
            for _ in range(50):  # up to ~5s; RTH market orders fill in <1s
                self.connection.ib.sleep(0.1)
                status = result.trade.orderStatus.status
                if status in terminal_bad:
                    err = "; ".join(
                        f"{log.status}:{log.message[:120]}"
                        for log in result.trade.log
                        if log.message
                    ) or status
                    logger.warning(
                        f"{opportunity.symbol}: order rejected post-submit — {err}"
                    )
                    if action == OrderAction.BUY:
                        self._release_cash(
                            opportunity.symbol, quantity, opportunity.current_price
                        )
                    result.success = False
                    result.message = f"Rejected: {err}"
                    return result
                if status == "Filled":
                    filled = True
                    break
            if not filled:
                # BUY accepted but unfilled within the window — do NOT attach a
                # naked stop (it'd be rejected as a short). The next risk-check
                # reconcile places the stop once the fill lands.
                logger.warning(
                    f"{opportunity.symbol}: BUY not filled within wait window "
                    f"(status={result.trade.orderStatus.status}); "
                    f"stop deferred to reconcile"
                )

        # Attach native trailing stop server-side. Survives bot/gateway crashes
        # AND ratchets up automatically as price moves favourably. Only once the
        # entry has actually filled (see fill-wait above).
        if result.success and filled and opportunity.stop_loss_price:
            stop_action = (
                OrderAction.SELL if action == OrderAction.BUY else OrderAction.BUY
            )
            trail_amount = self.config.atr_stop_multiplier * opportunity.atr_value
            if trail_amount > 0:
                self.order_manager.place_trailing_stop_order(
                    symbol=opportunity.symbol,
                    action=stop_action,
                    quantity=quantity,
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
                    quantity=quantity,
                    stop_price=opportunity.stop_loss_price,
                    reason="Fallback fixed stop (ATR=0)",
                )

        return result

    def top_up_position(
        self, opportunity: TradeOpportunity, held_qty: int
    ) -> OrderResult:
        """Buy an existing position up to target, then re-cover it with one stop.

        Why this exists: ``opportunity.position_size`` is the FULL target, and
        the rebalance loop used to skip any symbol already held. Positions
        therefore froze at whatever size they were opened at and never grew
        into a new target — after the 2026-07-27 sizing fix the book sat at 36%
        deployed while the engine was asking for ~73%.

        Sequencing is the whole safety story here:

        1. BUY the shortfall first. The shares already held stay covered by
           their existing stop for the entire buy — there is no window where
           the original position is unprotected.
        2. Only once the buy has FILLED, swap the stop. Placing the new
           full-size stop before cancelling the old one would put more SELL
           quantity on the book than we hold, which a cash account rejects as
           an attempted short (Error 201).

        If the buy does not fill in the wait window, the stop is deliberately
        left alone: the original shares keep their cover and the quantity-aware
        reconcile picks up the unprotected remainder within 30s.
        """
        target = opportunity.position_size
        delta = target - held_qty

        if delta <= 0:
            return OrderResult(
                success=False, message=f"no top-up needed ({held_qty}/{target})"
            )

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would top up {opportunity.symbol}: "
                f"+{delta} shares ({held_qty} -> {target})"
            )
            return OrderResult(
                success=True, message=f"[DRY RUN] top-up {opportunity.symbol}"
            )

        # Don't buy more of a name that is about to stop out: the added shares
        # would carry almost no distance to the (kept) ratchet, so a stop-out
        # turns the whole top-up into commission. Checked before the settled-
        # cash read so a skip costs nothing.
        gate = self._topup_stop_buffer_ok(opportunity)
        if gate is not None:
            return OrderResult(success=False, message=gate)

        # Cash account: buy what settled cash covers rather than have IBKR
        # reject the whole top-up (subject to the min_partial_topup_pct floor).
        delta = self._affordable_quantity(
            opportunity.symbol, delta, opportunity.current_price, is_new_entry=False
        )
        if delta <= 0:
            return OrderResult(
                success=False,
                message="Insufficient settled cash — top-up deferred until proceeds settle",
            )
        target = held_qty + delta  # the stop must cover exactly what we'll hold

        result = self.order_manager.place_market_order(
            symbol=opportunity.symbol,
            action=OrderAction.BUY,
            quantity=delta,
            reason=(
                f"Top-up to target: {held_qty} -> {target} "
                f"(signal {opportunity.signal_score:+.2f})"
            ),
        )
        if not result.success or result.trade is None:
            return result
        self._commit_cash(opportunity.symbol, delta, opportunity.current_price)

        filled = False
        terminal_bad = {"Cancelled", "ApiCancelled", "Inactive"}
        for _ in range(50):  # ~5s, same budget as a fresh entry
            self.connection.ib.sleep(0.1)
            status = result.trade.orderStatus.status
            if status in terminal_bad:
                err = "; ".join(
                    f"{log.status}:{log.message[:120]}"
                    for log in result.trade.log
                    if log.message
                ) or status
                logger.warning(
                    f"{opportunity.symbol}: top-up rejected post-submit — {err}"
                )
                self._release_cash(opportunity.symbol, delta, opportunity.current_price)
                result.success = False
                result.message = f"Rejected: {err}"
                return result
            if status == "Filled":
                filled = True
                break

        if not filled:
            # filled_quantity stays 0 so the caller can tell "order accepted"
            # from "shares actually acquired" — result.success only means IBKR
            # took the order. Conflating the two logged "Topped up COPA to 14"
            # on 2026-07-28 while 10 of those shares were still unfilled.
            result.filled_quantity = 0
            logger.warning(
                f"{opportunity.symbol}: top-up BUY not filled within wait window "
                f"(status={result.trade.orderStatus.status}); existing stop left "
                f"in place, reconcile will extend cover once the fill lands"
            )
            return result

        result.filled_quantity = delta

        trail_amount = self.config.atr_stop_multiplier * opportunity.atr_value
        if trail_amount <= 0:
            logger.error(
                f"{opportunity.symbol}: ATR=0 after top-up — cannot size a trail; "
                f"leaving reconcile to protect the added shares"
            )
            return result

        self.order_manager.replace_trailing_stop(
            symbol=opportunity.symbol,
            action=OrderAction.SELL,
            quantity=target,
            trail_amount=trail_amount,
            initial_stop_price=opportunity.stop_loss_price,
            reason=f"Top-up: re-cover full position of {target}",
        )
        return result

    def _topup_stop_buffer_ok(self, opportunity: TradeOpportunity):
        """Top-up buffer gate. Returns None when the top-up may proceed, or the
        skip reason (str) when the name sits within ``topup_min_stop_buffer_atr``
        ATRs of the stop it will actually carry after the swap.

        Why: a falling price RAISES the share target, so a position drifting
        down toward its own ratcheted stop is exactly when it crosses the 30%
        drift line — AIGA on 2026-09-11 sat at 70.4% of target and 1.2% above
        its 7.05 ratchet. Because the swap keeps the ratchet (f5208d3) the
        added shares would carry ~$0.04 of risk each against a $4 commission;
        a stop-out minutes later turns the whole top-up into fee.

        The stop the position will carry is max(ratcheted trigger, fresh
        price-3xATR level) — the same rule ``replace_trailing_stop`` applies.
        Fail-open: no ATR, no readable stops, or an IBKR error lets the old
        path run (logged) — a wrongly-skipped top-up retries tomorrow, a
        wrongly-allowed one costs $4; neither is worth a hard failure.
        """
        min_mult = float(getattr(self.config, "topup_min_stop_buffer_atr", 0.0) or 0.0)
        if min_mult <= 0:
            return None
        symbol = opportunity.symbol
        atr = float(opportunity.atr_value or 0.0)
        price = float(opportunity.current_price or 0.0)
        if atr <= 0 or price <= 0:
            logger.warning(
                f"{symbol}: cannot judge the stop buffer (price={price}, ATR={atr}) "
                f"— top-up gate not applied"
            )
            return None
        try:
            stops = self.order_manager.protective_stops_for(symbol, OrderAction.SELL.value)
            ratchet = OrderManager._ratcheted_trigger(stops, OrderAction.SELL)
        except Exception as e:  # fail-open, same as the settled-cash read
            logger.warning(
                f"{symbol}: could not read working stops for the top-up gate "
                f"({e}) — top-up gate not applied"
            )
            return None
        levels = [float(v) for v in (ratchet, opportunity.stop_loss_price) if v]
        if not levels:
            return None  # naked and no fresh level — reconcile's job, not a reason to skip
        carried_stop = max(levels)
        buffer = price - carried_stop
        required = min_mult * atr
        if buffer + 1e-9 < required:
            logger.info(
                f"  {symbol}: price {price:.3f} is only {buffer:.3f} "
                f"({buffer / atr:.2f}xATR, {buffer / price:.1%}) above the "
                f"{carried_stop:.3f} stop it would carry — top-up skipped "
                f"(floor {min_mult:g}xATR); a stop-out would turn the added "
                f"shares into pure commission"
            )
            return (
                f"Skipped: only {buffer / atr:.2f}xATR above the {carried_stop:.3f} "
                f"stop (floor {min_mult:g}xATR)"
            )
        return None

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
