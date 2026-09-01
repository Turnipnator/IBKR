"""
Main Trading Bot - Trend-following / momentum strategy.
Daily rebalance at 14:00 Europe/London (mid-LSE session) with intraday risk checks.
"""

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import logging
import os
import time
import signal
import sys
from datetime import datetime, time as dtime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .connection import ConnectionManager
from .engine import DecisionEngine
from .database import Database
from .config import ibkr_config, telegram_config, trading_config, currency_symbol
from .telegram_bot import TelegramNotifier, get_notifier, check_telegram_commands
from .gateway_monitor import GatewayMonitor
from .data_health_checker import DataHealthChecker

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Trend-following trading bot with daily rebalance scheduling.

    Schedule:
    - Daily rebalance at 14:00 Europe/London (mid-LSE session, configurable)
    - Intraday risk checks every 4 hours (trailing stops, drawdown)
    - Checks for Telegram commands continuously

    Usage:
        bot = TradingBot(dry_run=True)
        bot.run_scheduled()
    """

    # LSE trading hours (local Europe/London — handles BST/GMT automatically)
    MARKET_OPEN = dtime(8, 0)
    MARKET_CLOSE = dtime(16, 30)
    MARKET_TZ = ZoneInfo("Europe/London")

    def __init__(
        self,
        dry_run: bool = True,
        run_interval_minutes: int = 60,
        enable_telegram: bool = True,
    ):
        self.dry_run = dry_run
        self.run_interval = run_interval_minutes * 60
        self.running = False

        self.connection = ConnectionManager()
        self.engine = DecisionEngine(
            connection=self.connection,
            dry_run=dry_run,
        )
        self.db = Database()
        self.notifier = get_notifier() if enable_telegram else None

        self._last_summary_date: Optional[str] = None
        self._last_rebalance_date: Optional[str] = None
        self._last_rebalance_at: Optional[datetime] = None
        self._last_risk_check: Optional[datetime] = None
        self._consecutive_failures = 0
        self._failure_alert_threshold = 1
        self._last_failure_alert: Optional[str] = None
        self._base_currency: Optional[str] = None
        self._started_at: datetime = datetime.now()
        self._daily_loss_halt_date: Optional[str] = None
        self._last_nlv_drift: Optional[float] = None
        self._last_parity_status: Optional[str] = None

        # Gateway auto-restart on connection failure
        self.gateway_monitor = GatewayMonitor(notifier=self.notifier)
        self.connection.on_reconnect_failed(self.gateway_monitor.restart_gateway)

        # Data-farm health probe (catches "TCP up but data farm dead")
        self.data_health = DataHealthChecker(
            connection=self.connection,
            gateway_monitor=self.gateway_monitor,
            notifier=self.notifier,
        )

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received, stopping bot...")
        self.running = False

    def _is_market_hours(self) -> bool:
        """Check if currently within LSE market hours."""
        now_local = datetime.now(self.MARKET_TZ)
        if now_local.weekday() >= 5:
            return False
        return self.MARKET_OPEN <= now_local.time() <= self.MARKET_CLOSE

    def _is_rebalance_time(self) -> bool:
        """Check if it's time for the daily rebalance."""
        now_local = datetime.now(self.MARKET_TZ)
        today = now_local.strftime('%Y-%m-%d')

        # Already rebalanced today
        if self._last_rebalance_date == today:
            return False

        rebalance_time = dtime(
            trading_config.rebalance_hour,
            trading_config.rebalance_minute,
        )
        # Trigger within a 5-minute window of the rebalance time
        current = now_local.time()
        target_minutes = rebalance_time.hour * 60 + rebalance_time.minute
        current_minutes = current.hour * 60 + current.minute
        return 0 <= (current_minutes - target_minutes) < 5

    def _is_risk_check_time(self) -> bool:
        """Check if it's time for an intraday risk check."""
        if self._last_risk_check is None:
            return True

        hours = trading_config.risk_check_interval_hours
        elapsed = (datetime.now() - self._last_risk_check).total_seconds() / 3600
        return elapsed >= hours

    def _check_paper_trades(self) -> int:
        """
        Check open paper trades — ATR trailing stops only (no TP).
        Returns number of trades closed.
        """
        open_trades = self.db.get_open_paper_trades()
        if not open_trades:
            return 0

        closed_count = 0
        logger.info(f"Checking {len(open_trades)} open paper trades...")

        for trade in open_trades:
            symbol = trade['symbol']
            trade_id = trade['id']
            entry_price = trade['entry_price']
            stop_loss = trade['stop_loss']
            is_long = trade['action'] == 'BUY'

            # Check minimum hold period (still track prices and tighten stops)
            in_min_hold = False
            min_exit_date = trade.get('min_exit_date')
            if min_exit_date:
                min_dt = datetime.fromisoformat(min_exit_date)
                if datetime.now() < min_dt:
                    in_min_hold = True

            # Fetch enough bars for ATR(20) ratchet below — "5 D" silently
            # disabled the trailing-stop tighten because len(df) never reached 20.
            try:
                df = self.engine.fetcher.get_historical_data(
                    symbol, duration="60 D", bar_size="1 day"
                )
                if df is None or df.empty:
                    continue
                current_price = float(df['close'].iloc[-1])
            except Exception as e:
                logger.warning(f"Could not get price for {symbol}: {e}")
                continue

            # ATR trailing stop logic
            if stop_loss:
                best_price = trade.get('best_price') or entry_price

                # Update best price
                if is_long:
                    best_price = max(best_price, current_price)
                else:
                    best_price = min(best_price, current_price)

                # Compute new trailing stop from ATR
                from .indicators import atr as compute_atr
                if len(df) >= 20:
                    atr_val = compute_atr(df['high'], df['low'], df['close'], period=20).iloc[-1]
                    if not pd.isna(atr_val) and atr_val > 0:
                        multiplier = trading_config.atr_stop_multiplier
                        if is_long:
                            new_sl = round(best_price - multiplier * atr_val, 2)
                            if new_sl > stop_loss:
                                logger.info(f"  #{trade_id} {symbol}: Trail ${stop_loss:.2f} -> ${new_sl:.2f} (best ${best_price:.2f})")
                                stop_loss = new_sl
                        else:
                            new_sl = round(best_price + multiplier * atr_val, 2)
                            if new_sl < stop_loss:
                                logger.info(f"  #{trade_id} {symbol}: Trail ${stop_loss:.2f} -> ${new_sl:.2f} (best ${best_price:.2f})")
                                stop_loss = new_sl

                self.db.update_paper_trade_stop(trade_id, stop_loss, best_price)

                if in_min_hold:
                    logger.info(f"  #{trade_id} {symbol}: Min hold (tracking ${current_price:.2f}, stop ${stop_loss:.2f})")

            # Check stop loss hit (always enforced, even during min hold)
            sl_hit = False
            if stop_loss:
                if is_long and current_price <= stop_loss:
                    sl_hit = True
                elif not is_long and current_price >= stop_loss:
                    sl_hit = True

            if sl_hit:
                result = self.db.close_paper_trade(trade_id, current_price, "CLOSED_SL")
                closed_count += 1
                direction = "LONG" if is_long else "SHORT"
                logger.info(f"Paper trade #{trade_id} {symbol} ({direction}) hit TRAILING STOP @ ${current_price:.2f}")

                if self.notifier and self.notifier.enabled:
                    self.notifier.notify_paper_trade_closed(
                        trade_id=trade_id, symbol=symbol, action=trade['action'],
                        quantity=trade['quantity'], entry_price=entry_price,
                        exit_price=current_price,
                        pnl_amount=result['pnl_amount'],
                        pnl_percent=result['pnl_percent'],
                        exit_reason="CLOSED_SL",
                    )

            self.connection.ib.sleep(0.5)

        return closed_count

    def _send_daily_summary(self):
        """Send daily summary at market close."""
        today = datetime.now(self.MARKET_TZ).strftime('%Y-%m-%d')
        if self._last_summary_date == today:
            return

        overall = self.db.get_paper_trade_stats()
        today_stats = self.db.get_today_stats()
        if overall['total_trades'] == 0 and today_stats['opened_today'] == 0:
            return

        if self.notifier and self.notifier.enabled:
            self.notifier.notify_daily_summary(
                date=today,
                trades_opened=today_stats['opened_today'],
                trades_closed=today_stats['closed_today'],
                winning_trades=today_stats['won_today'],
                losing_trades=today_stats['lost_today'],
                day_pnl=today_stats['realized_pnl_today'],
                total_pnl=overall['total_pnl'],
                win_rate=overall['win_rate'],
            )
            self._last_summary_date = today

    def _get_base_currency(self) -> Optional[str]:
        """Return the account base currency (ISO-4217), cached after first lookup."""
        if self._base_currency:
            return self._base_currency
        if not self.connection.ensure_connected():
            return None
        summary = self.connection.get_account_summary()
        code = (summary.get("NetLiquidation") or {}).get("currency")
        if code:
            self._base_currency = code
        return self._base_currency

    def _get_live_positions(self):
        """Live IBKR positions normalised to the paper-trade shape for /positions.

        Returns None in dry-run (so the Telegram handler falls back to the
        paper_trades table) and None on any error. In live mode it returns a
        list (possibly empty) of real held positions, with the active trailing
        stop price attached where one exists.
        """
        if self.dry_run:
            return None
        if not self.connection.ensure_connected():
            return None
        try:
            # Map symbol -> active protective stop price (TRAIL/STP), if any.
            # IBKR reports unset numeric fields as UNSET_DOUBLE (~1.8e308) —
            # e.g. trailStopPrice before the server has computed it — and that
            # sentinel is truthy, so it must be filtered before display (it
            # once reached Telegram as "SL $1797693…e+308 (-inf% away)").
            def _px(v):
                return v if v is not None and 0 < v < 1e300 else None

            stops: dict[str, float] = {}
            for tr in self.connection.ib.openTrades():
                o = tr.order
                if o.orderType not in ("TRAIL", "STP", "STP LMT"):
                    continue
                if o.orderType == "TRAIL":
                    # auxPrice on a TRAIL is the trail *amount*, not a price —
                    # never fall back to it. Unset here just means "trailing,
                    # server hasn't reported the level yet" -> show N/A.
                    sp = _px(getattr(o, "trailStopPrice", None))
                else:
                    sp = _px(getattr(o, "auxPrice", None))
                if sp:
                    stops[tr.contract.symbol] = sp
            out = []
            for p in self.engine.position_manager.get_positions():
                if p.quantity == 0:
                    continue
                out.append({
                    "symbol": p.symbol,
                    "quantity": abs(p.quantity),
                    "action": "BUY" if p.quantity > 0 else "SELL",
                    "entry_price": p.avg_cost,
                    "stop_loss": stops.get(p.symbol),
                    "take_profit": None,
                })
            return out
        except Exception as e:
            logger.warning(f"Could not build live positions: {e}")
            return None

    def _get_account_summary(self) -> dict:
        """Return the live IBKR account summary for /balance. Empty if not connected."""
        if not self.connection.ensure_connected():
            return {}
        try:
            return self.connection.get_account_summary() or {}
        except Exception as e:
            logger.debug(f"get_account_summary failed: {e}")
            return {}

    def _get_bot_status(self) -> dict:
        """Return a snapshot of bot/connection state for /health."""
        try:
            connected = bool(self.connection.ib.isConnected())
        except Exception:
            connected = False
        return {
            "connected": connected,
            "dry_run": self.dry_run,
            "uptime_seconds": (datetime.now() - self._started_at).total_seconds(),
            "last_rebalance": self._last_rebalance_at,
            "last_risk_check": self._last_risk_check,
            "last_probe_time": getattr(self.data_health, "_last_probe_time", None),
            "probe_failures": getattr(self.data_health, "consecutive_failures", 0),
        }

    def connect(self) -> bool:
        logger.info("Connecting to IBKR...")
        return self.connection.connect()

    def disconnect(self):
        logger.info("Disconnecting from IBKR...")
        self.connection.disconnect()

    def run_once(self) -> dict:
        """Run a single analysis/rebalance cycle."""
        if not self.connection.ensure_connected():
            logger.error("Failed to connect to IBKR")
            self._consecutive_failures += 1
            today = datetime.now().strftime('%Y-%m-%d')
            if (self._consecutive_failures >= self._failure_alert_threshold
                    and self._last_failure_alert != today):
                if self.notifier and self.notifier.enabled:
                    self.notifier.notify_error(
                        f"Connection failed {self._consecutive_failures} consecutive times.",
                        "Connection",
                    )
                self._last_failure_alert = today
            return {"success": False, "error": "Connection failed"}

        self._consecutive_failures = 0

        logger.info("=" * 50)
        logger.info(f"TREND-FOLLOWING BOT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}")
        logger.info("=" * 50)

        # Check trailing stops on existing positions
        if self.dry_run:
            import pandas as pd  # needed for pd.isna in _check_paper_trades
            closed_count = self._check_paper_trades()
            if closed_count > 0:
                logger.info(f"Closed {closed_count} paper trades (trailing stop)")

        # Live-mode safety nets — order parity + NLV reconcile run before the
        # rebalance so any drift/orphans are surfaced or healed first.
        try:
            self._check_order_parity()
        except Exception as e:
            logger.error(f"Pre-rebalance order-parity check failed: {e}")
        try:
            self._check_nlv_reconciliation()
        except Exception as e:
            logger.error(f"Pre-rebalance NLV reconcile failed: {e}")

        # Run signal analysis and generate rebalance opportunities
        opportunities = self.engine.run_analysis()

        # Drawdown circuit breaker — flatten everything if engine flagged HALT
        if (
            not self.engine.state.market_ok
            and "HALT" in self.engine.state.market_reason
        ):
            self._handle_drawdown_halt()
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "dry_run": self.dry_run,
                "halted": True,
                "halt_reason": self.engine.state.market_reason,
                "symbols_analyzed": self.engine.state.symbols_analyzed,
                "opportunities": 0,
                "trades_executed": 0,
                "opportunities_detail": [],
            }

        # Daily-loss gate — softer than drawdown halt, blocks only new entries
        try:
            entries_allowed = self._check_daily_loss()
        except Exception as e:
            logger.error(f"Daily-loss check failed: {e}")
            entries_allowed = True

        results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "symbols_analyzed": self.engine.state.symbols_analyzed,
            "opportunities": len(opportunities),
            "trades_executed": 0,
            "opportunities_detail": [],
        }

        for opp in opportunities:
            detail = {
                "symbol": opp.symbol,
                "decision": opp.decision.value,
                "price": opp.current_price,
                "size": opp.position_size,
                "signal": opp.signal_score,
                "reasons": opp.reasons,
                "stop_loss": opp.stop_loss_price,
            }
            results["opportunities_detail"].append(detail)

            logger.info(
                f"Candidate: {opp.decision.value} {opp.position_size} {opp.symbol} "
                f"@ ${opp.current_price:.2f} (signal {opp.signal_score:+.2f})"
            )

            # In dry run: open paper trades
            if self.dry_run:
                if not entries_allowed:
                    logger.info(f"  → Skipped: daily-loss halt active")
                    continue
                # Check max open positions
                open_trades = self.db.get_open_paper_trades()
                if len(open_trades) >= self.config_max_positions:
                    logger.info(f"  → Skipped: max positions reached ({len(open_trades)}/{self.config_max_positions})")
                    continue

                if not self.db.has_open_paper_trade(opp.symbol):
                    trade_id = self.db.save_paper_trade(
                        symbol=opp.symbol,
                        action=opp.decision.value,
                        quantity=opp.position_size,
                        entry_price=opp.current_price,
                        stop_loss=opp.stop_loss_price,
                        take_profit=None,
                        reasons=opp.reasons,
                        signal_score=opp.signal_score,
                        min_hold_days=trading_config.min_hold_days,
                    )
                    logger.info(f"  → Opened paper trade #{trade_id}")

                    if self.notifier and self.notifier.enabled:
                        self.notifier.notify_paper_trade_opened(
                            trade_id=trade_id, symbol=opp.symbol,
                            action=opp.decision.value,
                            quantity=opp.position_size,
                            entry_price=opp.current_price,
                            stop_loss=opp.stop_loss_price,
                            take_profit=None,
                        )
                else:
                    logger.info(f"  → Skipped: already have open paper trade for {opp.symbol}")

        # Execute if live
        if not self.dry_run and opportunities:
            if not entries_allowed:
                logger.warning(
                    "Daily-loss halt active — skipping all live entries this cycle"
                )
                opportunities = []
            self._execute_live(opportunities, results)

        # Send analysis notification
        if self.notifier and self.notifier.enabled:
            self.notifier.notify_analysis_complete(
                symbols_analyzed=results["symbols_analyzed"],
                opportunities=results["opportunities"],
                trades_executed=results["trades_executed"],
                dry_run=self.dry_run,
            )

        logger.info("\n" + self.engine.get_status_report())
        return results

    def _execute_live(self, opportunities: list, results: dict) -> None:
        """Place live orders for this cycle's opportunities, then sweep for
        naked lots.

        Each name is isolated: an unexpected exception on one symbol must not
        abort the rest of the book, and must not skip the post-rebalance stop
        sweep — an entry BUY may already be resting at IBKR when the exception
        fires, which is exactly the case the sweep exists for. IBKR-level order
        failures are already caught inside OrderManager/engine and returned as
        OrderResult; this guards the code path *around* them (2026-08-28
        review: previously one raise here skipped every later name and the
        sweep, leaving any filled BUY naked until the next risk check).
        """
        # Quantities, not just symbols — a held symbol may still be far
        # below target and need topping up in the execution loop below.
        held_qty = {
            p.symbol: p.quantity
            for p in self.engine.position_manager.get_positions()
            if p.quantity != 0
        }
        # Pending entry orders not yet filled (e.g. submitted just before a
        # crash). Same-symbol guard prevents duplicate entries on restart.
        try:
            pending_entries = {
                t.contract.symbol
                for t in self.engine.order_manager.get_open_orders()
                if t.order.orderType == "MKT"
            }
        except Exception as e:
            logger.warning(f"Could not fetch open orders for dedupe: {e}")
            pending_entries = set()

        force_sweep = False
        try:
            for opp in opportunities:
                try:
                    self._execute_one(opp, held_qty, pending_entries, results)
                except Exception as e:
                    # Unknown state for this name: an order may or may not be
                    # resting. Carry on with the rest and force the sweep.
                    force_sweep = True
                    logger.error(
                        f"Live execution failed for {opp.symbol}: {e} — "
                        f"continuing with remaining names; forcing the "
                        f"post-rebalance stop sweep",
                        exc_info=True,
                    )
                    if self.notifier and self.notifier.enabled:
                        try:
                            self.notifier.notify_error(
                                f"Live execution failed for {opp.symbol}: {e}. "
                                f"Remaining names still processed; protective-"
                                f"stop sweep forced.",
                                "Live execution",
                            )
                        except Exception as ne:
                            logger.warning(
                                f"Could not send execution-failure alert: {ne}"
                            )
        finally:
            if results["trades_executed"] > 0 or force_sweep:
                self._post_rebalance_stop_sweep()

    def _execute_one(
        self, opp, held_qty: dict, pending_entries: set, results: dict
    ) -> None:
        """Top up, skip, or enter a single opportunity (live mode)."""
        if opp.symbol in held_qty:
            held = held_qty.get(opp.symbol, 0)
            target = opp.position_size
            threshold = trading_config.topup_drift_threshold
            # Top up only when materially short of target, and only
            # upwards — trimming an oversized position is the trailing
            # stop's job, and selling here would fight it.
            if held > 0 and target > 0 and held < target * (1 - threshold):
                logger.info(
                    f"  → Top-up: {opp.symbol} held {held} vs target "
                    f"{target} ({held / target:.0%}) — buying "
                    f"{target - held}"
                )
                result = self.engine.top_up_position(opp, held)
                if result.success:
                    # Count it either way: an accepted-but-unfilled BUY
                    # is exactly the case the post-rebalance sweep has
                    # to run for, and the sweep is gated on this count.
                    results["trades_executed"] += 1
                    if result.filled_quantity:
                        # filled_quantity may be < target-held if the
                        # top-up was trimmed to settled cash.
                        logger.info(
                            f"  → Topped up {opp.symbol} to "
                            f"{held + result.filled_quantity}"
                            + (
                                f" (target {target})"
                                if held + result.filled_quantity != target
                                else ""
                            )
                        )
                    else:
                        # The engine may have trimmed the BUY to settled
                        # cash, so target-held can overstate the resting
                        # order (claimed "+13" for a 3-share BUY, 2026-09-01).
                        # The Trade carries the quantity actually placed.
                        placed = (
                            int(result.trade.order.totalQuantity)
                            if result.trade is not None
                            else target - held
                        )
                        logger.info(
                            f"  → Top-up BUY placed for {opp.symbol} "
                            f"(+{placed}) but not yet filled — "
                            f"cover extends on reconcile"
                        )
                else:
                    logger.info(f"  → Top-up failed: {result.message}")
            else:
                logger.info(
                    f"  → Skipped: already holding {opp.symbol} "
                    f"({held}/{target})"
                )
            return
        if opp.symbol in pending_entries:
            logger.info(
                f"  → Skipped: entry order already pending for {opp.symbol}"
            )
            return
        result = self.engine.execute_opportunity(opp)
        if result.success:
            results["trades_executed"] += 1
            logger.info(f"  → Executed: live order placed for {opp.symbol}")
        else:
            logger.info(f"  → Failed: {result.message}")

    def _post_rebalance_stop_sweep(self) -> None:
        # Catch entry-race deferrals: BUYs that filled after the engine's 5s
        # wait window have no protective stop yet. Without this, the next
        # reconcile sometimes didn't run for hours (intraday risk-checks fire
        # on boot/rebalance/reconnect, not a fixed interval — see 2026-06-09
        # incident where AIGI/COPA sat naked ~18h). Two passes cover the
        # tail of slow fills cheaply.
        for delay_s in (30, 90):
            try:
                self.connection.ib.sleep(delay_s)
                # Log unconditionally: _reconcile_protective_stops() is
                # silent when it finds nothing to heal, so without this
                # there is no way to tell "swept, all clean" from
                # "never ran" (2026-07-27 healthcheck spent a while
                # timing the 120s gap to prove it was firing).
                healed = self._reconcile_protective_stops()
                logger.info(
                    f"Post-rebalance stop sweep (+{delay_s}s): "
                    f"{healed} stop(s) placed"
                )
            except Exception as e:
                logger.warning(
                    f"Post-rebalance reconcile sweep failed: {e}"
                )
                break

    @property
    def config_max_positions(self) -> int:
        return trading_config.max_open_positions

    def _reconcile_protective_stops(self) -> int:
        """
        On startup, ensure every open live position has a working TRAIL/STP order.

        Closes the gap where the bot crashed between entry-fill and stop-placement,
        leaving naked positions on IBKR. Re-running is idempotent — already-protected
        positions are skipped.

        Returns the number of stops placed (0 if nothing needed).
        """
        if self.dry_run:
            return 0
        if not self.connection.ensure_connected():
            return 0

        positions = self.engine.position_manager.get_positions()
        if not positions:
            return 0

        from .orders import OrderAction
        from .indicators import TrendFollowingAnalyzer

        placed = 0
        for pos in positions:
            if pos.quantity == 0:
                continue
            needed_action = "SELL" if pos.quantity > 0 else "BUY"
            needed_qty = abs(pos.quantity)

            # Compare COVERED SHARES, not merely "is there a stop?". Presence
            # alone was the old test, and it silently passed a position whose
            # stop covered fewer shares than were held — exactly what a top-up
            # creates if its stop swap fails. Anything short of full cover is
            # treated as naked and rebuilt.
            covered = self.engine.order_manager.covered_quantity(
                pos.symbol, needed_action
            )
            if covered >= needed_qty:
                continue

            if covered > 0:
                logger.warning(
                    f"Reconcile: {pos.symbol} ({pos.quantity:+d} shares) only has "
                    f"{covered} share(s) protected — replacing with full cover"
                )
            else:
                logger.warning(
                    f"Reconcile: {pos.symbol} ({pos.quantity:+d} shares) has no "
                    f"protective stop — placing one now"
                )

            # Fresh ATR + price from daily bars (matches engine sizing)
            try:
                df = self.engine.fetcher.get_historical_data(
                    pos.symbol,
                    duration=trading_config.data_duration,
                    bar_size=trading_config.bar_size,
                )
                if df is None or df.empty:
                    logger.error(
                        f"Reconcile: no historical data for {pos.symbol}, skipping"
                    )
                    continue
                analyzer = TrendFollowingAnalyzer(
                    df, atr_period=trading_config.atr_period,
                )
                atr_val = analyzer.compute_atr()
                current_price = analyzer.get_current_price()
            except Exception as e:
                logger.error(
                    f"Reconcile: failed to fetch data for {pos.symbol}: {e}"
                )
                continue

            if atr_val <= 0 or current_price <= 0:
                logger.error(
                    f"Reconcile: bad ATR ({atr_val}) or price ({current_price}) "
                    f"for {pos.symbol}, skipping"
                )
                continue

            trail_amount = trading_config.atr_stop_multiplier * atr_val
            if pos.quantity > 0:
                initial_stop = round(current_price - trail_amount, 2)
                action = OrderAction.SELL
            else:
                initial_stop = round(current_price + trail_amount, 2)
                action = OrderAction.BUY

            # replace_ rather than place_: when cover is PARTIAL the existing
            # short-quantity stop must come off first, or the book would carry
            # more SELL quantity than shares held (Error 201). With zero cover
            # there is nothing to cancel and this degrades to a plain place.
            result = self.engine.order_manager.replace_trailing_stop(
                symbol=pos.symbol,
                action=action,
                quantity=abs(pos.quantity),
                trail_amount=trail_amount,
                initial_stop_price=initial_stop,
                reason="Reconciliation: position was not fully protected",
            )
            if result.success:
                placed += 1
                logger.warning(
                    f"Reconcile: placed trailing stop on {pos.symbol} "
                    f"trail=${trail_amount:.2f} init=${initial_stop:.2f}"
                )
            else:
                logger.error(
                    f"Reconcile: failed to place stop on {pos.symbol}: "
                    f"{result.message}"
                )

        if placed > 0 and self.notifier and self.notifier.enabled:
            self.notifier.notify_error(
                f"Reconciliation placed {placed} missing protective stop(s). "
                f"Likely a previous rebalance BUY that filled after the 5s wait "
                f"window (entry-race deferral), or the bot crashed mid-rebalance.",
                "Protective-stop reconciliation",
            )

        return placed

    def _handle_drawdown_halt(self):
        """Flatten all positions on drawdown halt. Idempotent."""
        reason = self.engine.state.market_reason
        logger.warning(f"=== DRAWDOWN HALT ===\n{reason}")

        if self.dry_run:
            open_trades = self.db.get_open_paper_trades()
            for trade in open_trades:
                try:
                    df = self.engine.fetcher.get_historical_data(
                        trade["symbol"], duration="2 D", bar_size="1 day",
                    )
                    px = (
                        float(df["close"].iloc[-1])
                        if df is not None and not df.empty
                        else trade["entry_price"]
                    )
                    self.db.close_paper_trade(trade["id"], px, "CLOSED_HALT")
                except Exception as e:
                    logger.error(
                        f"Failed to halt-close paper trade #{trade['id']}: {e}"
                    )
            logger.warning(f"Halt: closed {len(open_trades)} paper trades")
        else:
            cancelled = self.engine.order_manager.cancel_all_orders()
            if cancelled:
                logger.warning(f"Halt: cancelled {cancelled} working orders")
            close_results = self.engine.position_manager.close_all_positions()
            ok = sum(1 for r in close_results if r.success)
            logger.warning(
                f"Halt: closed {ok}/{len(close_results)} live positions"
            )

        if self.notifier and self.notifier.enabled:
            self.notifier.notify_error(
                f"DRAWDOWN HALT — {reason}\nAll positions flattened.",
                "Risk halt",
            )

    # ------------------------------------------------------------------
    # Live-mode safety nets — healthcheck items #5–#7
    # ------------------------------------------------------------------

    def _check_order_parity(self) -> str:
        """
        Parity #5: every live position must have a working protective stop;
        every working stop must back a held position.

        Auto-heals naked positions (reuses startup reconciliation) and logs
        orphan stops (working stop with no matching position).

        Returns a short status string, also stored on self._last_parity_status.
        Always called; in dry_run it's a no-op + 'n/a'.
        """
        if self.dry_run:
            self._last_parity_status = "n/a (dry_run)"
            return self._last_parity_status
        if not self.connection.ensure_connected():
            self._last_parity_status = "skip (no conn)"
            return self._last_parity_status

        # Auto-heal any naked positions
        placed = 0
        try:
            placed = self._reconcile_protective_stops()
        except Exception as e:
            logger.error(f"Order-parity heal failed: {e}")

        # Orphan check: working TRAIL/STP for symbols with no position
        try:
            open_orders = self.engine.order_manager.get_open_orders()
            positions = {
                p.symbol: p.quantity
                for p in self.engine.position_manager.get_positions()
                if p.quantity != 0
            }
            orphans = []
            for t in open_orders:
                if t.order.orderType not in ("TRAIL", "STP"):
                    continue
                sym = t.contract.symbol
                if positions.get(sym, 0) == 0:
                    orphans.append(f"{sym}#{t.order.orderId}")
        except Exception as e:
            logger.error(f"Order-parity orphan check failed: {e}")
            orphans = []

        if orphans:
            msg = f"Orphan protective stop(s) with no position: {', '.join(orphans)}"
            logger.warning(f"Order-parity: {msg}")
            if self.notifier and self.notifier.enabled:
                self.notifier.notify_error(msg, "Order parity")

        if placed > 0 and orphans:
            status = f"healed {placed}, {len(orphans)} orphans"
        elif placed > 0:
            status = f"healed {placed}"
        elif orphans:
            status = f"{len(orphans)} orphans"
        else:
            status = "OK"

        self._last_parity_status = status
        logger.info(f"Order-parity: {status}")
        return status

    def _check_nlv_reconciliation(self, drift_threshold: float = 0.02) -> Optional[float]:
        """
        Parity #6: IBKR's live NetLiquidation vs our latest portfolio snapshot.
        Alerts (Telegram + log) if drift >= threshold (default 2%).

        Returns the drift fraction (or None if either side missing).
        """
        if not self.connection.ensure_connected():
            return None

        summary = self.connection.get_account_summary()
        nlv_entry = summary.get("NetLiquidation") or {}
        nlv_str = nlv_entry.get("value")
        if not nlv_str:
            logger.debug("NLV reconcile: no NetLiquidation in account summary")
            return None
        try:
            live_nlv = float(nlv_str)
        except ValueError:
            logger.debug(f"NLV reconcile: bad NetLiquidation value '{nlv_str}'")
            return None

        cur = currency_symbol(nlv_entry.get("currency"))
        snap = self.db.get_latest_portfolio_snapshot()
        if not snap or not snap.get("equity") or snap["equity"] <= 0:
            logger.info(f"NLV reconcile: live {cur}{live_nlv:,.2f}, no snapshot to compare")
            return None

        last_equity = float(snap["equity"])
        drift = abs(live_nlv - last_equity) / last_equity
        self._last_nlv_drift = drift
        logger.info(
            f"NLV reconcile: live {cur}{live_nlv:,.2f} vs snapshot {cur}{last_equity:,.2f} "
            f"({snap.get('created_at')}) drift={drift:.2%}"
        )
        if drift >= drift_threshold:
            msg = (
                f"NLV drift {drift:.1%} — live {cur}{live_nlv:,.2f} vs last snapshot "
                f"{cur}{last_equity:,.2f} (threshold {drift_threshold:.0%}). "
                f"Investigate: stale feed, manual trade, or fee/transfer."
            )
            logger.warning(f"NLV reconcile: {msg}")
            if self.notifier and self.notifier.enabled:
                self.notifier.notify_error(msg, "NLV reconcile")
        return drift

    def _compute_session_pnl(self) -> tuple[float, float]:
        """
        Today's realized + unrealized P&L (base currency).

        Realized: closed paper_trades dated today (paper mode) or IBKR
        RealizedPnL (live, BASE-currency row).
        Unrealized: dry_run uses the latest cached signal price as the mark;
        falls back to best_price then entry. Live mode reads IBKR
        UnrealizedPnL (BASE-currency row, i.e. converted to account base).
        """
        realized = self.db.get_daily_pnl()
        unrealized = 0.0

        if self.dry_run:
            # Pull latest signal prices once for a cheap mark-to-market
            latest_prices: dict = {}
            try:
                latest_prices = self.db.get_latest_signal_prices()
            except AttributeError:
                pass  # method might not exist yet — fall back below

            for trade in self.db.get_open_paper_trades():
                entry = trade.get("entry_price") or 0.0
                qty = trade.get("quantity") or 0
                sym = trade.get("symbol")
                mark = latest_prices.get(sym) or trade.get("best_price") or entry
                if trade.get("action") == "BUY":
                    unrealized += (mark - entry) * qty
                else:
                    unrealized += (entry - mark) * qty
        else:
            try:
                accounts = self.connection.ib.managedAccounts() or []
                if accounts:
                    # IBKR reports RealizedPnL/UnrealizedPnL once per holding
                    # currency PLUS a consolidated "BASE" row already converted to
                    # the account base currency (GBP here). Use BASE so the figure
                    # — and the max_daily_loss comparison — is in the right
                    # currency. Previously this skipped BASE and grabbed the first
                    # non-BASE row (USD), printing the raw USD number with a £ sign.
                    # Fall back to the first non-BASE row only if BASE is absent,
                    # so we never silently read 0 (which would fail unsafe).
                    base_real = base_unreal = None
                    fb_real = fb_unreal = None
                    for v in self.connection.ib.accountValues(accounts[0]):
                        if v.tag not in ("RealizedPnL", "UnrealizedPnL"):
                            continue
                        try:
                            val = float(v.value)
                        except ValueError:
                            continue
                        if v.currency == "BASE":
                            if v.tag == "RealizedPnL":
                                base_real = val
                            else:
                                base_unreal = val
                        elif v.currency != "":
                            if v.tag == "RealizedPnL" and fb_real is None:
                                fb_real = val
                            elif v.tag == "UnrealizedPnL" and fb_unreal is None:
                                fb_unreal = val
                    if base_real is not None:
                        realized = base_real
                    elif fb_real is not None:
                        realized = fb_real
                    if base_unreal is not None:
                        unrealized = base_unreal
                    elif fb_unreal is not None:
                        unrealized = fb_unreal
            except Exception as e:
                logger.debug(f"Could not fetch IBKR PnL values: {e}")

        return realized, unrealized

    def _check_daily_loss(self) -> bool:
        """
        Parity #7: halt new entries if today's session P&L breaches max_daily_loss.

        Side-effects:
        - Notifies on first breach of the day
        - Sets self._daily_loss_halt_date to today's date when halted
        - Returns False while halted, True otherwise
        - DOES NOT close positions (different from drawdown halt — this gates
          only new entries to stop bleeding)
        """
        today = datetime.now().strftime('%Y-%m-%d')

        # Auto-clear stale halt from a previous day
        if self._daily_loss_halt_date and self._daily_loss_halt_date != today:
            logger.info(f"Daily-loss halt cleared (was {self._daily_loss_halt_date})")
            self._daily_loss_halt_date = None

        realized, unrealized = self._compute_session_pnl()
        session = realized + unrealized
        limit = -abs(trading_config.max_daily_loss)
        cur = currency_symbol(self._get_base_currency())
        logger.info(
            f"Daily P&L: realized={cur}{realized:+,.2f} unreal={cur}{unrealized:+,.2f} "
            f"session={cur}{session:+,.2f} (limit {cur}{limit:,.2f})"
        )

        if session <= limit:
            if self._daily_loss_halt_date != today:
                self._daily_loss_halt_date = today
                msg = (
                    f"DAILY LOSS HALT — session P&L {cur}{session:+,.2f} breaches "
                    f"limit {cur}{limit:,.2f}. New entries blocked for the rest of "
                    f"the day. Existing positions remain (trail-stops still active)."
                )
                logger.warning(msg)
                if self.notifier and self.notifier.enabled:
                    self.notifier.notify_error(msg, "Daily loss halt")
            return False
        return True

    @property
    def daily_loss_halted(self) -> bool:
        """True if the daily-loss halt has fired today and not yet cleared."""
        today = datetime.now().strftime('%Y-%m-%d')
        return self._daily_loss_halt_date == today

    def run_risk_check(self):
        """Run intraday risk check (trailing stops + live-mode safety nets)."""
        if not self.connection.ensure_connected():
            return

        logger.info("--- Intraday risk check ---")

        if self.dry_run:
            import pandas as pd
            closed = self._check_paper_trades()
            if closed > 0:
                logger.info(f"Risk check closed {closed} trades")
            else:
                logger.info("Risk check: all positions OK")

        # Live-mode safety nets (each method handles dry_run internally)
        try:
            self._check_order_parity()
        except Exception as e:
            logger.error(f"Order-parity check failed: {e}")
        try:
            self._check_nlv_reconciliation()
        except Exception as e:
            logger.error(f"NLV reconcile failed: {e}")
        try:
            self._check_daily_loss()
        except Exception as e:
            logger.error(f"Daily-loss check failed: {e}")

        self._last_risk_check = datetime.now()

    def _on_commission_report(self, trade, fill, report):
        """Fired by ib_insync whenever IBKR sends a CommissionReport.

        Listening here (not at execDetailsEvent) because the report carries
        IBKR's realizedPNL and commission in the underlying's trading
        currency (USD for USD-quoted UCITS, GBP for GBP-quoted) — display
        in that currency rather than re-FX'ing per fill, so the audit trail
        matches the IBKR statement. Filter to SELL fills of protective stops
        (TRAIL/STP) — the silent-fill case from the 2026-06-08 healthcheck.
        Other fills (entry BUYs, drawdown-halt MKT exits) have their own
        log paths.
        """
        try:
            order_type = getattr(trade.order, "orderType", "") or ""
            action = getattr(trade.order, "action", "") or ""
            if order_type not in ("TRAIL", "STP", "STP LMT"):
                return
            if action not in ("SELL", "BUY"):
                return

            symbol = trade.contract.symbol
            exec_obj = fill.execution
            exit_price = float(
                getattr(exec_obj, "avgPrice", None)
                or getattr(exec_obj, "price", 0.0)
            )
            quantity = int(
                getattr(exec_obj, "shares", None)
                or getattr(exec_obj, "cumQty", 0)
            )
            pnl = float(getattr(report, "realizedPNL", 0.0) or 0.0)
            commission = float(getattr(report, "commission", 0.0) or 0.0)
            ccy_code = (
                getattr(report, "currency", None)
                or getattr(trade.contract, "currency", None)
            )
            sym = currency_symbol(ccy_code)

            logger.info(
                f"Protective stop FILLED: {action} {quantity} {symbol} "
                f"@ {sym}{exit_price:.2f} (realizedPnL={sym}{pnl:+.2f}, "
                f"commission={sym}{commission:.2f}, "
                f"orderId={trade.order.orderId})"
            )

            try:
                self.db.save_trade(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=exit_price,
                    order_id=trade.order.orderId,
                    status="FILLED",
                    reason=f"{order_type} stop fill (realizedPnL={sym}{pnl:+.2f})",
                )
            except Exception as e:
                logger.warning(f"Could not log fill to trades table: {e}")

            # Block re-entry for a while — this stop-out means the trend failed.
            # Re-buying within days is what turned CNYA into 3 losing round-trips.
            try:
                cooldown_days = trading_config.reentry_cooldown_days
                if cooldown_days > 0:
                    self.db.set_symbol_cooldown(
                        symbol, cooldown_days, reason=f"{order_type} stop fill"
                    )
                    logger.info(
                        f"{symbol}: re-entry blocked for {cooldown_days} days "
                        f"after stop-out"
                    )
            except Exception as e:
                logger.warning(f"Could not set re-entry cooldown for {symbol}: {e}")

            if self.notifier and self.notifier.enabled:
                self.notifier.notify_position_closed(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    exit_price=exit_price,
                    pnl_amount=pnl,
                    commission=commission,
                    exit_reason="TRAILING_STOP" if order_type == "TRAIL" else "STOP_LOSS",
                    currency=sym,
                )
        except Exception as e:
            logger.error(f"Error handling commission report: {e}")

    def _register_fill_handlers(self):
        """Subscribe to ib_insync commissionReportEvent (LIVE only).

        The IB instance persists across reconnects (ConnectionManager doesn't
        recreate it), so one subscription at startup catches all subsequent
        fills — including server-side GTC trail-stops that fire while the bot
        is disconnected and report when it reconnects.
        """
        if self.dry_run:
            return
        try:
            ib = self.connection.ib
            try:
                ib.commissionReportEvent -= self._on_commission_report
            except (ValueError, Exception):
                pass
            ib.commissionReportEvent += self._on_commission_report
            logger.info("Subscribed to commissionReportEvent for LIVE fill alerts")
        except Exception as e:
            logger.error(f"Failed to subscribe to commissionReportEvent: {e}")

    def _check_watchdog(self):
        """Self-restart if the data probe has been failing continuously for too long.

        Background: ``data_health_checker`` calls ``gateway_monitor.restart_gateway()``
        after 2 consecutive probe failures, but the monitor caps at 3 restarts/day
        (in-memory). Once the cap trips, every subsequent probe failure is silent
        past the initial Telegram alert and the loop runs forever in a broken
        state (the 2026-05-23→25 weekend pattern).

        Trigger: ``consecutive_failures`` >= ``watchdog_timeout_min /
        probe_interval_min`` (floor of 3, to avoid one-shot transient kills).
        Crashing via ``sys.exit(1)`` lets Docker's ``restart: unless-stopped``
        policy recreate the container, which resets ``gateway_monitor``'s
        in-memory restart counter so auto-heal can try again. Tied to failure
        COUNT (not wall-clock since last success) so off-hours gaps don't
        false-fire on first probe of the new session.
        """
        failures = self.data_health.consecutive_failures
        if failures == 0:
            return

        timeout_min = trading_config.watchdog_timeout_min
        probe_interval_min = max(1, self.data_health.probe_interval_sec // 60)
        threshold_failures = max(3, timeout_min // probe_interval_min)
        if failures < threshold_failures:
            return

        last_success = self.data_health.time_since_last_success()
        last_success_str = (
            f"{last_success.total_seconds()/60:.0f} min ago"
            if last_success else "never since startup"
        )
        msg = (
            f"Watchdog: data probe failed {failures}x consecutively "
            f"(last success: {last_success_str}, threshold {timeout_min} min). "
            f"Auto-heal isn't working — self-restarting via sys.exit(1) so "
            f"Docker recreates the container."
        )
        logger.critical(msg)
        if self.notifier and self.notifier.enabled:
            try:
                self.notifier.notify_error(msg, "Watchdog")
            except Exception as e:
                logger.error(f"Watchdog: failed to send Telegram alert: {e}")
        sys.exit(1)

    def run_scheduled(self):
        """Run the bot on a schedule until stopped."""
        self.running = True

        if not self.connect():
            logger.error("Failed to connect, exiting")
            if self.notifier and self.notifier.enabled:
                self.notifier.notify_error("Failed to connect to IBKR", "Bot startup")
            return

        mode = "DRY RUN" if self.dry_run else "LIVE TRADING"
        strategy = "TREND-FOLLOWING"
        logger.info(f"Bot started — {strategy} mode: {mode}")
        logger.info(f"Rebalance: {trading_config.rebalance_hour}:{trading_config.rebalance_minute:02d} {self.MARKET_TZ} daily")
        logger.info(f"Risk checks: every {trading_config.risk_check_interval_hours}h")

        if self.notifier and self.notifier.enabled:
            self.notifier.notify_bot_started(f"{mode} — {strategy}")

        # Startup reconciliation: ensure no naked positions left over from a
        # previous crash (no-op in dry_run or with no positions).
        try:
            self._reconcile_protective_stops()
        except Exception as e:
            logger.error(f"Startup reconciliation failed: {e}")

        # Subscribe to commission-report events so server-side trail-stop
        # fills emit Telegram alerts (closes the 2026-06-08 silent-fill gap).
        try:
            self._register_fill_handlers()
        except Exception as e:
            logger.error(f"Fill-handler registration failed: {e}")

        try:
            while self.running:
                now_local = datetime.now(self.MARKET_TZ)

                # Daily summary near LSE close (16:30 local)
                if now_local.hour == 16 and now_local.minute >= 25:
                    self._send_daily_summary()

                # Outside market hours — just check Telegram commands
                if not self._is_market_hours():
                    logger.info("Outside market hours, waiting...")
                    self._consecutive_failures = 0
                    for _ in range(20):  # 20 × 3s = 60s between log lines
                        if not self.running:
                            break
                        check_telegram_commands(
                            self.db,
                            None,
                            self._get_base_currency,
                            self._get_account_summary,
                            self._get_bot_status,
                            self._get_live_positions,
                        )
                        time.sleep(3)
                    continue

                # Data-farm health probe (runs every 5 min; auto-restarts
                # gateway if CSPX data requests time out)
                if self.data_health.should_probe():
                    self.data_health.check_and_heal()

                # Watchdog: if self-heal isn't working, sys.exit so Docker recreates us
                self._check_watchdog()

                # Daily rebalance window
                if self._is_rebalance_time():
                    logger.info("=== DAILY REBALANCE ===")
                    self.run_once()
                    self._last_rebalance_date = now_local.strftime('%Y-%m-%d')
                    self._last_rebalance_at = datetime.now()
                    self._last_risk_check = datetime.now()

                # Intraday risk check
                elif self._is_risk_check_time():
                    self.run_risk_check()

                # Build price fetcher for Telegram commands
                def get_prices(symbols):
                    if self.engine and self.engine.fetcher and self.connection.ensure_connected():
                        return self.engine.fetcher.get_latest_prices(symbols)
                    return {}

                # Wait, checking Telegram every 3 seconds
                for _ in range(20):  # 20 × 3s = 60s
                    if not self.running:
                        break
                    check_telegram_commands(
                        self.db,
                        get_prices,
                        self._get_base_currency,
                        self._get_account_summary,
                        self._get_bot_status,
                        self._get_live_positions,
                    )
                    time.sleep(3)

        except Exception as e:
            logger.error(f"Bot error: {e}")
            if self.notifier and self.notifier.enabled:
                self.notifier.notify_error(str(e), "Bot runtime")
        finally:
            self.disconnect()
            logger.info("Bot stopped")
            if self.notifier and self.notifier.enabled:
                self.notifier.notify_bot_stopped("Scheduled shutdown")


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """Configure logging for the bot."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("ib_insync").setLevel(logging.WARNING)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="IBKR Trend-Following Bot")
    parser.add_argument("--live", action="store_true", help="Run in live mode")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=60, help="Minutes between checks (default: 60)")
    parser.add_argument("--log-file", type=str, default="logs/trading.log", help="Log file path")

    args = parser.parse_args()
    setup_logging(log_file=args.log_file)

    # Live mode requires IBKR_LIVE_CONFIRMED=true. Inside containers this is
    # the single source of truth (no --live flag needed). The --live CLI flag
    # also enables live, but still requires the env confirmation as a safety gate.
    # Important: do NOT key off IBKR_TRADING_MODE — that env var configures
    # the gateway image's login (paper vs live IBKR account) and is unrelated
    # to whether the bot places real orders.
    env_confirmed = os.getenv("IBKR_LIVE_CONFIRMED", "").lower() == "true"
    live = args.live or env_confirmed
    dry_run = not live
    if live:
        logger.warning("=" * 50)
        logger.warning("LIVE TRADING MODE - REAL ORDERS WILL BE PLACED")
        logger.warning("=" * 50)
        if not env_confirmed:
            logger.error(
                "Refusing to start live mode: IBKR_LIVE_CONFIRMED=true is not set. "
                "Set it in .env to confirm this is intentional."
            )
            return
        logger.warning("IBKR_LIVE_CONFIRMED=true — proceeding with live trading")
    else:
        logger.info("Running in PAPER/DRY-RUN mode (IBKR_LIVE_CONFIRMED != true)")

    bot = TradingBot(dry_run=dry_run, run_interval_minutes=args.interval)

    if args.once:
        bot.connect()
        try:
            bot._reconcile_protective_stops()
        except Exception as e:
            logger.error(f"Startup reconciliation failed: {e}")
        bot.run_once()
        bot.disconnect()
    else:
        bot.run_scheduled()


if __name__ == "__main__":
    main()
