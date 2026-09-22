"""
Attempt 9 — the forward-test sleeve: US absolute momentum in UCITS form, ring-fenced inside the
live account so it can run alongside the momentum strategy.

Registered in research/2026-09-17_forward_test/PREREG_9_ucits_forward_test.md (commit ff98f9f).
The rule, unchanged from attempt 5 which passed on 1928–2007 US data:

    on the first trading day of each month, compare VUAA's (S&P 500) total return over the last
    12 month-ends with IB01's (0–1yr US Treasuries, standing in for T-bills);
    hold VUAA if it is ahead, otherwise hold IBTM (7–10yr US Treasuries).

Ring-fencing (PREREG_9 §3), enforced here and at the momentum call sites:
  * the sleeve owns a cash reserve and its own positions; both are excluded from the momentum
    strategy's sizing capital and settled-cash view;
  * sleeve symbols never get protective stops and are skipped by parity/reconcile;
  * the sleeve is exempt from the momentum daily-loss gate and drawdown brakes;
  * it trades at most once per calendar month — the executed month is stored in the database, so a
    restart cannot repeat it;
  * a buy that settled cash cannot fund yet is retried at each later close (T+2), never abandoned.

Profit is not what this tests: it tests whether live trading matches the model.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from .config import sleeve_config
from .orders import OrderAction

logger = logging.getLogger(__name__)


@dataclass
class SleeveSignal:
    target: str
    r_stocks: float
    r_hurdle: float
    asof: str
    month_ends: int


class SleeveStrategy:
    """Monthly absolute-momentum sleeve. Owns only its own symbols and cash reserve."""

    def __init__(self, connection, order_manager, position_manager, fetcher, db, notifier=None, config=None):
        self.connection = connection
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.fetcher = fetcher
        self.db = db
        self.notifier = notifier
        self.config = config or sleeve_config
        self._last_deferral = None    # what the last "still waiting" line said, see _defer()

    # ------------------------------------------------------------------ helpers
    @property
    def symbols(self) -> set:
        """Symbols the sleeve may hold. The hurdle is never held."""
        return {self.config.equity_symbol, self.config.bond_symbol}

    def month_key(self, now: Optional[datetime] = None) -> str:
        return (now or datetime.now()).strftime("%Y-%m")

    def positions(self) -> dict:
        """Sleeve positions only: symbol -> quantity."""
        try:
            return {
                p.symbol: p.quantity
                for p in self.position_manager.get_positions()
                if p.symbol in self.symbols and p.quantity != 0
            }
        except Exception as e:
            logger.warning(f"Sleeve: could not read positions: {e}")
            return {}

    def reserve(self) -> float:
        """The sleeve's claim on account cash, in base currency."""
        try:
            return float(self.db.get_sleeve_reserve(self.config.capital_base))
        except Exception as e:
            logger.warning(f"Sleeve: could not read reserve, assuming 0: {e}")
            return 0.0

    def market_value(self) -> float:
        """Market value of sleeve positions in base currency (0 if unavailable)."""
        try:
            fx = self.connection.get_fx_rates() or {}
            total = 0.0
            for p in self.position_manager.get_positions():
                if p.symbol not in self.symbols or p.quantity == 0:
                    continue
                if p.market_value:
                    total += float(p.market_value)
                    continue
                price = (self.fetcher.get_latest_prices([p.symbol]) or {}).get(p.symbol)
                if price:
                    from .contracts import CONTRACT_REGISTRY
                    ccy = CONTRACT_REGISTRY.get(p.symbol, ("USD", "LSEETF"))[0]
                    rate = 1.0 if ccy == "GBP" else fx.get(ccy, 1.0)
                    total += p.quantity * float(price) * rate
            return total
        except Exception as e:
            logger.warning(f"Sleeve: could not value positions: {e}")
            return 0.0

    def claim_on_account(self) -> float:
        """What the momentum strategy must exclude: sleeve positions + sleeve cash reserve."""
        if not self.config.enabled:
            return 0.0
        return self.market_value() + self.reserve()

    # ------------------------------------------------------------------ signal
    def _month_end_closes(self, symbol: str) -> pd.Series:
        df = self.fetcher.get_historical_data(
            symbol, duration="2 Y", bar_size="1 day", what_to_show="ADJUSTED_LAST"
        )
        if df is None or df.empty:
            raise RuntimeError(f"no bars for {symbol}")
        s = df.set_index(pd.to_datetime(df["date"]))["close"].sort_index()
        return s.groupby([s.index.year, s.index.month]).last()

    def signal(self) -> SleeveSignal:
        """12-month total return of stocks vs the T-bill stand-in, on month-end closes."""
        lb = self.config.lookback_months
        eq = self._month_end_closes(self.config.equity_symbol)
        hu = self._month_end_closes(self.config.hurdle_symbol)
        if len(eq) < lb + 1 or len(hu) < lb + 1:
            raise RuntimeError(f"need {lb + 1} month-ends, have {len(eq)}/{len(hu)}")
        # the current month is partial: the signal uses completed month-ends only
        eq, hu = eq.iloc[:-1] if _partial(eq) else eq, hu.iloc[:-1] if _partial(hu) else hu
        r_stocks = float(eq.iloc[-1] / eq.iloc[-1 - lb] - 1.0)
        r_hurdle = float(hu.iloc[-1] / hu.iloc[-1 - lb] - 1.0)
        target = self.config.equity_symbol if r_stocks > r_hurdle else self.config.bond_symbol
        year, month = eq.index[-1]          # the index is (year, month) of numpy ints
        return SleeveSignal(target=target, r_stocks=r_stocks, r_hurdle=r_hurdle,
                            asof=f"{int(year):04d}-{int(month):02d}", month_ends=len(eq))

    # ------------------------------------------------------------------ scheduling
    def is_due(self, now_local: datetime) -> bool:
        """First trading day of the month, at or after the sleeve's time, and not yet run."""
        if not self.config.enabled:
            return False
        if (now_local.hour, now_local.minute) < (self.config.hour, self.config.minute):
            return False
        try:
            return not self.db.sleeve_month_done(self.month_key(now_local))
        except Exception as e:
            logger.warning(f"Sleeve: could not read month state, skipping: {e}")
            return False

    # ------------------------------------------------------------------ trading
    def _unit_base(self, symbol: str) -> float:
        """What one share costs in account base currency, including the cash buffer. 0 if unknown."""
        price = (self.fetcher.get_latest_prices([symbol]) or {}).get(symbol)
        if not price or price <= 0:
            return 0.0
        from .contracts import CONTRACT_REGISTRY
        ccy = CONTRACT_REGISTRY.get(symbol, ("USD", "LSEETF"))[0]
        fx = self.connection.get_fx_rates() or {}
        rate = 0.01 * fx.get("GBP", 1.0) if ccy == "GBX" else (1.0 if ccy == "GBP" else fx.get(ccy, 1.0))
        unit = float(price) * rate * (1.0 + self.config.cash_buffer)
        return unit if unit > 0 else 0.0

    def _affordable(self, symbol: str, budget_base: float, unit: Optional[float] = None) -> int:
        """Whole shares of `symbol` that `budget_base` covers, keeping the cash buffer.

        Pass `unit` (from `_unit_base`) when the caller already has it, to avoid fetching the price twice.
        """
        if unit is None:
            unit = self._unit_base(symbol)
        if unit <= 0:
            return 0
        return int(max(0.0, budget_base) / unit)

    def _committed(self) -> float:
        """Base-currency cost of sleeve BUYs IBKR has accepted but not yet reported on."""
        try:
            return float(self.db.sleeve_committed_base())
        except Exception as e:
            logger.warning(f"Sleeve: could not read committed cash, assuming 0: {e}")
            return 0.0

    def _uncommitted_reserve(self) -> float:
        """What is left of the reserve once orders already in flight are taken off it.

        The reserve only falls when an order settles, which can be minutes after placement (the
        2026-09-22 rejection took 4m38s) or later still if the bot restarts in between. Until then
        the money is spoken for, and sizing a second order as though it were not is how the
        ring-fence would be breached.
        """
        return max(0.0, self.reserve() - self._committed())

    def _spendable(self) -> float:
        """The sleeve can spend the lesser of its uncommitted reserve and the account's settled cash."""
        reserve = self._uncommitted_reserve()
        try:
            summary = self.connection.get_account_summary() or {}
            raw = (summary.get("AvailableFunds") or {}).get("value")
            settled = float(raw) if raw is not None else None
        except Exception:
            settled = None
        return reserve if settled is None else min(reserve, settled)

    def _fund_target(self, month: str, target: str, retry: bool = False) -> dict:
        """Buy as much of `target` as the reserve and settled cash allow, in sensible tranches.

        Card 9 says to buy "as settled cash allows (T+2), retrying at each later close until funded",
        so this runs whether or not the sleeve already holds some of the target — the reserve, not the
        position, is what says the job is done. IBKR charges a flat fee per order, so a tranche must be
        worth `min_topup_base`, unless it essentially finishes funding (>=85% of what is left), and is
        never placed below `min_order_base`.

        Returns {"status": "placed" | "pending_cash" | "buy_failed", ...}; never raises. "placed"
        means IBKR accepted the submission, NOT that it executed — the reserve moves only when
        `on_order_settled` hears what became of it.
        """
        reserve = self._uncommitted_reserve()
        spendable = self._spendable()
        # The daily hook asks on every loop pass, but settled cash only moves when proceeds settle. Two
        # cases defer whatever the share price is, so they are decided before fetching one:
        #  * below the order floor, no tranche can reach it;
        #  * below the tranche minimum, only a finishing tranche may go in, and a tranche (at least one
        #    share, leaving less than one share's worth) can only finish the reserve if the reserve is
        #    under twice the spendable cash.
        below_floor = spendable < self.config.min_order_base
        if below_floor or (spendable < self.config.min_topup_base and reserve >= 2 * spendable):
            why = (f"below the {self.config.min_order_base:,.0f} order floor" if below_floor else
                   f"below the {self.config.min_topup_base:,.0f} minimum and cannot be the last tranche")
            return self._defer(
                target, ("cash", round(spendable), round(reserve)),
                f"Sleeve: {target} buy deferred — {spendable:,.0f} base of settled cash against "
                f"{reserve:,.0f} still to invest is {why}, waiting for more settled cash rather than "
                "paying a flat commission on a small order"
            )
        unit = self._unit_base(target)
        qty = self._affordable(target, spendable, unit)
        # IBKR rejects an oversized order outright rather than trimming it, so a retry at the same
        # size would be rejected again on the same settled-cash figure. Staying strictly below the
        # smallest quantity it refused today makes the retry converge instead of looping.
        rejected_qty = self._rejected_qty_today(target)
        if rejected_qty is not None and qty >= rejected_qty:
            logger.info(
                f"Sleeve: IBKR rejected {rejected_qty} {target} earlier today — "
                f"trimming this attempt from {qty} to {rejected_qty - 1}"
            )
            qty = rejected_qty - 1
        if qty < 1:
            return self._defer(
                target, ("shares", round(spendable), round(reserve)),
                f"Sleeve: {target} buy deferred — reserve {reserve:,.0f} and settled cash cover 0 shares; "
                "will retry at the next close"
            )
        value = qty * unit
        finishes = (reserve - value) < unit      # nothing left that could buy another share
        if (value < self.config.min_topup_base and not finishes) or value < self.config.min_order_base:
            return self._defer(
                target, ("tranche", round(value), round(reserve)),
                f"Sleeve: {target} tranche would be {value:,.0f} base of {reserve:,.0f} still to invest "
                f"— below the {self.config.min_topup_base:,.0f} minimum and not the last tranche, waiting "
                "for more settled cash "
                "rather than paying a flat commission on a small order"
            )
        reason = "sleeve entry (retry)" if retry else "sleeve entry"
        res = self.order_manager.place_market_order(target, OrderAction.BUY, qty, reason=reason)
        out = {"status": "buy_failed", "quantity": qty, "symbol": target,
               "ok": bool(res.success), "order_id": res.order_id, "message": res.message}
        if not res.success:
            logger.error(f"Sleeve: BUY {target} failed: {res.message}")
            return out
        # `place_market_order` returns as soon as IBKR accepts the submission: it carries no fill
        # price and no filled quantity, and the order can still be rejected minutes later. So the
        # row is written as SUBMITTED with the cost it reserves, and the reserve itself only moves
        # in `on_order_settled`, once IBKR has said what actually happened.
        self.db.record_sleeve_order(month, "BUY", target, qty, res.order_id,
                                    fill_price=None, est_base=value, status="SUBMITTED")
        logger.info(
            f"Sleeve: placed BUY {qty} {target} (orderId={res.order_id}), "
            f"{value:,.0f} base reserved pending the fill"
        )
        out["status"] = "placed"
        return out

    def _rejected_qty_today(self, symbol: str):
        """Smallest quantity IBKR rejected for `symbol` today, or None."""
        try:
            return self.db.sleeve_min_rejected_qty_today(symbol)
        except Exception as e:
            logger.warning(f"Sleeve: could not read today's rejections: {e}")
            return None

    def _defer(self, target: str, key: tuple, message: str) -> dict:
        """Log a funding deferral, then report it as pending cash.

        The daily hook repeats itself every loop pass (~90 s), so a deferral already logged today with
        the same figures goes to DEBUG; a new day, reason or figure is logged at INFO.
        """
        key = (datetime.now().date(), target) + key
        if key == self._last_deferral:
            logger.debug(message)
        else:
            logger.info(message)
            self._last_deferral = key
        return {"status": "pending_cash", "quantity": 0}

    def _notify(self, headline: str, lines: Optional[list] = None) -> None:
        """Telegram, best effort — the sleeve never fails because a message could not be sent."""
        if not (self.notifier and getattr(self.notifier, "enabled", False)):
            return
        try:
            self.notifier.notify_sleeve(headline, lines or [])
        except Exception as e:
            logger.debug(f"Sleeve: Telegram notify failed: {e}")

    def rebalance(self, now_local: Optional[datetime] = None, dry_run: bool = False) -> dict:
        """Run this month's decision. Returns a summary dict; never raises."""
        month = self.month_key(now_local)
        out = {"month": month, "target": None, "orders": [], "status": "noop"}
        try:
            sig = self.signal()
        except Exception as e:
            logger.error(f"Sleeve: signal failed, no trade this month: {e}")
            out["status"] = "signal_failed"
            return out
        out["target"] = sig.target
        logger.info(
            f"Sleeve signal {month}: {self.config.equity_symbol} 12m {sig.r_stocks:+.2%} vs "
            f"{self.config.hurdle_symbol} {sig.r_hurdle:+.2%} (as of {sig.asof}) -> hold {sig.target}"
        )
        if dry_run:
            out["status"] = "dry_run"
            return out

        # Read the reserve before anything touches it: add_sleeve_reserve() creates the row from
        # whatever delta it is handed, so a switch-sell as the sleeve's very first action would seed
        # the reserve at the sale proceeds instead of the capital base.
        self.reserve()

        held = self.positions()
        # 1. sell anything that is not the target
        for symbol, qty in held.items():
            if symbol == sig.target or qty <= 0:
                continue
            res = self.order_manager.place_market_order(symbol, OrderAction.SELL, int(qty), reason="sleeve exit")
            out["orders"].append({"action": "SELL", "symbol": symbol, "quantity": int(qty),
                                  "ok": bool(res.success), "order_id": res.order_id, "message": res.message})
            if res.success:
                # Same rule as the BUY leg: placement is not execution. The proceeds are credited
                # to the reserve in `on_order_settled`. Booking them here credited (fill_price or 0)
                # x (filled_quantity or 0) == 0, so a switch would have sold the old line and left
                # the reserve empty, and the sleeve would never have bought the new one.
                self.db.record_sleeve_order(month, "SELL", symbol, int(qty), res.order_id,
                                            fill_price=None, est_base=None, status="SUBMITTED")
            else:
                logger.error(f"Sleeve: SELL {symbol} failed: {res.message}")

        # 2. put the reserve to work in the target, however much of it is currently spendable
        funded = self._fund_target(month, sig.target)
        if funded.get("quantity"):
            out["orders"].append({"action": "BUY", "symbol": sig.target, "quantity": funded["quantity"],
                                  "ok": funded.get("ok", False), "order_id": funded.get("order_id"),
                                  "message": funded.get("message")})
        out["status"] = funded["status"]
        # Fully invested already: nothing pending, so say so rather than leaving the month looking
        # like it is still waiting for cash.
        if (out["status"] == "pending_cash" and self.positions().get(sig.target, 0) > 0
                and self.reserve() < self.config.min_order_base):
            out["status"] = "already_held"

        self.db.set_sleeve_month(month, sig.target, out["status"])
        detail = [
            f"{self.config.equity_symbol} 12m {sig.r_stocks:+.1%} vs cash {sig.r_hurdle:+.1%} "
            f"(as of {sig.asof})",
        ]
        detail += [f"{o['action']} {o['quantity']} {o['symbol']} — "
                   f"{'ok' if o['ok'] else 'FAILED: ' + str(o['message'])}" for o in out["orders"]]
        if out["status"] == "pending_cash":
            detail.append("Waiting for settled cash before buying.")
        self._notify(f"<b>{month}</b> — hold <b>{sig.target}</b>", detail)
        return out

    def retry_pending(self) -> dict:
        """Keep funding this month's target as cash settles — the daily half of the card's rule.

        Called on every loop pass during market hours, so it is guarded three ways: it does nothing
        once the reserve is spent, nothing once a sleeve order has already gone in today (IBKR's
        account summary is a lagging cache, so two orders in quick succession would both size off the
        same stale settled-cash figure), and nothing for a month whose signal never resolved.

        An order IBKR REJECTED does not count as having gone in — it reached no book and consumed
        nothing, so it must not cost the rest of the day. `_fund_target` keeps that from looping by
        sizing the next attempt strictly below whatever was refused.
        """
        out = {"status": "noop"}
        if not self.config.enabled:
            return out
        try:
            state = self.db.get_sleeve_month(self.month_key())
        except Exception as e:
            logger.warning(f"Sleeve: could not read month state: {e}")
            return out
        if not state:
            return out
        target = state.get("target")
        if not target or state.get("status") in (None, "", "signal_failed", "dry_run"):
            return out
        if self.reserve() <= 0:
            return out
        try:
            if self.db.sleeve_orders_today():
                return out
        except Exception as e:
            logger.warning(f"Sleeve: could not count today's orders, skipping the retry: {e}")
            return out
        funded = self._fund_target(state["month"], target, retry=True)
        if funded["status"] == "placed":
            self.db.set_sleeve_month(state["month"], target, "placed")
            out = {"status": "placed", "symbol": target, "quantity": funded["quantity"]}
        return out

    # ------------------------------------------------------------------ settlement
    def on_order_settled(self, order_id, status: str, fill_price=None,
                         filled_quantity=None) -> dict:
        """Apply IBKR's verdict on one sleeve order. Idempotent, and never raises.

        This is the only place the reserve moves. It is driven by the order events the bot already
        subscribes to (`orderStatusEvent` / `commissionReportEvent`), because placement tells us
        nothing: on 2026-09-22 a BUY was accepted at 08:31:39 and rejected at 08:36:17, 4m38s later.
        A short synchronous wait after `placeOrder` could not have caught that.

        `settle_sleeve_order` only moves a row that is still SUBMITTED and hands it back, so IBKR
        re-sending an orderStatus, one commission report per partial fill, and a completed-order
        replay after a restart all collapse to a single reserve movement.
        """
        # Seed the reserve before anything moves it: add_sleeve_reserve() creates the row from
        # whatever delta it is handed, so a SELL settling against a fresh database would seed the
        # reserve at the sale proceeds instead of the capital base (the 2026-09-17 bug, reachable
        # again now that proceeds land here rather than at placement).
        self.reserve()
        try:
            row = self.db.settle_sleeve_order(order_id, status, fill_price, filled_quantity)
        except Exception as e:
            logger.warning(f"Sleeve: could not settle order {order_id}: {e}")
            return {"status": "error"}
        if row is None:
            return {"status": "ignored"}     # not a sleeve order, or already settled

        symbol, action = row["symbol"], row["action"]
        qty = int(filled_quantity or row["quantity"] or 0)
        px = float(fill_price or 0.0)

        if status != "FILLED":
            # Nothing reached the book, so nothing is owed. Dropping out of SUBMITTED releases the
            # cash this order had reserved; the next pass sizes against the real figure again.
            logger.warning(
                f"Sleeve: {action} {row['quantity']} {symbol} (orderId={order_id}) {status} — "
                f"reserve unchanged at {self.reserve():,.0f} base"
            )
            lines = ["Nothing was bought or sold; the reserve is unchanged."]
            if status == "REJECTED":
                lines.append("The next attempt will be sized smaller.")
            self._notify(f"{action} <b>{row['quantity']} {symbol}</b> was {status.lower()}", lines)
            return {"status": status.lower(), "symbol": symbol}

        delta_base = self._to_base(symbol, px * qty) if (px > 0 and qty > 0) else None
        if delta_base is None:
            # A fill we cannot value — no price, no quantity, or no FX rate. Charge the reserved
            # estimate rather than nothing: under-charging is what lets the sleeve overrun its
            # ring-fence, and that is the worse error. A SELL has no estimate, so it credits
            # nothing until the figure can be trusted.
            est = float(row["est_base"] or 0.0)
            logger.error(
                f"Sleeve: {action} {qty} {symbol} (orderId={order_id}) filled at {px} but could "
                f"not be valued in base — using the reserved estimate {est:,.0f} base"
            )
            delta = -est if action == "BUY" else 0.0
        else:
            delta = -delta_base if action == "BUY" else delta_base

        try:
            left = self.db.add_sleeve_reserve(delta) if delta else self.reserve()
        except Exception as e:
            logger.error(f"Sleeve: reserve update failed for order {order_id}: {e}")
            left = self.reserve()

        verb = "bought" if action == "BUY" else "sold"
        logger.info(
            f"Sleeve: {verb} {qty} {symbol} @ {px:,.4f}; {left:,.0f} base still to invest"
        )
        if action == "BUY":
            try:
                self.db.set_sleeve_month(row["month"], symbol, "traded")
            except Exception as e:
                logger.warning(f"Sleeve: could not mark {row['month']} traded: {e}")
        self._notify(
            f"{verb.capitalize()} <b>{qty} {symbol}</b>" + (f" @ {px:,.2f}" if px else ""),
            [f"{left:,.0f} of {self.config.capital_base:,.0f} still to invest"
             if left >= self.config.min_order_base else "Fully invested."],
        )
        return {"status": "filled", "symbol": symbol, "quantity": qty, "reserve": left}

    def report_unsettled(self) -> list:
        """Log sleeve orders still waiting on IBKR from an earlier day.

        Their cost stays reserved, so the sleeve under-spends rather than over-spends while one is
        outstanding — but an order that never settles would quietly shrink the sleeve, so say so.
        """
        try:
            rows = self.db.sleeve_unsettled_orders()
        except Exception as e:
            logger.warning(f"Sleeve: could not read unsettled orders: {e}")
            return []
        today = datetime.now().strftime("%Y-%m-%d")
        stale = [r for r in rows if not str(r.get("created_at") or "").startswith(today)]
        for r in stale:
            logger.warning(
                f"Sleeve: order {r.get('order_id')} ({r.get('action')} {r.get('quantity')} "
                f"{r.get('symbol')}, placed {r.get('created_at')}) has never settled — "
                f"{float(r.get('est_base') or 0):,.0f} base stays reserved against it"
            )
        return stale

    def _to_base(self, symbol: str, amount_local: float):
        """Convert an instrument-currency amount to account base. None if the rate is unknown.

        Deliberately NOT falling back to 1.0. For a USD line in a GBP account that is a 34% error
        in the direction that under-charges the reserve, which is the one failure the ring-fence
        cannot absorb — and it is silent. Callers charge the reserved estimate instead.
        """
        from .contracts import CONTRACT_REGISTRY
        ccy = CONTRACT_REGISTRY.get(symbol, ("USD", "LSEETF"))[0]
        if ccy == "GBP":
            return float(amount_local)
        fx = self.connection.get_fx_rates() or {}
        rate = 0.01 * fx.get("GBP") if ccy == "GBX" and fx.get("GBP") else fx.get(ccy)
        if not rate or rate <= 0:
            logger.error(f"Sleeve: no {ccy} FX rate available — cannot value {symbol} in base")
            return None
        return float(amount_local) * rate


def _partial(month_end_closes: pd.Series) -> bool:
    """True if the last month-end in the series is the current (incomplete) month."""
    try:
        year, month = month_end_closes.index[-1]
        now = datetime.now()
        return (year, month) == (now.year, now.month)
    except Exception:
        return False
