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

    def _affordable(self, symbol: str, budget_base: float) -> int:
        """Whole shares of `symbol` that `budget_base` covers, keeping the cash buffer."""
        unit = self._unit_base(symbol)
        if unit <= 0:
            return 0
        return int(max(0.0, budget_base) / unit)

    def _spendable(self) -> float:
        """The sleeve can spend the lesser of its reserve and the account's settled cash."""
        reserve = self.reserve()
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

        Returns {"status": ...}; never raises.
        """
        reserve = self.reserve()
        spendable = self._spendable()
        qty = self._affordable(target, spendable)
        if qty < 1:
            logger.info(
                f"Sleeve: {target} buy deferred — reserve {reserve:,.0f} and settled cash cover 0 shares; "
                "will retry at the next close"
            )
            return {"status": "pending_cash", "quantity": 0}
        unit = self._unit_base(target)
        value = qty * unit
        finishes = (reserve - value) < unit      # nothing left that could buy another share
        if (value < self.config.min_topup_base and not finishes) or value < self.config.min_order_base:
            logger.info(
                f"Sleeve: {target} tranche would be {value:,.0f} base of {reserve:,.0f} still to invest "
                f"— below the {self.config.min_topup_base:,.0f} minimum and not the last tranche, waiting "
                "for more settled cash "
                "rather than paying a flat commission on a small order"
            )
            return {"status": "pending_cash", "quantity": 0}
        reason = "sleeve entry (retry)" if retry else "sleeve entry"
        res = self.order_manager.place_market_order(target, OrderAction.BUY, qty, reason=reason)
        out = {"status": "buy_failed", "quantity": qty, "symbol": target,
               "ok": bool(res.success), "order_id": res.order_id, "message": res.message}
        if not res.success:
            logger.error(f"Sleeve: BUY {target} failed: {res.message}")
            return out
        cost = (res.fill_price or 0) * (res.filled_quantity or qty)
        self.db.record_sleeve_order(month, "BUY", target, qty, res.order_id, res.fill_price)
        left = self.reserve()
        if cost:
            left = self.db.add_sleeve_reserve(-self._to_base(target, cost))
            logger.info(f"Sleeve: bought {qty} {target}; {left:,.0f} base still to invest")
        out["status"] = "traded"
        self._notify(
            f"Bought <b>{qty} {target}</b>" + (f" @ {res.fill_price:,.2f}" if res.fill_price else ""),
            [f"{left:,.0f} of {self.config.capital_base:,.0f} still to invest"
             if left >= self.config.min_order_base else "Fully invested."],
        )
        return out

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
                proceeds = (res.fill_price or 0) * (res.filled_quantity or 0)
                self.db.record_sleeve_order(month, "SELL", symbol, int(qty), res.order_id, res.fill_price)
                if proceeds:
                    self.db.add_sleeve_reserve(self._to_base(symbol, proceeds))
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
        if funded["status"] == "traded":
            self.db.set_sleeve_month(state["month"], target, "traded")
            out = {"status": "traded", "symbol": target, "quantity": funded["quantity"]}
        return out

    def _to_base(self, symbol: str, amount_local: float) -> float:
        from .contracts import CONTRACT_REGISTRY
        ccy = CONTRACT_REGISTRY.get(symbol, ("USD", "LSEETF"))[0]
        if ccy == "GBP":
            return float(amount_local)
        fx = self.connection.get_fx_rates() or {}
        rate = 0.01 * fx.get("GBP", 1.0) if ccy == "GBX" else fx.get(ccy, 1.0)
        return float(amount_local) * rate


def _partial(month_end_closes: pd.Series) -> bool:
    """True if the last month-end in the series is the current (incomplete) month."""
    try:
        year, month = month_end_closes.index[-1]
        now = datetime.now()
        return (year, month) == (now.year, now.month)
    except Exception:
        return False
