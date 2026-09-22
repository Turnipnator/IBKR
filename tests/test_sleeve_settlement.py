"""
Sleeve order settlement — the 2026-09-22 healthcheck findings.

Three defects, all rooted in the same mistake: treating `OrderManager.place_market_order`'s
return as if it said what the order DID. It does not. It returns the moment IBKR accepts the
submission, carrying `success=True`, an orderId, and no fill price or filled quantity at all.

  1. `SleeveConfig.cash_buffer` was 0.02 against a measured IBKR need factor of 4.3-5.5%, so the
     first genuinely cash-constrained tranche was sized one share too large and rejected
     (Error 201, orderId 5982: 9 VUAA needing GBP 1,066.61 against GBP 1,055.66 available).
  2. A rejected order was booked as though it had happened: it took the day's one-order slot,
     wrote a `sleeve_orders` row, marked the month `traded` and sent "Bought 9 VUAA" to Telegram.
  3. Worst, and never reached in live trading only because (1) rejected the order: since
     `res.fill_price` is always None, `cost` was always 0, so `add_sleeve_reserve` was never
     called. The GBP 2,500 ring-fence (PREREG_9 section 3) could never decrement — the sleeve
     would have bought a fresh tranche every day until the account ran out of settled cash.

The suite missed all three because `FakeOrders` returned `fill_price=10.0, filled_quantity=qty`,
a contract production has never honoured. That fake now matches `OrderResult`, and these tests
drive the real `SleeveStrategy`, the real `Database` on a temp file, and the real
`TradingBot._mark_order_terminal`, replaying the live numbers.
"""
from types import SimpleNamespace

import pytest

from src.bot import TradingBot
from src.config import SleeveConfig
from src.database import Database
from src.orders import OrderAction, OrderResult
from src.sleeve import SleeveStrategy
from tests.test_sleeve import FakeFetcher, FakeNotifier, FakeOrders, cfg


# ---------------------------------------------------------------- the live 2026-09-22 numbers
VUAA_USD = 150.39          # quote at 08:31:38
FX_USD = 0.7467            # IBKR ExchangeRate USD -> GBP
RAW_BASE = VUAA_USD * FX_USD               # 112.296 — probe-confirmed
IBKR_AVAILABLE = 1055.66                   # from the Error 201 text
IBKR_NEED_PER_SHARE = 1066.61 / 9          # 118.512 — ditto, so IBKR's factor was 5.54%


def _sleeve(tmp_path, *, available=IBKR_AVAILABLE, price=VUAA_USD, fx=FX_USD,
            config=None, positions=(), notifier=None, db=None):
    db = db or Database(str(tmp_path / "sleeve.db"))
    connection = SimpleNamespace(
        get_fx_rates=lambda: {"USD": fx, "GBP": 1.0},
        get_account_summary=lambda: {"AvailableFunds": {"value": str(available)}},
        ensure_connected=lambda: True,
    )
    orders = FakeOrders()
    s = SleeveStrategy(
        connection, orders, SimpleNamespace(get_positions=lambda: list(positions)),
        FakeFetcher({}, {"VUAA": price, "IBTM": price}), db, notifier,
        config or cfg(cash_buffer=SleeveConfig.cash_buffer),
    )
    db.set_sleeve_month(s.month_key(), "VUAA", "pending_cash")
    return s, orders, db


def _order_id(db, action="BUY"):
    return [r["order_id"] for r in db.sleeve_unsettled_orders() if r["action"] == action][0]


def _fits(qty):
    """Would IBKR have accepted this many shares against the cash it reported?"""
    return qty * IBKR_NEED_PER_SHARE <= IBKR_AVAILABLE


# ================================================================ 1. the buffer (P1)
class TestCashBuffer:
    def test_the_shipped_buffer_sizes_an_order_ibkr_accepts(self, tmp_path):
        """The regression: at 2% this is 9 shares needing GBP 1,066.61 against GBP 1,055.66."""
        s, _, _ = _sleeve(tmp_path)
        qty = s._affordable("VUAA", IBKR_AVAILABLE)
        assert qty == 8
        assert _fits(qty), "the shipped buffer must not size an order IBKR will reject"
        assert IBKR_AVAILABLE - qty * IBKR_NEED_PER_SHARE == pytest.approx(107.56, abs=0.5)

    def test_the_old_two_percent_buffer_is_what_was_rejected(self, tmp_path):
        s, _, _ = _sleeve(tmp_path, config=cfg(cash_buffer=0.02))
        qty = s._affordable("VUAA", IBKR_AVAILABLE)
        assert qty == 9
        assert not _fits(qty)                       # exactly the live Error 201

    def test_the_sleeve_buffer_matches_the_momentum_gate(self):
        from src.config import trading_config
        assert SleeveConfig.cash_buffer == trading_config.settled_cash_buffer == 0.06

    def test_the_buffer_covers_the_measured_need_factor(self, tmp_path):
        """IBKR needed 5.54% over the raw quote that day; August measured 4.3-4.8%."""
        s, _, _ = _sleeve(tmp_path)
        unit = s._unit_base("VUAA")
        assert unit / RAW_BASE == pytest.approx(1.06, abs=1e-9)
        assert unit >= IBKR_NEED_PER_SHARE


# ================================================================ 2. the ring-fence (P2, worst)
class TestReserveIsRingFenced:
    def test_a_fill_decrements_the_reserve_by_what_it_actually_cost(self, tmp_path):
        """Before: `cost` was (None or 0) * ... == 0, so this never ran at all."""
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        assert s.retry_pending()["status"] == "placed"
        qty = orders.placed[0][2]
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(2500.0)   # placement moves nothing

        s.on_order_settled(_order_id(db), "FILLED", fill_price=VUAA_USD, filled_quantity=qty)
        spent = qty * VUAA_USD * FX_USD                                 # actual cost, not the estimate
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(2500.0 - spent)

    def test_the_sleeve_cannot_buy_the_same_money_twice_across_days(self, tmp_path):
        """The runaway. Day 1 buys and fills; day 2 must size off what is LEFT, not GBP 2,500."""
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        s.retry_pending()
        qty1 = orders.placed[0][2]
        s.on_order_settled(_order_id(db), "FILLED", fill_price=VUAA_USD, filled_quantity=qty1)

        _clear_today(db)                                      # a new day
        s.retry_pending()
        total = sum(o[2] for o in orders.placed) * VUAA_USD * FX_USD
        assert total <= 2500.0 + 1e-6, f"sleeve committed {total:,.0f} against a 2,500 ring-fence"

    def test_an_unsettled_order_still_counts_against_the_reserve(self, tmp_path):
        """If a settlement event is ever missed, the cost stays reserved — the sleeve under-spends
        rather than breaching the ring-fence."""
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        s.retry_pending()
        reserved = db.sleeve_committed_base()
        assert reserved > 0
        assert s._uncommitted_reserve() == pytest.approx(2500.0 - reserved)

        _clear_today(db)                                      # a new day, still unsettled
        s.retry_pending()
        total = sum(o[2] for o in orders.placed) * VUAA_USD * FX_USD
        assert total <= 2500.0 + 1e-6

    def test_funding_stops_once_the_reserve_is_genuinely_spent(self, tmp_path):
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        for _ in range(6):
            _clear_today(db)
            s.retry_pending()
            for r in db.sleeve_unsettled_orders():
                s.on_order_settled(r["order_id"], "FILLED",
                                   fill_price=VUAA_USD, filled_quantity=r["quantity"])
        spent = sum(o[2] for o in orders.placed) * VUAA_USD * FX_USD
        assert spent <= 2500.0 + 1e-6
        assert db.get_sleeve_reserve(2500.0) < s.config.min_order_base

    def test_a_fill_with_no_price_falls_back_to_the_reserved_estimate(self, tmp_path):
        """Charging nothing is the failure that breaks the ring-fence, so charge the estimate."""
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        s.retry_pending()
        est = db.sleeve_committed_base()
        s.on_order_settled(_order_id(db), "FILLED", fill_price=0.0, filled_quantity=0)
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(2500.0 - est)


# ================================================================ 3. idempotency
class TestSettlementIsIdempotent:
    def test_the_same_fill_reported_twice_moves_the_reserve_once(self, tmp_path):
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        s.retry_pending()
        qty, oid = orders.placed[0][2], _order_id(db)
        s.on_order_settled(oid, "FILLED", fill_price=VUAA_USD, filled_quantity=qty)
        after = db.get_sleeve_reserve(2500.0)

        # orderStatusEvent, then a commission report per partial, then a restart replay
        for _ in range(3):
            assert s.on_order_settled(oid, "FILLED", VUAA_USD, qty)["status"] == "ignored"
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(after)

    def test_a_rejection_after_a_fill_cannot_refund_the_reserve(self, tmp_path):
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        s.retry_pending()
        qty, oid = orders.placed[0][2], _order_id(db)
        s.on_order_settled(oid, "FILLED", fill_price=VUAA_USD, filled_quantity=qty)
        after = db.get_sleeve_reserve(2500.0)
        s.on_order_settled(oid, "REJECTED")
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(after)

    def test_an_unknown_order_is_ignored(self, tmp_path):
        """Momentum orders pour through the same funnel and must fall straight past."""
        s, _, db = _sleeve(tmp_path)
        assert s.on_order_settled(999_999, "FILLED", 12.0, 5)["status"] == "ignored"
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(2500.0)


# ================================================================ 4. rejection handling
class TestRejection:
    def test_a_rejection_leaves_the_reserve_alone_and_frees_the_day(self, tmp_path):
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        s.retry_pending()
        s.on_order_settled(_order_id(db), "REJECTED")
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(2500.0)
        assert db.sleeve_committed_base() == pytest.approx(0.0)
        assert db.sleeve_orders_today() == 0        # nothing reached the book, so nothing was used

    def test_the_retry_is_strictly_smaller_than_what_was_rejected(self, tmp_path):
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        s.retry_pending()
        first = orders.placed[0][2]
        s.on_order_settled(_order_id(db), "REJECTED")
        s.retry_pending()
        assert orders.placed[1][2] == first - 1

    def test_repeated_rejections_terminate_instead_of_looping(self, tmp_path):
        """The hook runs every ~90s. Sizing off an unchanged settled-cash figure would re-place the
        same doomed order forever, so each rejection must shrink the next attempt."""
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        for _ in range(400):
            out = s.retry_pending()
            if out["status"] != "placed":
                break
            s.on_order_settled(_order_id(db), "REJECTED")
        sizes = [o[2] for o in orders.placed]
        assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(sizes)
        assert out["status"] != "placed", "funding never stopped re-placing rejected orders"

    def test_a_rejection_is_announced_and_is_not_an_error_alert(self, tmp_path):
        n = FakeNotifier()
        s, orders, db = _sleeve(tmp_path, available=5000.0, notifier=n)
        s.retry_pending()
        assert n.sleeve_messages == []                  # placement claims nothing
        s.on_order_settled(_order_id(db), "REJECTED")
        headline, lines = n.sleeve_messages[0]
        assert "rejected" in headline.lower()
        assert any("reserve is unchanged" in line for line in lines)
        assert n.errors == []

    def test_the_month_is_not_marked_traded_until_something_fills(self, tmp_path):
        """On 2026-09-22 the month read `traded` with nothing bought."""
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        s.retry_pending()
        assert db.get_sleeve_month(s.month_key())["status"] == "placed"
        s.on_order_settled(_order_id(db), "REJECTED")
        assert db.get_sleeve_month(s.month_key())["status"] == "placed"

        s.retry_pending()
        s.on_order_settled(_order_id(db), "FILLED", fill_price=VUAA_USD,
                           filled_quantity=orders.placed[-1][2])
        assert db.get_sleeve_month(s.month_key())["status"] == "traded"


# ================================================================ 5. the switch (SELL leg)
class TestSwitchProceeds:
    def test_a_sale_credits_the_reserve_when_it_settles(self, tmp_path):
        """Booking proceeds at placement credited (None or 0) * (0 or 0) == 0, so a VUAA -> IBTM
        switch would have sold the old line and then had nothing to buy the new one with."""
        db = Database(str(tmp_path / "s.db"))
        db.get_sleeve_reserve(2500.0)
        db.add_sleeve_reserve(-2400.0)                      # fully invested: GBP 100 left
        s, orders, db = _sleeve(tmp_path, available=5000.0, db=db)
        db.record_sleeve_order(s.month_key(), "SELL", "IBTM", 20, 777,
                               fill_price=None, est_base=None, status="SUBMITTED")

        s.on_order_settled(777, "FILLED", fill_price=124.0, filled_quantity=20)
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(100.0 + 20 * 124.0)  # IBTM is GBP

    def test_a_sale_does_not_reseed_the_reserve_on_a_fresh_database(self, tmp_path):
        """add_sleeve_reserve() creates the row from its delta — reachable again now that proceeds
        land at settlement rather than at placement."""
        db = Database(str(tmp_path / "fresh.db"))
        s, _, db = _sleeve(tmp_path, db=db)
        db.record_sleeve_order(s.month_key(), "SELL", "IBTM", 20, 777,
                               fill_price=None, est_base=None, status="SUBMITTED")
        s.on_order_settled(777, "FILLED", fill_price=124.0, filled_quantity=20)
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(2500.0 + 20 * 124.0)


# ================================================================ 6. the bot wiring
class TestBotRoutesSettlement:
    @staticmethod
    def _trade(order_id, symbol, action, qty, status, avg=0.0, filled=0):
        return SimpleNamespace(
            order=SimpleNamespace(orderId=order_id, action=action,
                                  totalQuantity=float(qty), orderType="MKT"),
            contract=SimpleNamespace(symbol=symbol, currency="USD"),
            orderStatus=SimpleNamespace(status=status, avgFillPrice=avg, filled=filled),
            log=[],
        )

    def _bot(self, tmp_path, sleeve, db):
        bot = TradingBot.__new__(TradingBot)
        bot.db = db
        bot.sleeve = sleeve
        return bot

    def test_a_sleeve_fill_reaches_the_sleeve_through_the_real_handler(self, tmp_path):
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        s.retry_pending()
        qty, oid = orders.placed[0][2], _order_id(db)
        bot = self._bot(tmp_path, s, db)

        bot._mark_order_terminal(
            self._trade(oid, "VUAA", "BUY", qty, "Filled", avg=VUAA_USD, filled=qty), "FILLED",
            price=VUAA_USD,
        )
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(2500.0 - qty * VUAA_USD * FX_USD)

    def test_the_live_rejection_replayed_through_the_real_handler(self, tmp_path):
        """orderId 5982: accepted 08:31:39, rejected 08:36:17 — 4m38s later, which is why a short
        synchronous wait after placeOrder could never have caught it."""
        s, orders, db = _sleeve(tmp_path)
        s.retry_pending()
        oid = _order_id(db)
        bot = self._bot(tmp_path, s, db)

        bot._mark_order_terminal(
            self._trade(oid, "VUAA", "BUY", orders.placed[0][2], "Cancelled"), "REJECTED",
            note="Error 201: Order rejected",
        )
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(2500.0)
        assert db.sleeve_orders_today() == 0
        assert db.get_sleeve_month(s.month_key())["status"] != "traded"

    def test_a_momentum_order_is_untouched_by_the_sleeve(self, tmp_path):
        s, _, db = _sleeve(tmp_path)
        bot = self._bot(tmp_path, s, db)
        bot.db.save_trade(symbol="CMOD", action="BUY", quantity=26, price=0.0,
                          order_id=5818, status="SUBMITTED", reason="entry")
        bot._mark_order_terminal(
            self._trade(5818, "CMOD", "BUY", 26, "Filled", avg=36.3, filled=26), "FILLED",
            price=36.3,
        )
        assert db.get_sleeve_reserve(2500.0) == pytest.approx(2500.0)
        conn = db._get_connection()
        try:
            row = conn.execute("SELECT status, price FROM trades WHERE order_id = 5818").fetchone()
        finally:
            conn.close()
        assert row["status"] == "FILLED" and row["price"] == pytest.approx(36.3)

    def test_a_sleeve_failure_never_disturbs_the_trades_ledger(self, tmp_path):
        db = Database(str(tmp_path / "x.db"))
        bot = self._bot(tmp_path, SimpleNamespace(
            on_order_settled=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))), db)
        db.save_trade(symbol="CMOD", action="BUY", quantity=1, price=0.0,
                      order_id=4242, status="SUBMITTED", reason="entry")
        bot._mark_order_terminal(
            self._trade(4242, "CMOD", "BUY", 1, "Filled", avg=36.3, filled=1), "FILLED", price=36.3)
        conn = db._get_connection()
        try:
            row = conn.execute("SELECT status FROM trades WHERE order_id = 4242").fetchone()
        finally:
            conn.close()
        assert row["status"] == "FILLED"

    def test_a_bot_without_a_sleeve_is_unaffected(self, tmp_path):
        db = Database(str(tmp_path / "y.db"))
        bot = TradingBot.__new__(TradingBot)          # no `sleeve` attribute at all
        bot.db = db
        db.save_trade(symbol="CMOD", action="BUY", quantity=1, price=0.0,
                      order_id=11, status="SUBMITTED", reason="entry")
        bot._mark_order_terminal(
            self._trade(11, "CMOD", "BUY", 1, "Filled", avg=1.0, filled=1), "FILLED", price=1.0)


# ================================================================ 7. visibility
class TestUnsettledVisibility:
    def test_a_stale_unsettled_order_is_reported(self, tmp_path):
        s, _, db = _sleeve(tmp_path)
        db.record_sleeve_order("2026-09", "BUY", "VUAA", 9, 5982,
                               fill_price=None, est_base=1005.0, status="SUBMITTED")
        conn = db._get_connection()
        try:
            conn.execute("UPDATE sleeve_orders SET created_at = '2026-09-01 08:31:39' "
                         "WHERE order_id = 5982")
            conn.commit()
        finally:
            conn.close()
        stale = s.report_unsettled()
        assert [r["order_id"] for r in stale] == [5982]

    def test_todays_pending_order_is_not_reported_as_stale(self, tmp_path):
        s, orders, db = _sleeve(tmp_path, available=5000.0)
        s.retry_pending()
        assert s.report_unsettled() == []


def _clear_today(db):
    """Age every sleeve order out of today, so the next pass is a fresh day for the daily guard."""
    conn = db._get_connection()
    try:
        conn.execute("UPDATE sleeve_orders SET created_at = '2020-01-01 00:00:00'")
        conn.commit()
    finally:
        conn.close()
