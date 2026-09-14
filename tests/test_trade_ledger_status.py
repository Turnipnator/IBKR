"""
Tests for the 2026-09-14 trades-ledger status tracking.

Before: `orders.py` wrote one `status=SUBMITTED` row per placed order and
nothing ever moved it on — only protective-stop fills wrote a separate
`FILLED` execution row (fbff002). On the live DB that left 120 rows stuck at
SUBMITTED (62 entry BUYs, 58 stop placements) against 4 orders actually
working at IBKR, and no entry fill price recorded anywhere.

After: `Database.update_trade_status` moves a SUBMITTED row to
FILLED / CANCELLED / REJECTED; `TradingBot._on_order_status` (subscribed to
`ib.orderStatusEvent`) drives it from IBKR's terminal statuses, and
`_on_commission_report` also marks FILLED so stops that fired while the bot
was disconnected (replayed by reqCompletedOrders, which emits no
orderStatusEvent) are covered too.

These drive the REAL `Database` (on a tmp file) and the REAL bot handlers on
a `TradingBot` built via `__new__`; the ib_insync Trade/Fill/CommissionReport
objects are SimpleNamespace replays of the live cases named in each test.
"""

import logging
from types import SimpleNamespace

import pytest

from src.bot import TradingBot
from src.database import Database


# ---------------------------------------------------------------- helpers

def _db(tmp_path):
    return Database(str(tmp_path / "ledger.db"))


def _row(db, order_id, status=None):
    conn = db._get_connection()
    try:
        q = "SELECT * FROM trades WHERE order_id = ?"
        args = [order_id]
        if status:
            q += " AND status = ?"
            args.append(status)
        q += " ORDER BY id"
        return [dict(r) for r in conn.execute(q, args).fetchall()]
    finally:
        conn.close()


def _trade(order_id, symbol, action, qty, otype, status,
           avg=0.0, log=(), remaining=None):
    t = SimpleNamespace(
        order=SimpleNamespace(
            orderId=order_id, action=action, totalQuantity=float(qty),
            orderType=otype,
        ),
        contract=SimpleNamespace(symbol=symbol, currency="USD"),
        orderStatus=SimpleNamespace(status=status, avgFillPrice=avg),
        log=[SimpleNamespace(status=s, message=m, errorCode=c)
             for s, m, c in log],
    )
    if remaining is not None:
        t.remaining = lambda: remaining
    return t


def _bot(db):
    bot = TradingBot.__new__(TradingBot)
    bot.db = db
    bot.dry_run = False
    bot.notifier = None
    bot.connection = SimpleNamespace(ib=None)
    return bot


class _Event:
    """Minimal stand-in for an ib_insync Event: supports += / -= and len()."""

    def __init__(self):
        self.handlers = []

    def __iadd__(self, h):
        self.handlers.append(h)
        return self

    def __isub__(self, h):
        if h not in self.handlers:
            raise ValueError("handler not registered")
        self.handlers.remove(h)
        return self


# ---------------------------------------------------- Database.update_trade_status

class TestUpdateTradeStatus:

    def test_market_buy_gets_its_fill_price(self, tmp_path):
        # AIGI 2026-09-07: BUY 62 saved with price=0, filled @ 20.05
        db = _db(tmp_path)
        db.save_trade("AIGI", "BUY", 62, 0.0, order_id=5711,
                      status="SUBMITTED", reason="Trend signal: +0.93")
        before = _row(db, 5711)[0]["executed_at"]

        assert db.update_trade_status(5711, "FILLED", price=20.05) == 1

        row = _row(db, 5711)[0]
        assert row["status"] == "FILLED"
        assert row["price"] == pytest.approx(20.05)
        assert row["reason"] == "Trend signal: +0.93"      # no note -> unchanged
        assert row["executed_at"] >= before

    def test_recorded_stop_price_is_never_overwritten(self, tmp_path):
        # AIGI stop 5713: placed init=19.41, filled @ 19.83 on 09-10
        db = _db(tmp_path)
        db.save_trade("AIGI", "SELL", 62, 19.41, order_id=5713,
                      status="SUBMITTED", reason="Trailing stop 3.0xATR")

        assert db.update_trade_status(5713, "FILLED", price=19.83) == 1

        assert _row(db, 5713)[0]["price"] == pytest.approx(19.41)

    def test_idempotent_and_unknown_order_is_a_noop(self, tmp_path):
        db = _db(tmp_path)
        db.save_trade("CRUD", "BUY", 78, 0.0, order_id=5576, status="SUBMITTED")

        assert db.update_trade_status(5576, "FILLED", price=16.0) == 1
        assert db.update_trade_status(5576, "FILLED", price=99.0) == 0
        assert db.update_trade_status(5576, "CANCELLED") == 0
        assert db.update_trade_status(424242, "FILLED", price=1.0) == 0
        assert db.update_trade_status(None, "FILLED") == 0

        row = _row(db, 5576)[0]
        assert row["status"] == "FILLED"
        assert row["price"] == pytest.approx(16.0)

    def test_non_terminal_status_is_refused(self, tmp_path):
        db = _db(tmp_path)
        db.save_trade("CMOD", "BUY", 3, 0.0, order_id=5450, status="SUBMITTED")
        with pytest.raises(ValueError):
            db.update_trade_status(5450, "SUBMITTED")
        with pytest.raises(ValueError):
            db.update_trade_status(5450, "Filled")
        assert _row(db, 5450)[0]["status"] == "SUBMITTED"

    def test_note_is_appended_to_reason(self, tmp_path):
        # EIMU 4683, 2026-08-13: Error 201 settled-cash rejection
        db = _db(tmp_path)
        db.save_trade("EIMU", "BUY", 117, 0.0, order_id=4683,
                      status="SUBMITTED", reason="Trend signal: +0.69")

        db.update_trade_status(4683, "REJECTED",
                               note="Error 201: Order rejected - reason:Available settled cash")

        row = _row(db, 4683)[0]
        assert row["status"] == "REJECTED"
        assert row["reason"].startswith("Trend signal: +0.69 | Error 201:")
        assert row["price"] == 0.0

    def test_execution_rows_are_left_alone(self, tmp_path):
        # Stop fill inserts a FILLED execution row for the same order_id; the
        # status update must touch only the SUBMITTED placement row.
        db = _db(tmp_path)
        db.save_trade("AIGA", "SELL", 126, 6.79, order_id=3554,
                      status="SUBMITTED", reason="Trailing stop 3.0xATR")
        db.save_trade("AIGA", "SELL", 3, 7.0325, order_id=3554,
                      status="FILLED", reason="TRAIL stop fill (realizedPnL=$-2.88)")
        db.save_trade("AIGA", "SELL", 123, 7.0325, order_id=3554,
                      status="FILLED", reason="TRAIL stop fill (realizedPnL=$+45.81)")

        assert db.update_trade_status(3554, "FILLED", price=7.0325, note="x") == 1

        rows = _row(db, 3554)
        assert [r["status"] for r in rows] == ["FILLED", "FILLED", "FILLED"]
        assert rows[0]["price"] == pytest.approx(6.79)          # placement keeps trigger
        assert rows[0]["reason"] == "Trailing stop 3.0xATR | x"
        assert rows[1]["reason"] == "TRAIL stop fill (realizedPnL=$-2.88)"
        assert rows[2]["reason"] == "TRAIL stop fill (realizedPnL=$+45.81)"


# ------------------------------------------------------ TradingBot._on_order_status

class TestOnOrderStatus:

    def test_filled_entry_buy_records_price(self, tmp_path, caplog):
        db = _db(tmp_path)
        db.save_trade("AIGI", "BUY", 62, 0.0, order_id=5711,
                      status="SUBMITTED", reason="Trend signal: +0.93")
        bot = _bot(db)

        with caplog.at_level(logging.INFO):
            bot._on_order_status(
                _trade(5711, "AIGI", "BUY", 62, "MKT", "Filled", avg=20.05))

        row = _row(db, 5711)[0]
        assert row["status"] == "FILLED"
        assert row["price"] == pytest.approx(20.05)
        assert "Ledger: order 5711 BUY 62 AIGI MKT -> FILLED @ 20.0500" in caplog.text

    def test_error_201_rejection_is_rejected_not_cancelled(self, tmp_path):
        # Replays the 2026-08-13 EIMU rejection: PendingSubmit -> Inactive ->
        # Cancelled with the Error 201 text on the last log entry.
        db = _db(tmp_path)
        db.save_trade("EIMU", "BUY", 117, 0.0, order_id=4683,
                      status="SUBMITTED", reason="Trend signal: +0.69")
        bot = _bot(db)
        msg = ("Error 201, reqId 4683: Order rejected - reason:Available settled "
               "cash converted to base: 59.24 GBP Cash needed for this order and "
               "other pending orders: <br>671.82 GBP")
        trade = _trade(4683, "EIMU", "BUY", 117, "MKT", "Cancelled", log=[
            ("PendingSubmit", "", 0), ("Inactive", "", 0), ("Cancelled", msg, 201),
        ])

        # The Inactive step must not be treated as terminal.
        trade.orderStatus.status = "Inactive"
        bot._on_order_status(trade)
        assert _row(db, 4683)[0]["status"] == "SUBMITTED"

        trade.orderStatus.status = "Cancelled"
        bot._on_order_status(trade)
        row = _row(db, 4683)[0]
        assert row["status"] == "REJECTED"
        assert "| Error 201: Error 201, reqId 4683: Order rejected" in row["reason"]
        assert "<br>" not in row["reason"]

    def test_clean_cancel_is_cancelled(self, tmp_path):
        # CMOD stop 5306 cancelled by replace_trailing_stop on 2026-09-01
        db = _db(tmp_path)
        db.save_trade("CMOD", "SELL", 23, 32.91, order_id=5306,
                      status="SUBMITTED", reason="Trailing stop 4.0xATR")
        bot = _bot(db)

        bot._on_order_status(_trade(5306, "CMOD", "SELL", 23, "TRAIL", "ApiCancelled",
                                    log=[("PendingCancel", "", 0),
                                         ("Cancelled", "", 0)]))

        row = _row(db, 5306)[0]
        assert row["status"] == "CANCELLED"
        assert row["reason"] == "Trailing stop 4.0xATR"
        assert row["price"] == pytest.approx(32.91)

    def test_non_terminal_statuses_are_ignored(self, tmp_path):
        db = _db(tmp_path)
        db.save_trade("CRUD", "SELL", 78, 14.91, order_id=5578, status="SUBMITTED")
        bot = _bot(db)

        for st in ("PendingSubmit", "PreSubmitted", "Submitted", "Inactive", ""):
            bot._on_order_status(_trade(5578, "CRUD", "SELL", 78, "TRAIL", st))
            assert _row(db, 5578)[0]["status"] == "SUBMITTED", st

    def test_unknown_order_and_db_failure_never_raise(self, tmp_path, caplog):
        db = _db(tmp_path)
        bot = _bot(db)

        # Probe/manual order the bot never recorded -> silent no-op
        with caplog.at_level(logging.INFO):
            bot._on_order_status(_trade(999, "XXXX", "BUY", 1, "MKT", "Filled", avg=1.0))
        assert "Ledger:" not in caplog.text

        # Ledger blowing up must not escape an ib_insync event handler
        class _Boom:
            def update_trade_status(self, *a, **k):
                raise RuntimeError("disk full")
        bot.db = _Boom()
        with caplog.at_level(logging.WARNING):
            bot._on_order_status(_trade(5711, "AIGI", "BUY", 62, "MKT", "Filled", avg=20.05))
        assert "Could not update trades ledger for order 5711" in caplog.text

        # And a malformed trade object is swallowed too
        bot._on_order_status(SimpleNamespace())


# ------------------------------------------- _on_commission_report belt-and-braces

def _fill(avg, shares):
    return SimpleNamespace(execution=SimpleNamespace(
        avgPrice=avg, price=avg, shares=shares, cumQty=shares))


def _report(pnl, commission=4.0):
    return SimpleNamespace(realizedPNL=pnl, commission=commission, currency="USD")


class TestCommissionReportMarksLedger:

    def test_stop_fill_in_two_partials_marks_placement_once_complete(self, tmp_path):
        # AIGA 2026-09-14 07:01: SELL 3 then SELL 123 on orderId 3554
        db = _db(tmp_path)
        db.save_trade("AIGA", "SELL", 126, 6.79, order_id=3554,
                      status="SUBMITTED", reason="Trailing stop 3.0xATR")
        bot = _bot(db)

        # First partial: 123 still remaining -> placement row stays SUBMITTED,
        # but the execution row is inserted as before.
        t = _trade(3554, "AIGA", "SELL", 126, "TRAIL", "Submitted", remaining=123)
        bot._on_commission_report(t, _fill(7.0325, 3), _report(-2.88))
        assert [r["status"] for r in _row(db, 3554)] == ["SUBMITTED", "FILLED"]

        # Last partial: complete -> placement row FILLED, trigger price kept.
        t = _trade(3554, "AIGA", "SELL", 126, "TRAIL", "Filled", avg=7.0325, remaining=0)
        bot._on_commission_report(t, _fill(7.0325, 123), _report(45.81, 0.0))
        rows = _row(db, 3554)
        assert [r["status"] for r in rows] == ["FILLED", "FILLED", "FILLED"]
        assert rows[0]["price"] == pytest.approx(6.79)
        assert rows[0]["reason"] == "Trailing stop 3.0xATR"
        # Existing behaviour intact: cooldown was set by the stop-fill path.
        blocked, _ = db.is_symbol_in_cooldown("AIGA")
        assert blocked

    def test_entry_buy_fill_records_price_without_execution_row(self, tmp_path):
        # A market BUY's commission report must fill in the entry price on the
        # placement row and must NOT go down the stop-fill path (no execution
        # row, no cooldown — that would block the name we just bought).
        db = _db(tmp_path)
        db.save_trade("AIGI", "BUY", 62, 0.0, order_id=5711,
                      status="SUBMITTED", reason="Trend signal: +0.93")
        bot = _bot(db)

        t = _trade(5711, "AIGI", "BUY", 62, "MKT", "Filled", avg=20.05, remaining=0)
        bot._on_commission_report(t, _fill(20.05, 62), _report(0.0))

        rows = _row(db, 5711)
        assert len(rows) == 1
        assert rows[0]["status"] == "FILLED"
        assert rows[0]["price"] == pytest.approx(20.05)
        blocked, _ = db.is_symbol_in_cooldown("AIGI")
        assert not blocked


# --------------------------------------------------------- subscription wiring

class TestRegistration:

    def test_both_events_subscribed_once(self, tmp_path):
        db = _db(tmp_path)
        bot = _bot(db)
        ib = SimpleNamespace(commissionReportEvent=_Event(), orderStatusEvent=_Event())
        bot.connection = SimpleNamespace(ib=ib)

        bot._register_fill_handlers()
        bot._register_fill_handlers()   # re-registering must not double-subscribe

        assert ib.commissionReportEvent.handlers == [bot._on_commission_report]
        assert ib.orderStatusEvent.handlers == [bot._on_order_status]

    def test_dry_run_subscribes_nothing(self, tmp_path):
        db = _db(tmp_path)
        bot = _bot(db)
        bot.dry_run = True
        ib = SimpleNamespace(commissionReportEvent=_Event(), orderStatusEvent=_Event())
        bot.connection = SimpleNamespace(ib=ib)

        bot._register_fill_handlers()

        assert ib.commissionReportEvent.handlers == []
        assert ib.orderStatusEvent.handlers == []
