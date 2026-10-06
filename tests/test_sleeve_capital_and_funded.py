"""
Raising the sleeve's capital base, and calling a fully invested sleeve fully invested — 2026-10-06.

1. Owner's decision 2026-10-06: put the ~£957 of idle cash into the sleeve by raising
   SLEEVE_CAPITAL_BASE. The reserve was seeded from capital_base once (2026-09-17) and never looked
   at it again, so changing the env var alone would have done nothing. The difference now moves
   into the reserve exactly once.
2. The 10-04 healthcheck P3: with £133 left of £2,500 the daily hook logged "buy deferred … £133 of
   settled cash against £133 still to invest is below the £150 order floor, waiting for more settled
   cash" once a day, forever — more cash cannot help when the reserve itself is below the floor.

Replays the 2026-10-06 state: 21 VUAA held, reserve 132.95906739068 of a 2,500 capital base, IBKR
AvailableFunds ~£1,090. Drives the real SleeveStrategy and the real Database on a temp file.
"""
import logging
from datetime import datetime

import pandas as pd
import pytest

from src.database import Database
from src.orders import OrderAction, Position
from tests.test_sleeve import bars, cfg, make_sleeve

RESERVE = 132.95906739068
VUAA_USD = 151.0
FX_USD = 0.75          # make_sleeve's rate


def _today(tmp_path, *, capital_base=2500.0, available=1090.0, positions=(), frames=None, notifier=None):
    """The live sleeve as of 2026-10-06, read under `capital_base`."""
    db = Database(str(tmp_path / "sleeve.db"))
    db.get_sleeve_reserve(2500.0)                 # seeded 2026-09-17
    db.add_sleeve_reserve(RESERVE - 2500.0)       # spent on 21 VUAA since
    s, orders, db = make_sleeve(tmp_path, db=db, available=available, positions=positions,
                                prices={"VUAA": VUAA_USD}, frames=frames,
                                config=cfg(capital_base=capital_base, cash_buffer=0.06))
    s.notifier = notifier
    s.connection.get_base_currency = lambda: "GBP"
    return s, orders, db


def _vuaa21():
    return [Position(symbol="VUAA", quantity=21, avg_cost=150.2, market_value=3171.0,
                     unrealized_pnl=0.0, realized_pnl=0.0)]


# ---------------------------------------------------------------- capital base changes
def test_raising_the_capital_base_moves_the_difference_into_the_reserve_once(tmp_path, caplog):
    db = Database(str(tmp_path / "t.db"))
    db.get_sleeve_reserve(2500.0)
    db.add_sleeve_reserve(RESERVE - 2500.0)
    with caplog.at_level(logging.WARNING):
        assert db.get_sleeve_reserve(3400.0) == pytest.approx(RESERVE + 900.0)
    assert "Sleeve capital base changed 2,500.00 -> 3,400.00" in caplog.text
    # every later read sees no difference — the raise is never re-applied
    caplog.clear()
    for _ in range(3):
        assert db.get_sleeve_reserve(3400.0) == pytest.approx(RESERVE + 900.0)
    assert "capital base changed" not in caplog.text


def test_the_old_code_ignored_a_new_capital_base(tmp_path):
    """Pins what was broken: the stored base is what moves, not just the number in the env."""
    db = Database(str(tmp_path / "t.db"))
    db.get_sleeve_reserve(2500.0)
    db.add_sleeve_reserve(RESERVE - 2500.0)
    db.get_sleeve_reserve(3400.0)
    conn = db._get_connection()
    try:
        assert conn.execute("SELECT capital_base FROM sleeve_account").fetchone()[0] == 3400.0
    finally:
        conn.close()


def test_spending_after_a_raise_is_not_undone_by_the_next_read(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.get_sleeve_reserve(2500.0)
    db.get_sleeve_reserve(3400.0)                            # 3,400 to invest
    db.add_sleeve_reserve(-1000.0)
    assert db.get_sleeve_reserve(3400.0) == pytest.approx(2400.0)


def test_a_cut_reduces_the_reserve_and_never_takes_it_negative(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.get_sleeve_reserve(2500.0)
    db.add_sleeve_reserve(-1500.0)                           # 1,000 left
    assert db.get_sleeve_reserve(2200.0) == pytest.approx(700.0)
    assert db.get_sleeve_reserve(1000.0) == 0.0              # more than is left: floors at 0
    assert db.get_sleeve_reserve(1000.0) == 0.0


def test_a_raise_then_a_cut_back_lands_where_it_started(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.get_sleeve_reserve(2500.0)
    db.add_sleeve_reserve(RESERVE - 2500.0)
    db.get_sleeve_reserve(3400.0)
    assert db.get_sleeve_reserve(2500.0) == pytest.approx(RESERVE)


def test_after_the_raise_the_daily_hook_buys_the_next_tranche(tmp_path):
    """Today's state under the new base: £1,033 to invest, £1,090 settled → one finishing tranche."""
    s, orders, db = _today(tmp_path, capital_base=3400.0, positions=_vuaa21())
    db.set_sleeve_month(s.month_key(), "VUAA", "already_held")
    out = s.retry_pending()
    assert out["status"] == "placed"
    unit = VUAA_USD * FX_USD * 1.06
    qty = int((RESERVE + 900.0) / unit)                      # 8 shares, ~£960
    assert orders.placed == [("VUAA", OrderAction.BUY, qty, "sleeve entry (retry)")]
    # placement reserves the cost; the reserve itself moves only on settlement
    assert db.sleeve_committed_base() == pytest.approx(qty * unit)
    assert db.get_sleeve_reserve(3400.0) == pytest.approx(RESERVE + 900.0)
    # and the fill leaves less than the floor, so the sleeve is done after one order
    s.on_order_settled(out_order_id(db), "FILLED", fill_price=VUAA_USD, filled_quantity=qty)
    assert db.get_sleeve_reserve(3400.0) < s.config.min_order_base


def out_order_id(db):
    rows = db.sleeve_unsettled_orders()
    assert len(rows) == 1
    return rows[0]["order_id"]


def test_settled_cash_still_caps_a_raised_reserve(tmp_path):
    """Raising the base claims no cash the account does not have: settled cash still binds."""
    s, orders, db = _today(tmp_path, capital_base=5000.0, available=1090.0, positions=_vuaa21())
    db.set_sleeve_month(s.month_key(), "VUAA", "already_held")
    s.retry_pending()
    unit = VUAA_USD * FX_USD * 1.06
    assert orders.placed[0][2] == int(1090.0 / unit)         # sized to the cash, not the reserve


# ---------------------------------------------------------------- fully invested (P3)
def test_todays_sleeve_is_fully_invested_not_waiting_for_cash(tmp_path, caplog):
    """Replays the 10-02/10-05/10-06 07:00 line, with plenty of settled cash on hand."""
    s, orders, db = _today(tmp_path, available=5000.0, positions=_vuaa21())
    with caplog.at_level(logging.INFO):
        out = s._fund_target(s.month_key(), "VUAA", retry=True)
    assert out == {"status": "funded", "quantity": 0}
    assert "VUAA fully invested — £133 left in cash is below the £150 order floor" in caplog.text
    assert "waiting for more settled cash" not in caplog.text
    assert orders.placed == []


def test_the_daily_hook_is_silent_once_fully_invested(tmp_path, caplog):
    s, orders, db = _today(tmp_path, available=5000.0, positions=_vuaa21())
    db.set_sleeve_month(s.month_key(), "VUAA", "already_held")
    with caplog.at_level(logging.DEBUG):
        for _ in range(5):
            assert s.retry_pending() == {"status": "noop"}
    assert "Sleeve" not in caplog.text
    assert orders.placed == []
    assert s.fetcher.price_calls == []                       # never prices a buy it cannot place


def test_the_monthly_decision_reports_already_held_without_a_waiting_line(tmp_path):
    last = pd.Timestamp("2026-09-30")
    frames = {"VUAA": bars(last, 14, 100, 1.01), "IB01": bars(last, 14, 100, 1.002)}
    notes = []
    notifier = type("N", (), {"enabled": True,
                              "notify_sleeve": lambda self, h, lines: notes.append((h, lines))})()
    s, orders, db = _today(tmp_path, available=5000.0, positions=_vuaa21(), frames=frames,
                           notifier=notifier)
    out = s.rebalance(datetime(2026, 11, 2, 14, 6))
    assert out["status"] == "already_held"
    assert db.get_sleeve_month("2026-11")["status"] == "already_held"
    assert orders.placed == []
    assert not any("Waiting for settled cash" in line for _, lines in notes for line in lines)


def test_a_reserve_at_the_floor_still_trades(tmp_path):
    """Exactly £150 left is placeable — only below the floor is done."""
    db = Database(str(tmp_path / "sleeve.db"))
    db.get_sleeve_reserve(2500.0)
    db.add_sleeve_reserve(150.0 - 2500.0)
    s, orders, _ = make_sleeve(tmp_path, db=db, available=5000.0, prices={"VUAA": 100.0},
                               config=cfg(cash_buffer=0.0))
    out = s._fund_target(s.month_key(), "VUAA")
    assert out["status"] == "placed"
    assert orders.placed[0][2] == 2                          # 2 x £75, finishing tranche


def test_a_large_reserve_short_of_settled_cash_still_defers_and_says_why(tmp_path, caplog):
    """The genuine wait is unchanged: lots left to invest, too little settled cash right now."""
    s, orders, db = make_sleeve(tmp_path, available=133.0, prices={"VUAA": VUAA_USD},
                                config=cfg(cash_buffer=0.06))
    s.connection.get_base_currency = lambda: "GBP"
    with caplog.at_level(logging.INFO):
        out = s._fund_target(s.month_key(), "VUAA", retry=True)
    assert out["status"] == "pending_cash"
    assert "£133 of settled cash against £2,500 still to invest is below the £150 order floor" in caplog.text
    assert orders.placed == []
