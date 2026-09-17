"""
Forward-test sleeve (attempt 9) and its ring-fencing.

Registered in research/2026-09-17_forward_test/PREREG_9_ucits_forward_test.md. These tests drive the
real objects — SleeveStrategy, the real DecisionEngine and TradingBot methods via __new__, and a real
Database on a temp file — with mocked IBKR dependencies, per CLAUDE.md's live-only testing rule.

The ring-fencing rules under test (PREREG_9 §3):
  1. momentum sizing excludes sleeve positions and cash reserve
  2. the momentum settled-cash gate subtracts the sleeve reserve
  3. sleeve symbols get no protective stops and are not orphan-flagged
  4. a momentum drawdown halt does not liquidate the sleeve
  5. one decision per calendar month, restart-safe
"""
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import SleeveConfig, TradingConfig
from src.database import Database
from src.engine import DecisionEngine
from src.orders import OrderAction, OrderResult, Position
from src.sleeve import SleeveStrategy


def cfg(**kw):
    base = dict(enabled=True, capital_base=2500.0, equity_symbol="VUAA", bond_symbol="IBTM",
                hurdle_symbol="IB01", lookback_months=12, cash_buffer=0.02, hour=14, minute=5)
    base.update(kw)
    return SleeveConfig(**base)


def bars(last_month_end: pd.Timestamp, months: int, start: float, growth: float):
    """Daily bars whose month-end closes grow geometrically by `growth` a month."""
    rows = []
    price = start
    for i in range(months):
        m_end = (last_month_end - pd.DateOffset(months=months - 1 - i)) + pd.offsets.MonthEnd(0)
        price *= growth
        for day in (5, 15, m_end.day):
            rows.append({"date": m_end.replace(day=min(day, m_end.day)), "close": price})
    return pd.DataFrame(rows)


class FakeFetcher:
    def __init__(self, frames, prices=None):
        self.frames = frames
        self.prices = prices or {}
        self.calls = []

    def get_historical_data(self, symbol, duration="1 Y", bar_size="1 day", what_to_show="TRADES", **kw):
        self.calls.append((symbol, duration, bar_size, what_to_show))
        return self.frames.get(symbol)

    def get_latest_prices(self, symbols):
        return {s: self.prices.get(s) for s in symbols if self.prices.get(s)}


class FakeOrders:
    def __init__(self, result=None):
        self.placed = []
        self.result = result or (lambda symbol, action, qty: OrderResult(
            success=True, order_id=100 + len(self.placed), fill_price=10.0, filled_quantity=qty))

    def place_market_order(self, symbol, action, quantity, reason=None):
        self.placed.append((symbol, action, quantity, reason))
        return self.result(symbol, action, quantity)


def make_sleeve(tmp_path, *, positions=(), prices=None, frames=None, available=10_000.0, config=None, db=None):
    db = db or Database(str(tmp_path / "sleeve.db"))
    connection = SimpleNamespace(
        get_fx_rates=lambda: {"USD": 0.75, "GBP": 1.0},
        get_account_summary=lambda: {"AvailableFunds": {"value": str(available)}},
        ensure_connected=lambda: True,
    )
    orders = FakeOrders()
    pm = SimpleNamespace(get_positions=lambda: list(positions))
    fetcher = FakeFetcher(frames or {}, prices)
    s = SleeveStrategy(connection, orders, pm, fetcher, db, None, config or cfg())
    return s, orders, db


# ---------------------------------------------------------------- signal
def test_signal_holds_stocks_when_they_beat_the_hurdle(tmp_path):
    last = pd.Timestamp("2026-08-31")
    frames = {"VUAA": bars(last, 14, 100, 1.01), "IB01": bars(last, 14, 100, 1.002)}
    s, _, _ = make_sleeve(tmp_path, frames=frames)
    sig = s.signal()
    assert sig.target == "VUAA"
    assert sig.r_stocks > sig.r_hurdle
    # the signal must come from the adjusted series, as the card states
    assert all(call[3] == "ADJUSTED_LAST" for call in s.fetcher.calls)


def test_signal_switches_to_bonds_when_stocks_lag_the_hurdle(tmp_path):
    last = pd.Timestamp("2026-08-31")
    frames = {"VUAA": bars(last, 14, 100, 0.99), "IB01": bars(last, 14, 100, 1.003)}
    s, _, _ = make_sleeve(tmp_path, frames=frames)
    assert s.signal().target == "IBTM"


def test_signal_ignores_the_current_partial_month(tmp_path):
    now = datetime.now()
    this_month_end = pd.Timestamp(now.year, now.month, 1) + pd.offsets.MonthEnd(0)
    frames = {"VUAA": bars(this_month_end, 15, 100, 1.01), "IB01": bars(this_month_end, 15, 100, 1.002)}
    s, _, _ = make_sleeve(tmp_path, frames=frames)
    sig = s.signal()
    prev = this_month_end - pd.DateOffset(months=1)
    assert sig.asof == f"{prev.year:04d}-{prev.month:02d}"     # last completed month, not this one


def test_signal_failure_does_not_trade(tmp_path):
    s, orders, db = make_sleeve(tmp_path, frames={})       # no bars at all
    out = s.rebalance(datetime(2026, 10, 1, 14, 6))
    assert out["status"] == "signal_failed"
    assert orders.placed == []
    assert db.sleeve_month_done("2026-10") is False        # a failed signal may be retried


# ---------------------------------------------------------------- monthly guard
def test_is_due_respects_the_hour_and_the_month_guard(tmp_path):
    s, _, db = make_sleeve(tmp_path)
    assert s.is_due(datetime(2026, 10, 1, 14, 6)) is True
    assert s.is_due(datetime(2026, 10, 1, 13, 59)) is False     # before the sleeve's time
    db.set_sleeve_month("2026-10", "VUAA", "traded")
    assert s.is_due(datetime(2026, 10, 1, 14, 6)) is False      # already ran this month
    assert s.is_due(datetime(2026, 11, 2, 14, 6)) is True       # next month is due again


def test_disabled_sleeve_never_runs(tmp_path):
    s, _, _ = make_sleeve(tmp_path, config=cfg(enabled=False))
    assert s.is_due(datetime(2026, 10, 1, 15, 0)) is False
    assert s.claim_on_account() == 0.0


# ---------------------------------------------------------------- trading
def test_rebalance_sells_the_old_line_then_buys_the_target(tmp_path):
    last = pd.Timestamp("2026-08-31")
    frames = {"VUAA": bars(last, 14, 100, 1.01), "IB01": bars(last, 14, 100, 1.002)}
    held = [Position(symbol="IBTM", quantity=20, avg_cost=100.0, market_value=2000.0,
                     unrealized_pnl=0.0, realized_pnl=0.0)]
    s, orders, db = make_sleeve(tmp_path, positions=held, frames=frames,
                                prices={"VUAA": 147.0, "IBTM": 124.0})
    out = s.rebalance(datetime(2026, 9, 1, 14, 6))
    actions = [(sym, act, qty) for sym, act, qty, _ in orders.placed]
    assert actions[0] == ("IBTM", OrderAction.SELL, 20)
    assert actions[1][0] == "VUAA" and actions[1][1] == OrderAction.BUY
    assert out["status"] == "traded"
    assert db.sleeve_month_done("2026-09") is True


def test_buy_is_deferred_when_settled_cash_is_short(tmp_path):
    last = pd.Timestamp("2026-08-31")
    frames = {"VUAA": bars(last, 14, 100, 1.01), "IB01": bars(last, 14, 100, 1.002)}
    s, orders, db = make_sleeve(tmp_path, frames=frames, prices={"VUAA": 147.0}, available=10.0)
    out = s.rebalance(datetime(2026, 9, 1, 14, 6))
    assert out["status"] == "pending_cash"
    assert orders.placed == []
    assert db.get_sleeve_month("2026-09")["status"] == "pending_cash"


def test_retry_completes_the_deferred_buy_once_cash_settles(tmp_path):
    last = pd.Timestamp("2026-08-31")
    frames = {"VUAA": bars(last, 14, 100, 1.01), "IB01": bars(last, 14, 100, 1.002)}
    db = Database(str(tmp_path / "sleeve.db"))
    db.set_sleeve_month(datetime.now().strftime("%Y-%m"), "VUAA", "pending_cash")
    s, orders, _ = make_sleeve(tmp_path, frames=frames, prices={"VUAA": 147.0}, available=5000.0, db=db)
    out = s.retry_pending()
    assert out["status"] == "traded"
    assert orders.placed[0][0] == "VUAA" and orders.placed[0][1] == OrderAction.BUY
    assert db.get_sleeve_month(datetime.now().strftime("%Y-%m"))["status"] == "traded"


def test_spendable_is_capped_by_both_reserve_and_settled_cash(tmp_path):
    s, _, _ = make_sleeve(tmp_path, available=900.0)
    assert s._spendable() == pytest.approx(900.0)          # settled cash is the binding limit
    s2, _, _ = make_sleeve(tmp_path / "b", available=9_000.0)
    assert s2._spendable() == pytest.approx(2500.0)        # the reserve is


def test_affordable_uses_whole_shares_and_the_cash_buffer(tmp_path):
    s, _, _ = make_sleeve(tmp_path, prices={"VUAA": 100.0})
    # £1,000 budget, $100 share at 0.75 = £75, +2% buffer = £76.50 -> 13 shares
    assert s._affordable("VUAA", 1000.0) == 13


# ---------------------------------------------------------------- database
def test_reserve_seeds_once_then_tracks_spending(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    assert db.get_sleeve_reserve(2500.0) == pytest.approx(2500.0)
    assert db.add_sleeve_reserve(-1000.0) == pytest.approx(1500.0)
    assert db.get_sleeve_reserve(2500.0) == pytest.approx(1500.0)   # seeded value is not re-applied
    assert db.add_sleeve_reserve(-99_999.0) == 0.0                  # never negative


def test_sleeve_orders_are_recorded(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    db.record_sleeve_order("2026-09", "BUY", "VUAA", 17, 4321, 147.25)
    conn = db._get_connection()
    try:
        row = conn.execute("SELECT month, action, symbol, quantity, order_id, fill_price FROM sleeve_orders").fetchone()
    finally:
        conn.close()
    assert tuple(row) == ("2026-09", "BUY", "VUAA", 17, 4321, 147.25)


# ---------------------------------------------------------------- ring-fencing
def engine_with_sleeve(sleeve_positions, momentum_positions, *, reserve=2500.0, available=3000.0):
    eng = DecisionEngine.__new__(DecisionEngine)
    eng.config = TradingConfig()
    eng.config.symbols = {"equity": ["CSPX"], "bond": ["IDTM"]}
    positions = list(sleeve_positions) + list(momentum_positions)
    eng.position_manager = SimpleNamespace(get_positions=lambda: positions)
    eng.db = SimpleNamespace(get_active_cooldowns=lambda: {})
    eng.connection = SimpleNamespace(
        get_fx_rates=lambda: {"USD": 0.75, "GBP": 1.0},
        get_account_summary=lambda: {"AvailableFunds": {"value": str(available)}},
    )
    eng.sleeve = SimpleNamespace(
        config=cfg(), symbols={"VUAA", "IBTM"},
        claim_on_account=lambda: sum(abs(p.market_value) for p in sleeve_positions) + reserve,
        reserve=lambda: reserve,
    )
    return eng


def test_momentum_ignores_sleeve_holdings_when_filtering_held_symbols():
    sleeve_pos = [Position(symbol="VUAA", quantity=17, avg_cost=147.0, market_value=1875.0,
                           unrealized_pnl=0.0, realized_pnl=0.0)]
    eng = engine_with_sleeve(sleeve_pos, [])
    assert eng._sleeve_symbols() == {"VUAA", "IBTM"}
    signals = {"CSPX": {"combined": 0.9, "price": 800.0, "atr": 10.0, "volatility": 0.15}}
    targets = eng._calculate_target_positions(signals, 2200.0)
    assert "CSPX" in targets            # sleeve holding did not disturb momentum's own sizing
    assert "VUAA" not in targets


def test_settled_cash_excludes_the_sleeve_reserve():
    eng = engine_with_sleeve([], [], reserve=2500.0, available=3000.0)
    assert eng._get_settled_cash_base() == pytest.approx(500.0)


def test_settled_cash_is_unchanged_when_the_sleeve_is_off():
    eng = engine_with_sleeve([], [], reserve=2500.0, available=3000.0)
    eng.sleeve.config = cfg(enabled=False)
    assert eng._get_settled_cash_base() == pytest.approx(3000.0)
    assert eng._sleeve_claim() == 0.0


def test_sleeve_claim_failure_leaves_sizing_unadjusted():
    eng = engine_with_sleeve([], [])

    def boom():
        raise RuntimeError("IBKR down")

    eng.sleeve.claim_on_account = boom
    assert eng._sleeve_claim() == 0.0     # fails open, never blocks a rebalance


def bot_with_sleeve(positions):
    from src.bot import TradingBot
    bot = TradingBot.__new__(TradingBot)
    bot.dry_run = False
    bot.notifier = None
    bot.sleeve = SimpleNamespace(config=cfg(), symbols={"VUAA", "IBTM"})
    closed = []
    bot.connection = SimpleNamespace(ensure_connected=lambda: True,
                                     ib=SimpleNamespace(openTrades=lambda: []))
    bot.engine = SimpleNamespace(
        position_manager=SimpleNamespace(
            get_positions=lambda: list(positions),
            close_position=lambda symbol, reason=None: closed.append(symbol) or OrderResult(success=True),
        ),
        order_manager=SimpleNamespace(cancel_all_orders=lambda: 0, get_open_orders=lambda: []),
        _sleeve_symbols=lambda: {"VUAA", "IBTM"},
        state=SimpleNamespace(market_reason="HALT: 20.1% drawdown", market_ok=False),
    )
    return bot, closed


def test_stop_reconciliation_skips_sleeve_positions():
    sleeve_only = [Position(symbol="VUAA", quantity=17, avg_cost=147.0, market_value=1875.0,
                            unrealized_pnl=0.0, realized_pnl=0.0)]
    bot, _ = bot_with_sleeve(sleeve_only)
    assert bot._reconcile_protective_stops() == 0      # nothing to protect: the sleeve carries no stops


def test_drawdown_halt_leaves_the_sleeve_alone():
    positions = [
        Position(symbol="CMOD", quantity=52, avg_cost=35.0, market_value=1400.0,
                 unrealized_pnl=0.0, realized_pnl=0.0),
        Position(symbol="VUAA", quantity=17, avg_cost=147.0, market_value=1875.0,
                 unrealized_pnl=0.0, realized_pnl=0.0),
    ]
    bot, closed = bot_with_sleeve(positions)
    bot._handle_drawdown_halt()
    assert closed == ["CMOD"]


def test_parity_does_not_flag_a_sleeve_symbol_as_an_orphan_stop():
    positions = [Position(symbol="VUAA", quantity=17, avg_cost=147.0, market_value=1875.0,
                          unrealized_pnl=0.0, realized_pnl=0.0)]
    bot, _ = bot_with_sleeve(positions)
    bot._reconcile_protective_stops = lambda: 0
    stray = SimpleNamespace(order=SimpleNamespace(orderType="TRAIL", orderId=999),
                            contract=SimpleNamespace(symbol="VUAA"))
    bot.engine.order_manager.get_open_orders = lambda: [stray]
    status = bot._check_order_parity()
    assert "Orphan" not in status and "orphan" not in status


# ------------------------------------------------- funding in tranches (owner's decision 2026-09-17)
# The sleeve is now fed by momentum stop proceeds as they settle, so it must keep buying across days
# instead of stopping at whatever the first day's cash allowed. Card 9 §3 already required this:
# "buy the new one as settled cash allows (T+2), retrying at each later close until funded".
def _part_funded(tmp_path, *, available, reserve_spent=225.0, held_qty=2):
    """Sleeve holding a small VUAA position with most of its reserve still to invest."""
    last = pd.Timestamp("2026-08-31")
    frames = {"VUAA": bars(last, 14, 100, 1.01), "IB01": bars(last, 14, 100, 1.002)}
    db = Database(str(tmp_path / "sleeve.db"))
    db.get_sleeve_reserve(2500.0)                 # seed, then spend what the first tranche cost
    db.add_sleeve_reserve(-reserve_spent)
    db.set_sleeve_month(datetime.now().strftime("%Y-%m"), "VUAA", "traded")
    held = [Position(symbol="VUAA", quantity=held_qty, avg_cost=147.0, market_value=147.0 * held_qty,
                     unrealized_pnl=0.0, realized_pnl=0.0)]
    s, orders, _ = make_sleeve(tmp_path, positions=held, frames=frames, prices={"VUAA": 147.0},
                               available=available, db=db)
    return s, orders, db


def test_sleeve_keeps_buying_a_partly_funded_position_as_cash_settles(tmp_path):
    s, orders, db = _part_funded(tmp_path, available=800.0)
    out = s.retry_pending()
    assert out["status"] == "traded"
    symbol, action, qty, _ = orders.placed[0]
    assert (symbol, action) == ("VUAA", OrderAction.BUY)
    assert qty == 7                                # 800 / (147 x 0.75 x 1.02) = 7 whole shares
    assert db.get_sleeve_reserve(2500.0) < 2275.0  # the reserve fell by what was spent


def test_a_small_tranche_waits_rather_than_paying_a_flat_commission(tmp_path):
    """£225 of buying power against £2,275 still to invest: a $4 fee on that is ~1.8%."""
    s, orders, _ = _part_funded(tmp_path, available=300.0)
    assert s.retry_pending()["status"] == "noop"
    assert orders.placed == []


def test_the_last_tranche_is_allowed_even_though_it_is_small(tmp_path):
    """Once what is left cannot buy another share, the small order is the finishing one."""
    s, orders, _ = _part_funded(tmp_path, available=5000.0, reserve_spent=2200.0)
    out = s.retry_pending()                        # £300 left, shares cost ~£112.5
    assert out["status"] == "traded"
    assert orders.placed[0][2] == 2


def test_only_one_sleeve_order_a_day(tmp_path):
    """The hook runs every loop pass, and IBKR's settled-cash figure is a lagging cache."""
    s, orders, db = _part_funded(tmp_path, available=800.0)
    assert s.retry_pending()["status"] == "traded"
    assert s.retry_pending()["status"] == "noop"   # same day, second pass
    assert s.retry_pending()["status"] == "noop"
    assert len(orders.placed) == 1


def test_funding_stops_once_the_reserve_is_spent(tmp_path):
    s, orders, _ = _part_funded(tmp_path, available=5000.0, reserve_spent=2500.0)
    assert s.retry_pending()["status"] == "noop"
    assert orders.placed == []


def test_a_month_with_no_decision_is_never_funded(tmp_path):
    """No month row, or a failed signal, must not trigger a buy."""
    s, orders, db = _part_funded(tmp_path, available=5000.0)
    db.set_sleeve_month(datetime.now().strftime("%Y-%m"), "VUAA", "signal_failed")
    assert s.retry_pending()["status"] == "noop"
    assert orders.placed == []


def test_monthly_rebalance_and_the_daily_hook_share_one_funding_path(tmp_path):
    """The monthly buy is the same tranche logic, so it cannot place a token order either."""
    last = pd.Timestamp("2026-08-31")
    frames = {"VUAA": bars(last, 14, 100, 1.01), "IB01": bars(last, 14, 100, 1.002)}
    s, orders, db = make_sleeve(tmp_path, frames=frames, prices={"VUAA": 147.0}, available=300.0)
    out = s.rebalance(datetime(2026, 9, 1, 14, 6))
    assert out["status"] == "pending_cash"         # £225 against a £2,500 reserve — too small
    assert orders.placed == []
    assert db.sleeve_month_done("2026-09") is True # the decision was still made and recorded


def test_a_switch_sell_as_the_first_action_does_not_reseed_the_reserve(tmp_path):
    """Regression: add_sleeve_reserve() creates the row from its delta, so the reserve must be
    seeded at capital_base before any proceeds land — otherwise the sleeve would spend its life
    investing the proceeds of one sale instead of its £2,500."""
    last = pd.Timestamp("2026-08-31")
    frames = {"VUAA": bars(last, 14, 100, 1.01), "IB01": bars(last, 14, 100, 1.002)}
    held = [Position(symbol="IBTM", quantity=20, avg_cost=100.0, market_value=2000.0,
                     unrealized_pnl=0.0, realized_pnl=0.0)]
    s, orders, db = make_sleeve(tmp_path, positions=held, frames=frames,
                                prices={"VUAA": 147.0, "IBTM": 124.0})
    s.rebalance(datetime(2026, 9, 1, 14, 6))       # fresh database: no sleeve_account row yet
    buys = [o for o in orders.placed if o[1] == OrderAction.BUY]
    assert buys and buys[0][2] == 24               # (2500 + 200 proceeds) / (147 x 0.75 x 1.02)
