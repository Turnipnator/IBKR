"""
Sleeve valued and reported in the account's base currency — the 2026-09-25 healthcheck finding.

`Sleeve.market_value()` took IBKR's portfolio `marketValue` as if it were pounds. It is in the
instrument's own currency, and VUAA is a dollar line, so the ring-fence the momentum strategy sizes
around read 3,283.97 on 09-23 and 3,249.95 on 09-24 for a sleeve worth about 2,490 in pounds. Its
fallback path also assumed an FX rate of 1.0 when one was missing — the same silent 34% error that
the 09-22 settlement fix removed from `_to_base`.

Replays the read-only probe of 2026-09-25 ~09:15 UTC: VUAA 21 sh, marketValue $3,140.09, IBKR
ExchangeRate USD 0.7557584, NetLiquidation currency GBP, sleeve reserve 132.95906739068.

These drive the real SleeveStrategy, the real Database on a temp file, the real
DecisionEngine._sleeve_claim and the real ConnectionManager.get_base_currency, per CLAUDE.md.
"""
import logging
from types import SimpleNamespace

import pytest

from src.connection import ConnectionManager
from src.database import Database
from src.engine import DecisionEngine
from src.orders import Position
from src.sleeve import SleeveStrategy
from tests.test_sleeve import FakeFetcher, FakeNotifier, FakeOrders, cfg

VUAA_QTY = 21
VUAA_MV_USD = 3140.09
FX_USD = 0.7557584
RESERVE = 132.95906739068


def _vuaa(market_value=VUAA_MV_USD):
    return Position(symbol="VUAA", quantity=VUAA_QTY, avg_cost=150.2, market_value=market_value,
                    unrealized_pnl=0.0, realized_pnl=0.0)


def _sleeve(tmp_path, *, positions=(), fx=None, prices=None, base="GBP", notifier=None):
    db = Database(str(tmp_path / "sleeve.db"))
    rates = {"USD": FX_USD, "GBP": 1.0} if fx is None else fx
    connection = SimpleNamespace(
        get_fx_rates=lambda: dict(rates),
        get_base_currency=lambda: base,
        get_account_summary=lambda: {"AvailableFunds": {"value": "5000"}},
        ensure_connected=lambda: True,
    )
    s = SleeveStrategy(connection, FakeOrders(), SimpleNamespace(get_positions=lambda: list(positions)),
                       FakeFetcher({}, prices or {}), db, notifier, cfg(cash_buffer=0.06))
    s.reserve()                                  # seed at the 2,500 capital base first
    db.add_sleeve_reserve(RESERVE - 2500.0)      # then move to today's figure
    return s, db


# ---------------------------------------------------------------- valuation
def test_todays_sleeve_is_valued_in_pounds(tmp_path):
    s, _ = _sleeve(tmp_path, positions=[_vuaa()])
    assert s.market_value() == pytest.approx(VUAA_MV_USD * FX_USD)          # 2,373.15
    assert s.claim_on_account() == pytest.approx(VUAA_MV_USD * FX_USD + RESERVE)   # 2,506.11
    # the old code's answer: dollars added to pounds
    assert s.claim_on_account() < VUAA_MV_USD + RESERVE - 700


def test_the_claim_matches_the_capital_it_was_funded_with(tmp_path):
    """£2,500 went in; a sleeve worth ~£2,506 is plausible, £3,273 is not."""
    s, _ = _sleeve(tmp_path, positions=[_vuaa()])
    assert s.claim_on_account() == pytest.approx(2500.0, rel=0.02)


def test_no_fx_rate_is_an_error_not_a_rate_of_one(tmp_path):
    s, _ = _sleeve(tmp_path, positions=[_vuaa()], fx={"GBP": 1.0})
    with pytest.raises(RuntimeError, match="FX rate"):
        s.market_value()


def test_the_price_fallback_is_converted_too(tmp_path):
    """No marketValue from the portfolio feed → quantity x quote, still in dollars, then converted."""
    s, _ = _sleeve(tmp_path, positions=[_vuaa(market_value=0.0)], prices={"VUAA": 149.528})
    assert s.market_value() == pytest.approx(VUAA_QTY * 149.528 * FX_USD)


def test_no_price_is_an_error_not_zero(tmp_path):
    s, _ = _sleeve(tmp_path, positions=[_vuaa(market_value=0.0)], prices={})
    with pytest.raises(RuntimeError, match="no price"):
        s.market_value()


def test_a_pound_line_needs_no_rate(tmp_path):
    held = Position(symbol="IBTM", quantity=20, avg_cost=100.0, market_value=2000.0,
                    unrealized_pnl=0.0, realized_pnl=0.0)
    s, _ = _sleeve(tmp_path, positions=[held], fx={})
    assert s.market_value() == pytest.approx(2000.0)


def test_non_sleeve_holdings_are_not_counted(tmp_path):
    cspx = Position(symbol="CSPX", quantity=1, avg_cost=800.0, market_value=833.9,
                    unrealized_pnl=0.0, realized_pnl=0.0)
    s, _ = _sleeve(tmp_path, positions=[_vuaa(), cspx])
    assert s.market_value() == pytest.approx(VUAA_MV_USD * FX_USD)


def test_the_engine_excludes_the_pound_figure(tmp_path):
    s, _ = _sleeve(tmp_path, positions=[_vuaa()])
    eng = DecisionEngine.__new__(DecisionEngine)
    eng.sleeve = s
    assert eng._sleeve_claim() == pytest.approx(VUAA_MV_USD * FX_USD + RESERVE)


def test_an_unvaluable_sleeve_is_logged_by_the_engine(tmp_path, caplog):
    s, _ = _sleeve(tmp_path, positions=[_vuaa()], fx={"GBP": 1.0})
    eng = DecisionEngine.__new__(DecisionEngine)
    eng.sleeve = s
    with caplog.at_level(logging.WARNING):
        eng._sleeve_claim()
    assert "Sleeve claim unavailable" in caplog.text


# ---------------------------------------------------------------- unit cost
def test_unit_cost_without_a_rate_defers_instead_of_guessing(tmp_path):
    s, _ = _sleeve(tmp_path, fx={"GBP": 1.0}, prices={"VUAA": 150.0})
    assert s._unit_base("VUAA") == 0.0            # was 150 x 1.0 x 1.06 = "£159" a share
    assert s._affordable("VUAA", 2000.0) == 0


def test_unit_cost_with_a_rate_is_unchanged(tmp_path):
    s, _ = _sleeve(tmp_path, prices={"VUAA": 150.0})
    assert s._unit_base("VUAA") == pytest.approx(150.0 * FX_USD * 1.06)


# ---------------------------------------------------------------- wording
def test_money_is_shown_in_pounds(tmp_path):
    s, _ = _sleeve(tmp_path)
    assert s._money(1554.32) == "£1,554"
    assert s._price("VUAA", 150.3502) == "$150.35"


def test_unknown_base_currency_says_base_rather_than_guessing_a_symbol(tmp_path):
    s, _ = _sleeve(tmp_path, base="")
    assert s._money(133) == "133 base"


def test_deferral_line_is_in_pounds(tmp_path, caplog):
    """This morning's line read '133 base of settled cash against 133 still to invest'."""
    s, db = _sleeve(tmp_path, prices={"VUAA": 150.0})
    db.set_sleeve_month(s.month_key(), "VUAA", "pending_cash")
    with caplog.at_level(logging.INFO):
        s._fund_target(s.month_key(), "VUAA", retry=True)
    assert "£133 of settled cash against £133 still to invest" in caplog.text
    assert "below the £150 order floor" in caplog.text


def test_fill_log_and_telegram_are_in_pounds(tmp_path, caplog):
    """Replays the 09-23 tranche: 13 VUAA @ $150.3502."""
    notifier = FakeNotifier()
    s, db = _sleeve(tmp_path, notifier=notifier)
    db.add_sleeve_reserve(1600.96 - RESERVE)
    month = s.month_key()
    db.record_sleeve_order(month, "BUY", "VUAA", 13, 6003, fill_price=None, est_base=1554.32,
                           status="SUBMITTED")
    with caplog.at_level(logging.INFO):
        s.on_order_settled(6003, "FILLED", fill_price=150.3502, filled_quantity=13)
    cost = 13 * 150.3502 * FX_USD
    assert f"(£{cost:,.0f}); £{1600.96 - cost:,.0f} still to invest" in caplog.text
    headline, lines = notifier.sleeve_messages[-1]
    assert headline == f"Bought <b>13 VUAA</b> @ $150.35 = £{cost:,.0f}"
    left = 1600.96 - cost                                   # £124 at today's rate: under the floor
    assert lines == [f"Fully invested — £{left:,.0f} stays in cash, below the £150 minimum order."]


def test_a_partly_funded_fill_says_what_is_left_in_pounds(tmp_path):
    notifier = FakeNotifier()
    s, db = _sleeve(tmp_path, notifier=notifier)
    db.add_sleeve_reserve(2500.0 - RESERVE)                 # back to the full 2,500
    db.record_sleeve_order(s.month_key(), "BUY", "VUAA", 8, 5990, fill_price=None, est_base=953.81,
                           status="SUBMITTED")
    s.on_order_settled(5990, "FILLED", fill_price=150.1104, filled_quantity=8)
    left = 2500.0 - 8 * 150.1104 * FX_USD
    assert notifier.sleeve_messages[-1][1] == [f"£{left:,.0f} of £2,500 still to invest"]


# ---------------------------------------------------------------- the connection
def _conn(values):
    c = ConnectionManager.__new__(ConnectionManager)
    c._base_ccy = ""
    c.ensure_connected = lambda: True
    c.ib = SimpleNamespace(accountValues=lambda: values)
    return c


def test_base_currency_is_read_from_net_liquidation():
    rows = [SimpleNamespace(tag="ExchangeRate", currency="BASE", value="1.00"),
            SimpleNamespace(tag="ExchangeRate", currency="USD", value="0.7557584"),
            SimpleNamespace(tag="NetLiquidation", currency="GBP", value="4668.55")]
    assert _conn(rows).get_base_currency() == "GBP"


def test_base_currency_is_cached_once_known():
    c = _conn([SimpleNamespace(tag="NetLiquidation", currency="GBP", value="4668.55")])
    assert c.get_base_currency() == "GBP"
    c.ib = SimpleNamespace(accountValues=lambda: [])       # a momentary gap
    assert c.get_base_currency() == "GBP"


def test_base_currency_unreadable_is_empty():
    c = _conn([])
    assert c.get_base_currency() == ""
    c.ib = SimpleNamespace(accountValues=lambda: (_ for _ in ()).throw(RuntimeError("loop")))
    assert c.get_base_currency() == ""
