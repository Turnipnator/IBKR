"""
Tests for the daily-loss gate's P&L source (2026-11-06 healthcheck).

`_compute_session_pnl` read IBKR's `UnrealizedPnL` account value as "today's
unrealized". That value is unrealized SINCE AVERAGE COST, not today's move.
On 2026-11-06 a read-only probe showed UnrealizedPnL BASE -£181.03 against
`reqPnL().dailyPnL` -£41.15; the 08:00 risk check logged
`session=£-196.86 (limit £-200.00)` — about £3 from a false DAILY LOSS HALT
that would have latched every day while the sleeve sat >£200 under cost.

The gate now uses reqPnL `dailyPnL` (realized today + today's unrealized
move); the since-cost figure is only a loudly-logged fallback while the
subscription has not delivered a value. These drive the REAL
`_compute_session_pnl` / `_check_daily_loss` on a TradingBot built via
`__new__` with a stub IB that mimics ib_insync's reqPnL/pnl contract.
"""

import logging
from types import SimpleNamespace

import pytest

from src.bot import TradingBot
from src.config import trading_config

ACCT = "U23125294"
NAN = float("nan")
UNSET_DOUBLE = 1.7976931348623157e308


class FakeIB:
    """reqPnL registers a PnL whose dailyPnL is NaN until IBKR pushes one,
    asserts on a duplicate key (as ib_insync does), and pnl() lists them."""

    def __init__(self, values, daily=None, arrive_after_sleeps=None,
                 pre_subscribed=False):
        self._values = values
        self._daily = daily
        self._arrive_after = arrive_after_sleeps
        self.subs = []
        self.req_calls = 0
        self.sleeps = 0
        if pre_subscribed:
            self._subscribe()

    def _subscribe(self):
        p = SimpleNamespace(account=ACCT, modelCode="", dailyPnL=NAN,
                            unrealizedPnL=NAN, realizedPnL=NAN)
        if self._daily is not None and self._arrive_after is None:
            p.dailyPnL = self._daily
        self.subs.append(p)
        return p

    def managedAccounts(self):
        return [ACCT]

    def accountValues(self, account):
        return self._values

    def pnl(self, account="", modelCode=""):
        return [p for p in self.subs if not account or p.account == account]

    def reqPnL(self, account, modelCode=""):
        self.req_calls += 1
        if self.subs:
            raise AssertionError  # ib_insync: key already subscribed
        return self._subscribe()

    def sleep(self, secs):
        self.sleeps += 1
        if self._arrive_after is not None and self.sleeps >= self._arrive_after:
            for p in self.subs:
                p.dailyPnL = self._daily


def av(tag, ccy, value):
    return SimpleNamespace(tag=tag, currency=ccy, value=str(value), account=ACCT)


# Real 2026-11-06 ~10:00 UTC account values (read-only probe, clientId 25).
VALUES_1106 = [
    av("RealizedPnL", "BASE", "0.00"), av("RealizedPnL", "GBP", "0.00"),
    av("RealizedPnL", "USD", "0.00"),
    av("UnrealizedPnL", "BASE", "-181.03"), av("UnrealizedPnL", "GBP", "0.00"),
    av("UnrealizedPnL", "USD", "-242.32"),
]


def _bot(ib):
    bot = TradingBot.__new__(TradingBot)
    bot.dry_run = False
    bot.notifier = None
    bot._daily_loss_halt_date = None
    bot.db = SimpleNamespace(get_daily_pnl=lambda: 0.0)
    bot.connection = SimpleNamespace(ib=ib)
    bot._get_base_currency = lambda: "GBP"
    return bot


class TestDailyPnLSource:

    def test_replay_1106_uses_todays_move_not_since_cost(self):
        ib = FakeIB(VALUES_1106, daily=-41.15, pre_subscribed=True)
        realized, unreal = _bot(ib)._compute_session_pnl()
        assert realized == 0.0
        assert unreal == pytest.approx(-41.15)

    def test_replay_1106_gate_stays_open(self, caplog):
        """Old code: session -181.03 (and -196.86 at 08:00) vs -200 cap.
        New: -41.15 — nowhere near the cap; no halt."""
        bot = _bot(FakeIB(VALUES_1106, daily=-41.15, pre_subscribed=True))
        with caplog.at_level(logging.INFO, logger="src.bot"):
            assert bot._check_daily_loss() is True
        assert bot._daily_loss_halt_date is None
        assert "session=£-41.15" in caplog.text

    def test_old_source_would_have_read_since_cost(self):
        """Control: with no reqPnL value the figure is the since-cost one —
        the number the gate used to compare against the daily cap."""
        ib = FakeIB(VALUES_1106, daily=None, pre_subscribed=True)
        _, unreal = _bot(ib)._compute_session_pnl()
        assert unreal == pytest.approx(-181.03)

    def test_realized_today_is_split_out_of_daily(self):
        """11-03 shape: CSPX stop realized -17.12 today; dailyPnL already
        includes it, so unrealized-today = daily - realized and the session
        total equals dailyPnL (not double-counted)."""
        values = [av("RealizedPnL", "BASE", "-17.12"),
                  av("UnrealizedPnL", "BASE", "-99.05")]
        ib = FakeIB(values, daily=-60.00, pre_subscribed=True)
        realized, unreal = _bot(ib)._compute_session_pnl()
        assert realized == pytest.approx(-17.12)
        assert unreal == pytest.approx(-42.88)
        assert realized + unreal == pytest.approx(-60.00)

    def test_real_daily_breach_still_halts(self):
        ib = FakeIB([av("RealizedPnL", "BASE", "0"),
                     av("UnrealizedPnL", "BASE", "+50")],
                    daily=-250.0, pre_subscribed=True)
        bot = _bot(ib)
        assert bot._check_daily_loss() is False
        assert bot.daily_loss_halted

    def test_profitable_since_cost_cannot_mask_a_bad_day(self):
        """The old source failed UNSAFE here: +£300 since cost hid a -£250 day."""
        ib = FakeIB([av("RealizedPnL", "BASE", "0"),
                     av("UnrealizedPnL", "BASE", "+300")],
                    daily=-250.0, pre_subscribed=True)
        assert _bot(ib)._check_daily_loss() is False


class TestSubscription:

    def test_first_call_subscribes_once(self):
        ib = FakeIB(VALUES_1106, daily=-41.15)
        bot = _bot(ib)
        bot._compute_session_pnl()
        bot._compute_session_pnl()
        assert ib.req_calls == 1
        assert len(ib.subs) == 1

    def test_resubscribes_after_reconnect_reset(self):
        """ib_insync's wrapper.reset() on reconnect drops reqId2PnL."""
        ib = FakeIB(VALUES_1106, daily=-41.15)
        bot = _bot(ib)
        bot._compute_session_pnl()
        ib.subs.clear()
        _, unreal = bot._compute_session_pnl()
        assert ib.req_calls == 2
        assert unreal == pytest.approx(-41.15)

    def test_gate_waits_for_first_value(self, caplog):
        """Gate path (wait=True): value lands after 2 ib.sleep polls."""
        ib = FakeIB(VALUES_1106, daily=-41.15, arrive_after_sleeps=2)
        bot = _bot(ib)
        with caplog.at_level(logging.WARNING, logger="src.bot"):
            assert bot._check_daily_loss() is True
        assert ib.sleeps == 2
        assert "SINCE COST" not in caplog.text

    def test_status_path_never_sleeps(self):
        """Telegram/status path (wait=False) must not run the event loop."""
        ib = FakeIB(VALUES_1106, daily=-41.15, arrive_after_sleeps=1)
        _, unreal = _bot(ib)._compute_session_pnl()
        assert ib.sleeps == 0
        assert unreal == pytest.approx(-181.03)  # fallback, logged

    def test_fallback_is_loud(self, caplog):
        ib = FakeIB(VALUES_1106, daily=None)
        with caplog.at_level(logging.WARNING, logger="src.bot"):
            assert _bot(ib)._check_daily_loss() is True
        assert ib.sleeps == 6
        assert "SINCE COST" in caplog.text

    @pytest.mark.parametrize("bad", [NAN, UNSET_DOUBLE, None])
    def test_unset_values_are_ignored(self, bad):
        ib = FakeIB(VALUES_1106, daily=bad, pre_subscribed=True)
        if bad is None:
            ib.subs[0].dailyPnL = None
        _, unreal = _bot(ib)._compute_session_pnl()
        assert unreal == pytest.approx(-181.03)

    def test_duplicate_request_assertion_is_tolerated(self):
        """pnl() empty for a non-blank modelCode sub but key already held."""
        ib = FakeIB(VALUES_1106, daily=-41.15)
        ib.subs.append(SimpleNamespace(account=ACCT, modelCode="M1",
                                       dailyPnL=-1.0))
        _, unreal = _bot(ib)._compute_session_pnl()
        assert unreal == pytest.approx(-181.03)  # no blank-model sub; fallback
