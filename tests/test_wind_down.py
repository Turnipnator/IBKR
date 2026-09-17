"""
Momentum wind-down switch (owner's decision 2026-09-17, after attempt 13).

`TradingConfig.momentum_entries_enabled = False` must mean: signals are still computed and stored
(the audit trail and the Telegram rankings carry on), but no opportunity is generated — so the bot
opens no position and tops none up — and nothing is sold, because opportunities are the only thing
bot.py acts on and there is no sell loop over holdings.

These drive the real DecisionEngine.run_analysis via __new__ + SimpleNamespace mocks, per CLAUDE.md.
"""
from types import SimpleNamespace

import pandas as pd

from src.config import TradingConfig
from src.engine import DecisionEngine, EngineState


SIGNALS = {
    "CMOD": dict(tsmom=1.0, csmom=0.9, combined=0.95, price=36.0, atr=0.9, volatility=0.22,
                 reasons=["strong uptrend"]),
    "CRUD": dict(tsmom=0.8, csmom=0.7, combined=0.75, price=17.0, atr=0.5, volatility=0.25,
                 reasons=["uptrend"]),
}


def make_engine(entries_enabled, *, targets=None):
    saved, calls = [], []
    eng = DecisionEngine.__new__(DecisionEngine)
    eng.config = TradingConfig(momentum_entries_enabled=entries_enabled)
    eng.connection = SimpleNamespace(ensure_connected=lambda: True, ib=SimpleNamespace(sleep=lambda s: None))
    eng.position_manager = SimpleNamespace(
        get_portfolio_value=lambda: {"net_liquidation": 4700.0, "sizing_capital": 4700.0,
                                     "accrued_cash": 0.0},
        get_positions=lambda: [],
    )
    eng.fetcher = SimpleNamespace(
        get_historical_data=lambda symbol, duration=None, bar_size=None, **kw: pd.DataFrame(
            {"close": [10.0] * 400, "high": [10.5] * 400, "low": [9.5] * 400, "open": [10.0] * 400}
        )
    )
    eng.db = SimpleNamespace(
        save_ohlcv=lambda df, symbol: None,
        save_instrument_signal=lambda **kw: saved.append(kw["symbol"]),
        get_active_cooldowns=lambda: {},
    )
    eng.sleeve = None
    eng.state = EngineState()
    eng.reset_cash_committed = lambda: None
    eng._check_portfolio_risk = lambda net_liq: (True, "OK — no drawdown brake")
    eng._get_all_symbols = lambda: list(SIGNALS)
    eng._compute_all_signals = lambda data: SIGNALS

    def target_spy(signals, capital):
        calls.append(capital)
        return targets or {}
    eng._calculate_target_positions = target_spy
    return eng, saved, calls


def test_wind_down_records_signals_but_generates_no_orders():
    eng, saved, calls = make_engine(False)
    opportunities = eng.run_analysis()
    assert opportunities == []
    assert eng.state.opportunities == []
    assert sorted(saved) == ["CMOD", "CRUD"]      # audit trail is untouched
    assert calls == []                            # sizing never runs, so nothing can be entered or topped up


def test_switch_on_still_produces_opportunities():
    """Control: the ONLY difference is the switch — same mocks, same signals."""
    targets = {"CMOD": dict(target_shares=26, target_weight=0.2, direction="LONG", price=36.0,
                            atr=0.9, stop_price=33.3, signal_score=0.95, currency="USD", fx_to_base=0.75)}
    eng, saved, calls = make_engine(True, targets=targets)
    opportunities = eng.run_analysis()
    assert [o.symbol for o in opportunities] == ["CMOD"]
    assert opportunities[0].position_size == 26
    assert calls == [4700.0]
    assert sorted(saved) == ["CMOD", "CRUD"]


def test_wind_down_never_asks_to_sell_anything():
    """A held name that is no longer wanted must not turn into a SELL — it leaves on its stop."""
    eng, _, _ = make_engine(False)
    eng.position_manager.get_positions = lambda: [
        SimpleNamespace(symbol="CMOD", quantity=52), SimpleNamespace(symbol="CRUD", quantity=78)
    ]
    assert eng.run_analysis() == []


def test_default_config_has_entries_off():
    """The deployed default is the decision itself, not a flag someone has to remember to set."""
    assert TradingConfig().momentum_entries_enabled is False
