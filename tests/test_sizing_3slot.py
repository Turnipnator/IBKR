"""
Tests for the 2026-08-24 sizing change: max_open_positions 5 -> 3,
max_position_pct 0.18 -> 0.30, atr_stop_multiplier 4.0 -> 3.0.

These drive the REAL `DecisionEngine._calculate_target_positions` — the
engine is built via `__new__` so no IBKR connection is needed, and its four
collaborators (config / position_manager / db / connection) are SimpleNamespace
or MagicMock stand-ins. The signal snapshot is the genuine one the bot logged at
2026-08-24 13:01:00 UTC (23 instruments, real prices/ATRs/vols), with the real
FX rates and NLV from the same rebalance, so the assertions below describe what
production would actually have done.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config import TradingConfig
from src.engine import DecisionEngine


# --- the real 2026-08-24 13:01 UTC rebalance snapshot ------------------------
# symbol: (combined, price, atr, volatility)   prices/ATRs in contract currency
SNAPSHOT_20260824 = {
    "CSPX": (+0.64, 825.61, 7.13, 0.116),
    "EQQQ": (+0.06, 52210.00, 796.70, 0.190),
    "RTWO": (+0.67, 143.53, 1.06, 0.120),
    "EIMU": (+0.09, 7.34, 0.13, 0.241),
    "VEUR": (+0.71, 43.41, 0.25, 0.067),
    "CNYA": (-0.37, 5.93, 0.07, 0.180),
    "DTLA": (-0.34, 4.47, 0.04, 0.104),
    "IDTM": (-0.57, 169.42, 0.64, 0.039),
    "IBTA": (+0.75, 5.98, 0.01, 0.011),
    "LQDE": (-0.53, 99.26, 0.60, 0.054),
    "IHYU": (-1.00, 93.51, 0.51, 0.066),
    "JPEA": (+0.78, 6.57, 0.05, 0.044),
    "IDTP": (+0.13, 257.26, 0.95, 0.031),
    "IGLN": (+0.82, 90.30, 1.51, 0.244),
    "ISLN": (+0.17, 65.98, 1.85, 0.402),
    "CRUD": (+0.85, 15.32, 0.42, 0.384),
    "NGAS": (-0.96, 4.63, 0.14, 0.395),
    "AIGA": (+0.89, 6.92, 0.08, 0.133),
    "AIGI": (+0.20, 19.89, 0.23, 0.120),
    "CMOD": (+0.93, 34.74, 0.37, 0.143),
    "COPA": (+0.96, 56.96, 0.96, 0.161),
    "AIGS": (+1.00, 7.60, 0.09, 0.163),
    "IDUP": (+0.24, 33.33, 0.38, 0.130),
}

FX_20260824 = {"EUR": 0.8560, "GBP": 1.0000, "USD": 0.7330}
CAPITAL_20260824 = 4612.72          # sizing_capital logged that rebalance
HELD_20260824 = ["CMOD", "AIGS", "RTWO", "EIMU", "AIGA", "VEUR", "CSPX"]
COOLDOWNS_20260824 = {"AIGI": "2026-08-28T14:39:59"}

WATCHLIST = {
    "equity": ["CSPX", "EQQQ", "RTWO", "EIMU", "VEUR", "CNYA"],
    "bond": ["DTLA", "IDTM", "IBTA", "LQDE", "IHYU", "JPEA", "IDTP"],
    "commodity": ["IGLN", "ISLN", "CRUD", "NGAS", "AIGA", "AIGI",
                  "CMOD", "COPA", "AIGS"],
    "alt": ["IDUP"],
}


def _signals(snapshot=None):
    snap = snapshot or SNAPSHOT_20260824
    return {
        sym: {"combined": c, "price": p, "atr": a, "volatility": v}
        for sym, (c, p, a, v) in snap.items()
    }


def _engine(config=None, held=None, cooldowns=None, fx=None):
    """Real DecisionEngine object, no IBKR. Only the collaborators the sizing path
    actually touches are stubbed."""
    eng = DecisionEngine.__new__(DecisionEngine)
    cfg = config or TradingConfig()
    cfg.symbols = WATCHLIST
    eng.config = cfg
    eng.position_manager = SimpleNamespace(
        get_positions=lambda: [
            SimpleNamespace(symbol=s, quantity=1)
            for s in (HELD_20260824 if held is None else held)
        ]
    )
    eng.db = MagicMock()
    eng.db.get_active_cooldowns.return_value = (
        COOLDOWNS_20260824 if cooldowns is None else cooldowns
    )
    eng.connection = SimpleNamespace(
        get_fx_rates=lambda: (FX_20260824 if fx is None else fx)
    )
    return eng


def _weights(targets, capital=CAPITAL_20260824):
    return {s: abs(t["target_weight"]) for s, t in targets.items()}


def _notional_base(t):
    return abs(t["target_shares"]) * t["price"] * t["fx_to_base"]


# ============================================================
# Slot count and position size
# ============================================================


class TestSlotCountAndSizing:

    def test_config_values_are_the_deployed_ones(self):
        c = TradingConfig()
        assert c.max_open_positions == 3
        assert c.max_position_pct == 0.30
        assert c.atr_stop_multiplier == 3.0
        # unchanged by this work — asserted so a future edit can't drift them
        # without a test failing
        assert c.max_asset_class_pct == 0.60   # 0.40 -> 0.60 on 2026-08-28
        assert c.min_volatility == 0.08
        assert c.risk_budget == 0.20

    def test_fills_exactly_three_slots(self):
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        assert len(targets) == 3

    def test_picks_the_top_three_tradeable_longs(self):
        """AIGS +1.00, COPA +0.96, CMOD +0.93. AIGA (+0.89) and CRUD (+0.85)
        were slots 4 and 5 under the old config and must now drop out."""
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        assert set(targets) == {"AIGS", "COPA", "CMOD"}
        assert "AIGA" not in targets
        assert "CRUD" not in targets

    def test_each_position_is_about_one_third_of_the_asset_class_cap(self):
        """All three names are commodity, so the class cap binds and each
        position lands at ~cap/3 of NLV: 13.3% under the original 40% cap,
        ~20% since the 2026-08-28 move to 60% (2 x the per-name cap)."""
        eng = _engine()
        cap = eng.config.max_asset_class_pct
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        for sym, w in _weights(targets).items():
            assert cap / 3 - 0.03 <= w <= cap / 3 + 0.005, (
                f"{sym} weight {w:.3f} not ~{cap / 3:.3f}")

    def test_gross_exposure_still_lands_on_the_asset_class_cap(self):
        """The change must not alter total exposure — only how it is divided."""
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        gross = sum(_weights(targets).values())
        cap = eng.config.max_asset_class_pct
        assert cap - 0.04 <= gross <= cap + 1e-9, f"gross {gross:.3f} not ~{cap:.0%}"

    def test_positions_are_materially_bigger_than_under_the_old_config(self):
        """The whole point of the change: fee per round-trip is $8 / position,
        so position size is the lever. Expect a ~1.6x increase."""
        old = TradingConfig()
        old.max_open_positions, old.max_position_pct = 5, 0.18
        old_t = _engine(config=old)._calculate_target_positions(
            _signals(), CAPITAL_20260824)
        new_t = _engine()._calculate_target_positions(
            _signals(), CAPITAL_20260824)

        old_med = sorted(_notional_base(t) for t in old_t.values())[len(old_t) // 2]
        new_med = sorted(_notional_base(t) for t in new_t.values())[len(new_t) // 2]
        assert len(old_t) == 5 and len(new_t) == 3
        assert new_med > old_med * 1.4, (
            f"median position only {old_med:.0f} -> {new_med:.0f}")
        # and the fee burden falls correspondingly (round-trip ~£5.86 at 0.733)
        assert (5.86 / new_med) < 0.65 * (5.86 / old_med)


# ============================================================
# Nothing gets force-sold
# ============================================================


class TestNoForcedSelling:

    def test_dropped_holdings_get_no_target_at_all(self):
        """Held names outside the new top-3 must be ABSENT from targets, not
        present with target_shares=0 — a zero target is what would generate a
        sell opportunity downstream."""
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        for sym in ("RTWO", "EIMU", "VEUR", "CSPX", "AIGA"):
            assert sym not in targets, f"{sym} present — would be sized/sold"

    def test_no_target_carries_a_zero_or_negative_share_count(self):
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        for sym, t in targets.items():
            assert t["target_shares"] > 0, f"{sym} sized {t['target_shares']}"
            assert t["direction"] == "LONG"


# ============================================================
# Stop multiplier revert
# ============================================================


class TestStopMultiplier:

    def test_stop_prices_are_three_atr_below_entry(self):
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        for sym, t in targets.items():
            expected = round(t["price"] - 3.0 * t["atr"], 2)
            assert t["stop_price"] == expected, (
                f"{sym}: stop {t['stop_price']} != 3xATR {expected}")

    def test_reverting_the_multiplier_does_not_change_position_size(self):
        """CLAUDE.local.md claims the per-position cap binds for every name, so
        the 4.0 -> 3.0 revert should move stop distance only. Verify, rather
        than trust the comment."""
        four = TradingConfig()
        four.symbols = WATCHLIST
        four.atr_stop_multiplier = 4.0
        t4 = _engine(config=four)._calculate_target_positions(
            _signals(), CAPITAL_20260824)
        t3 = _engine()._calculate_target_positions(_signals(), CAPITAL_20260824)

        assert set(t3) == set(t4)
        for sym in t3:
            assert t3[sym]["target_shares"] == t4[sym]["target_shares"], (
                f"{sym} sizing moved with the stop multiplier")
            assert t3[sym]["stop_price"] > t4[sym]["stop_price"], (
                f"{sym} 3xATR stop should sit above the 4xATR one")


# ============================================================
# Existing filters must survive the change
# ============================================================


class TestFiltersStillApply:

    def test_low_vol_names_are_still_dropped(self):
        """IBTA (1.1%), JPEA (4.4%) and VEUR (6.7%) all out-rank CMOD (+0.93)
        or sit near it, and must stay filtered by the 8% floor."""
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        for sym in ("IBTA", "JPEA", "VEUR", "IDTP", "IDTM", "LQDE", "DTLA"):
            assert sym not in targets, f"{sym} passed the vol floor"

    def test_vol_floor_boundary_is_inclusive_at_exactly_eight_percent(self):
        snap = dict(SNAPSHOT_20260824)
        snap["COPA"] = (+0.96, 56.96, 0.96, 0.0800)   # exactly at the floor
        assert "COPA" in _engine()._calculate_target_positions(
            _signals(snap), CAPITAL_20260824)

        snap["COPA"] = (+0.96, 56.96, 0.96, 0.0799)   # one basis point under
        assert "COPA" not in _engine()._calculate_target_positions(
            _signals(snap), CAPITAL_20260824)

    def test_shorts_never_take_a_slot(self):
        """IHYU -1.00 and NGAS -0.96 out-rank every long on |signal|. With only
        3 slots a leak here would cost 2 of them."""
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        assert "IHYU" not in targets and "NGAS" not in targets

    def test_cooldown_still_blocks_an_unheld_name_and_backfills(self):
        """A cooldown on COPA (slot 2, not held) must free the slot to the
        next-ranked tradeable long — AIGA +0.89, which is slot 4 normally."""
        eng = _engine(held=[], cooldowns={"COPA": "2026-09-01T00:00:00"})
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        assert "COPA" not in targets
        assert len(targets) == 3
        assert set(targets) == {"AIGS", "CMOD", "AIGA"}

    def test_cooldown_never_filters_a_held_name(self):
        """Filtering a held symbol would zero its target and force a sale."""
        eng = _engine(held=["AIGS"], cooldowns={"AIGS": "2026-09-01T00:00:00"})
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        assert "AIGS" in targets

    def test_unaffordable_name_is_skipped_and_the_slot_backfills(self):
        """A name whose single share exceeds the 30% cap (~£1,384 at £4.6k NLV)
        must not burn a slot. Synthetic: CSPX at $25,000 (~£18k). EQQQ used to
        be the example here — but its 52,210 is PENCE (£522; IBKR
        priceMagnifier=100), so since the 2026-08-28 GBX fix it is affordable
        (2 shares) and correctly takes a slot; see test_gbx_pricing.py."""
        snap = dict(SNAPSHOT_20260824)
        snap["CSPX"] = (+1.00, 25000.00, 300.00, 0.116)
        targets = _engine()._calculate_target_positions(
            _signals(snap), CAPITAL_20260824)
        assert "CSPX" not in targets
        assert len(targets) == 3

    def test_eqqq_is_affordable_once_its_pence_quote_is_understood(self):
        """The flip side of the test above: the real 2026-08-24 snapshot with
        EQQQ promoted to the top signal now yields a 2-share (~£1,044, 22.6%)
        equity target instead of a silently burned slot."""
        snap = dict(SNAPSHOT_20260824)
        snap["EQQQ"] = (+1.00, 52210.00, 796.70, 0.190)
        targets = _engine()._calculate_target_positions(
            _signals(snap), CAPITAL_20260824)
        assert targets["EQQQ"]["target_shares"] == 2
        assert targets["EQQQ"]["fx_to_base"] == pytest.approx(0.01)
        assert len(targets) == 3


# ============================================================
# Currency handling — the 30% cap must not become 30%-of-the-wrong-currency
# ============================================================


class TestCurrencyHandling:

    def test_gbp_quoted_name_is_capped_in_base_not_local(self):
        """VEUR is the GBP share class. Give it the top signal and enough vol to
        clear the floor, then check its BASE weight respects the cap — a missing
        FX conversion would size it ~1/0.733 too large."""
        snap = dict(SNAPSHOT_20260824)
        snap["VEUR"] = (+1.00, 43.41, 0.25, 0.150)
        targets = _engine()._calculate_target_positions(
            _signals(snap), CAPITAL_20260824)
        assert "VEUR" in targets
        t = targets["VEUR"]
        assert t["currency"] == "GBP"
        assert t["fx_to_base"] == 1.0
        assert abs(t["target_weight"]) <= 0.30 + 1e-9

    def test_usd_names_convert_at_the_published_rate(self):
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(), CAPITAL_20260824)
        for sym, t in targets.items():
            assert t["currency"] == "USD"
            assert t["fx_to_base"] == pytest.approx(0.7330)
            assert t["target_weight"] == pytest.approx(
                _notional_base(t) / CAPITAL_20260824, rel=1e-6)


# ============================================================
# Degenerate inputs
# ============================================================


class TestDegenerateInputs:

    def test_fewer_tradeable_signals_than_slots(self):
        snap = {"AIGS": SNAPSHOT_20260824["AIGS"],
                "IHYU": SNAPSHOT_20260824["IHYU"]}
        targets = _engine(held=[])._calculate_target_positions(
            _signals(snap), CAPITAL_20260824)
        assert set(targets) == {"AIGS"}

    def test_no_tradeable_signals_returns_empty(self):
        snap = {"IHYU": SNAPSHOT_20260824["IHYU"],
                "NGAS": SNAPSHOT_20260824["NGAS"]}
        assert _engine(held=[])._calculate_target_positions(
            _signals(snap), CAPITAL_20260824) == {}

    def test_zero_atr_name_is_skipped_not_divided_by(self):
        snap = dict(SNAPSHOT_20260824)
        snap["AIGS"] = (+1.00, 7.60, 0.0, 0.163)
        targets = _engine()._calculate_target_positions(
            _signals(snap), CAPITAL_20260824)
        assert "AIGS" not in targets
        assert len(targets) == 3          # slot backfilled
