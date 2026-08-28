"""
Tests for the 2026-08-28 GBX (pence) price-unit fix.

IBKR quotes EQQQ and IJPN's LSE GBP lines in pence (ContractDetails
priceMagnifier == 100; read-only probe 2026-08-28: EQQQ close 53,603, IJPN
1,865, VEUR 43.51 with magnifier 1). The engine sized off the raw number, so
EQQQ was "a £53k share" (0 shares, silently never bought) and IJPN was dropped
from the universe on 2026-08-14 as "£1.9k/share, unaffordable".

These drive the REAL `DecisionEngine._calculate_target_positions`,
`_fx_to_base` and `_affordable_quantity` (engine via `__new__`, collaborators
stubbed) with the genuine 2026-08-28 prices/ATRs/FX/NLV.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.engine as engine_mod
from src.config import TradingConfig, currency_symbol
from src.contracts import CONTRACT_REGISTRY, GBX_PER_GBP, IBKR_CURRENCY, resolve_contract
from src.engine import DecisionEngine

FX_20260828 = {"EUR": 0.8560, "GBP": 1.0000, "USD": 0.7329}
NLV_20260828 = 4710.57

# symbol: (combined, price, atr, volatility) — prices/ATRs in the contract's
# quoted units (pence for EQQQ/IJPN), from instrument_signals 2026-08-28 and
# the IJPN probe the same day.
EQQQ = (+1.00, 53308.0, 693.45, 0.186)
IJPN = (+1.00, 1865.0, 22.0, 0.150)
CSPX = (+0.75, 834.11, 6.00, 0.108)
QUIET = {  # below signal_threshold — never targets, present so the universe is realistic
    "RTWO": (+0.10, 143.81, 0.961, 0.116), "EIMU": (+0.02, 7.47, 0.109, 0.203),
    "VEUR": (+0.10, 43.53, 0.235, 0.069), "CNYA": (+0.06, 6.05, 0.068, 0.164),
    "AIGS": (+0.10, 7.80, 0.11, 0.169), "IGLN": (+0.10, 89.08, 1.411, 0.232),
}

WATCHLIST = {
    "equity": ["CSPX", "EQQQ", "RTWO", "EIMU", "VEUR", "IJPN", "CNYA"],
    "bond": ["DTLA", "IDTM", "IBTA", "LQDE", "IHYU", "JPEA", "IDTP"],
    "commodity": ["IGLN", "ISLN", "CRUD", "NGAS", "AIGA", "AIGI", "CMOD", "COPA", "AIGS"],
    "alt": ["IDUP"],
}


def _signals(**active):
    snap = dict(QUIET)
    snap.update(active)
    return {
        sym: {"combined": c, "price": p, "atr": a, "volatility": v}
        for sym, (c, p, a, v) in snap.items()
    }


def _engine(settled_cash=1200.0):
    eng = DecisionEngine.__new__(DecisionEngine)
    cfg = TradingConfig()
    cfg.symbols = WATCHLIST
    eng.config = cfg
    eng.position_manager = SimpleNamespace(get_positions=lambda: [])
    eng.db = MagicMock()
    eng.db.get_active_cooldowns.return_value = {}
    eng.connection = SimpleNamespace(
        get_fx_rates=lambda: FX_20260828,
        get_account_summary=lambda: {
            "AvailableFunds": {"value": str(settled_cash), "currency": "GBP"},
            "NetLiquidation": {"value": str(NLV_20260828), "currency": "GBP"},
        },
    )
    return eng


class TestRegistry:

    def test_pence_quoted_lines_are_tagged_gbx(self):
        assert CONTRACT_REGISTRY["EQQQ"][0] == "GBX"
        assert CONTRACT_REGISTRY["IJPN"][0] == "GBX"
        assert CONTRACT_REGISTRY["VEUR"][0] == "GBP"      # magnifier 1 — real pounds
        assert CONTRACT_REGISTRY["CSPX"][0] == "USD"

    def test_gbx_contracts_are_built_with_ibkr_currency_gbp(self):
        for sym in ("EQQQ", "IJPN"):
            c = resolve_contract(sym)
            assert c.currency == "GBP"
            assert c.primaryExchange == "LSEETF"
            assert c.exchange == "SMART"
        assert IBKR_CURRENCY == {"GBX": "GBP"}
        assert GBX_PER_GBP == 100.0

    def test_every_watchlist_symbol_resolves(self):
        for names in WATCHLIST.values():
            for sym in names:
                assert resolve_contract(sym).currency in ("USD", "GBP")

    def test_display_symbol(self):
        assert currency_symbol("GBX") == "p"
        assert currency_symbol("GBP") == "£"


class TestFxToBase:

    def test_gbx_is_one_hundredth_of_gbp(self):
        eng = _engine()
        assert eng._fx_to_base("GBX", FX_20260828) == pytest.approx(0.01)
        assert eng._fx_to_base("GBP", FX_20260828) == 1.0
        assert eng._fx_to_base("USD", FX_20260828) == 0.7329

    def test_gbx_without_a_gbp_rate_still_converts_pence(self):
        """GBP is the base so its rate is 1.0 even if IBKR omits it."""
        eng = _engine()
        assert eng._fx_to_base("GBX", {"USD": 0.7329}) == pytest.approx(0.01)


class TestSizing:

    def test_eqqq_sizes_to_two_shares_at_the_30pct_cap(self):
        """£4,710.57 × 30% = £1,413 → 141,317p / 53,308p = 2.65 → round 3 exceeds
        the cap → int 2. Before the fix: 141,317 / 5,330,800 → 0 shares."""
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(EQQQ=EQQQ), NLV_20260828)
        assert "EQQQ" in targets
        t = targets["EQQQ"]
        assert t["target_shares"] == 2
        assert t["fx_to_base"] == pytest.approx(0.01)
        assert t["target_weight"] == pytest.approx(2 * 53308.0 * 0.01 / NLV_20260828, rel=1e-6)
        assert 0.20 < t["target_weight"] < 0.30
        # stop stays in the contract's quoted units (pence) — that is what
        # IBKR expects on the order
        assert t["stop_price"] == pytest.approx(53308.0 - 3.0 * 693.45, abs=0.01)

    def test_ijpn_is_an_18_pound_etf_not_a_1900_pound_one(self):
        """141,317p / 1,865p = 75.8 → round 76 exceeds the cap → int 75."""
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(IJPN=IJPN), NLV_20260828)
        assert targets["IJPN"]["target_shares"] == 75
        assert targets["IJPN"]["target_weight"] == pytest.approx(75 * 18.65 / NLV_20260828, rel=1e-6)

    def test_gbx_names_are_measured_in_pounds_against_the_class_cap(self):
        """EQQQ (22.6%) + IJPN (29.7%) are both equity = 52.3% of NLV in GBP.
        Under the 60% class cap that is unscaled (2 + 75 shares); measured in
        pence it would be a 5,000% "exposure" and scaled to dust."""
        eng = _engine()
        cap = eng.config.max_asset_class_pct
        targets = eng._calculate_target_positions(_signals(EQQQ=EQQQ, IJPN=IJPN), NLV_20260828)
        assert set(targets) == {"EQQQ", "IJPN"}
        gross = sum(abs(t["target_weight"]) for t in targets.values())
        assert gross <= cap + 1e-9
        assert gross == pytest.approx((2 * 533.08 + 75 * 18.65) / NLV_20260828, rel=1e-3)
        assert targets["EQQQ"]["target_shares"] == 2
        assert targets["IJPN"]["target_shares"] == 75
        for t in targets.values():
            assert t["target_weight"] == pytest.approx(
                t["target_shares"] * t["price"] * t["fx_to_base"] / NLV_20260828, rel=1e-6)

    def test_gbx_names_are_scaled_when_they_do_exceed_the_class_cap(self, monkeypatch):
        eng = _engine()
        eng.config.max_asset_class_pct = 0.40
        targets = eng._calculate_target_positions(_signals(EQQQ=EQQQ, IJPN=IJPN), NLV_20260828)
        gross = sum(abs(t["target_weight"]) for t in targets.values())
        assert 0.30 < gross <= 0.40 + 1e-9
        assert targets["EQQQ"]["target_shares"] >= 1 and targets["IJPN"]["target_shares"] >= 1

    def test_mixed_book_ranks_and_sizes_across_units(self):
        eng = _engine()
        targets = eng._calculate_target_positions(
            _signals(EQQQ=EQQQ, IJPN=IJPN, CSPX=CSPX), NLV_20260828)
        assert set(targets) == {"EQQQ", "IJPN", "CSPX"}      # 3 slots, all clear threshold
        assert targets["CSPX"]["fx_to_base"] == 0.7329
        assert targets["CSPX"]["target_shares"] >= 1

    def test_old_registry_currency_reproduces_the_bug(self, monkeypatch):
        """Regression guard: with EQQQ tagged plain GBP the engine sizes it as a
        £53,308 share and drops it as unaffordable."""
        monkeypatch.setitem(engine_mod.CONTRACT_REGISTRY, "EQQQ", ("GBP", "LSEETF"))
        eng = _engine()
        targets = eng._calculate_target_positions(_signals(EQQQ=EQQQ), NLV_20260828)
        assert "EQQQ" not in targets


class TestAffordability:

    def test_settled_cash_check_uses_pounds_not_pence(self):
        """2 EQQQ ≈ £1,066 + 6% buffer = £1,130. Settled £1,200 covers both;
        before the fix the unit cost was computed as £56,506 → 0."""
        eng = _engine(settled_cash=1200.0)
        assert eng._affordable_quantity("EQQQ", 2, 53308.0, is_new_entry=True) == 2

    def test_partial_entry_trims_correctly_in_pence(self):
        eng = _engine(settled_cash=600.0)          # covers 1 of 2 (50% ≥ 50% floor)
        assert eng._affordable_quantity("EQQQ", 2, 53308.0, is_new_entry=True) == 1

    def test_below_floor_is_skipped(self):
        eng = _engine(settled_cash=400.0)          # covers 0 of 2
        assert eng._affordable_quantity("EQQQ", 2, 53308.0, is_new_entry=True) == 0

    def test_ijpn_top_up_trim(self):
        eng = _engine(settled_cash=500.0)          # £19.77 each with buffer → 25
        assert eng._affordable_quantity("IJPN", 75, 1865.0, is_new_entry=False) == 25
