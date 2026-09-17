#!/usr/bin/env python
"""
Post-registration robustness checks on attempt 8 (country absolute momentum), written and run AFTER
its registered result (all eight boxes passed, Sharpe 0.96, 0 of 1,000 random runs as good) was known.
They do not change the registered verdict.

Things that could manufacture a result this strong:
  R1  the safe asset: attempt 5's edge shrank when bonds were swapped for T-bills, because bonds rally
      while stocks trend down. Same swap here.
  R2  cash at 0% when out of stocks — the harshest version of the same question.
  R3  the declared execution limitation: monthly data forces a switch at the same month-end close that
      produced the signal. This variant delays every switch by a further month.
  R4  the January reset mechanic: sleeves are reset to equal weight each January, which the null shares,
      but the reset itself could matter. This variant resets only when the eligible set changes.
Each variant carries its own 1,000-run random-timing comparison built the same way (seeds 0–999).

Run on the VPS:
  docker run --rm --user root --cpus 3 -v /root/ibkr_research/countries:/c \
      -v /root/ibkr_research/long:/long ibkr_bot-trading-bot:latest python /c/robustness_8.py
"""
import json
import os
import sys
import time

sys.path.insert(0, "/c")
sys.path.insert(0, "/long")
os.environ.setdefault("RAW", "/long/raw")
import numpy as np
import pandas as pd

import countries_study as C

N_NULL = int(os.getenv("N_NULL", "1000"))
CRISES = {
    "1987 crash": ("1987-09", "1987-12"),
    "Japan bust 1990-92": ("1990-01", "1992-12"),
    "1998 LTCM": ("1998-07", "1998-10"),
    "2000-02 bear": ("2000-03", "2002-10"),
}


def setup():
    creturns, _ = C.country_returns()
    monthly = C.L.french("ff_monthly.zip", daily=False)
    rf = monthly["rf"]
    bonds = C.L.bond_returns()
    months = pd.period_range(C.FIRST_SIGNAL + 1, C.LAST_SIGNAL + 1, freq="M")
    sig_months = pd.period_range(C.FIRST_SIGNAL, C.LAST_SIGNAL, freq="M")
    countries = sorted(creturns)
    cret = np.full((len(countries), len(months)), np.nan)
    for i, name in enumerate(countries):
        cret[i] = creturns[name].reindex(months).to_numpy()
    bret = bonds.reindex(months).to_numpy()
    rf_m = rf.reindex(months).to_numpy()
    mat = np.full((len(countries), len(months)), np.nan)
    for i, name in enumerate(countries):
        mat[i] = C.signals_vec(creturns[name], rf, sig_months, 12).to_numpy()
    sig = np.where(np.isnan(cret), np.nan, mat)
    C.G.update(cret=cret, bret=np.nan_to_num(bret), rf_m=rf_m, months=months, countries=countries,
               sig={12: sig})
    return dict(sig=sig, cret=cret, bret=np.nan_to_num(bret), rf_m=rf_m, months=months, countries=countries)


def simulate_no_january(pos, slip_bps, buy_and_hold=False):
    """C.simulate, but sleeves are reset only when the eligible set changes (no January reset)."""
    c = C.COMMISSION + slip_bps / 1e4
    cr, br = C.G["cret"], C.G["bret"]
    n_c, n_m = cr.shape
    sleeve = np.zeros(n_c)
    prev = np.full(n_c, -1.0)
    prev_elig = np.zeros(n_c, dtype=bool)
    nlv = np.empty(n_m)
    total = C.START
    for t in range(n_m):
        elig = ~np.isnan(pos[:, t])
        if elig.sum() == 0:
            nlv[t] = total
            continue
        if t == 0 or (elig != prev_elig).any():
            target = np.where(elig, total / elig.sum(), 0.0)
            total -= c * np.abs(target - sleeve).sum()
            sleeve = np.where(elig, total / elig.sum(), 0.0)
            prev = np.where(elig, prev, -1.0)
        want = np.where(elig, pos[:, t] if not buy_and_hold else 1.0, -1.0)
        switched = elig & (want != prev)
        orders = np.where(switched & (prev < 0), 1.0, np.where(switched, 2.0, 0.0))
        sleeve = sleeve * (1.0 - c * orders)
        prev = np.where(elig, want, -1.0)
        r = np.where(want == 1.0, cr[:, t], br[t])
        sleeve = np.where(elig, sleeve * (1.0 + np.nan_to_num(r)), 0.0)
        prev_elig = elig
        total = sleeve.sum()
        nlv[t] = total
    return nlv, None


def shifted_signal(sig):
    """Delay every decision by one further month (tests the same-close execution limitation)."""
    out = np.full_like(sig, np.nan)
    out[:, 1:] = sig[:, :-1]
    return np.where(np.isnan(sig), np.nan, out)


def roll_null(sig, seed):
    rng = np.random.default_rng(seed)
    pos = sig.copy()
    for i in range(pos.shape[0]):
        ok = np.where(~np.isnan(pos[i]))[0]
        if len(ok) < 30:
            continue
        seq = pos[i, ok]
        off = int(rng.integers(12, max(13, len(seq) - 12) + 1)) % len(seq)
        pos[i, ok] = np.roll(seq, -off)
    return pos


def run_variant(name, S, sig, bret, sim=None):
    sim = sim or C.simulate
    C.G["bret"] = bret
    nlv, _ = sim(sig, C.SLIP_BPS)
    base = C.metrics(nlv)
    sharpes = []
    for seed in range(N_NULL):
        n, _ = sim(roll_null(sig, seed), C.SLIP_BPS)
        sharpes.append(C.metrics(n)["sharpe"])
    sharpes = np.array(sharpes)
    bh, _ = sim(sig, C.SLIP_BPS, buy_and_hold=True)
    bench = C.metrics(bh)
    res = dict(variant=name, sharpe=base["sharpe"], cagr=base["cagr"], maxdd=base["maxdd"],
               share_at_least_as_good=float((sharpes >= base["sharpe"]).mean()),
               null_sharpe_p50=float(np.median(sharpes)), null_sharpe_p95=float(np.percentile(sharpes, 95)),
               bench_sharpe=bench["sharpe"], bench_cagr=bench["cagr"], bench_maxdd=bench["maxdd"])
    print(f"{name:46s} Sharpe {res['sharpe']:.2f} CAGR {res['cagr']:+.1%} worst fall {res['maxdd']:.1%} | "
          f"{res['share_at_least_as_good']:.2%} of random runs >= (median {res['null_sharpe_p50']:.2f}) | "
          f"buy-and-hold Sharpe {res['bench_sharpe']:.2f} CAGR {res['bench_cagr']:+.1%}", flush=True)
    return res, nlv


def main():
    t0 = time.time()
    S = setup()
    sig, cret, bret, rf_m, months = S["sig"], S["cret"], S["bret"], S["rf_m"], S["months"]
    zeros = np.zeros_like(bret)
    out = dict(run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), n_null=N_NULL, variants=[])
    curves = {}
    for name, sg, br, sim in (("R0 as registered", sig, bret, None),
                              ("R1 T-bills as the safe asset", sig, rf_m, None),
                              ("R2 cash at 0% as the safe asset", sig, zeros, None),
                              ("R3 switch delayed one more month", shifted_signal(sig), bret, None),
                              ("R4 no January reset", sig, bret, simulate_no_january)):
        res, nlv = run_variant(name, S, sg, br, sim)
        out["variants"].append(res)
        curves[name] = nlv

    C.G["bret"] = bret
    in_stocks = float(np.nanmean(sig == 1.0))
    bh, _ = C.simulate(sig, C.SLIP_BPS, buy_and_hold=True)
    crisis = {}
    idx = {str(p): i for i, p in enumerate(months)}
    for label, (a, b) in CRISES.items():
        i0, i1 = idx.get(a), idx.get(b)
        if i0 is None or i1 is None:
            continue
        strat = curves["R0 as registered"]
        crisis[label] = dict(strategy=float(strat[i1] / strat[i0 - 1] - 1.0) if i0 else None,
                             buy_and_hold=float(bh[i1] / bh[i0 - 1] - 1.0) if i0 else None,
                             months_in_stocks=float(np.nanmean(sig[:, i0:i1 + 1] == 1.0)))
        print(f"{label:20s} strategy {crisis[label]['strategy']:+.1%} | buy-and-hold "
              f"{crisis[label]['buy_and_hold']:+.1%} | sleeves in stocks {crisis[label]['months_in_stocks']:.0%}")
    out["months_in_stocks_share"] = in_stocks
    out["crises"] = crisis
    json.dump(out, open("/c/out/robustness.json", "w"), indent=1, default=str)
    print(f"\nsleeve-months in stocks {in_stocks:.0%}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
