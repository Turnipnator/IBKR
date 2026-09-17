#!/usr/bin/env python
"""
Attempt 8, pre-registered 2026-09-17: attempt 5's absolute-momentum rule applied unchanged to 20
non-US country markets, 1975–2007.

Rules, data, costs, window, random-timing comparison, threshold (0.625%) and the eight pass-mark boxes
are fixed in PREREG_8_country_absolute_momentum.md, committed before this script produced any returns.
The US T-bill hurdle and the 10-year Treasury series are imported from the attempt 5 study so they are
the identical series, as the card states.

Run on the VPS:
  docker run --rm --user root --cpus 3 -e GIT_COMMIT=<hash> \
      -v /root/ibkr_research/countries:/c -v /root/ibkr_research/long:/long \
      ibkr_bot-trading-bot:latest python /c/countries_study.py
"""
import json
import math
import os
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, "/long")
os.environ.setdefault("RAW", "/long/raw")
import long_history_study as L   # french(), bond_returns() — the attempt 5 series

RAW_C = os.getenv("RAW_C", "/c/raw")
OUT = os.getenv("OUT", "/c/out")
N_NULL = int(os.getenv("N_NULL", "1000"))
NPROC = int(os.getenv("NPROC", "3"))
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")

THRESHOLD = 0.05 / 8
COMMISSION = 0.0007
SLIP_BPS, STRESS_BPS = 5.0, 15.0
FIRST_SIGNAL = pd.Period("1975-12", "M")
LAST_SIGNAL = pd.Period("2007-11", "M")
START = 100.0

G = {}


# ------------------------------------------------------------------ data
def country_returns():
    """Section 1 of each file: Value-Weight Dollar Returns, All 4 Data Items Not Reqd, column Mkt."""
    z = zipfile.ZipFile(f"{RAW_C}/intl_countries.zip")
    out, headers = {}, {}
    for name in z.namelist():
        txt = z.read(name).decode("latin-1").splitlines()
        start = None
        for i, line in enumerate(txt):
            t = line.strip()
            if not t:
                continue
            tok = t.split()[0]
            if tok.isdigit() and len(tok) == 6:
                start = i
                break
        head = " ".join(" ".join(l.strip() for l in txt[max(0, start - 3): start]).split())
        assert "Value-Weight Dollar Returns" in head and "Not Reqd" in head, (name, head[:80])
        headers[name] = head
        rows = {}
        for line in txt[start:]:
            t = line.strip()
            if not t:
                break
            parts = t.split()
            if not (parts[0].isdigit() and len(parts[0]) == 6):
                break
            v = float(parts[1])
            rows[pd.Period(f"{parts[0][:4]}-{parts[0][4:]}", "M")] = np.nan if v <= -99.98 else v / 100.0
        out[name.replace(".Dat", "")] = pd.Series(rows).sort_index()
    return out, headers


# ------------------------------------------------------------------ signals (two implementations)
def signals_vec(ret, rf, months, lookback):
    """1 = hold the country's market, 0 = hold US 10-year Treasuries, NaN = not eligible."""
    lr = np.log1p(ret).rolling(lookback, min_periods=lookback).sum()
    lb = np.log1p(rf).rolling(lookback, min_periods=lookback).sum()
    r12, b12 = np.expm1(lr).reindex(months), np.expm1(lb).reindex(months)
    out = pd.Series(np.where(r12 > b12, 1.0, 0.0), index=months)
    return out.where(r12.notna() & b12.notna())


def signals_loop(ret, rf, months, lookback):
    rv, bv = ret.to_dict(), rf.to_dict()
    out = []
    for m in months:
        win = [m - j for j in range(lookback)]
        vals = [rv.get(p) for p in win]
        if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
            out.append(np.nan)
            continue
        g = 1.0
        for p in reversed(win):
            g *= 1.0 + rv[p]
        gb = 1.0
        for p in reversed(win):
            gb *= 1.0 + bv[p]
        out.append(1.0 if g - 1.0 > gb - 1.0 else 0.0)
    return pd.Series(out, index=months)


# ------------------------------------------------------------------ simulator (monthly, equal-weight sleeves)
def simulate(pos, slip_bps, buy_and_hold=False):
    """pos: (n_countries, n_months) of 1 = country market, 0 = bonds, NaN = not eligible.
    Returns the pooled monthly value path plus each country's own sleeve path."""
    c = COMMISSION + slip_bps / 1e4
    cr, br = G["cret"], G["bret"]        # (n_c, n_m) and (n_m,)
    n_c, n_m = cr.shape
    sleeve = np.zeros(n_c)
    per_country = np.full((n_c, n_m), np.nan)
    prev = np.full(n_c, -1.0)
    prev_elig = np.zeros(n_c, dtype=bool)
    nlv = np.empty(n_m)
    total = START
    for t in range(n_m):
        elig = ~np.isnan(pos[:, t])
        if elig.sum() == 0:
            nlv[t] = total
            continue
        if t == 0 or (elig != prev_elig).any() or G["months"][t].month == 1:
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
        per_country[elig, t] = sleeve[elig]
        prev_elig = elig
        total = sleeve.sum()
        nlv[t] = total
    return nlv, per_country


def metrics(nlv):
    r = np.r_[nlv[0] / START - 1.0, nlv[1:] / nlv[:-1] - 1.0]
    ex = r - G["rf_m"]
    years = len(r) / 12.0
    peaks = np.maximum.accumulate(np.r_[START, nlv])[1:]
    return dict(sharpe=float(ex.mean() / ex.std(ddof=1) * math.sqrt(12)),
                cagr=float((nlv[-1] / START) ** (1 / years) - 1), maxdd=float((nlv / peaks - 1).min()),
                vol=float(r.std(ddof=1) * math.sqrt(12)), end=float(nlv[-1]))


def blocks(nlv):
    q = len(nlv) // 4
    bounds = [0, q, 2 * q, 3 * q, len(nlv)]
    out = []
    for i in range(4):
        v0 = START if bounds[i] == 0 else float(nlv[bounds[i] - 1])
        out.append(float(nlv[bounds[i + 1] - 1]) / v0 - 1.0)
    return out


def country_sharpes(per_country):
    out = {}
    for i, name in enumerate(G["countries"]):
        v = per_country[i]
        ok = ~np.isnan(v)
        if ok.sum() < 24:
            continue
        vals = v[ok]
        r = vals[1:] / vals[:-1] - 1.0
        # drop the January reset jumps: they are transfers between sleeves, not returns
        r = r[np.abs(r) < 0.9]
        if len(r) < 24 or r.std(ddof=1) == 0:
            continue
        out[name] = float(r.mean() / r.std(ddof=1) * math.sqrt(12))
    return out


def run_task(task):
    kind, slip, seed, lookback = task
    base = G["sig"][lookback]
    pos = base.copy()
    if seed is not None:
        rng = np.random.default_rng(seed)
        for i in range(pos.shape[0]):
            ok = np.where(~np.isnan(pos[i]))[0]
            if len(ok) < 30:
                continue
            seq = pos[i, ok]
            off = int(rng.integers(12, max(13, len(seq) - 12) + 1)) % len(seq)
            pos[i, ok] = np.roll(seq, -off)
    nlv, per_country = simulate(pos, slip)
    res = dict(kind=kind, slip=slip, seed=seed, lookback=lookback, blocks=blocks(nlv), **metrics(nlv))
    res["country_sharpes"] = country_sharpes(per_country)
    if kind == "base" and slip == SLIP_BPS and seed is None:
        res["curve"] = nlv
    return res


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    creturns, headers = country_returns()
    monthly = L.french("ff_monthly.zip", daily=False)
    rf = monthly["rf"]
    bonds = L.bond_returns()
    months = pd.period_range(FIRST_SIGNAL + 1, LAST_SIGNAL + 1, freq="M")   # execution months
    sig_months = pd.period_range(FIRST_SIGNAL, LAST_SIGNAL, freq="M")
    countries = sorted(creturns)

    sig = {}
    fid_disagree = 0
    for lb in (12, 11, 13):
        mat = np.full((len(countries), len(months)), np.nan)
        for i, name in enumerate(countries):
            s = signals_vec(creturns[name], rf, sig_months, lb)
            mat[i] = s.to_numpy()
            if lb == 12:
                loop = signals_loop(creturns[name], rf, sig_months, lb)
                a, b = s.to_numpy(), loop.to_numpy()
                fid_disagree += int(np.sum((np.isnan(a) != np.isnan(b)) | ((~np.isnan(a)) & (a != b))))
        sig[lb] = mat

    cret = np.full((len(countries), len(months)), np.nan)
    for i, name in enumerate(countries):
        cret[i] = creturns[name].reindex(months).to_numpy()
    bret = bonds.reindex(months).to_numpy()
    rf_m = rf.reindex(months).to_numpy()

    # data checks (box 7)
    holes = {}
    for i, name in enumerate(countries):
        used = ~np.isnan(sig[12][i])
        gaps = int(np.sum(used & np.isnan(cret[i])))
        if gaps:
            holes[name] = gaps
    us_mkt = monthly["mkt"].reindex(months).to_numpy()
    pooled = np.nanmean(cret, axis=0)
    ok = ~np.isnan(pooled) & ~np.isnan(us_mkt)
    us_corr = float(np.corrcoef(pooled[ok], us_mkt[ok])[0, 1])
    data_check = dict(country_gaps=holes, pooled_vs_us_corr=us_corr, bond_months=int(np.sum(~np.isnan(bret))),
                      passed=bool(not holes and us_corr < 0.95 and not np.isnan(bret).any()))
    fid = dict(disagree=int(fid_disagree), country_months=int(np.sum(~np.isnan(sig[12]))))
    print("countries:", len(countries), countries, flush=True)
    print("data check:", json.dumps(data_check), flush=True)
    print("fidelity:", fid, flush=True)
    if not data_check["passed"]:
        json.dump(dict(git_commit=GIT_COMMIT, withdrawn=True, reason="Part 8 data check failed",
                       data_check=data_check, fidelity=fid), open(f"{OUT}/results.json", "w"), indent=1, default=str)
        print("WITHDRAWN: data check failed; attempt 8 not run.")
        return

    G.update(cret=cret, bret=np.nan_to_num(bret), rf_m=rf_m, months=months, countries=countries, sig=sig)
    eligible_first = {c: str(months[np.argmax(~np.isnan(sig[12][i]))]) for i, c in enumerate(countries)}
    print("first eligible month:", eligible_first, flush=True)

    tasks = [("base", SLIP_BPS, None, 12), ("base", STRESS_BPS, None, 12),
             ("neighbour", SLIP_BPS, None, 11), ("neighbour", SLIP_BPS, None, 13)]
    for seed in range(N_NULL):
        tasks.append(("null", SLIP_BPS, seed, 12))
        tasks.append(("null", STRESS_BPS, seed, 12))
    t1 = time.time()
    run_task(tasks[0])
    print(f"one run {time.time() - t1:.2f}s; {len(tasks)} tasks on {NPROC} processes", flush=True)
    with Pool(NPROC) as pool:
        results = pool.map(run_task, tasks, chunksize=8)
    bh_nlv, bh_pc = simulate(sig[12], SLIP_BPS, buy_and_hold=True)
    bench = dict(blocks=blocks(bh_nlv), **metrics(bh_nlv))
    print(f"simulations done {time.time() - t0:.0f}s", flush=True)

    def pick(kind, slip, lookback):
        return next(r for r in results if r["kind"] == kind and r["slip"] == slip and r["lookback"] == lookback and r["seed"] is None)

    base, stress = pick("base", SLIP_BPS, 12), pick("base", STRESS_BPS, 12)
    nulls = np.array([r["sharpe"] for r in results if r["kind"] == "null" and r["slip"] == SLIP_BPS])
    nulls_s = np.array([r["sharpe"] for r in results if r["kind"] == "null" and r["slip"] == STRESS_BPS])
    null_cagr = np.array([r["cagr"] for r in results if r["kind"] == "null" and r["slip"] == SLIP_BPS])
    share, share_s = float((nulls >= base["sharpe"]).mean()), float((nulls_s >= stress["sharpe"]).mean())
    med = float(np.median(nulls))
    nbs = {str(lb): pick("neighbour", SLIP_BPS, lb)["sharpe"] for lb in (11, 13)}

    # breadth: each country's own strategy Sharpe vs its own random-timing median
    null_country = {}
    for r in results:
        if r["kind"] != "null" or r["slip"] != SLIP_BPS:
            continue
        for k, v in r["country_sharpes"].items():
            null_country.setdefault(k, []).append(v)
    breadth = {k: dict(strategy=v, null_median=float(np.median(null_country.get(k, [np.nan]))))
               for k, v in base["country_sharpes"].items()}
    above = [k for k, v in breadth.items() if v["strategy"] > v["null_median"]]
    switches_per_country = {}
    for i, name in enumerate(countries):
        s = sig[12][i]
        ok = ~np.isnan(s)
        seq = s[ok]
        switches_per_country[name] = int(np.sum(seq[1:] != seq[:-1]))
    avg_switches = float(np.mean(list(switches_per_country.values())))
    pos_blocks = sum(1 for x in base["blocks"] if x > 0)
    b2 = base["sharpe"] > bench["sharpe"] or (base["maxdd"] >= 0.5 * bench["maxdd"] and base["cagr"] >= bench["cagr"] - 0.02)

    boxes = {
        "1 beats random timing": (share <= THRESHOLD, f"{share:.2%} of {len(nulls)} random runs have Sharpe >= {base['sharpe']:.2f} (need <= {THRESHOLD:.3%})"),
        "2 beats buy-and-hold": (b2, f"Sharpe {base['sharpe']:.2f} vs {bench['sharpe']:.2f}; worst fall {base['maxdd']:.1%} vs {bench['maxdd']:.1%}; CAGR {base['cagr']:+.1%} vs {bench['cagr']:+.1%}"),
        "3 enough decisions": (len(months) >= 100 and avg_switches >= 15, f"{len(months)} months, {avg_switches:.1f} switches per country on average"),
        "4 sub-periods": (pos_blocks >= 3, f"{pos_blocks} of 4 positive: " + ", ".join(f"{x:+.1%}" for x in base["blocks"])),
        "5 stress slippage": (share_s <= THRESHOLD, f"{share_s:.2%} of random runs at 15 bps have Sharpe >= {stress['sharpe']:.2f}"),
        "6 neighbours": (all(v > med for v in nbs.values()), "neighbour Sharpe " + ", ".join(f"{k}: {v:.2f}" for k, v in nbs.items()) + f" vs random median {med:.2f}"),
        "7 fidelity and data": (fid["disagree"] == 0 and data_check["passed"], f"{fid['disagree']} signal disagreements in {fid['country_months']} country-months; pooled vs US corr {us_corr:.2f}"),
        "8 breadth": (len(above) >= math.ceil(2 / 3 * len(breadth)), f"{len(above)} of {len(breadth)} countries above their own random-timing median (need {math.ceil(2 / 3 * len(breadth))})"),
    }
    verdict = dict(passed=all(ok for ok, _ in boxes.values()), boxes={k: dict(ok=bool(v), detail=d) for k, (v, d) in boxes.items()},
                   base={k: v for k, v in base.items() if k not in ("curve", "country_sharpes")},
                   stress={k: v for k, v in stress.items() if k not in ("curve", "country_sharpes")},
                   null_sharpe=dict(p5=float(np.percentile(nulls, 5)), p50=med, p95=float(np.percentile(nulls, 95))),
                   null_cagr=dict(p5=float(np.percentile(null_cagr, 5)), p50=float(np.median(null_cagr)), p95=float(np.percentile(null_cagr, 95))),
                   neighbours=nbs, breadth=breadth, switches_per_country=switches_per_country)
    pd.DataFrame({"strategy": base["curve"], "buy_and_hold": bh_nlv}, index=months.to_timestamp()).round(4).to_csv(f"{OUT}/curves_monthly.csv")
    out = dict(git_commit=GIT_COMMIT, run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), withdrawn=False,
               countries=countries, first_eligible=eligible_first, data_check=data_check, fidelity=fid,
               threshold=THRESHOLD, n_null=N_NULL, window=dict(first_signal=str(sig_months[0]), last_signal=str(sig_months[-1]),
               first_month=str(months[0]), last_month=str(months[-1]), months=len(months)),
               benchmark=bench, verdict=verdict, runtime_s=time.time() - t0)
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=1, default=str)

    print(f"\nbuy-and-hold benchmark: CAGR {bench['cagr']:+.1%} Sharpe {bench['sharpe']:.2f} worst fall {bench['maxdd']:.1%}")
    print(f"\n=== country absolute momentum: {'PASS' if verdict['passed'] else 'FAIL'} ===  CAGR {base['cagr']:+.1%} "
          f"Sharpe {base['sharpe']:.2f} worst fall {base['maxdd']:.1%} vol {base['vol']:.1%} | random Sharpe p5/50/95 "
          f"{verdict['null_sharpe']['p5']:.2f}/{med:.2f}/{verdict['null_sharpe']['p95']:.2f} | random CAGR median {verdict['null_cagr']['p50']:+.1%}")
    for k, box in verdict["boxes"].items():
        print(f"  [{'x' if box['ok'] else ' '}] {k}: {box['detail']}")
    print("\nper-country Sharpe (strategy vs its own random median):")
    for k in sorted(breadth):
        v = breadth[k]
        print(f"  {k:10s} {v['strategy']:+.2f} vs {v['null_median']:+.2f} {'above' if v['strategy'] > v['null_median'] else 'below'}")
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
