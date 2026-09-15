#!/usr/bin/env python
"""
Attempts 2–4, pre-registered 2026-09-15: monthly trend timing on five asset classes (GTAA-5),
dual momentum, and a volatility-targeted 60/40.

The rules, costs, window, random-timing comparison, threshold and pass marks are fixed in
COMMON.md and the three PREREG cards in this folder, committed before this script produced any
returns. This script implements those documents and nothing else.

Run on the VPS after fetch_monthly.py:
  docker run --rm --user root --cpus 3 -e GIT_COMMIT=<hash> \
      -v /root/ibkr_research/oos:/study -v /root/ibkr_research/monthly:/monthly \
      ibkr_bot-trading-bot:latest python /monthly/monthly_study.py
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

BARS = os.getenv("BARS", "/study/bars")
GBPUSD_CSV = os.getenv("GBPUSD", "/study/gbpusd.csv")
OUT = os.getenv("OUT", "/monthly/out")
N_NULL = int(os.getenv("N_NULL", "1000"))
NPROC = int(os.getenv("NPROC", "3"))
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")

START_NLV = 4710.57
END = pd.Timestamp("2026-08-31")
LAST_SIGNAL = pd.Timestamp("2026-07-31")
FEE_MIN_USD, FEE_RATE = 4.0, 0.0005
SLIP_BPS, STRESS_BPS = 5.0, 15.0
BUFFER = 0.02
SETTLE_DAYS = 2
THRESHOLD = 0.05 / 4          # attempts 2–4 share the strictest threshold in the batch

GTAA_ASSETS = ["SPY", "EFA", "IEF", "VNQ", "GSG"]
TRADED = ["SPY", "EFA", "IEF", "VNQ", "GSG", "EXUS", "AGG"]
SPLICE_CANDIDATES = {"IEF": ["IEF-ARCA"], "AGG": ["AGG-ARCA"], "VEU": ["VEU-ARCA"], "ACWX": ["ACWX-ARCA"]}

G = {}   # filled in main, inherited by forked workers


# ------------------------------------------------------------------ data
def load_close(key):
    p = f"{BARS}/proxy_adj_{key}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    s = df.drop_duplicates("date").set_index("date")["close"].sort_index()
    s = s[s.index <= END]
    return s if len(s) else None


def splice(main, alt):
    common = main.index.intersection(alt.index)
    if len(common) < 250 or alt.index[0] >= main.index[0]:
        return None, None
    d0 = common[0]
    k = main.loc[d0] / alt.loc[d0]
    joined = pd.concat([alt[alt.index < d0] * k, main[main.index >= d0]])
    both = pd.concat([main.loc[common], alt.loc[common]], axis=1, keys=["m", "a"]).pct_change().dropna()
    return joined, dict(splice_date=str(d0.date()), alt_first=str(alt.index[0].date()),
                        overlap_days=int(len(common)), ret_corr=float(both.m.corr(both.a)))


def load_asset(sym, info):
    main = load_close(sym)
    if main is None:
        for alt_key in SPLICE_CANDIDATES.get(sym, []):
            alt = load_close(alt_key)
            if alt is not None:
                info[sym] = dict(note=f"default contract missing; using {alt_key} alone")
                return alt
        return None
    best = None
    for alt_key in SPLICE_CANDIDATES.get(sym, []):
        alt = load_close(alt_key)
        if alt is None:
            continue
        joined, meta = splice(main, alt)
        if joined is not None and (best is None or meta["ret_corr"] > best[1]["ret_corr"]):
            best = (joined, dict(meta, alt=alt_key))
    info[sym] = dict(first=str(main.index[0].date()), spliced=best[1] if best else None)
    return best[0] if best else main


# ------------------------------------------------------------------ signals (two implementations each)
def gtaa_vec(mc, length):
    sma = mc.rolling(length, min_periods=length).mean()
    sig = (mc > sma).astype(float)
    return sig.where(sma.notna() & mc.notna())


def gtaa_loop(mc, length):
    out = pd.DataFrame(np.nan, index=mc.index, columns=mc.columns)
    for a in mc.columns:
        vals = mc[a].tolist()
        col = out.columns.get_loc(a)
        for i in range(len(vals)):
            if i + 1 < length:
                continue
            window = vals[i - length + 1: i + 1]
            if any(math.isnan(v) for v in window):
                continue
            out.iat[i, col] = 1.0 if vals[i] > sum(window) / length else 0.0
    return out


def dm_vec(mc, lookback):
    r = mc / mc.shift(lookback) - 1
    valid = (r[["SPY", "EXUS", "BIL"]].notna().all(axis=1) & mc["AGG"].notna()).tolist()
    equity = np.where(r["SPY"] >= r["EXUS"], "SPY", "EXUS")
    labels = np.where(r["SPY"] > r["BIL"], equity, "AGG").tolist()
    # a plain list keeps None as None (Series.where can turn it into NaN, which is truthy)
    return pd.Series([lab if ok else None for lab, ok in zip(labels, valid)], index=mc.index, dtype=object)


def dm_loop(mc, lookback):
    spy, exus, bil, agg = (mc[c].tolist() for c in ("SPY", "EXUS", "BIL", "AGG"))
    out = []
    for i in range(len(spy)):
        if i < lookback:
            out.append(None)
            continue
        vals = [spy[i], spy[i - lookback], exus[i], exus[i - lookback], bil[i], bil[i - lookback], agg[i]]
        if any(math.isnan(v) for v in vals):
            out.append(None)
            continue
        r_spy = spy[i] / spy[i - lookback] - 1
        r_exus = exus[i] / exus[i - lookback] - 1
        r_bil = bil[i] / bil[i - lookback] - 1
        if r_spy > r_bil:
            out.append("SPY" if r_spy >= r_exus else "EXUS")
        else:
            out.append("AGG")
    return pd.Series(out, index=mc.index, dtype=object)


def vol_vec(mix, me_dates, target, window):
    sigma = mix.rolling(window, min_periods=window).std() * math.sqrt(252)
    return np.minimum(1.0, target / sigma.reindex(me_dates))


def vol_loop(mix, me_positions, me_dates, target, window):
    vals = mix.tolist()
    out = []
    for p in me_positions:
        if p + 1 < window:
            out.append(float("nan"))
            continue
        w = vals[p - window + 1: p + 1]
        if any(math.isnan(v) for v in w):
            out.append(float("nan"))
            continue
        mean = sum(w) / window
        sd = math.sqrt(sum((v - mean) ** 2 for v in w) / (window - 1)) * math.sqrt(252)
        out.append(min(1.0, target / sd))
    return pd.Series(out, index=me_dates)


# ------------------------------------------------------------------ simulator
def fee_usd(notional_usd):
    return max(FEE_MIN_USD, FEE_RATE * notional_usd)


def simulate(plan, mode, slip_bps, keep_curve=False):
    """plan: {day_index: {asset: target weight}}; mode: gtaa | dm | vol | bench."""
    px, gpu = G["px"], G["gpu"]
    col = G["col"]
    d0, d1 = G["first_trade"], G["end_idx"]
    slip = slip_bps / 1e4
    cash = START_NLV
    unsettled = []
    units = {a: 0 for a in TRADED}
    pending = {}
    orders = 0
    fees = 0.0
    round_trips = 0
    nlv = np.empty(d1 - d0 + 1)

    def price(a, d):
        return px[col[a], d]

    for j, d in enumerate(range(d0, d1 + 1)):
        g = gpu[d]
        if unsettled:
            cash += sum(v for t, v in unsettled if t <= d)
            unsettled = [(t, v) for t, v in unsettled if t > d]

        if d in plan:
            w = plan[d]
            value = cash + sum(v for _, v in unsettled) + sum(units[a] * price(a, d) * g for a in TRADED if units[a])
            pending = {}
            sells, buys = [], []
            for a in TRADED:
                tgt = w.get(a, 0.0)
                p = price(a, d)
                cur_val = units[a] * p * g
                cur_w = cur_val / value if value > 0 else 0.0
                delta = tgt * value - cur_val
                if mode in ("gtaa", "dm"):
                    if tgt == 0 and units[a] > 0:
                        sells.append((a, units[a]))
                    elif tgt > 0 and units[a] == 0:
                        buys.append((a, tgt * value))
                    elif mode == "gtaa" and tgt > 0 and not (0.75 * tgt <= cur_w <= 1.25 * tgt):
                        if delta < 0:
                            sells.append((a, min(units[a], round(-delta / (p * g)))))
                        else:
                            buys.append((a, delta))
                elif mode == "vol":
                    if abs(tgt - cur_w) > 0.05:
                        if delta < 0:
                            sells.append((a, min(units[a], round(-delta / (p * g)))))
                        else:
                            buys.append((a, delta))
                elif mode == "bench":
                    if abs(delta) >= p * g:
                        if delta < 0:
                            sells.append((a, min(units[a], round(-delta / (p * g)))))
                        else:
                            buys.append((a, delta))
            for a, qty in sells:
                if qty <= 0:
                    continue
                proceeds = qty * price(a, d) * (1 - slip)
                f = fee_usd(proceeds)
                unsettled.append((d + SETTLE_DAYS, (proceeds - f) * g))
                had = units[a]
                units[a] -= qty
                orders += 1
                fees += f * g
                if had > 0 and units[a] == 0:
                    round_trips += 1
            for a, val in buys:
                pending[a] = val

        for a in list(pending):
            p = price(a, d)
            unit_gbp = p * (1 + slip) * g
            budget = min(pending[a], cash / (1 + BUFFER))
            qty = int(budget // unit_gbp)
            while qty >= 1:
                cost_usd = qty * p * (1 + slip)
                total = (cost_usd + fee_usd(cost_usd)) * g
                if total <= cash:
                    break
                qty -= 1
            if qty >= 1:
                cost_usd = qty * p * (1 + slip)
                f = fee_usd(cost_usd)
                cash -= (cost_usd + f) * g
                units[a] += qty
                orders += 1
                fees += f * g
                pending[a] -= qty * p * g
            if pending[a] < p * g:
                del pending[a]

        nlv[j] = cash + sum(v for _, v in unsettled) + sum(units[a] * price(a, d) * g for a in TRADED if units[a])

    out = metrics(nlv)
    out.update(orders=orders, fees=fees, round_trips=round_trips,
               fee_pct_yr=fees / float(np.mean(nlv)) / out["years"] if out["years"] > 0 else 0.0,
               blocks=block_returns(nlv))
    if keep_curve:
        out["curve"] = nlv
    return out


def metrics(nlv):
    dates = G["dates"][G["first_trade"]: G["end_idx"] + 1]
    r = np.r_[nlv[0] / START_NLV - 1, nlv[1:] / nlv[:-1] - 1]
    sd = r.std(ddof=1)
    years = (dates[-1] - dates[0]).days / 365.25
    peaks = np.maximum.accumulate(np.r_[START_NLV, nlv])[1:]
    return dict(sharpe=float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
                cagr=float((nlv[-1] / START_NLV) ** (1 / years) - 1), years=years,
                maxdd=float((nlv / peaks - 1).min()), vol=float(sd * math.sqrt(252)), end_nlv=float(nlv[-1]))


def value_before(nlv, day_index):
    """Account value at the close of the day before trading day `day_index` (START before the first trade)."""
    j = day_index - 1 - G["first_trade"]
    return START_NLV if j < 0 else float(nlv[j])


def block_returns(nlv):
    starts = G["block_starts"]
    rets = []
    for b, k in enumerate(starts):
        v0 = value_before(nlv, G["trade_idx"][k])
        v1 = value_before(nlv, G["trade_idx"][starts[b + 1]]) if b + 1 < len(starts) else float(nlv[-1])
        rets.append(v1 / v0 - 1)
    return rets


# ------------------------------------------------------------------ plans
def plan_gtaa(sig_in):
    return {G["trade_idx"][k]: {a: (0.2 if sig_in[a][k] == 1.0 else 0.0) for a in GTAA_ASSETS} for k in range(G["n"])}


def plan_dm(labels):
    return {G["trade_idx"][k]: ({labels[k]: 1.0} if labels[k] else {}) for k in range(G["n"])}


def plan_vol(expo):
    return {G["trade_idx"][k]: ({"SPY": 0.6 * e, "IEF": 0.4 * e} if not math.isnan(e) else {})
            for k, e in enumerate(expo)}


def run_task(task):
    kind, strat, slip_bps, seed, param = task
    rng = np.random.default_rng(seed) if seed is not None else None
    n = G["n"]
    if strat == "gtaa":
        base = G["sig"]["gtaa"][param]
        seqs = {a: base[a] for a in GTAA_ASSETS}
        if rng is not None:
            seqs = {a: np.roll(np.asarray(seqs[a], dtype=float), -int(rng.integers(12, n - 12 + 1))) for a in GTAA_ASSETS}
        plan, mode = plan_gtaa(seqs), "gtaa"
    elif strat == "dm":
        labels = list(G["sig"]["dm"][param])
        if rng is not None:
            labels = list(np.roll(np.asarray(labels, dtype=object), -int(rng.integers(12, n - 12 + 1))))
        plan, mode = plan_dm(labels), "dm"
    elif strat == "vol":
        expo = list(G["sig"]["vol"][param])
        if rng is not None:
            expo = list(np.roll(np.asarray(expo, dtype=float), -int(rng.integers(12, n - 12 + 1))))
        plan, mode = plan_vol(expo), "vol"
    elif strat == "bench6040":
        plan, mode = {d: {"SPY": 0.6, "IEF": 0.4} for d in G["bench_days"]}, "bench"
    elif strat == "ew5":
        plan, mode = {d: {a: 0.2 for a in GTAA_ASSETS} for d in G["bench_days"]}, "bench"
    res = simulate(plan, mode, slip_bps, keep_curve=(kind == "base" and slip_bps == SLIP_BPS))
    res.update(kind=kind, strat=strat, slip=slip_bps, seed=seed, param=param)
    return res


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    info = {}
    series = {s: load_asset(s, info) for s in ["SPY", "EFA", "IEF", "VNQ", "GSG", "AGG", "BIL", "VEU", "ACWX"]}
    exus = "VEU" if series["VEU"] is not None and (series["ACWX"] is None or series["VEU"].index[0] <= series["ACWX"].index[0]) else "ACWX"
    missing = [s for s in ["SPY", "EFA", "IEF", "VNQ", "GSG", "AGG", "BIL"] if series[s] is None]
    assert not missing and series[exus] is not None, f"missing inputs: {missing} exus={exus}"
    closes = pd.DataFrame({"SPY": series["SPY"], "EFA": series["EFA"], "IEF": series["IEF"], "VNQ": series["VNQ"],
                           "GSG": series["GSG"], "EXUS": series[exus], "AGG": series["AGG"], "BIL": series["BIL"]})
    closes = closes[closes.index <= END].sort_index().ffill()
    cal = closes.index
    fx = pd.read_csv(GBPUSD_CSV)
    fx["date"] = pd.to_datetime(fx["date"]).dt.normalize()
    fx = fx.drop_duplicates("date").set_index("date")["gbpusd"].sort_index()
    gpu = (1.0 / fx.reindex(cal, method="ffill").bfill()).to_numpy()

    me_dates = pd.DatetimeIndex(cal.to_series().groupby([cal.year, cal.month]).max().values)
    me_dates = me_dates[me_dates <= LAST_SIGNAL]
    me_pos = cal.get_indexer(me_dates)
    mc = closes.loc[me_dates]
    mix = 0.6 * closes["SPY"].pct_change(fill_method=None) + 0.4 * closes["IEF"].pct_change(fill_method=None)

    # base signals + fidelity (two implementations must agree on every month)
    fid = {}
    g_vec, g_loop = gtaa_vec(mc[GTAA_ASSETS], 10), gtaa_loop(mc[GTAA_ASSETS], 10)
    both = g_vec.notna() & g_loop.notna()
    fid["gtaa"] = dict(disagree=int(((g_vec != g_loop) & both).sum().sum() + (g_vec.notna() ^ g_loop.notna()).sum().sum()),
                       months=int(len(mc)))
    d_vec, d_loop = dm_vec(mc, 12), dm_loop(mc, 12)
    fid["dm"] = dict(disagree=int(sum(1 for a, b in zip(d_vec, d_loop) if a != b)), months=int(len(mc)))
    v_vec, v_loop = vol_vec(mix, me_dates, 0.10, 21), vol_loop(mix, me_pos, me_dates, 0.10, 21)
    vv = pd.concat([v_vec, v_loop], axis=1, keys=["a", "b"])
    fid["vol"] = dict(disagree=int(((vv.a - vv.b).abs() > 1e-9).sum() + (vv.a.isna() ^ vv.b.isna()).sum()), months=int(len(mc)))

    # window: first month-end with every base signal valid and a next trading day
    valid = g_vec.notna().all(axis=1) & d_vec.notna() & v_vec.notna()
    first_k = int(np.argmax(valid.to_numpy()))
    assert valid.iloc[first_k], "no valid start month"
    months = me_dates[first_k:]
    trade_idx = [int(p) + 1 for p in me_pos[first_k:]]
    end_idx = int(cal.get_indexer([cal[cal <= END][-1]])[0])
    n = len(months)
    q = n // 4
    block_starts = [0, q, 2 * q, 3 * q]
    jan_days = [i for i in range(trade_idx[0] + 1, end_idx + 1) if cal[i].month == 1 and cal[i - 1].month == 12]

    def window(s):
        return s.iloc[first_k:].reset_index(drop=True) if isinstance(s, (pd.Series, pd.DataFrame)) else s[first_k:]

    sig = {"gtaa": {}, "dm": {}, "vol": {}}
    for L in (10, 9, 11):
        s = window(gtaa_vec(mc[GTAA_ASSETS], L))
        sig["gtaa"][L] = {a: [(x if not math.isnan(x) else 0.0) for x in s[a].tolist()] for a in GTAA_ASSETS}
    for L in (12, 11, 13):
        sig["dm"][L] = [x if isinstance(x, str) else None for x in window(dm_vec(mc, L)).tolist()]
    for tgt, win in ((0.10, 21), (0.08, 21), (0.12, 21), (0.10, 10), (0.10, 42)):
        sig["vol"][(tgt, win)] = [float(x) for x in window(vol_vec(mix, me_dates, tgt, win)).tolist()]

    G.update(px=closes[TRADED].to_numpy().T, gpu=gpu, col={a: i for i, a in enumerate(TRADED)}, dates=cal,
             first_trade=trade_idx[0], end_idx=end_idx, trade_idx=trade_idx, n=n, block_starts=block_starts,
             bench_days=[trade_idx[0]] + jan_days, sig=sig)
    assert not np.isnan(G["px"][:, trade_idx[0]: end_idx + 1]).any(), "a traded asset has no price inside the window"

    # hand-check rows
    hand = {}
    for label, k in (("first", 0), ("middle", n // 2), ("last", n - 1)):
        m = months[k]
        i = me_dates.get_loc(m)
        row = {"month_end": str(m.date()), "trade_day": str(cal[trade_idx[k]].date())}
        row["gtaa"] = {a: dict(close=round(float(mc[a].iloc[i]), 4),
                               sma10=round(float(mc[a].iloc[i - 9: i + 1].mean()), 4),
                               signal="in" if sig["gtaa"][10][a][k] == 1.0 else "out") for a in GTAA_ASSETS}
        row["dm"] = dict({c: round(float(mc[c].iloc[i] / mc[c].iloc[i - 12] - 1), 4) for c in ("SPY", "EXUS", "BIL")},
                         hold=sig["dm"][12][k])
        p = me_pos[i]
        w = mix.iloc[p - 20: p + 1]
        row["vol"] = dict(sigma=round(float(w.std() * math.sqrt(252)), 4), exposure=round(sig["vol"][(0.10, 21)][k], 4))
        hand[label] = row

    print("inputs:", json.dumps(info), "| ex-US:", exus, flush=True)
    print(f"window: signals {months[0].date()} -> {months[-1].date()} ({n} months), trades from "
          f"{cal[trade_idx[0]].date()}, valued to {cal[end_idx].date()}", flush=True)
    print("fidelity:", fid, flush=True)
    print("hand-check:", json.dumps(hand, indent=1), flush=True)

    tasks = []
    for strat, base_param, neighbours in (("gtaa", 10, (9, 11)), ("dm", 12, (11, 13)),
                                          ("vol", (0.10, 21), ((0.08, 21), (0.12, 21), (0.10, 10), (0.10, 42)))):
        tasks.append(("base", strat, SLIP_BPS, None, base_param))
        tasks.append(("base", strat, STRESS_BPS, None, base_param))
        for nb in neighbours:
            tasks.append(("neighbour", strat, SLIP_BPS, None, nb))
        for seed in range(N_NULL):
            tasks.append(("null", strat, SLIP_BPS, seed, base_param))
            tasks.append(("null", strat, STRESS_BPS, seed, base_param))
    tasks.append(("base", "bench6040", SLIP_BPS, None, None))
    tasks.append(("base", "ew5", SLIP_BPS, None, None))
    t1 = time.time()
    run_task(tasks[0])
    print(f"one run {time.time() - t1:.2f}s; {len(tasks)} tasks on {NPROC} processes", flush=True)
    with Pool(NPROC) as pool:
        results = pool.map(run_task, tasks, chunksize=8)
    print(f"simulations done {time.time() - t0:.0f}s", flush=True)

    def pick(kind, strat, slip, param):
        return next(r for r in results if r["kind"] == kind and r["strat"] == strat and r["slip"] == slip and r["param"] == param)

    bench = pick("base", "bench6040", SLIP_BPS, None)
    ew5 = pick("base", "ew5", SLIP_BPS, None)
    verdicts = {}
    for strat, base_param, neighbours in (("gtaa", 10, (9, 11)), ("dm", 12, (11, 13)),
                                          ("vol", (0.10, 21), ((0.08, 21), (0.12, 21), (0.10, 10), (0.10, 42)))):
        base = pick("base", strat, SLIP_BPS, base_param)
        stress = pick("base", strat, STRESS_BPS, base_param)
        nulls = np.array([r["sharpe"] for r in results if r["kind"] == "null" and r["strat"] == strat and r["slip"] == SLIP_BPS])
        nulls_s = np.array([r["sharpe"] for r in results if r["kind"] == "null" and r["strat"] == strat and r["slip"] == STRESS_BPS])
        null_cagr = np.array([r["cagr"] for r in results if r["kind"] == "null" and r["strat"] == strat and r["slip"] == SLIP_BPS])
        share = float((nulls >= base["sharpe"]).mean())
        share_s = float((nulls_s >= stress["sharpe"]).mean())
        nbs = {str(p): pick("neighbour", strat, SLIP_BPS, p)["sharpe"] for p in neighbours}
        median_null = float(np.median(nulls))
        if strat == "gtaa":
            trades_ok, trades_note = base["round_trips"] >= 100, f"{base['round_trips']} round trips"
        elif strat == "dm":
            labels = sig["dm"][12]
            switches = sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b)
            trades_ok, trades_note = (n >= 100 and switches >= 15), f"{n} monthly decisions, {switches} switches"
        else:
            trades_ok, trades_note = base["orders"] >= 100, f"{base['orders']} orders"
        positive_blocks = sum(1 for x in base["blocks"] if x > 0)
        b2 = base["sharpe"] > bench["sharpe"] or (base["maxdd"] >= 0.5 * bench["maxdd"] and base["cagr"] >= bench["cagr"] - 0.02)
        boxes = {
            "1 beats random timing": (share <= THRESHOLD, f"{share:.1%} of {len(nulls)} random runs have Sharpe >= {base['sharpe']:.2f} (need <= {THRESHOLD:.2%})"),
            "2 beats 60/40": (b2, f"Sharpe {base['sharpe']:.2f} vs {bench['sharpe']:.2f}; worst fall {base['maxdd']:.1%} vs {bench['maxdd']:.1%}; CAGR {base['cagr']:+.1%} vs {bench['cagr']:+.1%}"),
            "3 enough decisions": (trades_ok, trades_note),
            "4 sub-periods": (positive_blocks >= 3, f"{positive_blocks} of 4 positive: " + ", ".join(f"{x:+.1%}" for x in base["blocks"])),
            "5 stress slippage": (share_s <= THRESHOLD, f"{share_s:.1%} of random runs at 15 bps have Sharpe >= {stress['sharpe']:.2f}"),
            "6 neighbours": (all(v > median_null for v in nbs.values()), "neighbour Sharpe " + ", ".join(f"{k}: {v:.2f}" for k, v in nbs.items()) + f" vs random median {median_null:.2f}"),
            "7 fidelity": (fid[strat]["disagree"] == 0, f"{fid[strat]['disagree']} disagreements in {fid[strat]['months']} months"),
        }
        extra = {}
        if strat == "dm":
            k14 = next(k for k, m in enumerate(months) if m >= pd.Timestamp("2013-12-31"))
            curve = base["curve"]
            v0 = value_before(curve, trade_idx[k14])
            yrs = (cal[end_idx] - cal[trade_idx[k14] - 1]).days / 365.25
            extra["post_2014_cagr"] = float((curve[-1] / v0) ** (1 / yrs) - 1)
        verdicts[strat] = dict(
            passed=all(ok for ok, _ in boxes.values()), boxes={k: dict(ok=bool(ok), detail=d) for k, (ok, d) in boxes.items()},
            base={k: v for k, v in base.items() if k != "curve"}, stress={k: v for k, v in stress.items() if k != "curve"},
            null_sharpe=dict(p5=float(np.percentile(nulls, 5)), p50=median_null, p95=float(np.percentile(nulls, 95))),
            null_cagr=dict(p5=float(np.percentile(null_cagr, 5)), p50=float(np.median(null_cagr)), p95=float(np.percentile(null_cagr, 95))),
            neighbours=nbs, **extra)

    curves = pd.DataFrame({"gtaa": pick("base", "gtaa", SLIP_BPS, 10)["curve"], "dual_momentum": pick("base", "dm", SLIP_BPS, 12)["curve"],
                           "vol_target": pick("base", "vol", SLIP_BPS, (0.10, 21))["curve"], "bench_6040": bench["curve"],
                           "ew5": ew5["curve"]}, index=cal[trade_idx[0]: end_idx + 1])
    curves.resample("W-FRI").last().round(2).to_csv(f"{OUT}/curves_weekly.csv")
    out = dict(git_commit=GIT_COMMIT, run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), inputs=info, exus=exus,
               window=dict(first_signal=str(months[0].date()), last_signal=str(months[-1].date()), months=n,
                           first_trade=str(cal[trade_idx[0]].date()), valued_to=str(cal[end_idx].date()),
                           blocks=[str(months[k].date()) for k in block_starts]),
               fidelity=fid, hand_check=hand, threshold=THRESHOLD, n_null=N_NULL,
               benchmark_6040={k: v for k, v in bench.items() if k != "curve"}, ew5={k: v for k, v in ew5.items() if k != "curve"},
               verdicts=verdicts, runtime_s=time.time() - t0)
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=1, default=str)

    print(f"\n60/40 benchmark: CAGR {bench['cagr']:+.1%} Sharpe {bench['sharpe']:.2f} worst fall {bench['maxdd']:.1%} "
          f"fees {bench['fee_pct_yr']:.2%}/yr | EW5: CAGR {ew5['cagr']:+.1%} Sharpe {ew5['sharpe']:.2f} worst fall {ew5['maxdd']:.1%}")
    for strat, v in verdicts.items():
        b = v["base"]
        print(f"\n=== {strat}: {'PASS' if v['passed'] else 'FAIL'} ===  CAGR {b['cagr']:+.1%} Sharpe {b['sharpe']:.2f} "
              f"worst fall {b['maxdd']:.1%} vol {b['vol']:.1%} end £{b['end_nlv']:,.0f} orders {b['orders']} fees {b['fee_pct_yr']:.2%}/yr "
              f"| random Sharpe p5/50/95 {v['null_sharpe']['p5']:.2f}/{v['null_sharpe']['p50']:.2f}/{v['null_sharpe']['p95']:.2f}"
              + (f" | post-2014 CAGR {v['post_2014_cagr']:+.1%}" if "post_2014_cagr" in v else ""))
        for k, box in v["boxes"].items():
            print(f"  [{'x' if box['ok'] else ' '}] {k}: {box['detail']}")
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
