#!/usr/bin/env python
"""
Part B of attempts 5–7, pre-registered 2026-09-15: dual momentum for a UK investor (attempt 7) on
ISF / IWRD / IGLT, 2008–2026, in GBP.

Rules, data check, costs, window, random-timing comparison, threshold and pass marks are fixed in
COMMON.md and PREREG_7 in this folder, committed before this script produced any returns. The
account simulator is the one from ../2026-09-15_monthly_allocation/monthly_study.py, with GBP
prices and IBKR UK commission (max £3, 0.05%).

Run on the VPS:
  docker run --rm --user root --cpus 3 -e GIT_COMMIT=<hash> \
      -v /root/ibkr_research/long:/long ibkr_bot-trading-bot:latest python /long/uk_dual_momentum_study.py
"""
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

BARS = os.getenv("BARS", "/long/bars")
RAW = os.getenv("RAW", "/long/raw")
OUT = os.getenv("OUT", "/long/out_b")
N_NULL = int(os.getenv("N_NULL", "1000"))
NPROC = int(os.getenv("NPROC", "3"))
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")

START_NLV = 4710.57
END = pd.Timestamp("2026-08-31")
LAST_SIGNAL = pd.Timestamp("2026-07-31")
FEE_MIN_GBP, FEE_RATE = 3.0, 0.0005
SLIP_BPS, STRESS_BPS = 5.0, 15.0
BUFFER = 0.02
SETTLE_DAYS = 2
THRESHOLD = 0.05 / 7

TRADED = ["ISF", "IWRD", "IGLT"]
PENCE = {"ISF": 100.0, "IWRD": 100.0, "IGLT": 1.0}

G = {}


# ------------------------------------------------------------------ data
def load(kind, sym):
    df = pd.read_csv(f"{BARS}/{kind}_{sym}.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    s = df.drop_duplicates("date").set_index("date")["close"].sort_index()
    return s[s.index <= END] / PENCE[sym]


def uk_rate_monthly():
    d = pd.read_csv(f"{RAW}/fred_IR3TIB01GBM156N.csv")
    d.columns = ["date", "v"]
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    s = d.set_index(pd.PeriodIndex(pd.to_datetime(d["date"]), freq="M"))["v"].dropna()
    full = pd.period_range(s.index[0], pd.Period(LAST_SIGNAL, "M"), freq="M")
    return (s.reindex(full).ffill() / 1200.0), str(s.index[-1])


# ------------------------------------------------------------------ signals (two implementations)
def dm_vec(mc, hurdle_m, lookback):
    r = mc / mc.shift(lookback) - 1
    periods = mc.index.to_period("M")
    hl = np.log1p(hurdle_m).rolling(lookback, min_periods=lookback).sum()
    h = np.expm1(hl.reindex(periods)).to_numpy()
    valid = (r.notna().all(axis=1).to_numpy() & ~np.isnan(h)).tolist()
    equity = np.where(r["ISF"] >= r["IWRD"], "ISF", "IWRD")
    labels = np.where(r["ISF"].to_numpy() > h, equity, "IGLT").tolist()
    return pd.Series([lab if ok else None for lab, ok in zip(labels, valid)], index=mc.index, dtype=object)


def dm_loop(mc, hurdle_m, lookback):
    isf, iwrd, iglt = (mc[c].tolist() for c in TRADED)
    periods = mc.index.to_period("M")
    hmap = hurdle_m.to_dict()
    out = []
    for i in range(len(isf)):
        if i < lookback:
            out.append(None)
            continue
        vals = [isf[i], isf[i - lookback], iwrd[i], iwrd[i - lookback], iglt[i], iglt[i - lookback]]
        months = [periods[i] - j for j in range(lookback)]
        if any(math.isnan(v) for v in vals) or any(p not in hmap or math.isnan(hmap[p]) for p in months):
            out.append(None)
            continue
        g = 1.0
        for p in reversed(months):
            g *= 1.0 + hmap[p]
        h = g - 1.0
        r_isf = isf[i] / isf[i - lookback] - 1
        r_iwrd = iwrd[i] / iwrd[i - lookback] - 1
        if r_isf > h:
            out.append("ISF" if r_isf >= r_iwrd else "IWRD")
        else:
            out.append("IGLT")
    return pd.Series(out, index=mc.index, dtype=object)


# ------------------------------------------------------------------ account simulator (GBP)
def fee_gbp(notional):
    return max(FEE_MIN_GBP, FEE_RATE * notional)


def simulate(plan, mode, slip_bps, keep_curve=False):
    px, col = G["px"], G["col"]
    d0, d1 = G["first_trade"], G["end_idx"]
    slip = slip_bps / 1e4
    cash = START_NLV
    unsettled = []
    units = {a: 0 for a in TRADED}
    pending = {}
    orders = 0
    fees = 0.0
    nlv = np.empty(d1 - d0 + 1)

    def price(a, d):
        return px[col[a], d]

    for j, d in enumerate(range(d0, d1 + 1)):
        if unsettled:
            cash += sum(v for t, v in unsettled if t <= d)
            unsettled = [(t, v) for t, v in unsettled if t > d]
        if d in plan:
            w = plan[d]
            value = cash + sum(v for _, v in unsettled) + sum(units[a] * price(a, d) for a in TRADED if units[a])
            pending = {}
            sells, buys = [], []
            for a in TRADED:
                tgt = w.get(a, 0.0)
                p = price(a, d)
                cur_val = units[a] * p
                delta = tgt * value - cur_val
                if mode == "dm":
                    if tgt == 0 and units[a] > 0:
                        sells.append((a, units[a]))
                    elif tgt > 0 and units[a] == 0:
                        buys.append((a, tgt * value))
                elif mode == "bench":
                    if abs(delta) >= p:
                        if delta < 0:
                            sells.append((a, min(units[a], round(-delta / p))))
                        else:
                            buys.append((a, delta))
            for a, qty in sells:
                if qty <= 0:
                    continue
                proceeds = qty * price(a, d) * (1 - slip)
                f = fee_gbp(proceeds)
                unsettled.append((d + SETTLE_DAYS, proceeds - f))
                units[a] -= qty
                orders += 1
                fees += f
            for a, val in buys:
                pending[a] = val
        for a in list(pending):
            p = price(a, d)
            unit = p * (1 + slip)
            qty = int(min(pending[a], cash / (1 + BUFFER)) // unit)
            while qty >= 1 and qty * unit + fee_gbp(qty * unit) > cash:
                qty -= 1
            if qty >= 1:
                cost = qty * unit
                f = fee_gbp(cost)
                cash -= cost + f
                units[a] += qty
                orders += 1
                fees += f
                pending[a] -= qty * p
            if pending[a] < p:
                del pending[a]
        nlv[j] = cash + sum(v for _, v in unsettled) + sum(units[a] * price(a, d) for a in TRADED if units[a])

    out = metrics(nlv)
    out.update(orders=orders, fees=fees, fee_pct_yr=fees / float(np.mean(nlv)) / out["years"], blocks=blocks(nlv))
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
    j = day_index - 1 - G["first_trade"]
    return START_NLV if j < 0 else float(nlv[j])


def blocks(nlv):
    starts = G["block_starts"]
    res = []
    for b, k in enumerate(starts):
        v0 = value_before(nlv, G["trade_idx"][k])
        v1 = value_before(nlv, G["trade_idx"][starts[b + 1]]) if b + 1 < len(starts) else float(nlv[-1])
        res.append(v1 / v0 - 1)
    return res


def run_task(task):
    kind, strat, slip, seed, param = task
    if strat == "dm":
        labels = list(G["sig"][param])
        if seed is not None:
            rng = np.random.default_rng(seed)
            labels = list(np.roll(np.asarray(labels, dtype=object), -int(rng.integers(12, G["n"] - 12 + 1))))
        plan = {G["trade_idx"][k]: ({labels[k]: 1.0} if labels[k] else {}) for k in range(G["n"])}
        mode = "dm"
    elif strat == "bench6040":
        plan, mode = {d: {"IWRD": 0.6, "IGLT": 0.4} for d in G["bench_days"]}, "bench"
    else:
        plan, mode = {G["first_trade"]: {"ISF": 1.0}}, "bench"
    res = simulate(plan, mode, slip, keep_curve=(kind == "base" and slip == SLIP_BPS))
    res.update(kind=kind, strat=strat, slip=slip, seed=seed, param=param)
    return res


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    adj = {s: load("uk_adj", s) for s in TRADED}
    raw = {s: load("uk_raw", s) for s in TRADED}

    # data check (box 7): distributions present in the adjusted series
    ratios = {s: float(adj[s].iloc[0] / raw[s].loc[adj[s].index[0]]) for s in TRADED}
    data_check = dict(first_bar_adjusted_over_raw=ratios, passed=all(v < 0.95 for v in ratios.values()))
    hurdle, rate_last = uk_rate_monthly()
    data_check["uk_rate_last_observation"] = rate_last
    print("data check:", data_check, flush=True)

    closes = pd.DataFrame(adj).sort_index().ffill()
    cal = closes.index
    me_dates = pd.DatetimeIndex(cal.to_series().groupby([cal.year, cal.month]).max().values)
    me_dates = me_dates[me_dates <= LAST_SIGNAL]
    mc = closes.loc[me_dates]

    a, b = dm_vec(mc, hurdle, 12), dm_loop(mc, hurdle, 12)
    fid = dict(disagree=int(sum(1 for x, y in zip(a, b) if x != y)), months=int(len(mc)))
    print("fidelity:", fid, flush=True)
    if not data_check["passed"]:
        json.dump(dict(git_commit=GIT_COMMIT, withdrawn=True, reason="Part B distribution check failed",
                       data_check=data_check, fidelity=fid), open(f"{OUT}/results.json", "w"), indent=1)
        print("WITHDRAWN: Part B data check failed; attempt 7 not run.")
        return

    valid = a.notna().to_numpy()
    first_k = int(np.argmax(valid))
    months = me_dates[first_k:]
    me_pos = cal.get_indexer(months)
    trade_idx = [int(p) + 1 for p in me_pos]
    end_idx = int(cal.get_indexer([cal[cal <= END][-1]])[0])
    n = len(months)
    q = n // 4
    jan_days = [i for i in range(trade_idx[0] + 1, end_idx + 1) if cal[i].month == 1 and cal[i - 1].month == 12]
    sig = {}
    for L in (12, 11, 13):
        s = dm_vec(mc, hurdle, L).iloc[first_k:]
        sig[L] = [x if isinstance(x, str) else None for x in s.tolist()]
    assert all(sig[12]), "base signal missing inside the window"

    G.update(px=closes[TRADED].to_numpy().T, col={s: i for i, s in enumerate(TRADED)}, dates=cal, first_trade=trade_idx[0],
             end_idx=end_idx, trade_idx=trade_idx, n=n, block_starts=[0, q, 2 * q, 3 * q],
             bench_days=[trade_idx[0]] + jan_days, sig=sig)
    assert not np.isnan(G["px"][:, trade_idx[0]: end_idx + 1]).any(), "a line has no price inside the window"
    labels = sig[12]
    switches = sum(1 for x, y in zip(labels[:-1], labels[1:]) if x != y)
    mix = {s: labels.count(s) for s in TRADED}
    print(f"window: signals {months[0].date()} -> {months[-1].date()} ({n} months), trades from {cal[trade_idx[0]].date()}, "
          f"valued to {cal[end_idx].date()}; holdings by month {mix}; switches {switches}", flush=True)

    tasks = [("base", "dm", SLIP_BPS, None, 12), ("base", "dm", STRESS_BPS, None, 12),
             ("neighbour", "dm", SLIP_BPS, None, 11), ("neighbour", "dm", SLIP_BPS, None, 13),
             ("base", "bench6040", SLIP_BPS, None, None), ("base", "isf_bh", SLIP_BPS, None, None)]
    for seed in range(N_NULL):
        tasks.append(("null", "dm", SLIP_BPS, seed, 12))
        tasks.append(("null", "dm", STRESS_BPS, seed, 12))
    t1 = time.time()
    run_task(tasks[0])
    print(f"one run {time.time() - t1:.2f}s; {len(tasks)} tasks on {NPROC} processes", flush=True)
    with Pool(NPROC) as pool:
        results = pool.map(run_task, tasks, chunksize=8)
    print(f"simulations done {time.time() - t0:.0f}s", flush=True)

    def pick(kind, strat, slip, param):
        return next(r for r in results if r["kind"] == kind and r["strat"] == strat and r["slip"] == slip and r["param"] == param)

    base, stress = pick("base", "dm", SLIP_BPS, 12), pick("base", "dm", STRESS_BPS, 12)
    bench, isf = pick("base", "bench6040", SLIP_BPS, None), pick("base", "isf_bh", SLIP_BPS, None)
    nulls = np.array([r["sharpe"] for r in results if r["kind"] == "null" and r["slip"] == SLIP_BPS])
    nulls_s = np.array([r["sharpe"] for r in results if r["kind"] == "null" and r["slip"] == STRESS_BPS])
    null_cagr = np.array([r["cagr"] for r in results if r["kind"] == "null" and r["slip"] == SLIP_BPS])
    share, share_s = float((nulls >= base["sharpe"]).mean()), float((nulls_s >= stress["sharpe"]).mean())
    med = float(np.median(nulls))
    nbs = {str(p): pick("neighbour", "dm", SLIP_BPS, p)["sharpe"] for p in (11, 13)}
    pos_blocks = sum(1 for x in base["blocks"] if x > 0)
    b2 = base["sharpe"] > bench["sharpe"] or (base["maxdd"] >= 0.5 * bench["maxdd"] and base["cagr"] >= bench["cagr"] - 0.02)
    boxes = {
        "1 beats random timing": (share <= THRESHOLD, f"{share:.1%} of {len(nulls)} random runs have Sharpe >= {base['sharpe']:.2f} (need <= {THRESHOLD:.3%})"),
        "2 beats 60/40": (b2, f"Sharpe {base['sharpe']:.2f} vs {bench['sharpe']:.2f}; worst fall {base['maxdd']:.1%} vs {bench['maxdd']:.1%}; CAGR {base['cagr']:+.1%} vs {bench['cagr']:+.1%}"),
        "3 decisions and switches": (n >= 100 and switches >= 15, f"{n} monthly decisions, {switches} switches"),
        "4 sub-periods": (pos_blocks >= 3, f"{pos_blocks} of 4 positive: " + ", ".join(f"{x:+.1%}" for x in base["blocks"])),
        "5 stress slippage": (share_s <= THRESHOLD, f"{share_s:.1%} of random runs at 15 bps have Sharpe >= {stress['sharpe']:.2f}"),
        "6 neighbours": (all(v > med for v in nbs.values()), "neighbour Sharpe " + ", ".join(f"{k}: {v:.2f}" for k, v in nbs.items()) + f" vs random median {med:.2f}"),
        "7 fidelity and data": (fid["disagree"] == 0 and data_check["passed"], f"{fid['disagree']} signal disagreements; adjusted/raw ratios " + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items())),
    }
    verdict = dict(passed=all(ok for ok, _ in boxes.values()), boxes={k: dict(ok=bool(ok), detail=d) for k, (ok, d) in boxes.items()},
                   base={k: v for k, v in base.items() if k != "curve"}, stress={k: v for k, v in stress.items() if k != "curve"},
                   null_sharpe=dict(p5=float(np.percentile(nulls, 5)), p50=med, p95=float(np.percentile(nulls, 95))),
                   null_cagr=dict(p5=float(np.percentile(null_cagr, 5)), p50=float(np.median(null_cagr)), p95=float(np.percentile(null_cagr, 95))),
                   neighbours=nbs, holdings_by_month=mix, switches=switches)
    curves = pd.DataFrame({"uk_dual_momentum": base["curve"], "bench_6040": bench["curve"], "isf_buy_hold": isf["curve"]},
                          index=cal[trade_idx[0]: end_idx + 1])
    curves.resample("W-FRI").last().round(2).to_csv(f"{OUT}/curves_weekly.csv")
    out = dict(git_commit=GIT_COMMIT, run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), withdrawn=False,
               data_check=data_check, fidelity=fid, threshold=THRESHOLD, n_null=N_NULL,
               window=dict(first_signal=str(months[0].date()), last_signal=str(months[-1].date()), months=n,
                           first_trade=str(cal[trade_idx[0]].date()), valued_to=str(cal[end_idx].date())),
               benchmark_6040={k: v for k, v in bench.items() if k != "curve"}, isf_buy_hold={k: v for k, v in isf.items() if k != "curve"},
               verdict=verdict, runtime_s=time.time() - t0)
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=1, default=str)

    print(f"\n60/40 IWRD/IGLT: CAGR {bench['cagr']:+.1%} Sharpe {bench['sharpe']:.2f} worst fall {bench['maxdd']:.1%} end £{bench['end_nlv']:,.0f} | "
          f"ISF buy-and-hold: CAGR {isf['cagr']:+.1%} Sharpe {isf['sharpe']:.2f} worst fall {isf['maxdd']:.1%}")
    print(f"\n=== uk dual momentum: {'PASS' if verdict['passed'] else 'FAIL'} ===  CAGR {base['cagr']:+.1%} Sharpe {base['sharpe']:.2f} "
          f"worst fall {base['maxdd']:.1%} vol {base['vol']:.1%} end £{base['end_nlv']:,.0f} orders {base['orders']} fees {base['fee_pct_yr']:.2%}/yr "
          f"| random Sharpe p5/50/95 {verdict['null_sharpe']['p5']:.2f}/{med:.2f}/{verdict['null_sharpe']['p95']:.2f} | random CAGR median {verdict['null_cagr']['p50']:+.1%}")
    for k, box in verdict["boxes"].items():
        print(f"  [{'x' if box['ok'] else ' '}] {k}: {box['detail']}")
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
