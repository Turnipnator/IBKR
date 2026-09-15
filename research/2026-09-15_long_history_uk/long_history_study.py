#!/usr/bin/env python
"""
Part A of attempts 5–7, pre-registered 2026-09-15: US absolute momentum (attempt 5) and US 10-month
trend timing (attempt 6) over 1928–2007.

Rules, data checks, costs, window, random-timing comparison, threshold and pass marks are fixed in
COMMON.md and the PREREG_5 / PREREG_6 cards in this folder, committed before this script produced
any returns. This script implements those documents and nothing else.

Run on the VPS (raw files in /root/ibkr_research/long/raw):
  docker run --rm --user root --network host --cpus 3 -e GIT_COMMIT=<hash> \
      -v /root/ibkr_research/long:/long ibkr_bot-trading-bot:latest \
      sh -c "pip install -q xlrd >/dev/null 2>&1; python /long/long_history_study.py"
"""
import json
import math
import os
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

RAW = os.getenv("RAW", "/long/raw")
OUT = os.getenv("OUT", "/long/out_a")
N_NULL = int(os.getenv("N_NULL", "1000"))
NPROC = int(os.getenv("NPROC", "3"))
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")

THRESHOLD = 0.05 / 7
COMMISSION = 0.0007
SLIP_BPS, STRESS_BPS = 5.0, 15.0
FIRST_SIGNAL = pd.Period("1927-12", "M")
LAST_SIGNAL = pd.Period("2007-11", "M")
VALUE_END = pd.Timestamp("2007-12-31")
BOND_SWITCH = pd.Period("1953-04", "M")
ASSETS = ["stocks", "bonds", "bills"]

G = {}


# ------------------------------------------------------------------ data
def french(name, daily):
    z = zipfile.ZipFile(f"{RAW}/{name}")
    txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    h = next(i for i, line in enumerate(txt) if line.replace(" ", "").startswith(",Mkt-RF"))
    width = 8 if daily else 6
    rows = []
    for line in txt[h + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != width:
            break
        rows.append((parts[0], float(parts[1]), float(parts[4])))
    df = pd.DataFrame(rows, columns=["d", "mktrf", "rf"])
    df["mkt"] = (df.mktrf + df.rf) / 100.0
    df["rf"] = df.rf / 100.0
    if daily:
        df.index = pd.to_datetime(df.d, format="%Y%m%d")
    else:
        df.index = pd.PeriodIndex(pd.to_datetime(df.d, format="%Y%m"), freq="M")
    return df[["mkt", "rf"]]


def fred(series_id):
    d = pd.read_csv(f"{RAW}/fred_{series_id}.csv")
    d.columns = ["date", "v"]
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.set_index(pd.PeriodIndex(pd.to_datetime(d["date"]), freq="M"))["v"].dropna()


def par_bond_price(coupon, yld, years):
    """Price per 1 of face value, semi-annual coupons, fractional periods allowed."""
    n = 2.0 * years
    c, r = coupon / 2.0, yld / 2.0
    if abs(r) < 1e-12:
        return c * n + 1.0
    return c * (1.0 - (1.0 + r) ** (-n)) / r + (1.0 + r) ** (-n)


def bond_returns():
    lt, g10 = fred("LTGOVTBD"), fred("GS10")
    y = pd.concat([lt[lt.index < BOND_SWITCH], g10[g10.index >= BOND_SWITCH]]).sort_index() / 100.0
    out = {}
    for prev, cur in zip(y.index[:-1], y.index[1:]):
        if cur != prev + 1:
            continue
        out[cur] = y[prev] / 12.0 + par_bond_price(y[prev], y[cur], 10.0 - 1.0 / 12.0) - 1.0
    return pd.Series(out).sort_index()


def shiller_bond_returns():
    x = pd.read_excel(f"{RAW}/shiller_ie_data.xls", sheet_name="Data", header=None, engine="xlrd")
    hrow = next(i for i in range(25) if str(x.iat[i, 0]).strip() == "Date")
    labels = []
    for c in range(x.shape[1]):
        parts = [str(x.iat[r, c]) for r in range(max(0, hrow - 4), hrow + 1) if str(x.iat[r, c]) != "nan"]
        labels.append(" ".join(" ".join(parts).split()))
    col = next(c for c, lab in enumerate(labels) if "Bond" in lab and "Real" not in lab and "Annualized" not in lab)
    rows = {}
    for r in range(hrow + 1, x.shape[0]):
        d, v = x.iat[r, 0], x.iat[r, col]
        try:
            d, v = float(d), float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(d) or math.isnan(v):
            continue
        year = int(d)
        month = int(round((d - year) * 100))
        if 1 <= month <= 12:
            rows[pd.Period(f"{year}-{month:02d}", "M")] = v
    s = pd.Series(rows).sort_index()
    if s.median() > 0.5:          # stored as gross returns
        s = s - 1.0
    return s, labels[col]


# ------------------------------------------------------------------ signals (two implementations each)
def absmom_vec(m, lookback):
    ls = np.log1p(m["mkt"]).rolling(lookback, min_periods=lookback).sum()
    lb = np.log1p(m["rf"]).rolling(lookback, min_periods=lookback).sum()
    labels = np.where(np.expm1(ls) > np.expm1(lb), "stocks", "bonds").tolist()
    valid = (ls.notna() & lb.notna()).tolist()
    # plain lists keep None as None (Series.where can turn it into NaN)
    return pd.Series([x if ok else None for x, ok in zip(labels, valid)], index=m.index, dtype=object)


def absmom_loop(m, lookback):
    mk, rf = m["mkt"].tolist(), m["rf"].tolist()
    out = []
    for i in range(len(mk)):
        if i + 1 < lookback:
            out.append(None)
            continue
        gs = gb = 1.0
        for j in range(i - lookback + 1, i + 1):
            gs *= 1.0 + mk[j]
            gb *= 1.0 + rf[j]
        out.append("stocks" if gs - 1.0 > gb - 1.0 else "bonds")
    return pd.Series(out, index=m.index, dtype=object)


def trend_vec(m, length):
    tri = (1.0 + m["mkt"]).cumprod()
    sma = tri.rolling(length, min_periods=length).mean()
    labels = np.where(tri > sma, "stocks", "bills").tolist()
    valid = sma.notna().tolist()
    return pd.Series([x if ok else None for x, ok in zip(labels, valid)], index=m.index, dtype=object)


def trend_loop(m, length):
    level, tri = 1.0, []
    for r in m["mkt"].tolist():
        level *= 1.0 + r
        tri.append(level)
    out = []
    for i in range(len(tri)):
        if i + 1 < length:
            out.append(None)
            continue
        avg = sum(tri[i - length + 1: i + 1]) / length
        out.append("stocks" if tri[i] > avg else "bills")
    return pd.Series(out, index=m.index, dtype=object)


# ------------------------------------------------------------------ simulator (vectorised index returns)
def simulate(labels, slip_bps, keep_curve=False):
    """labels: asset held after each trade day (one per signal month)."""
    c = COMMISSION + slip_bps / 1e4
    days = G["n_days"]
    trade_pos = G["trade_pos"]                         # position of each trade day within the window
    codes = np.array([ASSETS.index(a) if a is not None else 3 for a in labels])
    k_of_day = np.searchsorted(trade_pos, np.arange(days), side="left") - 1   # last trade strictly before day
    held = np.where(k_of_day >= 0, codes[np.clip(k_of_day, 0, None)], 3)
    rets = G["R"][held, np.arange(days)]
    factor = 1.0 + rets
    prev = np.r_[3, codes[:-1]]
    orders = np.where(codes == prev, 0, np.where((prev == 3) | (codes == 3), 1, 2))
    factor[trade_pos] *= 1.0 - c * orders
    nlv = 100.0 * np.cumprod(factor)
    out = metrics(nlv)
    out.update(switches=int(np.sum(codes[1:] != codes[:-1])), orders=int(orders.sum()), blocks=blocks(nlv))
    if keep_curve:
        out["curve"] = nlv
    return out


def benchmark(kind, slip_bps, keep_curve=False):
    c = COMMISSION + slip_bps / 1e4
    s_ret, b_ret = G["R"][0], G["R"][1]
    days = G["n_days"]
    jan = set(G["jan_pos"])
    first = G["trade_pos"][0]
    nlv = np.empty(days)
    s_val = b_val = 0.0
    cash = 100.0
    for d in range(days):
        s_val *= 1.0 + s_ret[d]
        b_val *= 1.0 + b_ret[d]
        total = cash + s_val + b_val
        if d == first or (kind == "6040" and d in jan):
            ws = 0.6 if kind == "6040" else 1.0
            traded = abs(s_val - ws * total) + abs(b_val - (1.0 - ws) * total)
            total -= c * traded
            s_val, b_val, cash = ws * total, (1.0 - ws) * total, 0.0
        nlv[d] = cash + s_val + b_val
    out = metrics(nlv)
    if keep_curve:
        out["curve"] = nlv
    return out


def metrics(nlv):
    r = np.r_[nlv[0] / 100.0 - 1.0, nlv[1:] / nlv[:-1] - 1.0]
    ex = r - G["rf_daily"]
    years = G["years"]
    ann = math.sqrt(len(r) / years)
    peaks = np.maximum.accumulate(np.r_[100.0, nlv])[1:]
    return dict(sharpe=float(ex.mean() / ex.std(ddof=1) * ann), cagr=float((nlv[-1] / 100.0) ** (1 / years) - 1),
                maxdd=float((nlv / peaks - 1).min()), vol=float(r.std(ddof=1) * ann), end=float(nlv[-1]))


def blocks(nlv):
    starts = G["block_starts"]
    tp = G["trade_pos"]
    res = []
    for b, k in enumerate(starts):
        v0 = 100.0 if tp[k] == 0 else float(nlv[tp[k] - 1])
        v1 = float(nlv[tp[starts[b + 1]] - 1]) if b + 1 < len(starts) else float(nlv[-1])
        res.append(v1 / v0 - 1.0)
    return res


def run_task(task):
    kind, strat, slip, seed, param = task
    labels = list(G["sig"][strat][param])
    if seed is not None:
        rng = np.random.default_rng(seed)
        labels = list(np.roll(np.asarray(labels, dtype=object), -int(rng.integers(12, G["n"] - 12 + 1))))
    res = simulate(labels, slip, keep_curve=(kind == "base" and slip == SLIP_BPS))
    res.update(kind=kind, strat=strat, slip=slip, seed=seed, param=param)
    return res


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    monthly = french("ff_monthly.zip", daily=False)
    daily = french("ff_daily.zip", daily=True)
    bonds = bond_returns()

    # data check (box 7): constructed bond returns vs Shiller, calendar years 1928–2007
    shiller, shiller_label = shiller_bond_returns()
    yrs = range(1928, 2008)

    def annual(s):
        return pd.Series({y: float(np.prod(1.0 + s[(s.index.year == y)].to_numpy()) - 1.0) for y in yrs})
    a_mine, a_shiller = annual(bonds), annual(shiller)
    months_mine = sum(1 for p in bonds.index if 1928 <= p.year <= 2007)
    months_shiller = sum(1 for p in shiller.index if 1928 <= p.year <= 2007)
    bond_corr = float(a_mine.corr(a_shiller))
    data_check = dict(shiller_column=shiller_label, bond_months_mine=months_mine, bond_months_shiller=months_shiller,
                      annual_corr=bond_corr, passed=bool(bond_corr >= 0.90))   # the card's test; month counts are reported only
    print("data check:", data_check, flush=True)

    # fidelity (box 7): two implementations agree on every month
    m = monthly.copy()
    fid = {}
    for strat, vec, loop, param in (("absmom", absmom_vec, absmom_loop, 12), ("trend", trend_vec, trend_loop, 10)):
        a, b = vec(m, param), loop(m, param)
        fid[strat] = dict(disagree=int(sum(1 for x, y in zip(a, b) if x != y)), months=int(len(m)))
    print("fidelity:", fid, flush=True)

    if not data_check["passed"]:
        out = dict(git_commit=GIT_COMMIT, withdrawn=True, reason="Part A bond data check failed", data_check=data_check, fidelity=fid)
        json.dump(out, open(f"{OUT}/results.json", "w"), indent=1, default=str)
        print("WITHDRAWN: Part A data check failed; attempts 5 and 6 not run.")
        return

    # window and daily grid
    sig_months = pd.period_range(FIRST_SIGNAL, LAST_SIGNAL, freq="M")
    n = len(sig_months)
    ddates = daily.index
    month_of_day = ddates.to_period("M")
    month_end_day = {p: ddates[month_of_day == p][-1] for p in sig_months}
    trade_days = [ddates[ddates.get_loc(month_end_day[p]) + 1] for p in sig_months]
    start = ddates.get_loc(trade_days[0])
    end = ddates.get_loc(ddates[ddates <= VALUE_END][-1])
    window = ddates[start: end + 1]
    wmonths = window.to_period("M")
    n_in_month = pd.Series(1, index=month_of_day).groupby(level=0).sum()
    bond_daily = np.array([(1.0 + bonds[p]) ** (1.0 / n_in_month[p]) - 1.0 for p in wmonths])
    R = np.vstack([daily["mkt"].to_numpy()[start: end + 1], bond_daily, daily["rf"].to_numpy()[start: end + 1],
                   np.zeros(len(window))])
    trade_pos = np.array([window.get_loc(d) for d in trade_days])
    jan_pos = [i for i in range(1, len(window)) if window[i].month == 1 and window[i - 1].month == 12]
    years = (window[-1] - window[0]).days / 365.25

    sig = {"absmom": {}, "trend": {}}
    for L in (12, 11, 13):
        s = absmom_vec(m, L).reindex(sig_months)
        sig["absmom"][L] = [x if isinstance(x, str) else None for x in s.tolist()]
    for L in (10, 9, 11):
        s = trend_vec(m, L).reindex(sig_months)
        sig["trend"][L] = [x if isinstance(x, str) else None for x in s.tolist()]
    assert all(sig["absmom"][12]) and all(sig["trend"][10]), "base signal missing inside the window"

    G.update(n=n, n_days=len(window), trade_pos=trade_pos, jan_pos=jan_pos, R=R, rf_daily=R[2], years=years,
             block_starts=[0, 240, 480, 720], sig=sig)
    print(f"window: signals {sig_months[0]} -> {sig_months[-1]} ({n}), trades from {window[0].date()}, "
          f"valued to {window[-1].date()}, {len(window)} trading days, {years:.1f} years", flush=True)

    tasks = []
    for strat, base, neighbours in (("absmom", 12, (11, 13)), ("trend", 10, (9, 11))):
        tasks.append(("base", strat, SLIP_BPS, None, base))
        tasks.append(("base", strat, STRESS_BPS, None, base))
        for nb in neighbours:
            tasks.append(("neighbour", strat, SLIP_BPS, None, nb))
        for seed in range(N_NULL):
            tasks.append(("null", strat, SLIP_BPS, seed, base))
            tasks.append(("null", strat, STRESS_BPS, seed, base))
    t1 = time.time()
    run_task(tasks[0])
    print(f"one run {time.time() - t1:.3f}s; {len(tasks)} tasks on {NPROC} processes", flush=True)
    with Pool(NPROC) as pool:
        results = pool.map(run_task, tasks, chunksize=16)
    bench = benchmark("6040", SLIP_BPS, keep_curve=True)
    stocks_bh = benchmark("stocks", SLIP_BPS, keep_curve=True)
    print(f"simulations done {time.time() - t0:.0f}s", flush=True)

    def pick(kind, strat, slip, param):
        return next(r for r in results if r["kind"] == kind and r["strat"] == strat and r["slip"] == slip and r["param"] == param)

    verdicts = {}
    for strat, base, neighbours in (("absmom", 12, (11, 13)), ("trend", 10, (9, 11))):
        b = pick("base", strat, SLIP_BPS, base)
        st = pick("base", strat, STRESS_BPS, base)
        nulls = np.array([r["sharpe"] for r in results if r["kind"] == "null" and r["strat"] == strat and r["slip"] == SLIP_BPS])
        nulls_s = np.array([r["sharpe"] for r in results if r["kind"] == "null" and r["strat"] == strat and r["slip"] == STRESS_BPS])
        null_cagr = np.array([r["cagr"] for r in results if r["kind"] == "null" and r["strat"] == strat and r["slip"] == SLIP_BPS])
        share, share_s = float((nulls >= b["sharpe"]).mean()), float((nulls_s >= st["sharpe"]).mean())
        med = float(np.median(nulls))
        nbs = {str(p): pick("neighbour", strat, SLIP_BPS, p)["sharpe"] for p in neighbours}
        pos_blocks = sum(1 for x in b["blocks"] if x > 0)
        b2 = b["sharpe"] > bench["sharpe"] or (b["maxdd"] >= 0.5 * bench["maxdd"] and b["cagr"] >= bench["cagr"] - 0.02)
        boxes = {
            "1 beats random timing": (share <= THRESHOLD, f"{share:.1%} of {len(nulls)} random runs have Sharpe >= {b['sharpe']:.2f} (need <= {THRESHOLD:.3%})"),
            "2 beats 60/40": (b2, f"Sharpe {b['sharpe']:.2f} vs {bench['sharpe']:.2f}; worst fall {b['maxdd']:.1%} vs {bench['maxdd']:.1%}; CAGR {b['cagr']:+.1%} vs {bench['cagr']:+.1%}"),
            "3 decisions and switches": (n >= 100 and b["switches"] >= 15, f"{n} monthly decisions, {b['switches']} switches"),
            "4 sub-periods": (pos_blocks >= 3, f"{pos_blocks} of 4 positive: " + ", ".join(f"{x:+.1%}" for x in b["blocks"])),
            "5 stress slippage": (share_s <= THRESHOLD, f"{share_s:.1%} of random runs at 15 bps have Sharpe >= {st['sharpe']:.2f}"),
            "6 neighbours": (all(v > med for v in nbs.values()), "neighbour Sharpe " + ", ".join(f"{k}: {v:.2f}" for k, v in nbs.items()) + f" vs random median {med:.2f}"),
            "7 fidelity and data": (fid[strat]["disagree"] == 0 and data_check["passed"], f"{fid[strat]['disagree']} signal disagreements; bond check corr {bond_corr:.3f}"),
        }
        verdicts[strat] = dict(passed=all(ok for ok, _ in boxes.values()), boxes={k: dict(ok=bool(ok), detail=d) for k, (ok, d) in boxes.items()},
                               base={k: v for k, v in b.items() if k != "curve"}, stress={k: v for k, v in st.items() if k != "curve"},
                               null_sharpe=dict(p5=float(np.percentile(nulls, 5)), p50=med, p95=float(np.percentile(nulls, 95))),
                               null_cagr=dict(p5=float(np.percentile(null_cagr, 5)), p50=float(np.median(null_cagr)), p95=float(np.percentile(null_cagr, 95))),
                               neighbours=nbs)

    curves = pd.DataFrame({"absmom": pick("base", "absmom", SLIP_BPS, 12)["curve"], "trend": pick("base", "trend", SLIP_BPS, 10)["curve"],
                           "bench_6040": bench["curve"], "stocks": stocks_bh["curve"]}, index=window)
    curves.resample("ME").last().round(4).to_csv(f"{OUT}/curves_monthly.csv")
    out = dict(git_commit=GIT_COMMIT, run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), withdrawn=False,
               data_check=data_check, fidelity=fid, threshold=THRESHOLD, n_null=N_NULL,
               window=dict(first_signal=str(sig_months[0]), last_signal=str(sig_months[-1]), months=n,
                           first_trade=str(window[0].date()), valued_to=str(window[-1].date()), years=years),
               benchmark_6040={k: v for k, v in bench.items() if k != "curve"}, stocks_buy_hold={k: v for k, v in stocks_bh.items() if k != "curve"},
               verdicts=verdicts, runtime_s=time.time() - t0)
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=1, default=str)

    print(f"\n60/40 benchmark: CAGR {bench['cagr']:+.1%} Sharpe {bench['sharpe']:.2f} worst fall {bench['maxdd']:.1%} | "
          f"stocks buy-and-hold: CAGR {stocks_bh['cagr']:+.1%} Sharpe {stocks_bh['sharpe']:.2f} worst fall {stocks_bh['maxdd']:.1%}")
    for strat, v in verdicts.items():
        b = v["base"]
        print(f"\n=== {strat}: {'PASS' if v['passed'] else 'FAIL'} ===  CAGR {b['cagr']:+.1%} Sharpe {b['sharpe']:.2f} worst fall {b['maxdd']:.1%} "
              f"vol {b['vol']:.1%} switches {b['switches']} | random Sharpe p5/50/95 {v['null_sharpe']['p5']:.2f}/"
              f"{v['null_sharpe']['p50']:.2f}/{v['null_sharpe']['p95']:.2f} | random CAGR median {v['null_cagr']['p50']:+.1%}")
        for k, box in v["boxes"].items():
            print(f"  [{'x' if box['ok'] else ' '}] {k}: {box['detail']}")
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
