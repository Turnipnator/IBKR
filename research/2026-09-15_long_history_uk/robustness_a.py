#!/usr/bin/env python
"""
Post-registration robustness checks on attempt 5 (US absolute momentum), written and run AFTER its
registered result (all seven boxes passed) was known. They do not change the registered verdict.

The concern: FRED's GS10 and LTGOVTBD are monthly AVERAGE yields. A bond return built from two
monthly averages spans roughly mid-month to mid-month, so the "month t" bond return can include part
of month t−1's bond rally. Month t−1 is exactly when a stock fall trips the switch into bonds, so the
strategy could be credited with a rally that happened before it switched. Random-timing runs do not
get that benefit systematically, which would flatter box 1.

Variants, each with its own 1,000-run random-timing comparison built the same way (seeds 0–999):
  R0  registered data (should reproduce the registered run)
  R1  month-end yields from FRED DGS10 daily from 1962 onward (monthly averages before 1962)
  R2  the first month after each switch into bonds earns the T-bill return, not the bond return
  R3  R1 and R2 together
  R4  T-bills as the safe asset: no bond data at all

Run on the VPS:
  docker run --rm --user root --network host -v /root/ibkr_research/long:/long \
      ibkr_bot-trading-bot:latest sh -c "pip install -q xlrd >/dev/null 2>&1; python /long/robustness_a.py"
"""
import json
import os
import sys
import time

sys.path.insert(0, "/long")
import numpy as np
import pandas as pd

import long_history_study as L

N_NULL = int(os.getenv("N_NULL", "1000"))
CRASHES = {
    "1929-32 crash": ("1929-09-03", "1932-06-01"),
    "1937-38": ("1937-03-06", "1938-03-31"),
    "1946-49": ("1946-05-29", "1949-06-13"),
    "1968-70": ("1968-11-29", "1970-05-26"),
    "1973-74": ("1973-01-11", "1974-10-03"),
    "1987 crash": ("1987-08-25", "1987-12-04"),
    "2000-02": ("2000-03-24", "2002-10-09"),
}


def setup():
    monthly = L.french("ff_monthly.zip", daily=False)
    daily = L.french("ff_daily.zip", daily=True)
    bonds_avg = L.bond_returns()

    d10 = pd.read_csv(f"{L.RAW}/fred_DGS10.csv")
    d10.columns = ["date", "v"]
    d10["v"] = pd.to_numeric(d10["v"], errors="coerce")
    d10 = d10.dropna()
    d10["date"] = pd.to_datetime(d10["date"])
    month_end_yield = d10.groupby(d10["date"].dt.to_period("M"))["v"].last() / 100.0
    bonds_me = bonds_avg.copy()
    for cur in month_end_yield.index:
        prev = cur - 1
        if prev in month_end_yield.index and cur <= L.LAST_SIGNAL + 1:
            bonds_me[cur] = month_end_yield[prev] / 12.0 + L.par_bond_price(month_end_yield[prev], month_end_yield[cur], 10.0 - 1.0 / 12.0) - 1.0

    sig_months = pd.period_range(L.FIRST_SIGNAL, L.LAST_SIGNAL, freq="M")
    ddates = daily.index
    month_of_day = ddates.to_period("M")
    last_day = pd.Series(ddates, index=month_of_day).groupby(level=0).max()
    trade_days = [ddates[ddates.get_loc(last_day[p]) + 1] for p in sig_months]
    start = ddates.get_loc(trade_days[0])
    end = ddates.get_loc(ddates[ddates <= L.VALUE_END][-1])
    window = ddates[start: end + 1]
    wmonths = window.to_period("M")
    n_in_month = pd.Series(1, index=month_of_day).groupby(level=0).sum()

    def spread(bonds):
        return np.array([(1.0 + bonds[p]) ** (1.0 / n_in_month[p]) - 1.0 for p in wmonths])

    stocks = daily["mkt"].to_numpy()[start: end + 1]
    bills = daily["rf"].to_numpy()[start: end + 1]
    zeros = np.zeros(len(window))
    R_avg = np.vstack([stocks, spread(bonds_avg), bills, zeros])
    R_me = np.vstack([stocks, spread(bonds_me), bills, zeros])
    trade_pos = np.array([window.get_loc(d) for d in trade_days])
    jan_pos = [i for i in range(1, len(window)) if window[i].month == 1 and window[i - 1].month == 12]
    labels = [x if isinstance(x, str) else None for x in L.absmom_vec(monthly, 12).reindex(sig_months).tolist()]
    L.G.update(n=len(sig_months), n_days=len(window), trade_pos=trade_pos, jan_pos=jan_pos, R=R_avg, rf_daily=bills,
               years=(window[-1] - window[0]).days / 365.25, block_starts=[0, 240, 480, 720])
    return dict(window=window, R_avg=R_avg, R_me=R_me, labels=labels, bonds_avg=bonds_avg, bonds_me=bonds_me,
                n_me_months=int(sum(1 for p in bonds_me.index if p >= pd.Period("1962-02", "M") and p <= L.LAST_SIGNAL + 1)))


def simulate(labels, R, no_first_bond_month):
    """L.simulate with an optional rule: the first month after a switch into bonds earns T-bills."""
    days = L.G["n_days"]
    trade_pos = L.G["trade_pos"]
    codes = np.array([L.ASSETS.index(a) if a is not None else 3 for a in labels])
    k_of_day = np.searchsorted(trade_pos, np.arange(days), side="left") - 1
    kk = np.clip(k_of_day, 0, None)
    held = np.where(k_of_day >= 0, codes[kk], 3)
    rets = R[held, np.arange(days)]
    if no_first_bond_month:
        prev_code = np.r_[3, codes[:-1]]
        first_bond = (codes == 1) & (prev_code != 1)
        mask = (k_of_day >= 0) & first_bond[kk]
        rets = np.where(mask, R[2], rets)
    factor = 1.0 + rets
    prev = np.r_[3, codes[:-1]]
    orders = np.where(codes == prev, 0, np.where((prev == 3) | (codes == 3), 1, 2))
    factor[trade_pos] *= 1.0 - (L.COMMISSION + L.SLIP_BPS / 1e4) * orders
    nlv = 100.0 * np.cumprod(factor)
    out = L.metrics(nlv)
    out["blocks"] = L.blocks(nlv)
    out["curve"] = nlv
    return out


def run_variant(name, S, R, no_first_bond_month=False, bills_as_safe=False):
    base_labels = list(S["labels"])
    if bills_as_safe:
        base_labels = ["bills" if x == "bonds" else x for x in base_labels]
    L.G["R"] = R
    base = simulate(base_labels, R, no_first_bond_month)
    n = L.G["n"]
    sharpes, cagrs = [], []
    for seed in range(N_NULL):
        rng = np.random.default_rng(seed)
        rolled = list(np.roll(np.asarray(S["labels"], dtype=object), -int(rng.integers(12, n - 12 + 1))))
        if bills_as_safe:
            rolled = ["bills" if x == "bonds" else x for x in rolled]
        r = simulate(rolled, R, no_first_bond_month)
        sharpes.append(r["sharpe"])
        cagrs.append(r["cagr"])
    sharpes, cagrs = np.array(sharpes), np.array(cagrs)
    bench = L.benchmark("6040", L.SLIP_BPS)
    res = dict(variant=name, sharpe=base["sharpe"], cagr=base["cagr"], maxdd=base["maxdd"], blocks=base["blocks"],
               share_at_least_as_good=float((sharpes >= base["sharpe"]).mean()),
               null_sharpe_p50=float(np.median(sharpes)), null_sharpe_p95=float(np.percentile(sharpes, 95)),
               null_cagr_p50=float(np.median(cagrs)), bench_sharpe=bench["sharpe"], bench_cagr=bench["cagr"],
               bench_maxdd=bench["maxdd"])
    return res, base["curve"]


def main():
    t0 = time.time()
    S = setup()
    stocks_curve = 100.0 * np.cumprod(1.0 + S["R_avg"][0])
    out = dict(run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), month_end_yield_months=S["n_me_months"],
               months_in_stocks=int(sum(1 for x in S["labels"] if x == "stocks")), months=len(S["labels"]), variants=[])
    curves = {}
    for name, R, nfb, bills in (("R0 registered data", S["R_avg"], False, False),
                                ("R1 month-end yields from 1962", S["R_me"], False, False),
                                ("R2 no bond return in first month after switching", S["R_avg"], True, False),
                                ("R3 R1 + R2", S["R_me"], True, False),
                                ("R4 T-bills as the safe asset", S["R_avg"], False, True)):
        res, curve = run_variant(name, S, R, nfb, bills)
        out["variants"].append(res)
        curves[name] = curve
        print(f"{name:52s} Sharpe {res['sharpe']:.2f} CAGR {res['cagr']:+.1%} worst fall {res['maxdd']:.1%} | "
              f"{res['share_at_least_as_good']:.1%} of random runs >= (median {res['null_sharpe_p50']:.2f}, 95th {res['null_sharpe_p95']:.2f}) | "
              f"60/40 Sharpe {res['bench_sharpe']:.2f} | blocks " + ", ".join(f"{x:+.0%}" for x in res["blocks"]), flush=True)

    win = S["window"]
    crash = {}
    for label, (a, b) in CRASHES.items():
        i0, i1 = win.searchsorted(pd.Timestamp(a)), win.searchsorted(pd.Timestamp(b), side="right") - 1
        def ret(c):
            return float(c[i1] / c[i0] - 1)
        months = pd.period_range(pd.Timestamp(a).to_period("M"), pd.Timestamp(b).to_period("M"), freq="M")
        held = [S["labels"][k] for k, p in enumerate(pd.period_range(L.FIRST_SIGNAL, L.LAST_SIGNAL, freq="M")) if (p + 1) in set(months)]
        crash[label] = dict(stocks=ret(stocks_curve), absmom_registered=ret(curves["R0 registered data"]),
                            absmom_bills_safe=ret(curves["R4 T-bills as the safe asset"]),
                            months_in_bonds=sum(1 for x in held if x == "bonds"), months=len(held))
        print(f"{label:14s} stocks {crash[label]['stocks']:+.1%} | absmom {crash[label]['absmom_registered']:+.1%} | "
              f"T-bill version {crash[label]['absmom_bills_safe']:+.1%} | out of stocks {crash[label]['months_in_bonds']}/{crash[label]['months']} months")
    out["crashes"] = crash
    json.dump(out, open("/long/out_a/robustness.json", "w"), indent=1, default=str)
    print(f"months in stocks {out['months_in_stocks']}/{out['months']}; month-end-yield bond months {out['month_end_yield_months']}; runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
