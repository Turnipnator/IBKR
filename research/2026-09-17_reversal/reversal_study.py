#!/usr/bin/env python
"""
Attempt 11, pre-registered 2026-09-17: weekly short-term reversal on 40 large US shares, long only.

Rules, universe, costs, window, the two required comparisons, threshold (0.455%) and the eight boxes are
fixed in PREREG_11_short_term_reversal.md (commit 3856f81), committed before this script produced any
returns. This implements that card and nothing else.

    signal : five-day return to the close of the day before entry (normally Friday)
    entry  : the worst performer, bought at the next open (normally Monday)
    exit   : the open five trading days later, then repeat
    costs  : $1.00 per order (measured on this account), 5 bps slippage a side (15 bps stress)

Run on the VPS:
  docker run --rm --user root --cpus 3 -e GIT_COMMIT=<hash> \
      -v /root/ibkr_research/reversal:/rev ibkr_bot-trading-bot:latest python /rev/reversal_study.py
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd

BARS = os.getenv("BARS", "/rev/bars")
OUT = os.getenv("OUT", "/rev/out")
N_NULL = int(os.getenv("N_NULL", "1000"))
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")

THRESHOLD = 0.05 / 11
LOOKBACK = 5                    # trading days, also the holding period
COMMISSION_USD = 1.00
SLIP_BPS, STRESS_BPS = 5.0, 15.0
NOTIONAL_GBP, GBP_PER_USD = 2000.0, 0.75
MIN_ELIGIBLE = 20               # the window starts when this many names are eligible

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "XOM", "UNH", "JNJ", "V",
            "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "COST", "WMT", "BAC", "CRM", "MCD", "CSCO",
            "ACN", "LIN", "ADBE", "TMO", "ABT", "DHR", "WFC", "VZ", "TXN", "NEE", "PM", "INTC", "IBM"]

G = {}


# ------------------------------------------------------------------ data
def load():
    opens, closes, first = {}, {}, {}
    for sym in UNIVERSE:
        path = f"{BARS}/d_{sym}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop_duplicates("date").set_index("date").sort_index()
        opens[sym], closes[sym] = df["open"], df["close"]
        first[sym] = str(df.index[0].date())
    o = pd.DataFrame(opens).sort_index()
    c = pd.DataFrame(closes).sort_index()
    cal = o.index.intersection(c.index)
    return o.loc[cal], c.loc[cal], first


# ------------------------------------------------------------------ signals (two implementations)
def pick_vec(closes, t, lookback):
    """Worst five-day performer as of the close of day t-1; ties by alphabetical ticker."""
    prev, base = closes.iloc[t - 1], closes.iloc[t - 1 - lookback]
    r5 = (prev / base - 1.0).dropna()
    if r5.empty:
        return None, {}
    lowest = r5.min()
    winners = sorted(r5[r5 == lowest].index)
    return winners[0], r5.to_dict()


def pick_loop(closes, t, lookback):
    best_sym, best_r = None, None
    for sym in sorted(closes.columns):
        prev, base = closes[sym].iloc[t - 1], closes[sym].iloc[t - 1 - lookback]
        if not np.isfinite(prev) or not np.isfinite(base) or base == 0:
            continue
        r = prev / base - 1.0
        if best_r is None or r < best_r:
            best_sym, best_r = sym, r
    return best_sym


# ------------------------------------------------------------------ simulator
def entry_days(closes, lookback):
    """Every `lookback` trading days, once enough names are eligible."""
    eligible = closes.notna().sum(axis=1).to_numpy()
    start = None
    for t in range(lookback + 1, len(closes)):
        if eligible[t] >= MIN_ELIGIBLE:
            start = t
            break
    if start is None:
        return []
    return [t for t in range(start, len(closes) - lookback, lookback)]


def trade_return(sym, t, slip_bps, opens, lookback):
    buy = opens[sym].iloc[t]
    sell = opens[sym].iloc[t + lookback]
    if not np.isfinite(buy) or not np.isfinite(sell) or buy <= 0:
        return None
    slip = slip_bps / 1e4
    buy_px, sell_px = buy * (1 + slip), sell * (1 - slip)
    notional_usd = NOTIONAL_GBP / GBP_PER_USD
    qty = int(notional_usd // buy_px)
    if qty < 1:
        return None
    pnl = qty * (sell_px - buy_px) - 2.0 * COMMISSION_USD
    return pnl / notional_usd


def simulate(names_by_week, slip_bps, lookback=LOOKBACK):
    opens, closes, days = G["opens"], G["closes"], G["entry_days"]
    rets, trades, wins, gross = [], 0, 0, []
    for i, t in enumerate(days):
        sym = names_by_week[i]
        r = trade_return(sym, t, slip_bps, opens, lookback) if sym else None
        if r is None:
            rets.append(0.0)
            continue
        rets.append(r)
        trades += 1
        wins += r > 0
        gross.append(opens[sym].iloc[t + lookback] / opens[sym].iloc[t] - 1.0)
    rets = np.array(rets)
    equity = np.cumprod(1.0 + rets)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    sd = rets.std(ddof=1)
    years = len(rets) * lookback / 252.0
    return dict(sharpe=float(rets.mean() / sd * math.sqrt(252 / lookback)) if sd > 0 else 0.0,
                total_return=float(equity[-1] - 1.0),
                cagr=float(equity[-1] ** (1 / years) - 1.0) if years > 0 else 0.0,
                maxdd=float((equity / peaks - 1.0).min()), trades=trades,
                win_rate=float(wins / trades) if trades else 0.0,
                avg_net_per_trade=float(rets[rets != 0].mean()) if trades else 0.0,
                avg_gross_move=float(np.mean(gross)) if gross else 0.0,
                rets=rets, equity=equity)


def benchmark(slip_bps, frictionless=False):
    """Equal weight across eligible names, rebalanced on the first entry day of each calendar year."""
    opens, closes, days = G["opens"], G["closes"], G["entry_days"]
    slip = 0.0 if frictionless else slip_bps / 1e4
    comm = 0.0 if frictionless else COMMISSION_USD
    notional_usd = NOTIONAL_GBP / GBP_PER_USD
    value = notional_usd
    held = {}
    rets = []
    year = None
    for t in days:
        date = closes.index[t]
        px = opens.iloc[t]
        if held:
            new_value = sum(q * px[s] for s, q in held.items() if np.isfinite(px[s]))
            rets.append(new_value / value - 1.0)
            value = new_value
        else:
            rets.append(0.0)
        if date.year != year:
            year = date.year
            names = [s for s in closes.columns if np.isfinite(px[s]) and np.isfinite(closes[s].iloc[t])]
            if names:
                per = value / len(names)
                cost = len(names) * comm + value * slip
                value = max(0.0, value - cost)
                per = value / len(names)
                held = {s: per / (px[s] * (1 + slip)) for s in names}
    rets = np.array(rets)
    equity = np.cumprod(1.0 + rets)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    sd = rets.std(ddof=1)
    years = len(rets) * LOOKBACK / 252.0
    return dict(sharpe=float(rets.mean() / sd * math.sqrt(252 / LOOKBACK)) if sd > 0 else 0.0,
                total_return=float(equity[-1] - 1.0),
                cagr=float(equity[-1] ** (1 / years) - 1.0) if years > 0 else 0.0,
                maxdd=float((equity / peaks - 1.0).min()))


def blocks(rets):
    q = len(rets) // 4
    return [float(np.prod(1.0 + (rets[i * q:(i + 1) * q] if i < 3 else rets[3 * q:])) - 1.0) for i in range(4)]


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    opens, closes, first = load()
    G.update(opens=opens, closes=closes)
    missing = [s for s in UNIVERSE if s not in closes.columns]
    G["entry_days"] = entry_days(closes, LOOKBACK)
    days = G["entry_days"]
    assert days, "no entry days"

    picks, fid_disagree, r5_by_week = [], 0, []
    for t in days:
        sym, r5 = pick_vec(closes, t, LOOKBACK)
        picks.append(sym)
        r5_by_week.append(r5.get(sym) if sym else None)
        if sym != pick_loop(closes, t, LOOKBACK):
            fid_disagree += 1
    eligible_counts = [int(closes.iloc[t - 1].notna().sum()) for t in days]
    gap = float(np.nanmean([opens[p].iloc[t] / closes[p].iloc[t - 1] - 1.0 for p, t in zip(picks, days) if p]))
    fid = dict(disagree=int(fid_disagree), weeks=len(days), mean_entry_gap_bps=gap * 1e4,
               min_eligible=min(eligible_counts), max_eligible=max(eligible_counts))
    print(f"universe loaded: {len(closes.columns)}/{len(UNIVERSE)} names, missing {missing}", flush=True)
    print(f"window {closes.index[days[0]].date()} -> {closes.index[days[-1] + LOOKBACK].date()}, {len(days)} weekly trades", flush=True)
    print("fidelity:", fid, flush=True)
    if fid_disagree:
        json.dump(dict(git_commit=GIT_COMMIT, withdrawn=True, reason="signal implementations disagree", fidelity=fid),
                  open(f"{OUT}/results.json", "w"), indent=1, default=str)
        print("WITHDRAWN: signal implementations disagree")
        return

    base = simulate(picks, SLIP_BPS)
    stress = simulate(picks, STRESS_BPS)
    bench = benchmark(SLIP_BPS)
    bench_free = benchmark(SLIP_BPS, frictionless=True)

    # null 1: a random eligible name each week
    rng_names = []
    for t in days:
        avail = [s for s in closes.columns
                 if np.isfinite(closes[s].iloc[t - 1]) and np.isfinite(opens[s].iloc[t])
                 and np.isfinite(opens[s].iloc[t + LOOKBACK])]
        rng_names.append(avail)
    name_sharpes = np.empty(N_NULL)
    for seed in range(N_NULL):
        rng = np.random.default_rng(seed)
        choice = [avail[int(rng.integers(0, len(avail)))] if avail else None for avail in rng_names]
        name_sharpes[seed] = simulate(choice, SLIP_BPS)["sharpe"]
    share_names = float((name_sharpes >= base["sharpe"]).mean())

    name_sharpes_stress = np.empty(N_NULL)
    for seed in range(N_NULL):
        rng = np.random.default_rng(seed)
        choice = [avail[int(rng.integers(0, len(avail)))] if avail else None for avail in rng_names]
        name_sharpes_stress[seed] = simulate(choice, STRESS_BPS)["sharpe"]
    share_names_stress = float((name_sharpes_stress >= stress["sharpe"]).mean())

    # null 2: the same picks, shifted in time
    n = len(picks)
    lo, hi = 8, max(9, n - 8)
    time_sharpes = np.empty(N_NULL)
    for seed in range(N_NULL):
        rng = np.random.default_rng(10_000 + seed)
        rolled = list(np.roll(np.array(picks, dtype=object), -int(rng.integers(lo, hi + 1))))
        time_sharpes[seed] = simulate(rolled, SLIP_BPS)["sharpe"]
    share_time = float((time_sharpes >= base["sharpe"]).mean())
    med_name = float(np.median(name_sharpes))

    neighbours = {}
    for lb in (4, 10):
        G["entry_days"] = entry_days(closes, lb)
        nb_picks = [pick_vec(closes, t, lb)[0] for t in G["entry_days"]]
        neighbours[f"{lb}d"] = simulate(nb_picks, SLIP_BPS, lookback=lb)["sharpe"]
    G["entry_days"] = days

    pos_blocks = sum(1 for x in blocks(base["rets"]) if x > 0)
    b3 = base["sharpe"] > bench["sharpe"] or (base["maxdd"] >= 0.5 * bench["maxdd"] and base["cagr"] >= bench["cagr"] - 0.02)
    boxes = {
        "1 beats random names": (share_names <= THRESHOLD, f"{share_names:.2%} of {N_NULL} random-name runs have Sharpe >= {base['sharpe']:.2f} (need <= {THRESHOLD:.3%})"),
        "2 beats random timing": (share_time <= THRESHOLD, f"{share_time:.2%} of {N_NULL} shifted runs have Sharpe >= {base['sharpe']:.2f}"),
        "3 beats buy-and-hold": (b3, f"Sharpe {base['sharpe']:.2f} vs {bench['sharpe']:.2f}; worst fall {base['maxdd']:.1%} vs {bench['maxdd']:.1%}; CAGR {base['cagr']:+.1%} vs {bench['cagr']:+.1%}"),
        "4 enough trades": (base["trades"] >= 100, f"{base['trades']} round trips"),
        "5 sub-periods": (pos_blocks >= 3, f"{pos_blocks} of 4 positive: " + ", ".join(f"{x:+.1%}" for x in blocks(base["rets"]))),
        "6 stress slippage": (share_names_stress <= THRESHOLD, f"{share_names_stress:.2%} of random-name runs at 15 bps have Sharpe >= {stress['sharpe']:.2f}"),
        "7 neighbours": (all(v > med_name for v in neighbours.values()), "neighbour Sharpe " + ", ".join(f"{k}: {v:.2f}" for k, v in neighbours.items()) + f" vs random-name median {med_name:.2f}"),
        "8 fidelity and data": (fid_disagree == 0, f"{fid_disagree} disagreements in {len(days)} weeks; eligible names {fid['min_eligible']}-{fid['max_eligible']}; entry gap {fid['mean_entry_gap_bps']:.0f} bps"),
    }
    verdict = dict(passed=all(ok for ok, _ in boxes.values()),
                   boxes={k: dict(ok=bool(ok), detail=d) for k, (ok, d) in boxes.items()},
                   base={k: v for k, v in base.items() if k not in ("rets", "equity")},
                   stress={k: v for k, v in stress.items() if k not in ("rets", "equity")},
                   benchmark=bench, benchmark_frictionless=bench_free, neighbours=neighbours,
                   null_names=dict(p5=float(np.percentile(name_sharpes, 5)), p50=med_name,
                                   p95=float(np.percentile(name_sharpes, 95))),
                   null_timing=dict(p50=float(np.median(time_sharpes)), p95=float(np.percentile(time_sharpes, 95))),
                   picks_top=pd.Series(picks).value_counts().head(10).to_dict())
    json.dump(dict(git_commit=GIT_COMMIT, run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                   window=dict(first=str(closes.index[days[0]].date()), last=str(closes.index[days[-1] + LOOKBACK].date()),
                               weeks=len(days)), first_bar=first, missing=missing, fidelity=fid,
                   costs=dict(commission_usd=COMMISSION_USD, slip_bps=SLIP_BPS, stress_bps=STRESS_BPS,
                              notional_gbp=NOTIONAL_GBP),
                   threshold=THRESHOLD, n_null=N_NULL, verdict=verdict, runtime_s=time.time() - t0),
              open(f"{OUT}/results.json", "w"), indent=1, default=str)

    b = verdict["base"]
    print(f"\nbuy-and-hold 40 names (with costs): CAGR {bench['cagr']:+.1%} Sharpe {bench['sharpe']:.2f} worst fall {bench['maxdd']:.1%}"
          f" | frictionless: CAGR {bench_free['cagr']:+.1%} Sharpe {bench_free['sharpe']:.2f}")
    print(f"\n=== weekly reversal: {'PASS' if verdict['passed'] else 'FAIL'} ===  CAGR {b['cagr']:+.1%} Sharpe {b['sharpe']:.2f} "
          f"total {b['total_return']:+.1%} worst fall {b['maxdd']:.1%} trades {b['trades']} win {b['win_rate']:.0%} "
          f"net/trade {b['avg_net_per_trade']:+.3%} (gross move {b['avg_gross_move']:+.3%})")
    print(f"random-name Sharpe p5/50/95 {verdict['null_names']['p5']:.2f}/{med_name:.2f}/{verdict['null_names']['p95']:.2f}"
          f" | random-timing median {verdict['null_timing']['p50']:.2f}")
    for k, box in verdict["boxes"].items():
        print(f"  [{'x' if box['ok'] else ' '}] {k}: {box['detail']}")
    print("\nmost-picked names:", verdict["picks_top"])
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
