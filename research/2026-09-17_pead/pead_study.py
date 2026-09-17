#!/usr/bin/env python
"""
Attempt 13, pre-registered 2026-09-17: post-earnings announcement drift on 40 US shares.

Rules, threshold (+2.0%), holding period (20 days), universe, costs, comparisons, the 0.385% pass mark and the
eight boxes are fixed in PREREG_13_earnings_drift.md (commit recorded in §10), committed before this script
produced any returns.

    event   : an SEC 8-K tagged item 2.02, from EDGAR, with its acceptance timestamp
    day 0   : the filing's own trading day if accepted before 19:30 UTC, else the next one
    signal  : AR = that share's day-0 return minus the equal-weight universe return that day
    entry   : the open of day 1, only when AR >= +2.0%
    exit    : the open 20 trading days later; one position at a time, overlapping events skipped
    costs   : $1.00 per order (measured on this account), 5 bps slippage a side (15 bps stress)

Run on the VPS:
  docker run --rm --user root --cpus 3 -e GIT_COMMIT=<hash> \
      -v /root/ibkr_research/reversal:/rev -v /root/ibkr_research/pead:/pead \
      ibkr_bot-trading-bot:latest python /pead/pead_study.py
"""
import bisect
import csv
import datetime as dt
import json
import math
import os
import time

import numpy as np
import pandas as pd

BARS = os.getenv("BARS", "/rev/bars")
EARN = os.getenv("EARN", "/pead/earnings")
OUT = os.getenv("OUT", "/pead/out")
N_NULL = int(os.getenv("N_NULL", "1000"))
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")

THRESHOLD = 0.05 / 13
UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "XOM", "UNH", "JNJ", "V",
            "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "COST", "WMT", "BAC", "CRM", "MCD", "CSCO",
            "ACN", "LIN", "ADBE", "TMO", "ABT", "DHR", "WFC", "VZ", "TXN", "NEE", "PM", "INTC", "IBM"]
AR_THRESHOLD, HOLD = 0.02, 20
NEIGHBOURS = [(0.02, 10), (0.02, 40), (0.015, 20), (0.03, 20)]
CUTOFF_UTC = dt.time(19, 30)
MIN_HISTORY = 25
DRIFT_DAYS = 60
COMMISSION_USD = 1.00
SLIP_BPS, STRESS_BPS = 5.0, 15.0
NOTIONAL_GBP, GBP_PER_USD = 2000.0, 0.75
WINDOW_START, WINDOW_END = pd.Timestamp("2006-10-02"), pd.Timestamp("2026-09-10")

G = {}


# ------------------------------------------------------------------ data
def load_bars():
    o, c = {}, {}
    for sym in UNIVERSE:
        df = pd.read_csv(f"{BARS}/d_{sym}.csv")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop_duplicates("date").set_index("date").sort_index()
        o[sym], c[sym] = df["open"], df["close"]
    O, C = pd.DataFrame(o).sort_index(), pd.DataFrame(c).sort_index()
    ret = C / C.shift(1) - 1.0
    live = C.notna() & (C.notna().cumsum() >= MIN_HISTORY)
    live.iloc[:MIN_HISTORY] = False
    mkt = ret.where(live).mean(axis=1)               # equal-weight universe return, our market proxy
    abnormal = ret.sub(mkt, axis=0).where(live)
    return O, C, ret, mkt, abnormal, live


def read_filings():
    """(symbol, acceptance datetime UTC) for every item-2.02 8-K on disk."""
    rows = []
    for sym in UNIVERSE:
        path = f"{EARN}/{sym}.csv"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                rows.append((sym, dt.datetime.fromisoformat(r["acceptance_utc"].replace("Z", "+00:00"))))
    return rows


def map_events(filings):
    """Loop implementation: acceptance time -> day 0 index, with the 19:30 UTC cutoff."""
    cal = [d.date() for d in G["C"].index]
    live, n = G["live"], len(cal)
    seen, out = set(), []
    for sym, when in filings:
        target = when.date() if when.time() < CUTOFF_UTC else when.date() + dt.timedelta(days=1)
        i = bisect.bisect_left(cal, target)
        if i >= n or i < 1:
            continue
        if not live[sym].iat[i] or not live[sym].iat[i - 1]:
            continue
        key = (sym, i)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(symbol=sym, day0=i, filed=when))
    for e in out:
        e["ar"] = float(G["abnormal"][e["symbol"]].iat[e["day0"]])
    return [e for e in out if np.isfinite(e["ar"])]


def map_events_vectorised(filings):
    """Second, independent implementation of the same mapping, for box 8."""
    idx = G["C"].index
    syms = np.array([f[0] for f in filings])
    stamps = pd.to_datetime([f[1] for f in filings], utc=True)
    days = stamps.tz_convert("UTC")
    bump = (days.hour * 60 + days.minute) >= (CUTOFF_UTC.hour * 60 + CUTOFF_UTC.minute)
    targets = pd.DatetimeIndex(days.normalize().tz_localize(None)) + pd.to_timedelta(bump.astype(int), unit="D")
    pos = idx.searchsorted(targets, side="left")
    keep = set()
    for sym, p in zip(syms, pos):
        if 1 <= p < len(idx) and G["live"][sym].iat[p] and G["live"][sym].iat[p - 1]:
            keep.add((sym, int(p)))
    return keep


# ------------------------------------------------------------------ trading
def price_trade(sym, entry_idx, exit_idx, slip_bps):
    O = G["O"]
    buy_raw, sell_raw = O[sym].iat[entry_idx], O[sym].iat[exit_idx]
    if not np.isfinite(buy_raw) or not np.isfinite(sell_raw) or buy_raw <= 0:
        return None
    slip = slip_bps / 1e4
    buy, sell = buy_raw * (1 + slip), sell_raw * (1 - slip)
    notional_usd = NOTIONAL_GBP / GBP_PER_USD
    qty = int(notional_usd // buy)
    if qty < 1:
        return None
    return (qty * (sell - buy) - 2.0 * COMMISSION_USD) / notional_usd


def pick_trades(events, thr, hold):
    """Which events become trades: chronological, one position at a time, largest AR wins the day."""
    n_days = len(G["C"])
    hi = G["hi"]
    ordered = sorted([e for e in events if e["ar"] >= thr], key=lambda e: (e["day0"], -e["ar"], e["symbol"]))
    trades, busy_until = [], -1
    for e in ordered:
        entry = e["day0"] + 1
        if entry <= busy_until or entry > hi:
            continue
        exit_idx = min(entry + hold, hi, n_days - 1)
        if exit_idx <= entry:
            continue
        trades.append(dict(symbol=e["symbol"], day0=e["day0"], entry=entry, exit=exit_idx, ar=e["ar"]))
        busy_until = exit_idx
    return trades


def run(trades, slip_bps, names=None):
    """Price a fixed trade schedule; `names` overrides which share each trade buys (the nulls)."""
    lo, hi = G["lo"], G["hi"]
    rets = np.zeros(hi - lo + 1)
    done, wins = [], 0
    for i, t in enumerate(trades):
        sym = names[i] if names else t["symbol"]
        if sym is None:
            continue
        r = price_trade(sym, t["entry"], t["exit"], slip_bps)
        if r is None:
            continue
        rets[t["exit"] - lo] += r
        done.append(dict(t, symbol=sym, ret=float(r)))
        wins += r > 0
    equity = np.cumprod(1.0 + rets)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    sd = rets.std(ddof=1)
    years = len(rets) / 252.0
    return dict(sharpe=float(rets.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
                total_return=float(equity[-1] - 1.0),
                cagr=float(equity[-1] ** (1 / years) - 1.0) if years > 0 else 0.0,
                maxdd=float((equity / peaks - 1.0).min()), trades=len(done),
                win_rate=float(wins / len(done)) if done else 0.0,
                avg_per_trade=float(np.mean([x["ret"] for x in done])) if done else 0.0,
                rets=rets, trade_list=done)


def blocks(trade_list):
    if not trade_list:
        return [0.0] * 4
    q = max(1, len(trade_list) // 4)
    out = []
    for i in range(4):
        chunk = trade_list[i * q:(i + 1) * q] if i < 3 else trade_list[3 * q:]
        out.append(float(np.prod([1.0 + x["ret"] for x in chunk]) - 1.0))
    return out


def benchmark(slip_bps):
    O, C, live = G["O"], G["C"], G["live"]
    lo, hi = G["lo"], G["hi"]
    slip = slip_bps / 1e4
    value = NOTIONAL_GBP / GBP_PER_USD
    held, rets, year = {}, [], None
    for t in range(lo, hi + 1):
        px = O.iloc[t]
        if held:
            nv = sum(q * px[s] for s, q in held.items() if np.isfinite(px[s]))
            rets.append(nv / value - 1.0)
            value = nv
        else:
            rets.append(0.0)
        if C.index[t].year != year:
            year = C.index[t].year
            names = [s for s in UNIVERSE if np.isfinite(px[s]) and live[s].iat[t]]
            if names:
                value = max(0.0, value - len(names) * COMMISSION_USD - value * slip)
                per = value / len(names)
                held = {s: per / (px[s] * (1 + slip)) for s in names}
    rets = np.array(rets)
    equity = np.cumprod(1.0 + rets)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    sd = rets.std(ddof=1)
    return dict(sharpe=float(rets.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
                cagr=float(equity[-1] ** (252 / len(rets)) - 1.0), maxdd=float((equity / peaks - 1.0).min()),
                total_return=float(equity[-1] - 1.0))


def drift_curve(events, sign):
    """Average cumulative abnormal return from day 1 to day 60 — the effect itself, not the tradeable rule."""
    AB = G["abnormal"]
    n_days = len(AB)
    rows = []
    for e in events:
        if (sign > 0 and e["ar"] < AR_THRESHOLD) or (sign < 0 and e["ar"] > -AR_THRESHOLD):
            continue
        end = e["day0"] + DRIFT_DAYS
        if end >= n_days:
            continue
        seq = AB[e["symbol"]].iloc[e["day0"] + 1:end + 1].to_numpy(dtype=float, copy=True)
        if np.isnan(seq).any():
            continue
        rows.append((e["day0"], np.cumsum(seq)))
    if not rows:
        return dict(n=0)
    cum = np.vstack([r[1] for r in rows])
    early = np.vstack([r[1] for r in rows if G["C"].index[r[0]].year <= 2016])
    late = np.vstack([r[1] for r in rows if G["C"].index[r[0]].year >= 2017])
    return dict(n=len(rows),
                day5=float(cum[:, 4].mean()), day20=float(cum[:, 19].mean()), day60=float(cum[:, -1].mean()),
                day20_2006_2016=float(early[:, 19].mean()) if len(early) else None,
                day20_2017_2026=float(late[:, 19].mean()) if len(late) else None,
                day60_2006_2016=float(early[:, -1].mean()) if len(early) else None,
                day60_2017_2026=float(late[:, -1].mean()) if len(late) else None)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    O, C, ret, mkt, abnormal, live = load_bars()
    G.update(O=O, C=C, ret=ret, mkt=mkt, abnormal=abnormal, live=live)
    cal = C.index
    G["lo"] = int(cal.get_indexer([cal[cal >= WINDOW_START][0]])[0])
    G["hi"] = int(cal.get_indexer([cal[cal <= WINDOW_END][-1]])[0])

    filings = read_filings()
    events = [e for e in map_events(filings) if G["lo"] <= e["day0"] <= G["hi"]]
    qualifying = [e for e in events if e["ar"] >= AR_THRESHOLD]
    print(f"window {cal[G['lo']].date()} -> {cal[G['hi']].date()} | {len(filings)} filings -> "
          f"{len(events)} events in window, {len(qualifying)} qualifying (AR >= {AR_THRESHOLD:.1%})", flush=True)

    # ---- data quality (box 8)
    # Only COMPLETE years count: the window opens in October 2006 and closes in September 2026, and a name
    # listed mid-year has a short first year. Counting those as full years would flag every name for free.
    full_years = {}
    for sym in UNIVERSE:
        dates = C.index[C[sym].notna()]
        if len(dates) == 0:
            continue
        first, last = dates[0], dates[-1]
        full_years[sym] = {y for y in range(2007, 2026)
                           if first <= pd.Timestamp(year=y, month=1, day=31)
                           and last >= pd.Timestamp(year=y, month=12, day=1)
                           and cal[G["lo"]] <= pd.Timestamp(year=y, month=1, day=31)
                           and cal[G["hi"]] >= pd.Timestamp(year=y, month=12, day=1)}
    per_year = {}
    for e in events:
        yr = cal[e["day0"]].year
        if yr not in full_years.get(e["symbol"], set()):
            continue
        per_year[(e["symbol"], yr)] = per_year.get((e["symbol"], yr), 0) + 1
    for sym, years in full_years.items():          # a complete year with no filing at all is a coverage gap
        for y in years:
            per_year.setdefault((sym, y), 0)
    counts = {}
    for (sym, yr), n in per_year.items():
        counts.setdefault(sym, []).append(n)
    strict_bad = sorted({f"{sym} {yr}:{n}" for (sym, yr), n in per_year.items() if n < 3 or n > 6})
    median_bad = sorted(s for s, v in counts.items() if not 3 <= float(np.median(v)) <= 6)
    sample = [(e["symbol"], str(cal[e["day0"]].date()), e["filed"].isoformat(), round(e["ar"], 4))
              for e in sorted(events, key=lambda x: x["day0"])[:5] + sorted(events, key=lambda x: -x["day0"])[:5]]

    # ---- the strategy, run once
    trades = pick_trades(events, AR_THRESHOLD, HOLD)
    base = run(trades, SLIP_BPS)
    stress = run(trades, STRESS_BPS)
    bench = benchmark(SLIP_BPS)

    # ---- fidelity (box 8)
    vec = map_events_vectorised(filings)
    loop = {(e["symbol"], e["day0"]) for e in events}
    vec_in_window = {(s, i) for (s, i) in vec if G["lo"] <= i <= G["hi"]}
    problems = [f"{len(loop ^ vec_in_window)} event(s) differ between implementations"] if loop != vec_in_window else []
    for t in base["trade_list"]:
        if t["entry"] != t["day0"] + 1:
            problems.append(f"{t['symbol']}: entry {t['entry']} is not the day after day 0 {t['day0']}")
        if not np.isfinite(O[t["symbol"]].iat[t["entry"]]):
            problems.append(f"{t['symbol']}: entry price is not a real bar")
    for a, b in zip(base["trade_list"], base["trade_list"][1:]):
        if b["entry"] <= a["exit"]:
            problems.append(f"overlapping positions at {b['entry']}")

    # ---- nulls
    def eligible_at(idx):
        return [s for s in UNIVERSE if live[s].iat[idx] and np.isfinite(O[s].iat[idx])]

    def null_names(seed, sched):
        rng = np.random.default_rng(seed)
        out = []
        for t in sched:
            avail = eligible_at(t["entry"])
            out.append(avail[int(rng.integers(0, len(avail)))] if avail else None)
        return out

    name_sh = np.array([run(trades, SLIP_BPS, names=null_names(s, trades))["sharpe"] for s in range(N_NULL)])
    name_sh_stress = np.array([run(trades, STRESS_BPS, names=null_names(s, trades))["sharpe"] for s in range(N_NULL)])
    n = len(trades)
    time_sh = []
    for seed in range(N_NULL):
        rng = np.random.default_rng(50_000 + seed)
        off = int(rng.integers(8, max(9, n - 8))) if n > 20 else 1
        time_sh.append(run(trades, SLIP_BPS, names=[trades[(i + off) % n]["symbol"] for i in range(n)])["sharpe"])
    time_sh = np.array(time_sh)
    share_names = float((name_sh >= base["sharpe"]).mean())
    share_names_stress = float((name_sh_stress >= stress["sharpe"]).mean())
    share_time = float((time_sh >= base["sharpe"]).mean())
    null_median = float(np.median(name_sh))

    # ---- neighbours (box 7)
    neigh = {}
    for thr, hold in NEIGHBOURS:
        r = run(pick_trades(events, thr, hold), SLIP_BPS)
        neigh[f"AR>={thr:.1%}, hold {hold}d"] = dict(sharpe=r["sharpe"], trades=r["trades"], cagr=r["cagr"])

    # ---- boxes
    pos_blocks = sum(1 for x in blocks(base["trade_list"]) if x > 0)
    b3 = base["sharpe"] > bench["sharpe"] or (base["maxdd"] >= 0.5 * bench["maxdd"] and base["cagr"] >= bench["cagr"] - 0.02)
    boxes = {
        "1 beats random names": (share_names <= THRESHOLD, f"{share_names:.2%} of {N_NULL} random-name runs have Sharpe >= {base['sharpe']:.2f} (need <= {THRESHOLD:.3%}); random median {null_median:.2f}"),
        "2 beats random timing": (share_time <= THRESHOLD, f"{share_time:.2%} of shifted runs have Sharpe >= {base['sharpe']:.2f}"),
        "3 beats buy-and-hold": (b3, f"Sharpe {base['sharpe']:.2f} vs {bench['sharpe']:.2f}; worst fall {base['maxdd']:.1%} vs {bench['maxdd']:.1%}; CAGR {base['cagr']:+.1%} vs {bench['cagr']:+.1%}"),
        "4 enough trades": (base["trades"] >= 100, f"{base['trades']} round trips"),
        "5 sub-periods": (pos_blocks >= 3, f"{pos_blocks} of 4 positive: " + ", ".join(f"{x:+.1%}" for x in blocks(base["trade_list"]))),
        "6 stress slippage": (share_names_stress <= THRESHOLD, f"{share_names_stress:.2%} of random-name runs at 15 bps have Sharpe >= {stress['sharpe']:.2f}"),
        "7 neighbours": (all(v["sharpe"] > null_median for v in neigh.values()), "; ".join(f"{k} {v['sharpe']:.2f}" for k, v in neigh.items()) + f" vs random median {null_median:.2f}"),
        "8 fidelity and data": (not problems and not strict_bad, f"{len(problems)} implementation disagreement(s); {len(strict_bad)} name-year(s) outside 3-6 filings (median reading: {len(median_bad)} name(s) fail)"),
    }
    verdict = dict(passed=all(ok for ok, _ in boxes.values()),
                   boxes={k: dict(ok=bool(ok), detail=d) for k, (ok, d) in boxes.items()},
                   strategy={k: v for k, v in base.items() if k not in ("rets", "trade_list")},
                   stress={k: v for k, v in stress.items() if k not in ("rets", "trade_list")},
                   benchmark=bench, neighbours=neigh,
                   events=dict(filings=len(filings), in_window=len(events), qualifying=len(qualifying),
                               traded=base["trades"]),
                   drift_up=drift_curve(events, +1), drift_down=drift_curve(events, -1),
                   null_names=dict(p5=float(np.percentile(name_sh, 5)), p50=null_median,
                                   p95=float(np.percentile(name_sh, 95))),
                   null_timing=dict(p50=float(np.median(time_sh))),
                   data_quality=dict(strict_outliers=strict_bad[:40], median_failures=median_bad, sample=sample),
                   most_traded=dict(sorted(
                       {s: sum(1 for t in base["trade_list"] if t["symbol"] == s) for s in UNIVERSE}.items(),
                       key=lambda kv: -kv[1])[:10]))
    json.dump(dict(git_commit=GIT_COMMIT, run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                   window=[str(cal[G["lo"]].date()), str(cal[G["hi"]].date())],
                   threshold=THRESHOLD, n_null=N_NULL, verdict=verdict, runtime_s=time.time() - t0),
              open(f"{OUT}/results.json", "w"), indent=1, default=str)

    d = verdict["drift_up"]
    print(f"\ndrift after a +2% earnings jump ({d['n']} events): day+5 {d['day5']:+.2%}, day+20 {d['day20']:+.2%}, "
          f"day+60 {d['day60']:+.2%}  |  2006-16 vs 2017-26 at day+60: {d['day60_2006_2016']:+.2%} vs {d['day60_2017_2026']:+.2%}")
    dn = verdict["drift_down"]
    print(f"mirror (-2% jumps, not tradeable here, {dn['n']} events): day+20 {dn['day20']:+.2%}, day+60 {dn['day60']:+.2%}")
    print(f"buy-and-hold the 40 (with costs): CAGR {bench['cagr']:+.1%} Sharpe {bench['sharpe']:.2f} worst fall {bench['maxdd']:.1%}")
    print(f"\n=== earnings drift: {'PASS' if verdict['passed'] else 'FAIL'} ===  CAGR {base['cagr']:+.1%} "
          f"Sharpe {base['sharpe']:.2f} total {base['total_return']:+.1%} worst fall {base['maxdd']:.1%} "
          f"trades {base['trades']} win {base['win_rate']:.0%} per trade {base['avg_per_trade']:+.3%}")
    print(f"    random names p5/50/95 {verdict['null_names']['p5']:.2f}/{null_median:.2f}/{verdict['null_names']['p95']:.2f}"
          f" | random timing median {verdict['null_timing']['p50']:.2f}")
    for k, box in verdict["boxes"].items():
        print(f"  [{'x' if box['ok'] else ' '}] {k}: {box['detail']}")
    if strict_bad:
        print(f"  name-years outside 3-6 filings ({len(strict_bad)}): {', '.join(strict_bad[:15])}"
              f"{' ...' if len(strict_bad) > 15 else ''}")
    print("  most traded:", verdict["most_traded"])
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
