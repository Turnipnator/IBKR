#!/usr/bin/env python
"""
Attempt 10, pre-registered 2026-09-17: intraday momentum on US shares, long only.

Rules, costs, window, random-timing comparison, threshold (0.5%) and the eight pass-mark boxes are
fixed in PREREG_10_intraday_momentum.md (commit 812e667), committed before this script produced any
returns. This implements that card and nothing else.

    signal   : return of the 09:30-10:00 bar
    entry    : open of the 15:30 bar, when the signal is above the threshold
    exit     : close of the 15:30 bar (the session close), always
    costs    : $1.00 per order (measured on the live account), 3 bps slippage a side (10 bps stress)

Run on the VPS:
  docker run --rm --user root --cpus 3 -e GIT_COMMIT=<hash> \
      -v /root/ibkr_research/intraday:/intra ibkr_bot-trading-bot:latest python /intra/intraday_study.py
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd

BARS = os.getenv("BARS", "/intra/bars")
OUT = os.getenv("OUT", "/intra/out")
N_NULL = int(os.getenv("N_NULL", "1000"))
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")

THRESHOLD = 0.05 / 10          # attempt 10
PRIMARY = "AAPL"
BREADTH = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "JPM", "XOM"]
REPORTED = ["SPY"]
COMMISSION_USD = 1.00          # per order, measured 2026-09-17 via what-if orders
SLIP_BPS, STRESS_BPS = 3.0, 10.0
NOTIONAL_GBP, GBP_PER_USD = 1500.0, 0.75
BARS_PER_SESSION = 13


# ------------------------------------------------------------------ data
def sessions(symbol):
    """Per-session frame: first-bar open/close and last-bar open/close, complete sessions only."""
    df = pd.read_csv(f"{BARS}/m30_{symbol}.csv")
    ts = pd.to_datetime(df["date"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    df = df.assign(ts=ts, day=ts.dt.date, hm=ts.dt.strftime("%H:%M")).sort_values("ts")
    counts = df.groupby("day").size()
    complete = set(counts[counts == BARS_PER_SESSION].index)
    df = df[df["day"].isin(complete)]
    rows = []
    for day, g in df.groupby("day"):
        g = g.sort_values("ts")
        first, last, penult = g.iloc[0], g.iloc[-1], g.iloc[-2]
        if first["hm"] != "09:30" or last["hm"] != "15:30":
            continue
        rows.append(dict(day=day, first_open=first["open"], first_close=first["close"],
                         last_open=last["open"], last_close=last["close"], penult_close=penult["close"]))
    out = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    return out, dict(total_days=int(counts.size), complete_days=int(len(out)),
                     dropped=int(counts.size - len(out)))


# ------------------------------------------------------------------ signals (two implementations)
def signal_vec(s, threshold):
    return ((s["first_close"] / s["first_open"] - 1.0) > threshold).to_numpy()


def signal_loop(s, threshold):
    out = []
    for o, c in zip(s["first_open"].tolist(), s["first_close"].tolist()):
        out.append((c / o - 1.0) > threshold)
    return np.array(out)


# ------------------------------------------------------------------ simulator
def simulate(s, take, slip_bps):
    """One trade a session at most: buy the 15:30 open, sell the 15:30 close. Whole shares."""
    slip = slip_bps / 1e4
    notional_usd = NOTIONAL_GBP / GBP_PER_USD
    rets = np.zeros(len(s))
    trades = 0
    wins = 0
    gross = []
    entry_px = s["last_open"].to_numpy()
    exit_px = s["last_close"].to_numpy()
    for i in range(len(s)):
        if not take[i]:
            continue
        buy = entry_px[i] * (1.0 + slip)
        sell = exit_px[i] * (1.0 - slip)
        qty = int(notional_usd // buy)
        if qty < 1:
            continue
        pnl = qty * (sell - buy) - 2.0 * COMMISSION_USD
        rets[i] = pnl / notional_usd
        gross.append(exit_px[i] / entry_px[i] - 1.0)
        trades += 1
        wins += pnl > 0
    equity = np.cumprod(1.0 + rets)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    sd = rets.std(ddof=1)
    years = len(s) / 252.0
    return dict(sharpe=float(rets.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
                total_return=float(equity[-1] - 1.0),
                cagr=float(equity[-1] ** (1 / years) - 1.0),
                maxdd=float((equity / peaks - 1.0).min()),
                trades=trades, win_rate=float(wins / trades) if trades else 0.0,
                avg_net_per_trade=float(rets[rets != 0].mean()) if trades else 0.0,
                avg_gross_move=float(np.mean(gross)) if gross else 0.0,
                equity=equity, rets=rets)


def blocks(rets):
    q = len(rets) // 4
    out = []
    for i in range(4):
        seg = rets[i * q: (i + 1) * q] if i < 3 else rets[3 * q:]
        out.append(float(np.prod(1.0 + seg) - 1.0))
    return out


def null_shares(s, take, slip_bps, base_sharpe, n_null=N_NULL):
    n = len(take)
    lo, hi = 20, max(21, n - 20)
    sharpes = np.empty(n_null)
    for seed in range(n_null):
        rng = np.random.default_rng(seed)
        rolled = np.roll(take, -int(rng.integers(lo, hi + 1)))
        sharpes[seed] = simulate(s, rolled, slip_bps)["sharpe"]
    return float((sharpes >= base_sharpe).mean()), sharpes


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    data, meta = {}, {}
    for sym in sorted(set(BREADTH + REPORTED)):
        path = f"{BARS}/m30_{sym}.csv"
        if not os.path.exists(path):
            print(f"missing bars for {sym}")
            continue
        data[sym], meta[sym] = sessions(sym)
    assert PRIMARY in data, "no bars for the primary instrument"
    s = data[PRIMARY]

    # data checks (box 7)
    gap = float(np.mean(np.abs(s["last_open"] / s["penult_close"] - 1.0)))
    vec, loop = signal_vec(s, 0.0), signal_loop(s, 0.0)
    fid = dict(disagree=int((vec != loop).sum()), sessions=int(len(s)), mean_entry_gap_bps=gap * 1e4)
    data_check = dict(primary=meta[PRIMARY], mean_entry_gap_bps=fid["mean_entry_gap_bps"],
                      passed=bool(fid["disagree"] == 0 and gap < 0.0005 and meta[PRIMARY]["complete_days"] >= 400))
    print("sessions:", {k: v["complete_days"] for k, v in meta.items()}, flush=True)
    print("data check:", json.dumps(data_check), flush=True)
    if not data_check["passed"]:
        json.dump(dict(git_commit=GIT_COMMIT, withdrawn=True, data_check=data_check, fidelity=fid),
                  open(f"{OUT}/results.json", "w"), indent=1, default=str)
        print("WITHDRAWN: data check failed; attempt 10 not run.")
        return

    base = simulate(s, vec, SLIP_BPS)
    stress = simulate(s, signal_vec(s, 0.0), STRESS_BPS)
    always = simulate(s, np.ones(len(s), dtype=bool), SLIP_BPS)
    bh_ret = float(s["last_close"].iloc[-1] / s["first_open"].iloc[0] - 1.0)
    share, null_sharpes = null_shares(s, vec, SLIP_BPS, base["sharpe"])
    share_s, _ = null_shares(s, vec, STRESS_BPS, stress["sharpe"])
    med_null = float(np.median(null_sharpes))
    neighbours = {}
    for thr in (0.0005, -0.0005):
        r = simulate(s, signal_vec(s, thr), SLIP_BPS)
        neighbours[f"{thr:+.4f}"] = r["sharpe"]

    breadth = {}
    for sym in BREADTH:
        if sym not in data:
            continue
        si = data[sym]
        ti = signal_vec(si, 0.0)
        ri = simulate(si, ti, SLIP_BPS)
        sh, nulls_i = null_shares(si, ti, SLIP_BPS, ri["sharpe"], n_null=200)
        breadth[sym] = dict(sharpe=ri["sharpe"], null_median=float(np.median(nulls_i)),
                            trades=ri["trades"], total_return=ri["total_return"])
    above = [k for k, v in breadth.items() if v["sharpe"] > v["null_median"]]

    spy = None
    if "SPY" in data:
        ss = data["SPY"]
        spy_res = simulate(ss, signal_vec(ss, 0.0), SLIP_BPS)
        spy = {k: spy_res[k] for k in ("sharpe", "total_return", "trades", "win_rate", "avg_net_per_trade")}

    pos_blocks = sum(1 for x in blocks(base["rets"]) if x > 0)
    b2 = base["sharpe"] > always["sharpe"] and base["total_return"] > always["total_return"]
    boxes = {
        "1 beats random timing": (share <= THRESHOLD, f"{share:.2%} of {N_NULL} random runs have Sharpe >= {base['sharpe']:.2f} (need <= {THRESHOLD:.1%})"),
        "2 beats always-long-the-close": (b2, f"Sharpe {base['sharpe']:.2f} vs {always['sharpe']:.2f}; total return {base['total_return']:+.1%} vs {always['total_return']:+.1%}"),
        "3 enough trades": (base["trades"] >= 100, f"{base['trades']} round trips"),
        "4 sub-periods": (pos_blocks >= 3, f"{pos_blocks} of 4 positive: " + ", ".join(f"{x:+.1%}" for x in blocks(base["rets"]))),
        "5 stress slippage": (share_s <= THRESHOLD, f"{share_s:.2%} of random runs at 10 bps have Sharpe >= {stress['sharpe']:.2f}"),
        "6 neighbours": (all(v > med_null for v in neighbours.values()), "neighbour Sharpe " + ", ".join(f"{k}: {v:.2f}" for k, v in neighbours.items()) + f" vs random median {med_null:.2f}"),
        "7 fidelity and data": (data_check["passed"], f"{fid['disagree']} signal disagreements in {fid['sessions']} sessions; entry gap {fid['mean_entry_gap_bps']:.1f} bps; {meta[PRIMARY]['dropped']} incomplete sessions dropped"),
        "8 breadth": (len(above) >= 6, f"{len(above)} of {len(breadth)} names beat their own random median (need 6)"),
    }
    verdict = dict(passed=all(ok for ok, _ in boxes.values()),
                   boxes={k: dict(ok=bool(ok), detail=d) for k, (ok, d) in boxes.items()},
                   base={k: v for k, v in base.items() if k not in ("equity", "rets")},
                   stress={k: v for k, v in stress.items() if k not in ("equity", "rets")},
                   always_long={k: v for k, v in always.items() if k not in ("equity", "rets")},
                   buy_and_hold_return=bh_ret, spy=spy, neighbours=neighbours, breadth=breadth,
                   null_sharpe=dict(p5=float(np.percentile(null_sharpes, 5)), p50=med_null,
                                    p95=float(np.percentile(null_sharpes, 95))))
    json.dump(dict(git_commit=GIT_COMMIT, run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                   window=dict(first=str(s["day"].iloc[0]), last=str(s["day"].iloc[-1]), sessions=int(len(s))),
                   costs=dict(commission_usd=COMMISSION_USD, slip_bps=SLIP_BPS, stress_bps=STRESS_BPS,
                              notional_gbp=NOTIONAL_GBP),
                   data_check=data_check, fidelity=fid, threshold=THRESHOLD, n_null=N_NULL, verdict=verdict,
                   runtime_s=time.time() - t0), open(f"{OUT}/results.json", "w"), indent=1, default=str)

    print(f"\nwindow {s['day'].iloc[0]} -> {s['day'].iloc[-1]} ({len(s)} sessions)")
    print(f"always-long-the-close: Sharpe {always['sharpe']:.2f} total {always['total_return']:+.1%} "
          f"trades {always['trades']} | AAPL buy-and-hold over the window {bh_ret:+.1%}")
    if spy:
        print(f"SPY (published instrument, not tradeable here): Sharpe {spy['sharpe']:.2f} "
              f"total {spy['total_return']:+.1%} trades {spy['trades']}")
    b = verdict["base"]
    print(f"\n=== intraday momentum ({PRIMARY}): {'PASS' if verdict['passed'] else 'FAIL'} ===  Sharpe {b['sharpe']:.2f} "
          f"total {b['total_return']:+.1%} worst fall {b['maxdd']:.1%} trades {b['trades']} win {b['win_rate']:.0%} "
          f"net/trade {b['avg_net_per_trade']:+.3%} (gross move {b['avg_gross_move']:+.3%}) | random Sharpe "
          f"p5/50/95 {verdict['null_sharpe']['p5']:.2f}/{med_null:.2f}/{verdict['null_sharpe']['p95']:.2f}")
    for k, box in verdict["boxes"].items():
        print(f"  [{'x' if box['ok'] else ' '}] {k}: {box['detail']}")
    print("\nper-name (strategy Sharpe vs its own random median):")
    for k in sorted(breadth):
        v = breadth[k]
        print(f"  {k:6s} {v['sharpe']:+.2f} vs {v['null_median']:+.2f} {'above' if v['sharpe'] > v['null_median'] else 'below'}"
              f"  ({v['trades']} trades, total {v['total_return']:+.1%})")
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
