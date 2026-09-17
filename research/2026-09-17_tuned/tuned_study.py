#!/usr/bin/env python
"""
Attempt 12, pre-registered 2026-09-17: tuned entry/exit on 10 US shares — one global rule versus a rule
chosen per share, fitted on 2006–2016 and judged once on 2017–2026.

Rules, grid, costs, windows, comparisons, threshold (0.417%) and the eight boxes are fixed in
PREREG_12_tuned_entry_exit.md (commit recorded in §10), committed before this script produced any returns.

    grid      : lookback 3/5/10/20 x direction winner|loser x exit (fixed 3/5/10/20 or trail 1/2/3/4 x ATR20)
                = 64 combinations per share, 640 choices in total
    entry     : the open of the day AFTER the signal day
    portfolio : one position at a time; when several shares signal, the largest absolute N-day move wins
    costs     : $1.00 per order (measured on this account), 5 bps slippage a side (15 bps stress)

Run on the VPS (daily bars come from the attempt 11 fetch):
  docker run --rm --user root --cpus 3 -e GIT_COMMIT=<hash> \
      -v /root/ibkr_research/reversal:/rev -v /root/ibkr_research/tuned:/tuned \
      ibkr_bot-trading-bot:latest python /tuned/tuned_study.py
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd

BARS = os.getenv("BARS", "/rev/bars")
OUT = os.getenv("OUT", "/tuned/out")
N_NULL = int(os.getenv("N_NULL", "1000"))
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")

THRESHOLD = 0.05 / 12
UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "JPM", "XOM"]
LOOKBACKS = [3, 5, 10, 20]
DIRECTIONS = ["winner", "loser"]
EXITS = [("hold", 3), ("hold", 5), ("hold", 10), ("hold", 20),
         ("trail", 1.0), ("trail", 2.0), ("trail", 3.0), ("trail", 4.0)]
MAX_HOLD = 40
MIN_HISTORY = 25
COMMISSION_USD = 1.00
SLIP_BPS, STRESS_BPS = 5.0, 15.0
NOTIONAL_GBP, GBP_PER_USD = 2000.0, 0.75
TRAIN_END = pd.Timestamp("2016-12-30")
TEST_START = pd.Timestamp("2017-01-03")
TEST_END = pd.Timestamp("2026-09-10")  # the card's window; later bars exist but are not used

G = {}


def combos():
    return [(n, d, e) for n in LOOKBACKS for d in DIRECTIONS for e in EXITS]


# ------------------------------------------------------------------ data
def load():
    o, h, l, c = {}, {}, {}, {}
    for sym in UNIVERSE:
        df = pd.read_csv(f"{BARS}/d_{sym}.csv")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop_duplicates("date").set_index("date").sort_index()
        o[sym], h[sym], l[sym], c[sym] = df["open"], df["high"], df["low"], df["close"]
    O, H, L, C = (pd.DataFrame(x).sort_index() for x in (o, h, l, c))
    cal = C.index
    atr = {}
    for sym in UNIVERSE:
        prev = C[sym].shift(1)
        tr = pd.concat([H[sym] - L[sym], (H[sym] - prev).abs(), (L[sym] - prev).abs()], axis=1).max(axis=1)
        atr[sym] = tr.rolling(20).mean()
    return O, H, L, C, pd.DataFrame(atr), cal


def signal_ok(sym, t, n, direction):
    """That share's own N-day return, as of the close of day t. None when it cannot be computed."""
    C = G["C"]
    if t - n < 0:
        return None
    cur, base = C[sym].iat[t], C[sym].iat[t - n]
    if not np.isfinite(cur) or not np.isfinite(base) or base <= 0:
        return None
    r = cur / base - 1.0
    if direction == "winner" and r > 0:
        return r
    if direction == "loser" and r < 0:
        return r
    return None


def eligible(sym, t):
    C = G["C"]
    col = C[sym]
    return t >= MIN_HISTORY and np.isfinite(col.iat[t]) and col.iloc[:t + 1].notna().sum() >= MIN_HISTORY


# ------------------------------------------------------------------ one trade
def run_trade(sym, entry_idx, exit_rule, slip_bps, force_name=None):
    """Enter at the open of entry_idx; return (ret_on_notional, exit_idx) or None."""
    O, H, L, C, ATR = G["O"], G["H"], G["L"], G["C"], G["ATR"]
    sym = force_name or sym
    n_days = len(C)
    if entry_idx >= n_days:
        return None
    buy_raw = O[sym].iat[entry_idx]
    if not np.isfinite(buy_raw) or buy_raw <= 0:
        return None
    slip = slip_bps / 1e4
    buy = buy_raw * (1 + slip)
    notional_usd = NOTIONAL_GBP / GBP_PER_USD
    qty = int(notional_usd // buy)
    if qty < 1:
        return None
    kind, param = exit_rule
    if kind == "hold":
        exit_idx = min(entry_idx + int(param), n_days - 1)
        sell_raw = O[sym].iat[exit_idx]
        if not np.isfinite(sell_raw):
            return None
        sell = sell_raw * (1 - slip)
    else:
        atr0 = ATR[sym].iat[entry_idx - 1] if entry_idx > 0 else np.nan
        if not np.isfinite(atr0) or atr0 <= 0:
            return None
        stop = C[sym].iat[entry_idx] - param * atr0 if np.isfinite(C[sym].iat[entry_idx]) else buy_raw - param * atr0
        peak = C[sym].iat[entry_idx]
        exit_idx, sell = None, None
        for t in range(entry_idx + 1, min(entry_idx + MAX_HOLD, n_days - 1) + 1):
            low, op, cl = L[sym].iat[t], O[sym].iat[t], C[sym].iat[t]
            if not np.isfinite(low):
                continue
            if low <= stop:
                px = op if (np.isfinite(op) and op < stop) else stop
                exit_idx, sell = t, px * (1 - slip)
                break
            if np.isfinite(cl):
                peak = max(peak, cl)
                stop = max(stop, peak - param * atr0)
        if exit_idx is None:
            exit_idx = min(entry_idx + MAX_HOLD, n_days - 1)
            sell_raw = C[sym].iat[exit_idx]
            if not np.isfinite(sell_raw):
                return None
            sell = sell_raw * (1 - slip)
    pnl = qty * (sell - buy) - 2.0 * COMMISSION_USD
    return pnl / notional_usd, exit_idx


# ------------------------------------------------------------------ portfolio simulation
def simulate(rule_for, lo, hi, slip_bps, name_override=None, shift=0):
    """rule_for: symbol -> (n, direction, exit). One position at a time between indices lo..hi."""
    rets = np.zeros(hi - lo + 1)
    trades, wins, skipped = [], 0, set()
    t = lo
    while t <= hi:
        best = None
        for sym in UNIVERSE:
            if sym not in rule_for or not eligible(sym, t):
                continue
            n, direction, _ = rule_for[sym]
            r = signal_ok(sym, t, n, direction)
            if r is None:
                continue
            if best is None or abs(r) > abs(best[1]) or (abs(r) == abs(best[1]) and sym < best[0]):
                best = (sym, r)
        if best is None:
            t += 1
            continue
        sym = best[0]
        entry_idx = t + 1 + shift
        if entry_idx > hi:
            break
        chosen = name_override(entry_idx) if name_override else sym
        out = run_trade(sym, entry_idx, rule_for[sym][2], slip_bps, force_name=chosen)
        if out is None:
            skipped.add(t)
            t += 1
            continue
        ret, exit_idx = out
        if exit_idx - lo < len(rets):
            rets[exit_idx - lo] += ret
        trades.append(dict(symbol=chosen, entry=int(entry_idx), exit=int(exit_idx), ret=float(ret)))
        wins += ret > 0
        t = exit_idx + 1
    equity = np.cumprod(1.0 + rets)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    sd = rets.std(ddof=1)
    years = len(rets) / 252.0
    return dict(sharpe=float(rets.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
                total_return=float(equity[-1] - 1.0),
                cagr=float(equity[-1] ** (1 / years) - 1.0) if years > 0 else 0.0,
                maxdd=float((equity / peaks - 1.0).min()), trades=len(trades),
                win_rate=float(wins / len(trades)) if trades else 0.0,
                avg_per_trade=float(np.mean([x["ret"] for x in trades])) if trades else 0.0,
                rets=rets, trade_list=trades, skipped=skipped)


def fidelity_check(rule_for, lo, hi, trades, skipped):
    """Independent vectorised rebuild of the signal layer; audits the trade list the loop produced.

    Different code path on purpose: pandas shift/compare over whole columns instead of the day loop's
    scalar indexing. It re-derives which share should have been bought on each entry day and checks that
    no earlier eligible signal was passed over.
    """
    C, O = G["C"], G["O"]
    n_days = len(C)
    pos = np.arange(n_days)
    fires, strength = {}, {}
    for sym, (n, direction, _) in rule_for.items():
        col = C[sym]
        base = col.shift(n)
        r = col / base - 1.0
        hit = (r > 0) if direction == "winner" else (r < 0)
        elig = col.notna() & (col.notna().cumsum() >= MIN_HISTORY) & pd.Series(pos >= MIN_HISTORY, index=col.index)
        fires[sym] = (hit & r.notna() & (base > 0) & elig).to_numpy()
        strength[sym] = r.abs().to_numpy()

    def winner_on(day):
        cands = [(strength[x][day], x) for x in UNIVERSE if x in rule_for and fires[x][day]]
        if not cands:
            return None
        return sorted(cands, key=lambda z: (-z[0], z[1]))[0][1]

    problems, prev_exit = [], lo - 1
    for tr in trades:
        sym, entry, sig = tr["symbol"], tr["entry"], tr["entry"] - 1
        if sig < lo or entry > hi:
            problems.append(f"{sym} {entry}: outside the window")
            continue
        if not fires[sym][sig]:
            problems.append(f"{sym}: no signal on day {sig}")
        elif winner_on(sig) != sym:
            problems.append(f"{sym} {sig}: {winner_on(sig)} had the larger move")
        for t in range(prev_exit + 1, sig):
            if t not in skipped and winner_on(t) is not None:
                problems.append(f"day {t}: signal skipped before the {sym} trade")
                break
        if not np.isfinite(O[sym].iat[entry]):
            problems.append(f"{sym} {entry}: entry price is not a real bar")
        if entry <= sig:
            problems.append(f"{sym}: entry {entry} is not after the signal day {sig}")
        prev_exit = tr["exit"]
    return problems


def blocks(rets):
    q = len(rets) // 4
    return [float(np.prod(1.0 + (rets[i * q:(i + 1) * q] if i < 3 else rets[3 * q:])) - 1.0) for i in range(4)]


def benchmark(lo, hi, slip_bps):
    O, C = G["O"], G["C"]
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
            names = [s for s in UNIVERSE if np.isfinite(px[s]) and eligible(s, t)]
            if names:
                value = max(0.0, value - len(names) * COMMISSION_USD - value * slip)
                per = value / len(names)
                held = {s: per / (px[s] * (1 + slip)) for s in names}
    rets = np.array(rets)
    equity = np.cumprod(1.0 + rets)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    sd = rets.std(ddof=1)
    years = len(rets) / 252.0
    return dict(sharpe=float(rets.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
                cagr=float(equity[-1] ** (1 / years) - 1.0), maxdd=float((equity / peaks - 1.0).min()),
                total_return=float(equity[-1] - 1.0))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    O, H, L, C, ATR, cal = load()
    G.update(O=O, H=H, L=L, C=C, ATR=ATR)
    train_lo = MIN_HISTORY
    train_hi = int(cal.get_indexer([cal[cal <= TRAIN_END][-1]])[0])
    test_lo = int(cal.get_indexer([cal[cal >= TEST_START][0]])[0])
    test_hi = int(cal.get_indexer([cal[cal <= TEST_END][-1]])[0])
    print(f"train {cal[train_lo].date()} -> {cal[train_hi].date()} | test {cal[test_lo].date()} -> {cal[test_hi].date()}", flush=True)

    # ---- choose rules on TRAINING data only
    all_combos = combos()
    per_stock_train = {}
    for sym in UNIVERSE:
        scores = []
        for combo in all_combos:
            r = simulate({sym: combo}, train_lo, train_hi, SLIP_BPS)
            scores.append((r["sharpe"], r["trades"], combo))
        scores.sort(key=lambda x: (-x[0], str(x[2])))
        per_stock_train[sym] = dict(best=scores[0][2], sharpe=scores[0][0], trades=scores[0][1])
    global_scores = []
    for combo in all_combos:
        r = simulate({s: combo for s in UNIVERSE}, train_lo, train_hi, SLIP_BPS)
        global_scores.append((r["sharpe"], r["trades"], combo))
    global_scores.sort(key=lambda x: (-x[0], str(x[2])))
    global_combo = global_scores[0][2]
    per_rule = {s: per_stock_train[s]["best"] for s in UNIVERSE}
    global_rule = {s: global_combo for s in UNIVERSE}
    print("global rule chosen in training:", global_combo, f"(training Sharpe {global_scores[0][0]:.2f})", flush=True)
    for s in UNIVERSE:
        print(f"  {s:6s} {str(per_stock_train[s]['best']):34s} training Sharpe {per_stock_train[s]['sharpe']:.2f} "
              f"({per_stock_train[s]['trades']} trades)", flush=True)

    # ---- run once on TEST data
    per_train = simulate(per_rule, train_lo, train_hi, SLIP_BPS)
    glob_train = simulate(global_rule, train_lo, train_hi, SLIP_BPS)
    per_test = simulate(per_rule, test_lo, test_hi, SLIP_BPS)
    glob_test = simulate(global_rule, test_lo, test_hi, SLIP_BPS)
    per_stress = simulate(per_rule, test_lo, test_hi, STRESS_BPS)
    bench = benchmark(test_lo, test_hi, SLIP_BPS)

    # ---- comparisons on the test window
    def random_name_factory(rng):
        def pick(entry_idx):
            avail = [s for s in UNIVERSE if eligible(s, entry_idx) and np.isfinite(O[s].iat[entry_idx])]
            return avail[int(rng.integers(0, len(avail)))] if avail else None
        return pick

    name_sh = np.empty(N_NULL)
    for seed in range(N_NULL):
        rng = np.random.default_rng(seed)
        name_sh[seed] = simulate(per_rule, test_lo, test_hi, SLIP_BPS, name_override=random_name_factory(rng))["sharpe"]
    share_names = float((name_sh >= per_test["sharpe"]).mean())

    name_sh_stress = np.empty(N_NULL)
    for seed in range(N_NULL):
        rng = np.random.default_rng(seed)
        name_sh_stress[seed] = simulate(per_rule, test_lo, test_hi, STRESS_BPS, name_override=random_name_factory(rng))["sharpe"]
    share_names_stress = float((name_sh_stress >= per_stress["sharpe"]).mean())

    span = test_hi - test_lo
    time_sh = np.empty(N_NULL)
    for seed in range(N_NULL):
        rng = np.random.default_rng(20_000 + seed)
        time_sh[seed] = simulate(per_rule, test_lo, test_hi, SLIP_BPS, shift=int(rng.integers(5, max(6, span // 4))))["sharpe"]
    share_time = float((time_sh >= per_test["sharpe"]).mean())

    problems = fidelity_check(per_rule, test_lo, test_hi, per_test["trade_list"], per_test["skipped"])
    pos_blocks = sum(1 for x in blocks(per_test["rets"]) if x > 0)
    b3 = per_test["sharpe"] > bench["sharpe"] or (per_test["maxdd"] >= 0.5 * bench["maxdd"] and per_test["cagr"] >= bench["cagr"] - 0.02)
    boxes = {
        "1 per-stock beats random names": (share_names <= THRESHOLD, f"{share_names:.2%} of {N_NULL} random-name runs have Sharpe >= {per_test['sharpe']:.2f} (need <= {THRESHOLD:.3%})"),
        "2 per-stock beats the global rule": (per_test["sharpe"] > glob_test["sharpe"], f"test Sharpe {per_test['sharpe']:.2f} vs {glob_test['sharpe']:.2f}"),
        "3 beats buy-and-hold": (b3, f"Sharpe {per_test['sharpe']:.2f} vs {bench['sharpe']:.2f}; worst fall {per_test['maxdd']:.1%} vs {bench['maxdd']:.1%}; CAGR {per_test['cagr']:+.1%} vs {bench['cagr']:+.1%}"),
        "4 enough trades": (per_test["trades"] >= 100, f"{per_test['trades']} round trips in the test window"),
        "5 sub-periods": (pos_blocks >= 3, f"{pos_blocks} of 4 positive: " + ", ".join(f"{x:+.1%}" for x in blocks(per_test["rets"]))),
        "6 stress slippage": (share_names_stress <= THRESHOLD, f"{share_names_stress:.2%} of random-name runs at 15 bps have Sharpe >= {per_stress['sharpe']:.2f}"),
        "7 random timing": (share_time <= THRESHOLD, f"{share_time:.2%} of shifted runs have Sharpe >= {per_test['sharpe']:.2f}"),
        "8 fidelity": (not problems, f"{len(problems)} disagreement(s) between the two implementations over "
                                    f"{per_test['trades']} trades" + (f"; first: {problems[0]}" if problems else
                                    "; every entry is the open of the day after its signal, every fill a real bar")),
    }
    tax = dict(per_stock=dict(train=per_train["sharpe"], test=per_test["sharpe"]),
               global_rule=dict(train=glob_train["sharpe"], test=glob_test["sharpe"]))
    verdict = dict(passed=all(ok for ok, _ in boxes.values()),
                   boxes={k: dict(ok=bool(ok), detail=d) for k, (ok, d) in boxes.items()},
                   per_stock_test={k: v for k, v in per_test.items() if k not in ("rets", "trade_list")},
                   global_test={k: v for k, v in glob_test.items() if k not in ("rets", "trade_list")},
                   benchmark=bench, overfitting_tax=tax,
                   chosen=dict(global_rule=str(global_combo),
                               per_stock={s: str(per_stock_train[s]["best"]) for s in UNIVERSE}),
                   null_names=dict(p50=float(np.median(name_sh)), p95=float(np.percentile(name_sh, 95))),
                   null_timing=dict(p50=float(np.median(time_sh)), p95=float(np.percentile(time_sh, 95))))
    json.dump(dict(git_commit=GIT_COMMIT, run_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                   windows=dict(train=[str(cal[train_lo].date()), str(cal[train_hi].date())],
                                test=[str(cal[test_lo].date()), str(cal[test_hi].date())]),
                   threshold=THRESHOLD, n_null=N_NULL, verdict=verdict, runtime_s=time.time() - t0),
              open(f"{OUT}/results.json", "w"), indent=1, default=str)

    print(f"\noverfitting tax — per-stock: training Sharpe {tax['per_stock']['train']:.2f} -> test {tax['per_stock']['test']:.2f}"
          f" | global: training {tax['global_rule']['train']:.2f} -> test {tax['global_rule']['test']:.2f}")
    print(f"buy-and-hold the ten (test window): CAGR {bench['cagr']:+.1%} Sharpe {bench['sharpe']:.2f} worst fall {bench['maxdd']:.1%}")
    p = verdict["per_stock_test"]
    print(f"\n=== tuned per-stock: {'PASS' if verdict['passed'] else 'FAIL'} ===  CAGR {p['cagr']:+.1%} Sharpe {p['sharpe']:.2f} "
          f"total {p['total_return']:+.1%} worst fall {p['maxdd']:.1%} trades {p['trades']} win {p['win_rate']:.0%} "
          f"per trade {p['avg_per_trade']:+.3%}")
    g = verdict["global_test"]
    print(f"    global rule on the same window: CAGR {g['cagr']:+.1%} Sharpe {g['sharpe']:.2f} trades {g['trades']}")
    print(f"    random names: median Sharpe {verdict['null_names']['p50']:.2f} (95th {verdict['null_names']['p95']:.2f})")
    for k, box in verdict["boxes"].items():
        print(f"  [{'x' if box['ok'] else ' '}] {k}: {box['detail']}")
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
