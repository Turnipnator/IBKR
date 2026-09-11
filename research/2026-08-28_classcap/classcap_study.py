#!/usr/bin/env python
"""
Class-cap vs slot-count study (2026-08-28).

Question: with 3 slots and a 40% asset-class cap, the class cap (not the slot
count) binds whenever the top-3 are one class — deployment is then ~40% NLV.
Should the cap be raised for a 3-slot book?

Method: replay the REAL signal code (TrendFollowingAnalyzer, rank_cross_sectional,
compute_combined_signal) and the REAL DecisionEngine._calculate_target_positions
(via __new__) day by day over 3Y of real IBKR daily bars, then simulate the bot's
actual mechanics: enter at the day's close when a name enters the target set,
top up when held < 70% of target, exit only via a 3xATR trailing stop that
ratchets on the daily high, settled cash with T+2, 10-day re-entry cooldown after
a stop-out, 6% settled-cash buffer + 50% partial-entry floor, IBKR commission
max($4, 0.05% notional) per order. Runs in the bot image so the engine is the
deployed one (GBX pence handling included).
"""
import sys, json, math, time, itertools
sys.path.insert(0, "/app")
import numpy as np
import pandas as pd
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.config import TradingConfig, trading_config
from src.engine import DecisionEngine
from src.indicators import TrendFollowingAnalyzer, rank_cross_sectional, compute_combined_signal
from src.contracts import CONTRACT_REGISTRY

STUDY = "/study"
START_NLV = 4710.57
WINDOW_BARS = 256          # what "1 Y" from IBKR gives the engine
TOPUP_THRESHOLD = 0.30
BUFFER = 0.06
MIN_PARTIAL = 0.50
COOLDOWN_TDAYS = 7         # 10 calendar days
SETTLE_LAG = 2
USD_MIN_FEE = 4.0
FEE_RATE = 0.0005

WATCHLIST = {k: list(v) for k, v in trading_config.symbols.items()}   # real 23-name universe (data mounted)
UNIVERSE = sorted(s for v in WATCHLIST.values() for s in v)
CLASS_OF = {s: k for k, v in WATCHLIST.items() for s in v}

# ---------------------------------------------------------------- data
bars = pd.read_csv(f"{STUDY}/hist3y.csv")
bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
bars = bars[bars.symbol.isin(UNIVERSE)].sort_values(["symbol", "date"]).reset_index(drop=True)
frames = {s: g.reset_index(drop=True) for s, g in bars.groupby("symbol")}
dates = np.array(sorted(bars.date.unique()))
fx = pd.read_csv(f"{STUDY}/gbpusd.csv"); fx["date"] = pd.to_datetime(fx["date"]).dt.normalize()
fx = fx.set_index("date")["gbpusd"].reindex(dates).ffill().bfill()
usd_to_gbp = (1.0 / fx).to_dict()

def fx_for(sym, d):
    ccy = CONTRACT_REGISTRY[sym][0]
    if ccy == "USD": return usd_to_gbp[d]
    if ccy == "GBX": return 0.01
    return 1.0

# per-symbol date -> row index
idx_of = {s: {d: i for i, d in enumerate(f.date)} for s, f in frames.items()}

# ---------------------------------------------------------------- signals (variant-independent)
lookbacks = [trading_config.lookback_short, trading_config.lookback_medium, trading_config.lookback_long]
t0 = time.time()
signals_by_day = {}
for d in dates:
    tsmom, info = {}, {}
    for s, f in frames.items():
        i = idx_of[s].get(d)
        if i is None or i < 60:          # engine would get "insufficient data" -> 0 signal; skip cleanly
            continue
        win = f.iloc[max(0, i - WINDOW_BARS + 1): i + 1]
        an = TrendFollowingAnalyzer(win, lookbacks=lookbacks, atr_period=trading_config.atr_period)
        sc, _ = an.compute_tsmom_signal()
        tsmom[s] = sc
        info[s] = {"price": an.get_current_price(), "atr": an.compute_atr(), "volatility": an.compute_volatility()}
    cs = rank_cross_sectional(tsmom)
    sig = {}
    for s in tsmom:
        sig[s] = dict(info[s], combined=compute_combined_signal(tsmom[s], cs.get(s, 0.0),
                                                                  trading_config.tsmom_weight, trading_config.csmom_weight))
    signals_by_day[d] = sig
print(f"signals replayed for {len(dates)} days in {time.time()-t0:.0f}s", file=sys.stderr)

# first day on which >= 20 names have a full 1Y window -> study start
full = [d for d in dates if sum(1 for s in frames if (idx_of[s].get(d) or 0) >= WINDOW_BARS - 1) >= 20]
STUDY_START = full[0]
LIVE_START = pd.Timestamp("2026-05-22")
print(f"study window {STUDY_START.date()} -> {dates[-1].date()}", file=sys.stderr)

# ---------------------------------------------------------------- variant-independent: how often is the top-3 one class?
def tradeable(sig):
    return {s: v for s, v in sig.items()
            if v["combined"] >= trading_config.signal_threshold and v["volatility"] >= trading_config.min_volatility}

single_class_days = {3: 0, 4: 0, 5: 0}; n_days = 0; class_hist = {}
for d in dates:
    if d < STUDY_START: continue
    n_days += 1
    top = sorted(tradeable(signals_by_day[d]).items(), key=lambda kv: -kv[1]["combined"])
    for n in (3, 4, 5):
        names = [s for s, _ in top[:n]]
        if len(names) >= n and len({CLASS_OF[s] for s in names}) == 1:
            single_class_days[n] += 1
    if top:
        c = CLASS_OF[top[0][0]]; class_hist[c] = class_hist.get(c, 0) + 1

# ---------------------------------------------------------------- simulator
def make_engine(overrides):
    cfg = TradingConfig(); cfg.symbols = WATCHLIST
    for k, v in overrides.items(): setattr(cfg, k, v)
    eng = DecisionEngine.__new__(DecisionEngine); eng.config = cfg
    return eng

def fee_gbp(notional_gbp, d):
    usd = notional_gbp / usd_to_gbp[d]
    return max(USD_MIN_FEE, FEE_RATE * usd) * usd_to_gbp[d]

def simulate(overrides, start=STUDY_START, end=None, label=""):
    eng = make_engine(overrides)
    twin = make_engine(dict(overrides, max_asset_class_pct=1.0))   # same slots/cap, no class cap
    k = eng.config.atr_stop_multiplier
    last_close = {}; target_gross = []; bind_days = 0; topups = 0; giveback_n = 0; giveback_pct = []
    cash = START_NLV; unsettled = []; pos = {}; cooldown = {}
    trades = 0; fees = 0.0; realized = 0.0
    curve = []; deploy = []; npos = []; class_exp = []; wins = 0; losses = 0
    days = [d for d in dates if d >= start and (end is None or d <= end)]
    eng.position_manager = twin.position_manager = SimpleNamespace(get_positions=lambda: [SimpleNamespace(symbol=s, quantity=p["qty"]) for s, p in pos.items()])
    eng.db = twin.db = MagicMock()
    for di, d in enumerate(days):
        # settle
        due = [a for (t, a) in unsettled if t <= di]; unsettled = [(t, a) for (t, a) in unsettled if t > di]
        cash += sum(due)
        # stops on today's bar
        for s, p in list(pos.items()):
            i = idx_of[s].get(d)
            if i is None: continue
            b = frames[s].iloc[i]
            last_close[s] = b.close
            if b.low <= p["stop"]:
                px = min(p["stop"], b.open) if b.open < p["stop"] else p["stop"]
                notional = p["qty"] * px * fx_for(s, d); f = fee_gbp(notional, d)
                pnl = notional - p["cost"] - f
                realized += pnl; fees += f; trades += 1
                wins += pnl > 0; losses += pnl <= 0
                unsettled.append((di + SETTLE_LAG, notional - f))
                cooldown[s] = di + COOLDOWN_TDAYS
                del pos[s]
            else:
                p["stop"] = max(p["stop"], b.high - p["trail"])
        # mark
        gross = sum(p["qty"] * last_close[s] * fx_for(s, d) for s, p in pos.items())
        nlv = cash + sum(a for _, a in unsettled) + gross
        # targets from today's signals
        eng.db.get_active_cooldowns.return_value = {s: str(u) for s, u in cooldown.items() if u > di}
        eng.connection = SimpleNamespace(get_fx_rates=lambda d=d: {"GBP": 1.0, "USD": usd_to_gbp[d]})
        sig = {s: v for s, v in signals_by_day[d].items() if v["price"] > 0 and v["atr"] > 0}
        targets = eng._calculate_target_positions(sig, nlv)
        twin.connection = eng.connection
        tg = sum(abs(t["target_weight"]) for t in targets.values())
        tg_uncapped = sum(abs(t["target_weight"]) for t in twin._calculate_target_positions(sig, nlv).values())
        target_gross.append(tg); bind_days += tg < tg_uncapped - 1e-9
        settled = cash
        for s, t in sorted(targets.items(), key=lambda kv: -kv[1]["signal_score"]):
            if t["target_shares"] <= 0: continue
            px, f2b, atr = t["price"], t["fx_to_base"], t["atr"]
            unit = px * f2b * (1 + BUFFER)
            held = pos[s]["qty"] if s in pos else 0
            if held:
                if held < t["target_shares"] * (1 - TOPUP_THRESHOLD):
                    delta = t["target_shares"] - held
                    aff = min(delta, int(settled / unit))
                    if aff >= 1:
                        notional = aff * px * f2b; f = fee_gbp(notional, d)
                        cash -= notional + f; settled = cash; fees += f; trades += 1
                        p = pos[s]; p["qty"] += aff; p["cost"] += notional + f
                        topups += 1
                        new_stop = px - k * atr                              # bot replaces the stop with a fresh trail
                        if new_stop < p["stop"]:                             # ...which can sit below the old ratchet
                            giveback_n += 1; giveback_pct.append((p["stop"] - new_stop) / px)
                        p["trail"] = k * atr; p["stop"] = new_stop
                continue
            q = t["target_shares"]; aff = int(settled / unit)
            if aff < q:
                if aff <= 0 or aff / q < MIN_PARTIAL: continue
                q = aff
            notional = q * px * f2b; f = fee_gbp(notional, d)
            cash -= notional + f; settled = cash; fees += f; trades += 1
            pos[s] = {"qty": q, "cost": notional + f, "trail": k * atr, "stop": px - k * atr}
            last_close[s] = px
        gross = sum(p["qty"] * last_close[s] * fx_for(s, d) for s, p in pos.items())
        nlv = cash + sum(a for _, a in unsettled) + gross
        curve.append(nlv); deploy.append(gross / nlv); npos.append(len(pos))
        ce = {}
        for s, p in pos.items():
            ce[CLASS_OF[s]] = ce.get(CLASS_OF[s], 0) + p["qty"] * last_close[s] * fx_for(s, d) / nlv
        class_exp.append(max(ce.values()) if ce else 0.0)
    c = pd.Series(curve, index=days)
    r = c.pct_change().dropna()
    years = len(days) / 252
    dd = (c / c.cummax() - 1).min()
    return dict(label=label, end=c.iloc[-1], ret=c.iloc[-1] / START_NLV - 1, cagr=(c.iloc[-1] / START_NLV) ** (1 / years) - 1 if years > 0 else 0,
                maxdd=dd, vol=r.std() * math.sqrt(252), sharpe=(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0,
                deploy=float(np.mean(deploy)), npos=float(np.mean(npos)), trades=trades, fees=fees,
                fee_pct_yr=fees / START_NLV / years if years > 0 else 0, worst_class=float(np.max(class_exp)),
                avg_maxclass=float(np.mean(class_exp)), wins=wins, losses=losses, realized=realized, years=years,
                target_gross=float(np.mean(target_gross)), bind_pct=bind_days / len(days),
                topups=topups, giveback_n=giveback_n, giveback_pct=float(np.mean(giveback_pct)) if giveback_pct else 0.0, curve=c)

GRID = [(3, 0.30), (4, 0.225), (5, 0.18)]
CAPS = [0.40, 0.50, 0.60, 0.80, 1.00]
windows = {"2Y": (STUDY_START, None), "12M": (dates[-1] - pd.Timedelta(days=365), None), "LIVE": (LIVE_START, None)}

out = {"single_class_days": single_class_days, "n_days": n_days, "class_hist": class_hist,
       "study_start": str(STUDY_START.date()), "end": str(dates[-1].date()), "results": []}
for wname, (ws, we) in windows.items():
    for slots, cap in GRID:
        for cc in CAPS:
            res = simulate({"max_open_positions": slots, "max_position_pct": cap, "max_asset_class_pct": cc},
                           start=ws, end=we, label=f"{slots}s/{cap:.3f}/{cc:.2f}")
            res["window"] = wname; res["slots"] = slots; res["cap"] = cap; res["class_cap"] = cc
            out["results"].append({k: v for k, v in res.items() if k != "curve"})
            if wname == "2Y" and slots == 3 and cc in (0.40, 1.00):
                res["curve"].to_csv(f"{STUDY}/curve_{slots}s_{cc:.2f}.csv")
            print(f"{wname:4s} {res['label']:16s} end £{res['end']:8,.0f} ret {res['ret']:+7.1%} cagr {res['cagr']:+7.1%} "
                  f"dd {res['maxdd']:6.1%} sh {res['sharpe']:+5.2f} dep {res['deploy']:5.1%} npos {res['npos']:4.1f} "
                  f"tr {res['trades']:3d} fees £{res['fees']:6,.0f} ({res['fee_pct_yr']:.1%}/yr) worstcls {res['worst_class']:5.1%} "
                  f"tgt {res['target_gross']:5.1%} bind {res['bind_pct']:4.0%} topups {res['topups']:3d} ratchet-giveback {res['giveback_n']:3d} ({res['giveback_pct']:.1%}) W/L {res['wins']}/{res['losses']}", flush=True)
json.dump(out, open(f"{STUDY}/results.json", "w"), indent=1, default=str)
print("single-class top-N days:", single_class_days, "of", n_days, "| top-1 class hist:", class_hist)
