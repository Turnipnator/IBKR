#!/usr/bin/env python
"""
Out-of-sample study (2026-09-15): does the LIVE configuration have an edge before 2024?

Every structural parameter (3 slots x 30%, 60% class cap, 3xATR, 8% vol floor,
top-up gates) was chosen on 2024-09 -> 2026-08 data. This replays the frozen
live config over 2007-2023, which none of those decisions looked at.

Universe: the 23 live UCITS lines mapped to the US-listed ETFs they replaced
(dividend-adjusted IBKR bars, 20Y). AIGS (WisdomTree Softs) has no long-history
proxy and is excluded. Signals mirror the live bot: TSMOM on price-only closes
for lines that distribute (the bot reads unadjusted TRADES bars), total-return
closes for accumulating lines and ETCs; trading P&L is always total return.

Fidelity:
  V0  vectorised indicators == the real TrendFollowingAnalyzer on sampled days
  V1  UCITS-bar replay vs the signals the live bot saved in instrument_signals
  V2  proxy universe vs native UCITS universe over their common history
Mechanics (all from the deployed engine or mirrored from bot.py/engine.py):
  real DecisionEngine._calculate_target_positions (via __new__); fill at the
  signal close +/- slippage; 3xATR TRAIL ratcheting on the daily high, gap fills
  at the open; top-up only < 70% of target, skipped within 1xATR of the stop it
  would carry, stop re-armed at max(ratchet, fresh level); settled cash T+2 with
  the 6% buffer, per-cycle committed tally and 50% entry/top-up floors; 10-day
  cooldown after a stop; daily-loss gate (£200 fixed) blocks every order that
  cycle; drawdown REDUCE halves targets at 10% from the all-time peak; HALT at
  20% flattens and (as live — the stored peak never resets) stays halted.
Costs: max($4, 0.05%) per order + slippage (bps per side, default 5).
Nulls (same mechanics, same costs):
  SEL  scores permuted among vol-eligible names each month -> same number of
       names pass the threshold each day (timing kept), names chosen at random
  ALL  every vol-eligible name gets a random passing score each month ->
       always ~fully invested in random names (no timing, no selection)

Run on the VPS after fetch_bars.py:
  docker run --rm --user root --cpus 3 -v /root/ibkr_research/oos:/study \
      ibkr_bot-trading-bot:latest python /study/oos_study.py
"""
import json
import logging
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from types import SimpleNamespace

sys.path.insert(0, "/app")
import numpy as np
import pandas as pd

from src.config import TradingConfig
from src.contracts import CONTRACT_REGISTRY
from src.engine import DecisionEngine
from src.indicators import TrendFollowingAnalyzer, compute_combined_signal, rank_cross_sectional

logging.disable(logging.CRITICAL)   # the engine logs every target at INFO

STUDY = "/study"
OUT = f"{STUDY}/out"
START_NLV = 4710.57
WINDOW_BARS = int(os.getenv("WINDOW_BARS", "256"))   # bars a "1 Y" request gives the engine (checked against live logs)
LAST_DATE = pd.Timestamp("2026-09-14")   # drop the partial bar fetched mid-session
N_SEEDS = int(os.getenv("N_SEEDS", "200"))
NPROC = int(os.getenv("NPROC", "3"))

UCITS_WATCHLIST = {   # data/watchlist.json (v2_ucits) — hardcoded: the image has no data/ mount
    "equity": ["CSPX", "EQQQ", "RTWO", "EIMU", "VEUR", "CNYA", "IJPN"],
    "bond": ["DTLA", "IDTM", "IBTA", "LQDE", "IHYU", "JPEA", "IDTP"],
    "commodity": ["IGLN", "ISLN", "CRUD", "NGAS", "AIGA", "AIGI", "CMOD", "COPA", "AIGS"],
    "alt": ["IDUP"],
}
PROXY_OF = {
    "CSPX": "SPY", "EQQQ": "QQQ", "RTWO": "IWM", "EIMU": "EEM", "VEUR": "VGK", "CNYA": "FXI", "IJPN": "EWJ",
    "DTLA": "TLT", "IDTM": "IEF", "IBTA": "SHY", "LQDE": "LQD", "IHYU": "HYG", "JPEA": "EMB", "IDTP": "TIP",
    "IGLN": "GLD", "ISLN": "SLV", "CRUD": "USO", "NGAS": "UNG", "AIGA": "DBA", "AIGI": "DBB",
    "CMOD": "DJP",      # Invesco Bloomberg Commodity -> iPath Bloomberg Commodity ETN (2006)
    "COPA": "CPER",     # WT Copper -> US Copper Index Fund (2011-11; absent before)
    "IDUP": "VNQ",
    # "AIGS": WT Softs — no US softs ETF spans 2007-2023; excluded
}
# UCITS lines that DISTRIBUTE (IBKR longName / share class): the live signal on these
# is price-only, so their proxies' TSMOM uses unadjusted closes in the "live" mode.
DIST = {"EQQQ", "EIMU", "VEUR", "IJPN", "IDTM", "LQDE", "IHYU", "IDUP"}

# IBKR history for these proxies is truncated (TLT from 2016-02, IEF/SHY from 2017-08: the
# iShares Treasury ETFs changed primary listing). Before each one's first bar, splice on the
# candidate whose overlap daily returns correlate best with it (chosen on fit, never on
# strategy results). "-ARCA" keys are the same ETF requested on its old listing.
SPLICE_CANDIDATES = {
    "TLT": ["TLT-ARCA", "SPTL", "VGLT", "TLH", "EDV"],
    "IEF": ["IEF-ARCA", "SPTI", "IEI", "VGIT", "SCHR"],
    "SHY": ["SHY-ARCA", "BIL", "SCHO", "VGSH", "SPTS"],
}

CRISES = {
    "GFC 2007-10..2009-03": ("2007-10-09", "2009-03-09"),
    "2011 downgrade": ("2011-04-29", "2011-10-03"),
    "2015-16 selloff": ("2015-05-21", "2016-02-11"),
    "Q4 2018": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rates": ("2022-01-03", "2022-10-12"),
}

UNIV = {}   # filled in main, inherited by forked workers


# ------------------------------------------------------------------ data
def load_bars(kind, sym):
    p = f"{STUDY}/bars/{kind}_{sym}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df.date <= LAST_DATE].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df if len(df) else None


def splice(main, alt, alt_key):
    """Ratio-join alt's bars before main's first date, scaled so the closes meet on the first common day."""
    common = main[["date", "close"]].merge(alt[["date", "close"]], on="date", suffixes=("", "_alt"))
    if len(common) < 250 or alt.date.iloc[0] >= main.date.iloc[0]:
        return None, None
    d0 = common.date.iloc[0]
    k = common.close.iloc[0] / common.close_alt.iloc[0]
    pre = alt[alt.date < d0].copy()
    for c in ("open", "high", "low", "close"):
        pre[c] = pre[c] * k
    pre["symbol"] = main.symbol.iloc[0]
    r = common.set_index("date")[["close", "close_alt"]].pct_change().dropna()
    info = dict(alt=alt_key, splice_date=str(d0.date()), alt_first=str(alt.date.iloc[0].date()),
                overlap_days=len(common), ret_corr=float(r.close.corr(r.close_alt)),
                vol_ratio_main_over_alt=float(r.close.std() / r.close_alt.std()))
    return pd.concat([pre, main[main.date >= d0]], ignore_index=True), info


# ------------------------------------------------------------------ indicators (vectorised, validated in V0)
def tsmom_arr(close, lookbacks, window=WINDOW_BARS):
    """Literal re-statement of TrendFollowingAnalyzer.compute_tsmom_signal on a trailing window."""
    n = len(close)
    out = np.zeros(n)
    weights = [0.3, 0.3, 0.4]
    for i in range(n):
        m = min(i + 1, window)
        if m < min(lookbacks) + 2:
            continue
        signals = []
        for lb, w in zip(lookbacks, weights):
            if m < lb + 2:
                lb = m - 2
            if m < lb + 1:
                ret = 0.0
            else:
                cur, past = close[i], close[i - lb]
                ret = 0.0 if (past == 0 or np.isnan(past) or np.isnan(cur)) else (cur - past) / past
            signals.append(np.sign(ret) * w)
        out[i] = float(round(sum(signals), 3))
    return out


def atr_arr(df, period):
    prev = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    a = tr.rolling(window=period).mean().to_numpy(dtype=float, copy=True)   # pandas 3 views are read-only
    a[: period] = 0.0          # compute_atr returns 0 when len < period + 1
    return np.nan_to_num(a, nan=0.0)


def vol_arr(df, period=20):
    r = df.close.pct_change()
    v = (r.rolling(period).std() * np.sqrt(252)).to_numpy(dtype=float, copy=True)
    v[: period] = 0.0          # annualised_volatility returns 0 when len < period + 1
    return np.nan_to_num(v, nan=0.0)


def indicator_pack(px_df, sig_df, cfg):
    lookbacks = [cfg.lookback_short, cfg.lookback_medium, cfg.lookback_long]
    return dict(d_px=px_df.date.values, price=px_df.close.to_numpy(float), atr=atr_arr(px_df, cfg.atr_period),
                vol=vol_arr(px_df), d_sig=sig_df.date.values, tsm=tsmom_arr(sig_df.close.to_numpy(float), lookbacks))


def validate_indicators(frames_px, frames_sig, cfg, n_samples=400, seed=7):
    """V0: vectorised arrays must equal the real analyzer on trailing windows."""
    rng = random.Random(seed)
    lookbacks = [cfg.lookback_short, cfg.lookback_medium, cfg.lookback_long]
    worst = {"tsmom_mismatch": 0, "price_mismatch": 0, "atr_rel": 0.0, "vol_rel": 0.0, "n": 0}
    syms = list(frames_px)
    packs = {s: indicator_pack(frames_px[s], frames_sig[s], cfg) for s in syms}
    for _ in range(n_samples):
        s = rng.choice(syms)
        f, fs = frames_px[s], frames_sig[s]
        i = rng.randrange(len(f))
        win = f.iloc[max(0, i - WINDOW_BARS + 1): i + 1]
        an = TrendFollowingAnalyzer(win, lookbacks=lookbacks, atr_period=cfg.atr_period)
        j = int(np.searchsorted(fs.date.values, f.date.values[i], side="right") - 1)
        swin = fs.iloc[max(0, j - WINDOW_BARS + 1): j + 1]
        ts, _ = TrendFollowingAnalyzer(swin, lookbacks=lookbacks, atr_period=cfg.atr_period).compute_tsmom_signal()
        P = packs[s]
        worst["n"] += 1
        worst["tsmom_mismatch"] += int(ts != P["tsm"][j])
        worst["price_mismatch"] += int(an.get_current_price() != P["price"][i])
        for key, real, mine in (("atr_rel", an.compute_atr(), P["atr"][i]), ("vol_rel", an.compute_volatility(), P["vol"][i])):
            rel = abs(real - mine) / max(abs(real), 1e-12) if (real or mine) else 0.0
            worst[key] = max(worst[key], rel)
    return worst


def build_signals(order, packs, dates, cfg):
    """Per date: {sym: {price, atr, volatility, tsmom, combined}} exactly as _compute_all_signals builds it."""
    min_bars = cfg.lookback_short + 5
    out = []
    for d in dates:
        tsm, info = {}, {}
        for s in order:            # insertion order = watchlist order (CSMOM ties break on it, as live)
            P = packs.get(s)
            if P is None:
                continue
            i = int(np.searchsorted(P["d_px"], d, side="right") - 1)
            if i < 0 or i + 1 < min_bars or (d - P["d_px"][i]) > np.timedelta64(10, "D"):
                continue
            j = int(np.searchsorted(P["d_sig"], d, side="right") - 1)
            if j < 0:
                continue
            tsm[s] = P["tsm"][j]
            info[s] = (float(P["price"][i]), float(P["atr"][i]), float(P["vol"][i]))
        cs = rank_cross_sectional(tsm)
        out.append({s: {"price": info[s][0], "atr": info[s][1], "volatility": info[s][2], "tsmom": tsm[s],
                        "combined": compute_combined_signal(tsm[s], cs.get(s, 0.0), cfg.tsmom_weight, cfg.csmom_weight)}
                    for s in tsm})
    return out


# ------------------------------------------------------------------ simulator
def make_engine(overrides, watchlist):
    cfg = TradingConfig()
    cfg.symbols = watchlist
    for k, v in overrides.items():
        setattr(cfg, k, v)
    eng = DecisionEngine.__new__(DecisionEngine)
    eng.config = cfg
    return eng


def apply_null(sig, d, mode, rng, cache, min_vol):
    key = (d.year, d.month)
    if key not in cache:
        elig = [s for s, v in sig.items() if v["volatility"] >= min_vol]
        if mode == "sel":
            perm = elig[:]
            rng.shuffle(perm)
            cache[key] = dict(zip(elig, perm))
        else:
            cache[key] = {s: rng.uniform(0.5, 1.0) for s in elig}
    m = cache[key]
    out = {}
    for s, v in sig.items():
        v2 = dict(v)
        if mode == "sel":
            src = m.get(s)
            if src is not None and src in sig:
                v2["combined"] = sig[src]["combined"]
        else:
            v2["combined"] = m.get(s, 0.0)
        out[s] = v2
    return out


def span(dates, start, end):
    i0 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(start)), side="left"))
    i1 = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(end)), side="right") - 1)
    return i0, i1


def run_spec(spec):
    U = UNIV[spec["univ"]]
    dates = U["dates"]
    i0, i1 = span(dates, spec["start"], spec["end"])
    eng = make_engine(spec.get("cfg", {}), U["watchlist"])
    cfg = eng.config
    k = cfg.atr_stop_multiplier
    slip = spec.get("slip_bps", 5.0) / 1e4
    fee_min = spec.get("fee_min_usd", 4.0)
    fee_rate = spec.get("fee_rate", 0.0005)
    cap0 = spec.get("capital", START_NLV)
    daily_loss = spec.get("daily_loss", cfg.max_daily_loss)
    brakes = spec.get("brakes", "nohalt")
    null = spec.get("null")
    rng = random.Random(null[1]) if null else None
    cache = {}
    sigs = U["sig_" + spec.get("sigmode", "live")]
    O, H, L, C, col, gpu = U["O"], U["H"], U["L"], U["C"], U["col"], U["gbp_per_usd"]
    fx_for = U["fx_for"]

    cash = cap0
    unsettled = []
    pos, cooldown, last_close = {}, {}, {}
    peak, prev_nlv, halted_on = cap0, cap0, None
    st = dict(fees=0.0, orders=0, topups=0, topup_gated=0, reduce_days=0, loss_block_days=0, entries=0)
    trips = []
    n = i1 - i0 + 1
    curve, deploy, npos = np.zeros(n), np.zeros(n), np.zeros(n)

    def fee_of(notional_base, g):
        if fee_min == 0 and fee_rate == 0:
            return 0.0
        return max(fee_min, fee_rate * notional_base / g) * g

    def mark(g):
        gross = sum(p["qty"] * last_close[s] * fx_for(s, g) for s, p in pos.items())
        return cash + sum(a for _, a in unsettled) + gross, gross

    def close_pos(s, px, d, di, g, reason):
        nonlocal cash
        p = pos.pop(s)
        gross = p["qty"] * px * fx_for(s, g)
        f = fee_of(gross, g)
        st["fees"] += f
        st["orders"] += 1
        unsettled.append((di + 2, gross - f))
        trips.append(dict(sym=s, entry=str(p["entry"])[:10], exit=str(d)[:10], qty=p["qty"], cost=p["cost"],
                          pnl=gross - f - p["cost"], reason=reason, topups=p["topups"]))
        if reason == "stop":
            cooldown[s] = d + pd.Timedelta(days=cfg.reentry_cooldown_days)

    for j, di in enumerate(range(i0, i1 + 1)):
        d = pd.Timestamp(dates[di])
        g = gpu[di]
        if unsettled:
            cash += sum(a for t, a in unsettled if t <= di)
            unsettled = [(t, a) for t, a in unsettled if t > di]
        for s in list(pos):
            c_ = col[s]
            lo = L[c_, di]
            if np.isnan(lo):
                continue
            p = pos[s]
            if lo <= p["stop"]:
                o = O[c_, di]
                px = (o if o < p["stop"] else p["stop"]) * (1 - slip)
                close_pos(s, px, d, di, g, "stop")
            else:
                p["stop"] = max(p["stop"], H[c_, di] - p["trail"])
                last_close[s] = C[c_, di]
        nlv, gross = mark(g)

        if halted_on is None:
            peak = max(peak, nlv)
            dd = (peak - nlv) / peak
            if brakes == "live" and dd >= cfg.drawdown_halt_pct:
                for s in list(pos):
                    close_pos(s, last_close[s] * (1 - slip), d, di, g, "halt")
                halted_on = str(d)[:10]
            else:
                reduce = brakes != "off" and dd >= cfg.drawdown_reduce_pct
                st["reduce_days"] += int(reduce)
                if nlv - prev_nlv < -daily_loss:
                    st["loss_block_days"] += 1
                else:
                    sig = sigs[di]
                    if null:
                        sig = apply_null(sig, d, null[0], rng, cache, cfg.min_volatility)
                    sig = {s: v for s, v in sig.items() if v["price"] > 0 and v["atr"] > 0}
                    held = [SimpleNamespace(symbol=s, quantity=p["qty"]) for s, p in pos.items()]
                    cds = {s: str(u)[:10] for s, u in cooldown.items() if d < u}
                    eng.position_manager = SimpleNamespace(get_positions=lambda held=held: held)
                    eng.db = SimpleNamespace(get_active_cooldowns=lambda cds=cds: cds)
                    eng.connection = SimpleNamespace(get_fx_rates=lambda g=g: {"USD": g, "GBP": 1.0})
                    targets = eng._calculate_target_positions(sig, nlv)
                    if reduce:
                        for t in targets.values():
                            t["target_shares"] = int(t["target_shares"] * 0.5)
                    settled = cash
                    for s, t in targets.items():          # rank order, as bot._execute_live iterates
                        qt = t["target_shares"]
                        if qt <= 0 or np.isnan(C[col[s], di]):
                            continue
                        px0, f2b, atr = t["price"], t["fx_to_base"], t["atr"]
                        unit = px0 * f2b * (1 + cfg.settled_cash_buffer)
                        aff = int(max(settled, 0.0) / unit) if unit > 0 else 0
                        if s in pos:
                            p = pos[s]
                            if not p["qty"] < qt * (1 - cfg.topup_drift_threshold):
                                continue
                            delta = qt - p["qty"]
                            carried = max(p["stop"], px0 - k * atr)
                            if cfg.topup_min_stop_buffer_atr > 0 and (px0 - carried) < cfg.topup_min_stop_buffer_atr * atr - 1e-9:
                                st["topup_gated"] += 1
                                continue
                            q = delta
                            if aff < delta:
                                if aff <= 0 or (cfg.min_partial_topup_pct > 0 and aff / delta < cfg.min_partial_topup_pct):
                                    continue
                                q = aff
                            notional = q * px0 * (1 + slip) * f2b
                            f = fee_of(notional, g)
                            cash -= notional + f
                            settled -= q * unit
                            st["fees"] += f
                            st["orders"] += 1
                            st["topups"] += 1
                            p["qty"] += q
                            p["cost"] += notional + f
                            p["topups"] += 1
                            p["stop"] = max(p["stop"], px0 - k * atr)
                            p["trail"] = k * atr
                        else:
                            q = qt
                            if aff < qt:
                                if aff <= 0 or aff / qt < cfg.min_partial_entry_pct:
                                    continue
                                q = aff
                            notional = q * px0 * (1 + slip) * f2b
                            f = fee_of(notional, g)
                            cash -= notional + f
                            settled -= q * unit
                            st["fees"] += f
                            st["orders"] += 1
                            st["entries"] += 1
                            pos[s] = dict(qty=q, cost=notional + f, stop=px0 - k * atr, trail=k * atr, entry=d, topups=0)
                            last_close[s] = C[col[s], di]
            nlv, gross = mark(g)
        curve[j] = nlv
        deploy[j] = gross / nlv if nlv > 0 else 0.0
        npos[j] = len(pos)
        prev_nlv = nlv

    idx = pd.DatetimeIndex(dates[i0: i1 + 1])
    res = dict(name=spec["name"], univ=spec["univ"], start=str(idx[0])[:10], end=str(idx[-1])[:10],
               halted_on=halted_on, **perf(pd.Series(curve, index=idx), cap0), **trip_stats(trips),
               fees=st["fees"], orders=st["orders"], entries=st["entries"], topups=st["topups"],
               topup_gated=st["topup_gated"], reduce_pct=st["reduce_days"] / n, loss_block_days=st["loss_block_days"],
               deploy=float(deploy.mean()), npos=float(npos.mean()))
    years = res["years"]
    res["fee_pct_nlv_yr"] = st["fees"] / float(np.mean(curve)) / years if years > 0 else 0.0
    res["orders_per_yr"] = st["orders"] / years if years > 0 else 0.0
    if null:
        res["null"] = null[0]
        res["seed"] = null[1]
        res["monthly"] = pd.Series(curve, index=idx).resample("ME").last().round(2).tolist()
    if spec.get("keep"):
        res["curve"] = pd.Series(curve, index=idx)
        res["trip_list"] = trips
    return res


def perf(c, cap0):
    years = (c.index[-1] - c.index[0]).days / 365.25
    r = pd.concat([pd.Series([c.iloc[0] / cap0 - 1]), c.pct_change().iloc[1:]]).to_numpy()
    sd = r.std(ddof=1)
    peaks = np.maximum.accumulate(np.r_[cap0, c.to_numpy()])[1:]
    return dict(years=years, end_nlv=float(c.iloc[-1]), ret=float(c.iloc[-1] / cap0 - 1),
                cagr=float((c.iloc[-1] / cap0) ** (1 / years) - 1) if years > 0 else 0.0,
                vol=float(sd * math.sqrt(252)), sharpe=float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
                maxdd=float((c.to_numpy() / peaks - 1).min()))


def trip_stats(trips):
    if not trips:
        return dict(trips=0, win_rate=0.0, payoff=0.0, exp_per_trip=0.0, avg_win=0.0, avg_loss=0.0,
                    avg_trip_ret=0.0, med_trip_ret=0.0)
    p = np.array([t["pnl"] for t in trips])
    rr = np.array([t["pnl"] / t["cost"] for t in trips if t["cost"] > 0])
    w, l = p[p > 0], p[p <= 0]
    return dict(trips=len(p), win_rate=float(len(w) / len(p)), avg_win=float(w.mean()) if len(w) else 0.0,
                avg_loss=float(l.mean()) if len(l) else 0.0,
                payoff=float(w.mean() / -l.mean()) if len(w) and len(l) and l.mean() < 0 else 0.0,
                exp_per_trip=float(p.mean()), avg_trip_ret=float(rr.mean()), med_trip_ret=float(np.median(rr)))


# ------------------------------------------------------------------ benchmarks (GBP, same fee model)
def bench_series(U, weights_fn, i0, i1, cap0, rebalance, fees=True, slip_bps=5.0):
    """Generic long-only benchmark on adjusted closes. weights_fn(di) -> {sym: w}."""
    C, col, gpu = U["C"], U["col"], U["gbp_per_usd"]
    dates = U["dates"]
    cash, units, out = cap0, {}, []
    slip = slip_bps / 1e4
    last_key = None
    for di in range(i0, i1 + 1):
        d = pd.Timestamp(dates[di])
        g = gpu[di]
        px = {s: C[col[s], di] for s in U["syms"]}
        val = cash + sum(u * px[s] * g for s, u in units.items() if not np.isnan(px[s]))
        key = rebalance(d)
        if key != last_key:
            last_key = key
            w = {s: x for s, x in weights_fn(di).items() if not np.isnan(px[s])}
            tot = sum(w.values())
            for s in set(units) | set(w):
                if np.isnan(px[s]):
                    continue
                tgt_val = val * w.get(s, 0.0) / tot if tot else 0.0
                cur_val = units.get(s, 0.0) * px[s] * g
                dv = tgt_val - cur_val
                if abs(dv) < 1e-6 or (s not in w and cur_val == 0):
                    continue
                cost = (max(4.0, 0.0005 * abs(dv) / g) * g if fees else 0.0) + (abs(dv) * slip if fees else 0.0)
                units[s] = units.get(s, 0.0) + dv / (px[s] * g)
                cash -= dv + cost
            val = cash + sum(u * px[s] * g for s, u in units.items() if not np.isnan(px[s]))
        out.append(val)
    return pd.Series(out, index=pd.DatetimeIndex(dates[i0: i1 + 1]))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    cfg = TradingConfig()
    frozen = {k: getattr(cfg, k) for k in (
        "max_open_positions", "max_position_pct", "max_asset_class_pct", "max_gross_exposure", "atr_stop_multiplier",
        "atr_period", "min_volatility", "signal_threshold", "tsmom_weight", "csmom_weight", "lookback_short",
        "lookback_medium", "lookback_long", "risk_budget", "reentry_cooldown_days", "topup_drift_threshold",
        "settled_cash_buffer", "min_partial_entry_pct", "min_partial_topup_pct", "topup_min_stop_buffer_atr",
        "drawdown_reduce_pct", "drawdown_halt_pct", "max_daily_loss", "enable_shorting")}
    expect = dict(max_open_positions=3, max_position_pct=0.30, max_asset_class_pct=0.60, atr_stop_multiplier=3.0,
                  min_volatility=0.08, signal_threshold=0.5, reentry_cooldown_days=10, topup_min_stop_buffer_atr=1.0,
                  drawdown_halt_pct=0.20, max_daily_loss=200.0, enable_shorting=False)
    for k_, v in expect.items():
        assert frozen[k_] == v, f"config drift: {k_}={frozen[k_]} expected {v}"
    print("frozen live config:", frozen, flush=True)

    fx = pd.read_csv(f"{STUDY}/gbpusd.csv")
    fx["date"] = pd.to_datetime(fx["date"]).dt.normalize()
    fx = fx.set_index("date")["gbpusd"]

    # ---- proxy universe
    order_ucits = [s for v in UCITS_WATCHLIST.values() for s in v]
    px_adj, px_raw, missing = {}, {}, []
    for u in order_ucits:
        p = PROXY_OF.get(u)
        if p is None:
            continue
        a, r = load_bars("proxy_adj", p), load_bars("proxy_raw", p)
        if a is None:
            missing.append(p)
            continue
        px_adj[p] = a
        px_raw[p] = r if r is not None else a
    assert not missing, f"missing proxy bars: {missing}"
    splices = {}
    for p, alts in SPLICE_CANDIDATES.items():
        if p not in px_adj:
            continue
        best = None
        for a in alts:
            fa = load_bars("proxy_adj", a)
            if fa is None:
                continue
            joined, info = splice(px_adj[p], fa, a)
            if joined is not None and (best is None or info["ret_corr"] > best[1]["ret_corr"]):
                best = (a, info, joined)
        if best is None:
            splices[p] = "no usable candidate — history stays truncated"
            continue
        a, info, joined = best
        px_adj[p] = joined
        fr = load_bars("proxy_raw", a)
        rawj, _ = splice(px_raw[p], fr if fr is not None else load_bars("proxy_adj", a), a)
        if rawj is not None:
            px_raw[p] = rawj
        splices[p] = info
    print("splices:", json.dumps(splices), flush=True)
    split_flags = {}
    for p in px_adj:
        m = px_adj[p][["date", "close"]].merge(px_raw[p][["date", "close"]], on="date", suffixes=("_a", "_r"))
        ratio = (m.close_a / m.close_r)
        jumps = (ratio / ratio.shift(1) - 1).abs()
        if (jumps > 0.15).any():
            split_flags[p] = str(m.date[jumps.idxmax()].date())
    proxy_order = [PROXY_OF[u] for u in order_ucits if u in PROXY_OF]
    ucits_of = {v: k_ for k_, v in PROXY_OF.items()}
    watch_proxy = {c: [PROXY_OF[s] for s in v if s in PROXY_OF] for c, v in UCITS_WATCHLIST.items()}
    sig_frame_live = {p: (px_raw[p] if ucits_of[p] in DIST and p not in split_flags else px_adj[p]) for p in proxy_order}
    print("split artefacts in raw bars (fall back to adjusted):", split_flags, flush=True)

    v0 = validate_indicators({p: px_adj[p] for p in proxy_order}, sig_frame_live, cfg)
    print("V0 indicator validation (proxies):", v0, flush=True)
    assert v0["tsmom_mismatch"] == 0 and v0["price_mismatch"] == 0 and v0["atr_rel"] < 1e-9 and v0["vol_rel"] < 1e-9

    def build_univ(name, order, frames_px, sig_frames_by_mode, watchlist, fx_for):
        dates = np.array(sorted(set().union(*[set(f.date.values) for f in frames_px.values()])))
        col = {s: i for i, s in enumerate(order)}
        O = np.full((len(order), len(dates)), np.nan)
        H, L, C = O.copy(), O.copy(), O.copy()
        pos_of = {d: i for i, d in enumerate(dates)}
        for s, f in frames_px.items():
            ii = np.array([pos_of[d] for d in f.date.values])
            O[col[s], ii], H[col[s], ii], L[col[s], ii], C[col[s], ii] = f.open, f.high, f.low, f.close
        gpu = (1.0 / fx.reindex(pd.DatetimeIndex(dates), method="ffill").bfill()).to_numpy()
        U = dict(name=name, syms=order, dates=dates, col=col, O=O, H=H, L=L, C=C, gbp_per_usd=gpu,
                 watchlist=watchlist, fx_for=fx_for)
        for mode, sframes in sig_frames_by_mode.items():
            packs = {s: indicator_pack(frames_px[s], sframes[s], cfg) for s in order if s in frames_px}
            U["sig_" + mode] = build_signals(order, packs, dates, cfg)
        return U

    UNIV["proxy"] = build_univ("proxy", proxy_order, px_adj, {"live": sig_frame_live, "tr": px_adj},
                               watch_proxy, lambda s, g: g)
    print(f"proxy universe built ({len(UNIV['proxy']['dates'])} days) {time.time()-t0:.0f}s", flush=True)

    # ---- native UCITS universe (V1/V2)
    ucits = {s: load_bars("ucits", s) for s in order_ucits}
    ucits = {s: f for s, f in ucits.items() if f is not None}

    def fx_ucits(s, g):
        ccy = CONTRACT_REGISTRY[s][0]
        return g if ccy == "USD" else (0.01 if ccy == "GBX" else 1.0)
    UNIV["ucits"] = build_univ("ucits", [s for s in order_ucits if s in ucits], ucits, {"live": ucits},
                               UCITS_WATCHLIST, fx_ucits)
    print(f"ucits universe built {time.time()-t0:.0f}s", flush=True)

    if os.getenv("RECON"):
        # (a) reconcile with the 2026-08-28 class-cap study's 2Y window (it had no slippage)
        for name, univ, a, b, kw in (("0828 window ucits slip0", "ucits", "2024-09-02", "2026-08-28", dict(slip_bps=0.0)),
                                     ("0828 window ucits slip5", "ucits", "2024-09-02", "2026-08-28", {}),
                                     ("0828 window proxy slip0", "proxy", "2024-09-02", "2026-08-28", dict(slip_bps=0.0))):
            r = run_spec(dict(name=name, univ=univ, start=a, end=b, brakes="off", **kw))
            print(f"RECON {name:24s} ret {r['ret']:+.1%} cagr {r['cagr']:+.1%} dd {r['maxdd']:.1%} trips {r['trips']} "
                  f"win {r['win_rate']:.0%} orders/yr {r['orders_per_yr']:.0f} fees {r['fee_pct_nlv_yr']:.1%}/yr "
                  f"dep {r['deploy']:.0%} topups {r['topups']} gated {r['topup_gated']}", flush=True)
        # (b) do the SEL nulls trade like the strategy? (a turnover gap alone could favour them)
        oos = ("2008-01-09", "2023-12-29")
        for label, kw in (("live costs", {}), ("frictionless", dict(slip_bps=0.0, fee_min_usd=0.0, fee_rate=0.0))):
            base = run_spec(dict(name="strat", univ="proxy", start=oos[0], end=oos[1], brakes="off", **kw))
            ns = [run_spec(dict(name="null", univ="proxy", start=oos[0], end=oos[1], brakes="off", null=("sel", sd), **kw))
                  for sd in range(30)]

            def med(k):
                return float(np.median([x[k] for x in ns]))
            print(f"TURNOVER {label:12s} strat orders/yr {base['orders_per_yr']:.1f} trips {base['trips']} win {base['win_rate']:.1%} "
                  f"tripret {base['avg_trip_ret']:+.2%} dep {base['deploy']:.0%} npos {base['npos']:.2f} topups {base['topups']} | "
                  f"null_sel median orders/yr {med('orders_per_yr'):.1f} trips {med('trips'):.0f} win {med('win_rate'):.1%} "
                  f"tripret {med('avg_trip_ret'):+.2%} dep {med('deploy'):.0%} npos {med('npos'):.2f} topups {med('topups'):.0f}", flush=True)
        return

    # V1: replay vs the live bot's saved signals
    v1 = {}
    dbs = pd.read_csv(f"{STUDY}/db_signals.csv")
    dbs["signal_date"] = pd.to_datetime(dbs.signal_date)
    Uu = UNIV["ucits"]
    dbs = dbs[dbs.signal_date <= LAST_DATE]

    def top3(sc):
        ok = [(s_, v) for s_, v in sc.items() if v >= cfg.signal_threshold]
        return set(s_ for s_, _ in sorted(ok, key=lambda kv: -kv[1])[:3])

    def v1_compare(lag):
        """Re-rank CSMOM over the symbols the bot actually had that day; lag 0 = that day's
        full bar (the bot saw a 14:00 partial one), lag 1 = the previous close."""
        rows, top_agree, top_n = [], 0, 0
        for d, grp in dbs.groupby("signal_date"):
            g = grp.drop_duplicates("symbol", keep="last").set_index("symbol")
            di = int(np.searchsorted(Uu["dates"], np.datetime64(d), side="right") - 1) - lag
            base = Uu["sig_live"][di]
            syms = [s_ for s_ in Uu["syms"] if s_ in g.index and s_ in base]
            tsm = {s_: base[s_]["tsmom"] for s_ in syms}
            cs = rank_cross_sectional(tsm)
            mine = {s_: compute_combined_signal(tsm[s_], cs.get(s_, 0.0), cfg.tsmom_weight, cfg.csmom_weight) for s_ in syms}
            for s_ in syms:
                pm = abs(g.at[s_, "price"] - base[s_]["price"]) <= 1e-4 * abs(base[s_]["price"])
                rows.append((g.at[s_, "tsmom_score"] == tsm[s_], abs(g.at[s_, "combined_score"] - mine[s_]), pm))
            db_sc = {s_: g.at[s_, "combined_score"] for s_ in syms if g.at[s_, "volatility"] >= cfg.min_volatility}
            my_sc = {s_: mine[s_] for s_ in syms if base[s_]["volatility"] >= cfg.min_volatility}
            top_agree += int(top3(db_sc) == top3(my_sc))
            top_n += 1
        pm_rows = [a for a, _, c in rows if c]
        return dict(pairs=len(rows), tsmom_exact=float(np.mean([a for a, _, _ in rows])),
                    combined_mae=float(np.mean([b for _, b, _ in rows])), top3_same_days=top_agree, days=top_n,
                    price_match_rate=float(np.mean([c for _, _, c in rows])),
                    tsmom_exact_when_price_matches=float(np.mean(pm_rows)) if pm_rows else None)
    v1 = {"same_day_bar": v1_compare(0), "prior_day_bar": v1_compare(1)}
    print("V1 replay vs live instrument_signals:", v1, flush=True)

    # V2: proxy vs UCITS fidelity over common history
    Up = UNIV["proxy"]
    weekly = {}
    for u, f in ucits.items():
        p = PROXY_OF.get(u)
        if p is None:
            continue
        ccy = CONTRACT_REGISTRY[u][0]
        uc = f.set_index("date").close
        if ccy == "GBP":
            uc = uc * fx.reindex(uc.index, method="ffill")
        elif ccy == "GBX":
            uc = uc * fx.reindex(uc.index, method="ffill") / 100
        pc = px_adj[p].set_index("date").close
        both = pd.concat([uc.resample("W-FRI").last(), pc.resample("W-FRI").last()], axis=1, keys=["u", "p"], sort=True).dropna()
        rr = both.pct_change().dropna()
        weekly[u] = dict(proxy=p, weeks=len(rr), corr=float(rr.u.corr(rr.p)) if len(rr) > 20 else None,
                         ann_ret_u=float((1 + rr.u).prod() ** (52 / len(rr)) - 1) if len(rr) > 20 else None,
                         ann_ret_p=float((1 + rr.p).prod() ** (52 / len(rr)) - 1) if len(rr) > 20 else None)
    n_full = []
    for di, d in enumerate(Uu["dates"]):
        cnt = sum(1 for s, f in ucits.items() if np.searchsorted(f.date.values, d, side="right") >= 254)
        n_full.append(cnt)
    common_start = pd.Timestamp(Uu["dates"][next(i for i, c in enumerate(n_full) if c >= 20)])
    agree, jac = [], []
    for di_u, d in enumerate(Uu["dates"]):
        if d < np.datetime64(common_start):
            continue
        di_p = int(np.searchsorted(Up["dates"], d, side="right") - 1)
        su, sp = Uu["sig_live"][di_u], Up["sig_live"][di_p]
        pass_u = {s for s, v in su.items() if v["combined"] >= 0.5 and v["volatility"] >= 0.08}
        pass_p = {ucits_of[s] for s, v in sp.items() if v["combined"] >= 0.5 and v["volatility"] >= 0.08}
        names = [s for s in su if s in PROXY_OF and PROXY_OF[s] in sp]
        agree.extend(int((s in pass_u) == (s in pass_p)) for s in names)
        tu = set(sorted([s for s in pass_u if s in PROXY_OF], key=lambda s: -su[s]["combined"])[:3])
        tp = set(sorted(pass_p, key=lambda s: -sp[PROXY_OF[s]]["combined"])[:3])
        if tu or tp:
            jac.append(len(tu & tp) / len(tu | tp))
    v2 = dict(weekly=weekly, common_start=str(common_start.date()), pass_agree=float(np.mean(agree)),
              top3_jaccard=float(np.mean(jac)))
    print("V2 fidelity:", json.dumps({k_: v for k_, v in v2.items() if k_ != "weekly"}), flush=True)
    for u, w in weekly.items():
        print(f"   {u:5s} vs {w['proxy']:5s} weeks {w['weeks']:4d} corr {w['corr'] if w['corr'] is None else round(w['corr'], 3)} "
              f"ann u {w['ann_ret_u'] if w['ann_ret_u'] is None else round(w['ann_ret_u'], 3)} "
              f"p {w['ann_ret_p'] if w['ann_ret_p'] is None else round(w['ann_ret_p'], 3)}", flush=True)

    # ---- periods
    dates = Up["dates"]
    full254 = [sum(1 for s in proxy_order if (np.searchsorted(px_adj[s].date.values, d, side="right")) >= 254) for d in dates]
    study_start = pd.Timestamp(dates[next(i for i, c in enumerate(full254) if c >= 18)])
    P = {
        "ALL": (study_start, LAST_DATE),
        "OOS": (study_start, "2023-12-29"),
        "OOS_A": (study_start, "2012-12-31"),
        "OOS_B": ("2013-01-01", "2017-12-29"),
        "OOS_C": ("2018-01-01", "2023-12-29"),
        "IS": ("2024-01-01", LAST_DATE),
        "COMMON": (common_start, LAST_DATE),
    }
    print("periods:", {k_: (str(pd.Timestamp(a).date()), str(pd.Timestamp(b).date())) for k_, (a, b) in P.items()}, flush=True)

    specs = []

    def add(name, period, **kw):
        s, e = P[period]
        specs.append(dict(name=name, period=period, start=s, end=e, univ=kw.pop("univ", "proxy"), **kw))
    FRIC = dict(slip_bps=0.0, fee_min_usd=0.0, fee_rate=0.0)
    K50 = dict(capital=50000.0, daily_loss=200.0 * 50000.0 / START_NLV)
    add("strat_live_brakes", "ALL", brakes="live", keep=True)
    add("strat", "ALL", keep=True)
    add("strat_no_brakes", "ALL", brakes="off", keep=True)
    for per in ("OOS", "OOS_A", "OOS_B", "OOS_C", "IS"):
        add("strat", per, keep=(per == "OOS"))
        add("strat_no_brakes", per, brakes="off", keep=(per == "OOS"))
    add("strat_live_brakes", "OOS", brakes="live")
    add("strat_frictionless", "OOS", **FRIC)
    add("strat_frictionless_no_brakes", "OOS", brakes="off", **FRIC)
    add("strat_50k", "OOS", **K50)
    add("strat_50k_no_brakes", "OOS", brakes="off", **K50)
    add("strat_slip15_no_brakes", "OOS", brakes="off", slip_bps=15.0)
    add("strat_tr_signals_no_brakes", "OOS", brakes="off", sigmode="tr")
    add("nb_threshold_0.3", "OOS", brakes="off", cfg=dict(signal_threshold=0.3))
    add("nb_threshold_0.7", "OOS", brakes="off", cfg=dict(signal_threshold=0.7))
    add("nb_atr_2x", "OOS", brakes="off", cfg=dict(atr_stop_multiplier=2.0))
    add("nb_atr_4x", "OOS", brakes="off", cfg=dict(atr_stop_multiplier=4.0))
    add("nb_5slots_18pct", "OOS", brakes="off", cfg=dict(max_open_positions=5, max_position_pct=0.18))
    add("fid_proxy", "COMMON", brakes="off", keep=True)
    add("fid_ucits", "COMMON", univ="ucits", brakes="off", keep=True)
    for seed in range(N_SEEDS):
        add("null_sel", "OOS", null=("sel", seed))
        add("null_sel_no_brakes", "OOS", brakes="off", null=("sel", seed))
        add("null_all_no_brakes", "OOS", brakes="off", null=("all", seed))
        add("null_sel_frictionless_no_brakes", "OOS", brakes="off", null=("sel", seed), **FRIC)
        add("null_sel_50k_no_brakes", "OOS", brakes="off", null=("sel", seed), **K50)
        add("null_sel_no_brakes", "IS", brakes="off", null=("sel", seed))

    t1 = time.time()
    t_one = run_spec(specs[1])
    print(f"one ALL run took {time.time()-t1:.1f}s; {len(specs)} specs on {NPROC} procs", flush=True)
    with Pool(NPROC) as pool:
        results = pool.map(run_spec, specs, chunksize=4)
    for r, s in zip(results, specs):
        r["period"] = s["period"]
    print(f"simulations done {time.time()-t0:.0f}s", flush=True)

    # ---- benchmarks per period
    bench = {}
    Up_syms = proxy_order

    def spy_w(di):
        return {"SPY": 1.0}

    def w6040(di):
        return {"SPY": 0.6, "IEF": 0.4}

    def ew(di):
        return {s: 1.0 for s in Up_syms if not np.isnan(Up["C"][Up["col"][s], di])}
    for per in ("ALL", "OOS", "OOS_A", "OOS_B", "OOS_C", "IS"):
        i0, i1 = span(dates, *P[per])
        bench[per] = {
            "SPY_bh": bench_series(Up, spy_w, i0, i1, START_NLV, lambda d: 0),
            "6040_annual": bench_series(Up, w6040, i0, i1, START_NLV, lambda d: d.year),
            "EW_monthly_frictionless": bench_series(Up, ew, i0, i1, START_NLV, lambda d: (d.year, d.month), fees=False),
        }

    # ---- assemble
    def pick(name, period):
        return next(r for r in results if r["name"] == name and r["period"] == period and "null" not in r)
    table = []
    for r in results:
        if "null" in r:
            continue
        table.append({k_: v for k_, v in r.items() if k_ not in ("curve", "trip_list")})
    btable = []
    for per, bs in bench.items():
        for bname, c in bs.items():
            btable.append(dict(name=bname, period=per, **perf(c, START_NLV)))

    nulls = {}
    for (nname, per, cmp_name) in (("null_sel", "OOS", "strat"),
                                   ("null_sel_no_brakes", "OOS", "strat_no_brakes"),
                                   ("null_all_no_brakes", "OOS", "strat_no_brakes"),
                                   ("null_sel_frictionless_no_brakes", "OOS", "strat_frictionless_no_brakes"),
                                   ("null_sel_50k_no_brakes", "OOS", "strat_50k_no_brakes"),
                                   ("null_sel_no_brakes", "IS", "strat_no_brakes")):
        rs = [r for r in results if r["name"] == nname and r["period"] == per]
        strat = pick(cmp_name, per)
        cagr = np.array([r["cagr"] for r in rs])
        sh = np.array([r["sharpe"] for r in rs])
        mdd = np.array([r["maxdd"] for r in rs])
        tret = np.array([r["avg_trip_ret"] for r in rs])
        nulls[f"{nname}@{per}"] = dict(
            n=len(rs), vs=cmp_name, strat_cagr=strat["cagr"], strat_sharpe=strat["sharpe"], strat_maxdd=strat["maxdd"],
            null_cagr_p5=float(np.percentile(cagr, 5)), null_cagr_p50=float(np.median(cagr)), null_cagr_p95=float(np.percentile(cagr, 95)),
            null_sharpe_p5=float(np.percentile(sh, 5)), null_sharpe_p50=float(np.median(sh)), null_sharpe_p95=float(np.percentile(sh, 95)),
            null_maxdd_p50=float(np.median(mdd)), strat_trip_ret=strat["avg_trip_ret"],
            null_trip_ret_p5=float(np.percentile(tret, 5)), null_trip_ret_p50=float(np.median(tret)),
            null_trip_ret_p95=float(np.percentile(tret, 95)),
            p_trip_ret=float((1 + (tret >= strat["avg_trip_ret"]).sum()) / (len(rs) + 1)),
            p_cagr=float((1 + (cagr >= strat["cagr"]).sum()) / (len(rs) + 1)),
            p_sharpe=float((1 + (sh >= strat["sharpe"]).sum()) / (len(rs) + 1)),
            monthly_p5=np.percentile(np.array([r["monthly"] for r in rs]), 5, axis=0).round(2).tolist(),
            monthly_p50=np.percentile(np.array([r["monthly"] for r in rs]), 50, axis=0).round(2).tolist(),
            monthly_p95=np.percentile(np.array([r["monthly"] for r in rs]), 95, axis=0).round(2).tolist(),
        )

    # crisis windows and calendar years on the ALL run (no terminal halt) vs benchmarks
    strat_all = pick("strat", "ALL")["curve"]
    nb_all = pick("strat_no_brakes", "ALL")["curve"]
    live_all = pick("strat_live_brakes", "ALL")["curve"]
    ball = bench["ALL"]
    crises = {}
    for label, (a, b) in CRISES.items():
        def wret(c):
            s = c[(c.index >= a) & (c.index <= b)]
            return float(s.iloc[-1] / s.iloc[0] - 1) if len(s) > 1 else None
        crises[label] = dict(strat=wret(strat_all), strat_no_brakes=wret(nb_all), SPY=wret(ball["SPY_bh"]), p6040=wret(ball["6040_annual"]),
                             EW=wret(ball["EW_monthly_frictionless"]))
    years = {}
    for y in range(pd.Timestamp(study_start).year + 1, 2027):
        def yret(c):
            s = c[c.index.year == y]
            prev = c[c.index.year == y - 1]
            return float(s.iloc[-1] / prev.iloc[-1] - 1) if len(s) and len(prev) else None
        years[y] = dict(strat=yret(strat_all), strat_no_brakes=yret(nb_all), SPY=yret(ball["SPY_bh"]), p6040=yret(ball["6040_annual"]),
                        EW=yret(ball["EW_monthly_frictionless"]))

    trips_oos = pick("strat", "OOS")["trip_list"]
    pd.DataFrame(trips_oos).to_csv(f"{OUT}/trips_oos.csv", index=False)
    curves = pd.DataFrame({"strat_all": strat_all, "strat_no_brakes_all": nb_all, "strat_live_brakes_all": live_all,
                           "SPY_bh": ball["SPY_bh"], "p6040": ball["6040_annual"], "EW": ball["EW_monthly_frictionless"]})
    curves.resample("W-FRI").last().round(2).to_csv(f"{OUT}/curves_weekly.csv")
    fidc = pd.DataFrame({"proxy": pick("fid_proxy", "COMMON")["curve"], "ucits": pick("fid_ucits", "COMMON")["curve"]}).dropna()
    fid_corr = float(fidc.pct_change().dropna().corr().iloc[0, 1])
    fid_wcorr = float(fidc.resample("W-FRI").last().pct_change().dropna().corr().iloc[0, 1])

    out = dict(frozen_config=frozen, split_flags=split_flags, splices=splices, V0=v0, V1=v1, V2=v2,
               fidelity_strategy=dict(daily_corr=fid_corr, weekly_corr=fid_wcorr,
                                      proxy={k_: pick("fid_proxy", "COMMON")[k_] for k_ in ("cagr", "sharpe", "maxdd", "trips", "win_rate")},
                                      ucits={k_: pick("fid_ucits", "COMMON")[k_] for k_ in ("cagr", "sharpe", "maxdd", "trips", "win_rate")}),
               periods={k_: (str(pd.Timestamp(a).date()), str(pd.Timestamp(b).date())) for k_, (a, b) in P.items()},
               strategies=table, benchmarks=btable, nulls=nulls, crises=crises, years=years,
               null_month_index=[str(x.date()) for x in pick("strat", "OOS")["curve"].resample("ME").last().index],
               runtime_s=time.time() - t0, n_seeds=N_SEEDS)
    json.dump(out, open(f"{OUT}/results.json", "w"), indent=1, default=str)

    # ---- print summary
    def fmt(r):
        return (f"{r['name']:24s} {r['period']:6s} cagr {r['cagr']:+6.1%} sh {r['sharpe']:+5.2f} dd {r['maxdd']:6.1%} "
                f"vol {r['vol']:5.1%} end £{r['end_nlv']:>9,.0f}")
    print("\n=== STRATEGIES ===")
    for r in table:
        extra = (f" trips {r['trips']:4d} win {r['win_rate']:5.1%} payoff {r['payoff']:4.2f} exp £{r['exp_per_trip']:+6.2f} "
                 f"fees {r['fee_pct_nlv_yr']:5.1%}/yr orders/yr {r['orders_per_yr']:5.1f} dep {r['deploy']:4.0%} "
                 f"reduce {r['reduce_pct']:4.0%} lossblock {r['loss_block_days']:3d} tripret {r['avg_trip_ret']:+.2%} halted {r['halted_on']}")
        print(fmt(r) + extra)
    print("\n=== BENCHMARKS ===")
    for r in btable:
        print(fmt(r))
    print("\n=== NULLS ===")
    for k_, v in nulls.items():
        print(f"{k_:32s} n={v['n']} strat cagr {v['strat_cagr']:+.1%} sh {v['strat_sharpe']:+.2f} | null cagr p5/50/95 "
              f"{v['null_cagr_p5']:+.1%}/{v['null_cagr_p50']:+.1%}/{v['null_cagr_p95']:+.1%} sharpe "
              f"{v['null_sharpe_p5']:+.2f}/{v['null_sharpe_p50']:+.2f}/{v['null_sharpe_p95']:+.2f} | p(cagr) {v['p_cagr']:.3f} p(sharpe) {v['p_sharpe']:.3f} "
              f"| trip ret strat {v['strat_trip_ret']:+.2%} null p5/50/95 {v['null_trip_ret_p5']:+.2%}/{v['null_trip_ret_p50']:+.2%}/{v['null_trip_ret_p95']:+.2%} p {v['p_trip_ret']:.3f}")
    print("\n=== CRISES (ALL run, no terminal halt) ===")
    for k_, v in crises.items():
        print(f"{k_:24s} " + " ".join(f"{n_} {x:+.1%}" if x is not None else f"{n_} n/a" for n_, x in v.items()))
    print("\n=== YEARS ===")
    for y, v in years.items():
        print(f"{y} " + " ".join(f"{n_} {x:+.1%}" if x is not None else f"{n_} n/a" for n_, x in v.items()))
    print(f"\nfidelity strategy corr daily {fid_corr:.3f} weekly {fid_wcorr:.3f}; runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
