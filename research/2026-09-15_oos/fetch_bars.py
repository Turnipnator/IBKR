#!/usr/bin/env python
"""
Bar fetch for the 2026-09-15 out-of-sample study (read-only session).

Pulls, from the live IB Gateway on a spare clientId with readonly=True:
  - UCITS contract metadata (longName, priceMagnifier) for the live universe
  - 20Y of daily TRADES bars for every UCITS line (whatever history IBKR has)
  - 20Y of daily ADJUSTED_LAST (dividend-adjusted) and TRADES bars for the
    US-listed ETFs used as pre-UCITS proxies
  - 20Y of GBPUSD daily midpoints

Requests are spaced SPACING seconds apart (<= 50 per 10 min, under IBKR's
60-per-10-min historical pacing limit) so the bot's own 5-minute data probe
never trips a pacing violation -> probe failure -> gateway restart.
One CSV per (kind, symbol) under /study/bars; existing files are skipped,
so the script can be re-run to resume.

Run on the VPS:
  docker run --rm --user root --network host -v /root/ibkr_research/oos:/study \
      ibkr_bot-trading-bot:latest python /study/fetch_bars.py
"""
import os
import sys
import time

sys.path.insert(0, "/app")
import pandas as pd
from ib_insync import IB, Stock, Forex, util

from src.contracts import CONTRACT_REGISTRY, resolve_contract

OUT = "/study"
BARS = f"{OUT}/bars"
CLIENT_ID = 23
SPACING = 12.0

PROXIES = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ", "FXI", "ASHR",
    "TLT", "IEF", "SHY", "LQD", "HYG", "EMB", "TIP",
    "GLD", "SLV", "USO", "UNG", "DBA", "DBB", "PDBC", "DJP", "GSG", "DBC", "CPER", "JJC",
    "VNQ",
]
QUIET_CODES = {2104, 2106, 2107, 2108, 2119, 2158}

os.makedirs(BARS, exist_ok=True)
logf = open(f"{OUT}/fetch.log", "a")


def say(*parts):
    msg = f"{time.strftime('%H:%M:%S')} " + " ".join(str(p) for p in parts)
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


ib = IB()
ib.connect("127.0.0.1", 4001, clientId=CLIENT_ID, readonly=True, timeout=20)
errors = []
ib.errorEvent += lambda reqId, code, msg, c: (
    None if code in QUIET_CODES else errors.append((code, msg[:160]))
)
_last = [0.0]


def pace():
    wait = SPACING - (time.time() - _last[0])
    if wait > 0:
        ib.sleep(wait)
    _last[0] = time.time()


def qualify_us(sym):
    for c in (Stock(sym, "SMART", "USD"),
              Stock(sym, "SMART", "USD", primaryExchange="ARCA"),
              Stock(sym, "SMART", "USD", primaryExchange="NASDAQ"),
              Stock(sym, "SMART", "USD", primaryExchange="NYSE")):
        q = ib.qualifyContracts(c)
        if q:
            return q[0]
    return None


def fetch(kind, sym, contract, what):
    path = f"{BARS}/{kind}_{sym}.csv"
    if os.path.exists(path):
        say("skip (exists)", kind, sym)
        return
    pace()
    n_err = len(errors)
    bars = ib.reqHistoricalData(contract, endDateTime="", durationStr="20 Y",
                                barSizeSetting="1 day", whatToShow=what,
                                useRTH=True, formatDate=1, timeout=240)
    df = util.df(bars) if bars else None
    if df is None or df.empty:
        say("NO DATA", kind, sym, errors[n_err:])
        return
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df.insert(0, "symbol", sym)
    df.to_csv(path, index=False)
    say(f"{kind:9s} {sym:5s} {len(df):5d} bars {df.date.iloc[0]} -> {df.date.iloc[-1]}")


# ---- UCITS metadata + native bars
meta = []
ucits_contracts = {}
for sym in CONTRACT_REGISTRY:
    q = ib.qualifyContracts(resolve_contract(sym))
    if not q:
        say("UCITS qualify failed", sym)
        continue
    ucits_contracts[sym] = q[0]
    cd = ib.reqContractDetails(q[0])
    d = cd[0] if cd else None
    meta.append({"symbol": sym, "conId": q[0].conId, "currency": q[0].currency,
                 "longName": d.longName if d else "", "priceMagnifier": d.priceMagnifier if d else None})
pd.DataFrame(meta).to_csv(f"{OUT}/ucits_meta.csv", index=False)
say("ucits meta written", len(meta))

pace()
fx = ib.reqHistoricalData(Forex("GBPUSD"), endDateTime="", durationStr="20 Y",
                          barSizeSetting="1 day", whatToShow="MIDPOINT", useRTH=True,
                          formatDate=1, timeout=240)
fxdf = util.df(fx)[["date", "close"]].rename(columns={"close": "gbpusd"})
fxdf.to_csv(f"{OUT}/gbpusd.csv", index=False)
say("gbpusd", len(fxdf), fxdf.date.iloc[0], "->", fxdf.date.iloc[-1])

for sym, c in ucits_contracts.items():
    fetch("ucits", sym, c, "TRADES")

# ---- US proxies: adjusted (total return) and raw (price-only) bars
for sym in PROXIES:
    c = qualify_us(sym)
    if c is None:
        say("PROXY qualify failed", sym, errors[-1:] if errors else "")
        continue
    fetch("proxy_adj", sym, c, "ADJUSTED_LAST")
    fetch("proxy_raw", sym, c, "TRADES")

say("done; non-informational errors:", len(errors))
for e in errors:
    say("  ", e)
ib.disconnect()
