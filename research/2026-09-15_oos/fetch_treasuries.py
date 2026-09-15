#!/usr/bin/env python
"""
Second fetch for the 2026-09-15 OOS study: IBKR's history for the iShares Treasury
ETFs is truncated (TLT from 2016-02, IEF/SHY from 2017-08 — their primary listing
moved), so the 2008-2016 universe had no Treasuries. This pulls (read-only, paced
like fetch_bars.py) the same ETFs pinned to their old ARCA listing, plus
long-history substitutes, for oos_study.py to splice on before the first real bar.

  docker run --rm --user root --network host -v /root/ibkr_research/oos:/study \
      ibkr_bot-trading-bot:latest python /study/fetch_treasuries.py
"""
import os
import time

import pandas as pd
from ib_insync import IB, Stock, util

OUT = "/study"
BARS = f"{OUT}/bars"
CLIENT_ID = 23
SPACING = 12.0
QUIET_CODES = {2104, 2106, 2107, 2108, 2119, 2158}

# (file key, contract)
REQUESTS = [
    ("TLT-ARCA", Stock("TLT", "ARCA", "USD")),
    ("IEF-ARCA", Stock("IEF", "ARCA", "USD")),
    ("SHY-ARCA", Stock("SHY", "ARCA", "USD")),
] + [(s, Stock(s, "SMART", "USD")) for s in
     ("SPTL", "VGLT", "TLH", "EDV", "SPTI", "IEI", "VGIT", "SCHR", "BIL", "SCHO", "VGSH", "SPTS")]

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


for key, contract in REQUESTS:
    q = ib.qualifyContracts(contract)
    if not q:
        say("qualify failed", key, errors[-1:] if errors else "")
        continue
    c = q[0]
    for kind, what in (("proxy_adj", "ADJUSTED_LAST"), ("proxy_raw", "TRADES")):
        path = f"{BARS}/{kind}_{key}.csv"
        if os.path.exists(path):
            say("skip (exists)", kind, key)
            continue
        pace()
        n_err = len(errors)
        bars = ib.reqHistoricalData(c, endDateTime="", durationStr="20 Y", barSizeSetting="1 day",
                                    whatToShow=what, useRTH=True, formatDate=1, timeout=240)
        df = util.df(bars) if bars else None
        if df is None or df.empty:
            say("NO DATA", kind, key, errors[n_err:])
            continue
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df.insert(0, "symbol", key)
        df.to_csv(path, index=False)
        say(f"{kind:9s} {key:9s} conId {c.conId} {len(df):5d} bars {df.date.iloc[0]} -> {df.date.iloc[-1]}")

say("treasury fetch done; non-informational errors:", len(errors))
ib.disconnect()
