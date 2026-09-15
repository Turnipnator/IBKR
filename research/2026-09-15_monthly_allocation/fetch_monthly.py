#!/usr/bin/env python
"""
Bar fetch for attempts 2–4 (read-only session, paced like ../2026-09-15_oos/fetch_bars.py).

The other inputs (SPY, EFA, IEF + IEF-ARCA, VNQ, GSG, BIL) were fetched for the 2026-09-15
out-of-sample study and are reused from the same bars directory. This adds the dual-momentum
inputs, each also requested on its ARCA listing in case IBKR's default contract has truncated
history (as TLT/IEF/SHY did). Only dividend-adjusted bars are needed.

  docker run --rm --user root --network host -v /root/ibkr_research/oos:/study \
      -v /root/ibkr_research/monthly:/monthly ibkr_bot-trading-bot:latest \
      python /monthly/fetch_monthly.py
"""
import os
import time

from ib_insync import IB, Stock, util

BARS = "/study/bars"
CLIENT_ID = 23
SPACING = 12.0
QUIET_CODES = {2104, 2106, 2107, 2108, 2119, 2158}

REQUESTS = [
    ("VEU", Stock("VEU", "SMART", "USD")),
    ("VEU-ARCA", Stock("VEU", "ARCA", "USD")),
    ("ACWX", Stock("ACWX", "SMART", "USD")),
    ("ACWX-ARCA", Stock("ACWX", "ARCA", "USD")),
    ("AGG", Stock("AGG", "SMART", "USD")),
    ("AGG-ARCA", Stock("AGG", "ARCA", "USD")),
]

logf = open("/monthly/fetch.log", "a")


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
    path = f"{BARS}/proxy_adj_{key}.csv"
    if os.path.exists(path):
        say("skip (exists)", key)
        continue
    q = ib.qualifyContracts(contract)
    if not q:
        say("qualify failed", key, errors[-1:] if errors else "")
        continue
    c = q[0]
    pace()
    n_err = len(errors)
    bars = ib.reqHistoricalData(c, endDateTime="", durationStr="20 Y", barSizeSetting="1 day",
                                whatToShow="ADJUSTED_LAST", useRTH=True, formatDate=1, timeout=240)
    df = util.df(bars) if bars else None
    if df is None or df.empty:
        say("NO DATA", key, errors[n_err:])
        continue
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df.insert(0, "symbol", key)
    df.to_csv(path, index=False)
    say(f"{key:9s} conId {c.conId} {len(df):5d} bars {df.date.iloc[0]} -> {df.date.iloc[-1]}")

say("monthly fetch done; non-informational errors:", len(errors))
ib.disconnect()
