#!/usr/bin/env python
"""
Attempt 13 data fetch: earnings announcement dates from SEC EDGAR.

Every 8-K tagged item 2.02 ("Results of Operations and Financial Condition") for each name's CIK, with
EDGAR's own acceptanceDateTime — which is what decides whether the market could react that session or the
next one. No vendor, no API key, no estimates: the filing itself.

Predecessor CIKs matter and were found by checking, not guessing: Alphabet's filings before October 2015 sit
under Google Inc, Broadcom's under two earlier entities, Accenture plc's under Accenture Ltd and Accenture SCA,
and the XOM ticker now resolves to a 2026 re-registration holding a single filing while twenty years of Exxon
Mobil sit under the old CIK. Without these the early years go silently missing, which the study's per-year
coverage check would then report as a data failure.

Writes /root/ibkr_research/pead/earnings/<SYM>.csv with columns: symbol, filing_date, acceptance_utc, accession.
"""
import csv
import json
import os
import time
import urllib.request

OUT = os.getenv("OUT", "/root/ibkr_research/pead/earnings")
UA = "IBKR-Bot research script"
UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "XOM", "UNH", "JNJ", "V",
            "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "COST", "WMT", "BAC", "CRM", "MCD", "CSCO",
            "ACN", "LIN", "ADBE", "TMO", "ABT", "DHR", "WFC", "VZ", "TXN", "NEE", "PM", "INTC", "IBM"]
EXTRA_CIKS = {                      # predecessor entities, so the early years are not silently empty
    "GOOGL": ["0001288776"],        # Google Inc, before the 2015 Alphabet reorganisation
    "AVGO": ["0001649338", "0001441634"],   # Broadcom Ltd, Avago Technologies
    "ACN": ["0001134538", "0001143908"],    # Accenture Ltd and Accenture SCA, before the 2009 Irish plc
    "LIN": ["0000884905"],          # Praxair, before the 2018 Linde merger
    "XOM": ["0000034088"],          # the long-standing Exxon Mobil Corp entity; the ticker now maps to a 2026 re-registration
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        import gzip
        raw = gzip.decompress(raw)
    return json.loads(raw)


def ticker_map():
    d = get("https://www.sec.gov/files/company_tickers.json")
    return {v["ticker"]: f"{int(v['cik_str']):010d}" for v in d.values()}


def filings_for(cik):
    """All 8-K item-2.02 filings for one CIK, including the older overflow files."""
    out = []
    doc = get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    blocks = [doc["filings"]["recent"]]
    for extra in doc["filings"].get("files", []):
        time.sleep(0.2)
        blocks.append(get(f"https://data.sec.gov/submissions/{extra['name']}"))
    for b in blocks:
        for i, form in enumerate(b["form"]):
            items = b["items"][i] or ""
            if form == "8-K" and "2.02" in items.split(","):
                out.append((b["filingDate"][i], b["acceptanceDateTime"][i], b["accessionNumber"][i]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    tmap = ticker_map()
    missing, totals = [], {}
    for sym in UNIVERSE:
        ciks = [tmap.get(sym)] + EXTRA_CIKS.get(sym, [])
        ciks = [c for c in ciks if c]
        if not ciks:
            missing.append(sym)
            continue
        rows = []
        for cik in ciks:
            time.sleep(0.25)
            try:
                rows += filings_for(cik)
            except Exception as exc:
                print(f"  {sym} CIK {cik}: {exc}", flush=True)
        rows = sorted(set(rows))
        with open(f"{OUT}/{sym}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["symbol", "filing_date", "acceptance_utc", "accession"])
            for d, a, acc in rows:
                w.writerow([sym, d, a, acc])
        totals[sym] = len(rows)
        span = f"{rows[0][0]} -> {rows[-1][0]}" if rows else "none"
        print(f"{sym:6s} {len(rows):4d} filings  {span}  (CIKs {','.join(ciks)})", flush=True)
    print(f"\nfetched {len(totals)}/{len(UNIVERSE)}; missing: {missing}")
    print("total item-2.02 filings:", sum(totals.values()))


if __name__ == "__main__":
    main()
