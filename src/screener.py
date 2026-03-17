"""
Daily Universe Validator & Signal Pre-Calculator.

Runs daily at 14:00 UTC (30 min before US market open).
For the fixed 30-ETF trend-following universe:
1. Validates all instruments are liquid and tradeable
2. Pre-computes TSMOM signals from yfinance (free data)
3. Writes validated universe + signals to data/watchlist.json
4. Sends Telegram summary

Usage:
    python -m src.screener              # Run and update watchlist
    python -m src.screener --dry-run    # Print results without writing
"""

import argparse
import json
import logging
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("screener")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fixed 30-ETF universe across 5 asset classes
UNIVERSE = {
    "equity": ["SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ", "FXI"],
    "bond": ["TLT", "IEF", "SHY", "LQD", "HYG", "EMB"],
    "commodity": ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "PDBC", "CPER"],
    "fx": ["UUP", "FXE", "FXY", "FXB"],
    "alt": ["VNQ", "BITO", "DBC", "TIP"],
}

ALL_SYMBOLS = [s for syms in UNIVERSE.values() for s in syms]

# Minimum average daily dollar volume to be considered liquid
MIN_DAILY_DOLLAR_VOLUME = 5_000_000  # $5M

# Lookback periods for TSMOM (trading days)
LOOKBACKS = [21, 63, 252]  # 1M, 3M, 12M

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
WATCHLIST_PATH = Path(os.getenv("WATCHLIST_PATH", "data/watchlist.json"))
BACKUP_DIR = DATA_DIR / "backups"
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ---------------------------------------------------------------------------
# Telegram helper
# ---------------------------------------------------------------------------

def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured, skipping notification")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8")).get("ok", False)
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Data fetching and signal computation
# ---------------------------------------------------------------------------

def fetch_and_compute(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch 1Y of daily data via yfinance and compute TSMOM signals.

    Returns dict of symbol -> {price, volume, dollar_volume, tsmom_score,
    tsmom_details, atr, volatility, liquid}
    """
    import yfinance as yf

    results = {}
    logger.info(f"Fetching data for {len(symbols)} instruments...")

    tickers = yf.Tickers(" ".join(symbols))

    for symbol in symbols:
        try:
            ticker = tickers.tickers.get(symbol)
            if ticker is None:
                logger.warning(f"{symbol}: not found in yfinance")
                continue

            hist = ticker.history(period="1y", interval="1d")
            if hist is None or len(hist) < 60:
                logger.warning(f"{symbol}: insufficient data ({len(hist) if hist is not None else 0} bars)")
                continue

            close = hist["Close"]
            high = hist["High"]
            low = hist["Low"]
            volume = hist["Volume"]
            price = float(close.iloc[-1])

            # Average daily dollar volume (20-day)
            avg_vol = volume.tail(20).mean()
            dollar_volume = avg_vol * price
            liquid = dollar_volume >= MIN_DAILY_DOLLAR_VOLUME

            # ATR (20-day)
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_20 = float(tr.tail(20).mean())

            # Annualised volatility
            returns = close.pct_change().dropna()
            vol = float(returns.tail(20).std() * np.sqrt(252))

            # TSMOM signal (blended multi-lookback)
            tsmom_details = []
            signals = []
            weights = [0.3, 0.3, 0.4]

            for lookback, weight in zip(LOOKBACKS, weights):
                if len(close) >= lookback + 1:
                    ret = (close.iloc[-1] - close.iloc[-lookback - 1]) / close.iloc[-lookback - 1]
                    sig = np.sign(ret)
                    signals.append(sig * weight)
                    period_name = {21: "1M", 63: "3M", 252: "12M"}.get(lookback, f"{lookback}d")
                    direction = "+" if ret > 0 else "-"
                    tsmom_details.append(f"{period_name}: {direction}{abs(ret):.1%}")
                else:
                    signals.append(0)
                    tsmom_details.append(f"{lookback}d: N/A")

            tsmom_score = round(sum(signals), 3)

            results[symbol] = {
                "price": round(price, 2),
                "dollar_volume": round(dollar_volume, 0),
                "liquid": liquid,
                "tsmom_score": tsmom_score,
                "tsmom_details": tsmom_details,
                "atr": round(atr_20, 4),
                "volatility": round(vol, 4),
            }

            direction = "LONG" if tsmom_score > 0.3 else "SHORT" if tsmom_score < -0.3 else "FLAT"
            liq_mark = "OK" if liquid else "LOW"
            logger.info(
                f"  {symbol}: ${price:.2f} | TSMOM {tsmom_score:+.2f} ({direction}) | "
                f"ATR ${atr_20:.2f} | Vol {vol:.1%} | DolVol ${dollar_volume/1e6:.1f}M [{liq_mark}]"
            )

        except Exception as e:
            logger.error(f"  {symbol}: error - {e}")

    return results


# ---------------------------------------------------------------------------
# Watchlist update
# ---------------------------------------------------------------------------

def load_previous_watchlist() -> dict:
    try:
        if WATCHLIST_PATH.exists():
            with open(WATCHLIST_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def backup_watchlist() -> bool:
    if not WATCHLIST_PATH.exists():
        return True
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
        backup_path = BACKUP_DIR / f"watchlist_{ts}.json"
        shutil.copy2(WATCHLIST_PATH, backup_path)
        logger.info(f"Backed up watchlist to {backup_path}")

        # Clean old backups (keep 14 days)
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=14)
        for f in BACKUP_DIR.glob("watchlist_*.json"):
            try:
                file_date = datetime.strptime(
                    f.stem.replace("watchlist_", ""), "%Y%m%d_%H%M"
                ).replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    f.unlink()
            except (ValueError, OSError):
                pass
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False


def update_watchlist(data: dict[str, dict]) -> bool:
    """Write validated universe + signals to watchlist.json."""
    # Build symbols dict (only liquid instruments)
    symbols_by_class: dict[str, list[str]] = {}
    for asset_class, syms in UNIVERSE.items():
        liquid_syms = [s for s in syms if s in data and data[s]["liquid"]]
        if liquid_syms:
            symbols_by_class[asset_class] = liquid_syms

    watchlist_data = {
        "updated_at": datetime.now(tz=timezone.utc).isoformat() + "Z",
        "strategy": "trend_following",
        "symbols": symbols_by_class,
        "signals": {
            sym: {
                "tsmom_score": d["tsmom_score"],
                "tsmom_details": d["tsmom_details"],
                "price": d["price"],
                "atr": d["atr"],
                "volatility": d["volatility"],
                "dollar_volume": d["dollar_volume"],
            }
            for sym, d in data.items() if d["liquid"]
        },
    }

    try:
        if not backup_watchlist():
            logger.error("Backup failed — aborting update")
            return False

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = WATCHLIST_PATH.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(watchlist_data, f, indent=2)
        with open(tmp_path) as f:
            json.load(f)  # validate
        tmp_path.rename(WATCHLIST_PATH)
        logger.info(f"Watchlist updated: {WATCHLIST_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to write watchlist: {e}")
        try:
            WATCHLIST_PATH.with_suffix(".tmp").unlink(missing_ok=True)
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Telegram notification
# ---------------------------------------------------------------------------

def build_telegram_message(data: dict[str, dict], dry_run: bool = False) -> str:
    now = datetime.now(tz=timezone.utc)

    # Separate into long/short/flat
    longs = [(s, d) for s, d in data.items() if d["liquid"] and d["tsmom_score"] > 0.3]
    shorts = [(s, d) for s, d in data.items() if d["liquid"] and d["tsmom_score"] < -0.3]
    flat = [(s, d) for s, d in data.items() if d["liquid"] and abs(d["tsmom_score"]) <= 0.3]
    illiquid = [(s, d) for s, d in data.items() if not d["liquid"]]

    longs.sort(key=lambda x: x[1]["tsmom_score"], reverse=True)
    shorts.sort(key=lambda x: x[1]["tsmom_score"])

    prefix = "[DRY RUN] " if dry_run else ""
    lines = [f"\U0001F4CA <b>{prefix}Daily Trend-Following Scan</b>"]
    lines.append(f"\n<b>Universe:</b> {len(data)} instruments scanned")

    if longs:
        lines.append(f"\n\U0001F7E2 <b>LONG signals ({len(longs)}):</b>")
        for sym, d in longs[:10]:
            lines.append(f"  {sym}: {d['tsmom_score']:+.2f} ({', '.join(d['tsmom_details'][:2])})")

    if shorts:
        lines.append(f"\n\U0001F534 <b>SHORT signals ({len(shorts)}):</b>")
        for sym, d in shorts[:10]:
            lines.append(f"  {sym}: {d['tsmom_score']:+.2f} ({', '.join(d['tsmom_details'][:2])})")

    if flat:
        lines.append(f"\n\u26AA <b>FLAT ({len(flat)}):</b> {', '.join(s for s, _ in flat)}")

    if illiquid:
        lines.append(f"\n\u26A0\uFE0F <b>Illiquid ({len(illiquid)}):</b> {', '.join(s for s, _ in illiquid)}")

    lines.append(f"\n<code>{now.strftime('%Y-%m-%d %H:%M:%S')} UTC</code>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging(log_file: str | None = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


def run_screener(dry_run: bool = False) -> bool:
    logger.info("=" * 60)
    logger.info(f"TREND-FOLLOWING SCREENER - {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"Universe: {len(ALL_SYMBOLS)} instruments across {len(UNIVERSE)} asset classes")
    logger.info("=" * 60)

    # Fetch data and compute signals
    data = fetch_and_compute(ALL_SYMBOLS)
    if not data:
        send_telegram("\U0001F6A8 <b>Screener Error</b>\n\nFailed to fetch data.\nKeeping yesterday's watchlist.")
        return False

    logger.info(f"\nGot data for {len(data)}/{len(ALL_SYMBOLS)} instruments")

    # Summary
    liquid = [s for s, d in data.items() if d["liquid"]]
    longs = [s for s in liquid if data[s]["tsmom_score"] > 0.3]
    shorts = [s for s in liquid if data[s]["tsmom_score"] < -0.3]
    logger.info(f"Liquid: {len(liquid)} | Long signals: {len(longs)} | Short signals: {len(shorts)}")

    # Update watchlist
    if dry_run:
        logger.info("\n--- DRY RUN: Not writing watchlist ---")
    else:
        if not update_watchlist(data):
            send_telegram("\U0001F6A8 <b>Screener Error</b>\n\nFailed to update watchlist.")
            return False

    # Telegram
    msg = build_telegram_message(data, dry_run=dry_run)
    send_telegram(msg)

    logger.info("\nScreener completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Trend-Following Screener")
    parser.add_argument("--dry-run", action="store_true", help="Don't write watchlist")
    args = parser.parse_args()

    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    log_file = LOG_DIR / f"screener_{today}.log"
    setup_logging(str(log_file))

    try:
        success = run_screener(dry_run=args.dry_run)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception(f"Screener crashed: {e}")
        send_telegram(f"\U0001F6A8 <b>Screener Crashed</b>\n\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
