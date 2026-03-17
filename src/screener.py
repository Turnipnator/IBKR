"""
Dynamic Instrument Screener - Selects best instruments to trade each day.

Runs daily at 14:00 UTC (30 min before US market open).
Screens a universe of liquid large-caps using yfinance and writes
the top candidates to data/watchlist.json for the bot to pick up.

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

import pandas as pd

logger = logging.getLogger("screener")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCREENING_UNIVERSE = [
    # AI / Semiconductors
    "NVDA", "AMD", "AVGO", "TSM", "QCOM", "INTC", "MU", "ARM", "ANET",
    # Mega-cap Tech
    "MSFT", "GOOGL", "META", "AMZN", "AAPL", "TSLA",
    # Enterprise / Cloud
    "CRM", "NOW", "SNOW", "PLTR", "PANW", "CRWD", "NET",
    # Diversified / Other
    "V", "MA", "JPM", "GS", "NVO", "LLY", "XOM",
]

# Sector mapping for the output watchlist (mirrors bot's sector structure)
SECTOR_MAP = {
    # AI / Semiconductors
    "NVDA": "ai", "AMD": "ai", "AVGO": "ai", "TSM": "ai",
    "QCOM": "ai", "INTC": "ai", "MU": "ai", "ARM": "ai", "ANET": "ai",
    # Mega-cap Tech
    "MSFT": "tech", "GOOGL": "tech", "META": "tech", "AMZN": "tech",
    "AAPL": "tech", "TSLA": "tech",
    # Enterprise / Cloud
    "CRM": "cloud", "NOW": "cloud", "SNOW": "cloud", "PLTR": "cloud",
    "PANW": "cloud", "CRWD": "cloud", "NET": "cloud",
    # Diversified
    "V": "diversified", "MA": "diversified", "JPM": "diversified",
    "GS": "diversified", "NVO": "diversified", "LLY": "diversified",
    "XOM": "diversified",
}

# Preferred sectors get a scoring bonus (tech/AI historically 63% win rate)
PREFERRED_SECTORS = {"ai", "tech", "cloud"}

# Screening thresholds
MIN_MARKET_CAP = 10e9       # $10B minimum
MIN_REL_VOLUME = 1.3        # 1.3x 20-day avg volume
RELAXED_REL_VOLUME = 1.0    # Relaxed threshold if <4 pass
MIN_PRICE = 15.0
MAX_PRICE = 800.0
MIN_ATR_PCT = 1.5           # 1.5% daily ATR relative to price
MIN_PREMARKET_MOVE = 0.5    # 0.5% pre-market move (bonus, not mandatory)

# Output limits
MIN_OUTPUT = 4
MAX_OUTPUT = 12  # Matches bot's current max watchlist size

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
WATCHLIST_PATH = Path(os.getenv("WATCHLIST_PATH", "data/watchlist.json"))
BACKUP_DIR = DATA_DIR / "backups"
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))

# Telegram (read from env — same as the bot)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ---------------------------------------------------------------------------
# Telegram helper (standalone — no dependency on bot's telegram_bot.py)
# ---------------------------------------------------------------------------

def send_telegram(text: str) -> bool:
    """Send a Telegram message. Returns True on success."""
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
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("ok", False)
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Screening logic
# ---------------------------------------------------------------------------

def fetch_screening_data(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch screening data for all symbols via yfinance.

    Returns dict of symbol -> {market_cap, avg_volume, current_volume,
    rel_volume, atr_pct, premarket_move, price, name}.
    """
    import yfinance as yf

    results = {}

    # Download recent daily bars for ATR and volume (25 days gives us 20 trading days)
    logger.info(f"Downloading daily data for {len(symbols)} symbols...")
    tickers = yf.Tickers(" ".join(symbols))

    for symbol in symbols:
        try:
            ticker = tickers.tickers.get(symbol)
            if ticker is None:
                logger.warning(f"{symbol}: ticker not found in yfinance")
                continue

            # Get info for market cap and current price
            info = ticker.info or {}
            market_cap = info.get("marketCap", 0) or 0
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            name = info.get("shortName", symbol)

            # Get pre-market change if available
            pre_market_price = info.get("preMarketPrice")
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
            premarket_move = 0.0
            if pre_market_price and prev_close and prev_close > 0:
                premarket_move = ((pre_market_price - prev_close) / prev_close) * 100

            # Historical data for volume and ATR
            hist = ticker.history(period="1mo", interval="1d")
            if hist is None or len(hist) < 10:
                logger.warning(f"{symbol}: insufficient historical data ({len(hist) if hist is not None else 0} bars)")
                continue

            # Use previous completed day's volume (last bar may be today's partial session)
            # Average over days excluding the most recent (which may be incomplete)
            if len(hist) >= 3:
                avg_volume = hist["Volume"].iloc[:-1].tail(20).mean()
                current_volume = hist["Volume"].iloc[-2]  # Last complete day
            else:
                avg_volume = hist["Volume"].mean()
                current_volume = hist["Volume"].iloc[-1]

            # Relative volume (last complete day vs 20-day average)
            rel_volume = (current_volume / avg_volume) if avg_volume > 0 else 0

            # ATR% (14-day ATR / price)
            high = hist["High"]
            low = hist["Low"]
            close = hist["Close"]
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_14 = tr.tail(14).mean()
            atr_pct = (atr_14 / price * 100) if price > 0 else 0

            results[symbol] = {
                "name": name,
                "price": price,
                "market_cap": market_cap,
                "avg_volume": avg_volume,
                "current_volume": current_volume,
                "rel_volume": round(rel_volume, 2),
                "atr_pct": round(atr_pct, 2),
                "premarket_move": round(premarket_move, 2),
            }

            logger.info(
                f"  {symbol}: ${price:.2f} | MCap ${market_cap/1e9:.0f}B | "
                f"RelVol {rel_volume:.1f}x | ATR% {atr_pct:.1f}% | PM {premarket_move:+.1f}%"
            )

        except Exception as e:
            logger.error(f"  {symbol}: error fetching data - {e}")

    return results


def screen_and_score(
    data: dict[str, dict],
    volume_threshold: float = MIN_REL_VOLUME,
) -> list[dict]:
    """
    Apply screening criteria and score candidates.

    Returns sorted list of dicts with symbol, score, and reasons.
    """
    candidates = []

    for symbol, d in data.items():
        reasons_pass = []
        reasons_fail = []

        # --- Mandatory filters ---

        # Market cap
        if d["market_cap"] < MIN_MARKET_CAP:
            reasons_fail.append(f"MCap ${d['market_cap']/1e9:.1f}B < ${MIN_MARKET_CAP/1e9:.0f}B")
            logger.info(f"  {symbol} REJECTED: {reasons_fail[-1]}")
            continue

        # Price range
        if d["price"] < MIN_PRICE or d["price"] > MAX_PRICE:
            reasons_fail.append(f"Price ${d['price']:.2f} outside ${MIN_PRICE}-${MAX_PRICE}")
            logger.info(f"  {symbol} REJECTED: {reasons_fail[-1]}")
            continue

        # Relative volume
        if d["rel_volume"] < volume_threshold:
            reasons_fail.append(f"RelVol {d['rel_volume']:.1f}x < {volume_threshold}x")
            logger.info(f"  {symbol} REJECTED: {reasons_fail[-1]}")
            continue

        # ATR%
        if d["atr_pct"] < MIN_ATR_PCT:
            reasons_fail.append(f"ATR% {d['atr_pct']:.1f}% < {MIN_ATR_PCT}%")
            logger.info(f"  {symbol} REJECTED: {reasons_fail[-1]}")
            continue

        # --- Scoring ---
        score = 0.0

        # Relative volume score (higher = better, capped at 3x)
        vol_score = min(d["rel_volume"] / 3.0, 1.0) * 30
        score += vol_score
        reasons_pass.append(f"{d['rel_volume']:.1f}x rel vol")

        # ATR% score (higher = more tradeable, capped at 5%)
        atr_score = min(d["atr_pct"] / 5.0, 1.0) * 25
        score += atr_score
        reasons_pass.append(f"{d['atr_pct']:.1f}% ATR")

        # Pre-market move bonus
        if abs(d["premarket_move"]) >= MIN_PREMARKET_MOVE:
            pm_score = min(abs(d["premarket_move"]) / 3.0, 1.0) * 20
            score += pm_score
            reasons_pass.append(f"{d['premarket_move']:+.1f}% pre-market")

        # Sector preference bonus
        sector = SECTOR_MAP.get(symbol, "other")
        if sector in PREFERRED_SECTORS:
            score += 15
            reasons_pass.append(f"preferred sector ({sector})")

        # Market cap liquidity bonus (bigger = more liquid)
        if d["market_cap"] > 100e9:
            score += 10
            reasons_pass.append("mega-cap liquidity")

        candidates.append({
            "symbol": symbol,
            "sector": sector,
            "score": round(score, 1),
            "reasons": reasons_pass,
            "data": d,
        })

    # Sort by score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def check_open_losing_positions(candidates: list[dict]) -> list[dict]:
    """
    Remove symbols that have open losing paper trades in the database.
    If database is not accessible, return candidates unchanged.
    """
    try:
        from .database import Database
        db = Database()
        open_trades = db.get_open_paper_trades()
        losing_symbols = set()
        for trade in open_trades:
            # If we have best_price tracking and it's below entry, it's losing
            # Simple heuristic: skip any symbol with an open position
            losing_symbols.add(trade["symbol"])

        if losing_symbols:
            before = len(candidates)
            candidates = [c for c in candidates if c["symbol"] not in losing_symbols]
            removed = before - len(candidates)
            if removed:
                logger.info(f"Removed {removed} symbols with open positions: {losing_symbols}")

    except Exception as e:
        logger.warning(f"Could not check open positions (non-fatal): {e}")

    return candidates


# ---------------------------------------------------------------------------
# Watchlist update
# ---------------------------------------------------------------------------

def load_previous_watchlist() -> dict:
    """Load the previous watchlist for comparison."""
    try:
        if WATCHLIST_PATH.exists():
            with open(WATCHLIST_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def backup_watchlist() -> bool:
    """Create a timestamped backup of the current watchlist."""
    if not WATCHLIST_PATH.exists():
        return True  # Nothing to back up

    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
        backup_path = BACKUP_DIR / f"watchlist_{ts}.json"
        shutil.copy2(WATCHLIST_PATH, backup_path)
        logger.info(f"Backed up watchlist to {backup_path}")

        # Clean old backups (keep last 14 days)
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=14)
        for f in BACKUP_DIR.glob("watchlist_*.json"):
            try:
                file_date = datetime.strptime(f.stem.replace("watchlist_", ""), "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    f.unlink()
                    logger.info(f"Cleaned old backup: {f.name}")
            except (ValueError, OSError):
                pass

        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False


def update_watchlist(candidates: list[dict]) -> bool:
    """
    Write the new watchlist to data/watchlist.json.
    Preserves sector grouping for the bot.
    """
    # Build sector-grouped symbol dict
    symbols_by_sector: dict[str, list[str]] = {}
    for c in candidates:
        sector = c["sector"]
        if sector not in symbols_by_sector:
            symbols_by_sector[sector] = []
        symbols_by_sector[sector].append(c["symbol"])

    watchlist_data = {
        "updated_at": datetime.now(tz=timezone.utc).isoformat() + "Z",
        "symbols": symbols_by_sector,
        "details": {
            c["symbol"]: {
                "score": c["score"],
                "reasons": c["reasons"],
                "rel_volume": c["data"]["rel_volume"],
                "atr_pct": c["data"]["atr_pct"],
                "premarket_move": c["data"]["premarket_move"],
                "price": c["data"]["price"],
            }
            for c in candidates
        },
    }

    try:
        # Backup first
        if not backup_watchlist():
            logger.error("Backup failed - aborting watchlist update")
            return False

        # Write atomically (write to temp, then rename)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = WATCHLIST_PATH.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(watchlist_data, f, indent=2)

        # Validate the written file
        with open(tmp_path) as f:
            json.load(f)  # Will raise if invalid JSON

        tmp_path.rename(WATCHLIST_PATH)
        logger.info(f"Watchlist updated: {WATCHLIST_PATH}")
        return True

    except Exception as e:
        logger.error(f"Failed to write watchlist: {e}")
        # Clean up temp file
        try:
            tmp_path = WATCHLIST_PATH.with_suffix(".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Telegram notification
# ---------------------------------------------------------------------------

def build_telegram_message(
    candidates: list[dict],
    previous: dict,
    dry_run: bool = False,
) -> str:
    """Build the Telegram notification message."""
    now = datetime.now(tz=timezone.utc)
    market_open_mins = max(0, 30 - now.minute) if now.hour == 13 else 0
    if now.hour < 14:
        market_open_mins = (14 - now.hour) * 60 + (30 - now.minute)

    selected = [c["symbol"] for c in candidates]
    prev_symbols = []
    for sector_syms in previous.get("symbols", {}).values():
        prev_symbols.extend(sector_syms)

    added = [s for s in selected if s not in prev_symbols]
    dropped = [s for s in prev_symbols if s not in selected]

    lines = []
    prefix = "[DRY RUN] " if dry_run else ""
    lines.append(f"\U0001F4CA <b>{prefix}Daily Instrument Screen Complete</b>")
    lines.append("")
    lines.append(f"<b>Selected ({len(selected)}):</b> {', '.join(selected)}")
    lines.append("")
    lines.append("<b>Reasoning:</b>")
    for c in candidates[:8]:  # Limit detail to top 8
        reasons_short = ", ".join(c["reasons"][:3])
        lines.append(f"  \u2022 {c['symbol']} - {reasons_short} (score: {c['score']})")
    if len(candidates) > 8:
        lines.append(f"  ... and {len(candidates) - 8} more")

    if dropped:
        lines.append(f"\n<b>Dropped from yesterday:</b> {', '.join(dropped)}")
    if added:
        lines.append(f"<b>Added vs yesterday:</b> {', '.join(added)}")
    if not dropped and not added:
        lines.append("\n<i>No changes from yesterday's watchlist.</i>")

    if market_open_mins > 0:
        lines.append(f"\nMarket opens in ~{market_open_mins} minutes.")

    lines.append(f"\n<code>{now.strftime('%Y-%m-%d %H:%M:%S')} UTC</code>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging(log_file: str | None = None):
    """Configure logging to both console and file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    # Suppress noisy libraries
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


def run_screener(dry_run: bool = False) -> bool:
    """
    Main screener entry point.

    Returns True if successful, False on failure.
    """
    logger.info("=" * 60)
    logger.info(f"DAILY INSTRUMENT SCREENER - {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    logger.info(f"Universe: {len(SCREENING_UNIVERSE)} symbols")
    logger.info("=" * 60)

    # Load previous watchlist for comparison
    previous = load_previous_watchlist()

    # Step 1: Fetch data
    logger.info("\n--- Step 1: Fetching screening data ---")
    data = fetch_screening_data(SCREENING_UNIVERSE)
    if not data:
        msg = "Failed to fetch any screening data"
        logger.error(msg)
        send_telegram(f"\U0001F6A8 <b>Screener Error</b>\n\n{msg}\n\nKeeping yesterday's watchlist.")
        return False

    logger.info(f"Got data for {len(data)}/{len(SCREENING_UNIVERSE)} symbols")

    # Step 2: Screen and score
    logger.info("\n--- Step 2: Screening with standard thresholds ---")
    candidates = screen_and_score(data, volume_threshold=MIN_REL_VOLUME)

    # If fewer than MIN_OUTPUT, relax volume filter
    if len(candidates) < MIN_OUTPUT:
        logger.info(f"\nOnly {len(candidates)} candidates - relaxing volume to {RELAXED_REL_VOLUME}x")
        candidates = screen_and_score(data, volume_threshold=RELAXED_REL_VOLUME)

    if not candidates:
        msg = "No symbols passed screening criteria"
        logger.warning(msg)
        send_telegram(
            f"\U0001F6A8 <b>Screener Warning</b>\n\n{msg}\n\nKeeping yesterday's watchlist."
        )
        return False

    # Step 3: Filter out symbols with open losing positions
    logger.info("\n--- Step 3: Checking open positions ---")
    candidates = check_open_losing_positions(candidates)

    # Step 4: Limit to MAX_OUTPUT
    candidates = candidates[:MAX_OUTPUT]
    logger.info(f"\n--- Final selection: {len(candidates)} symbols ---")
    for i, c in enumerate(candidates, 1):
        logger.info(f"  {i}. {c['symbol']} (score: {c['score']}) - {', '.join(c['reasons'])}")

    # Step 5: Update watchlist (or just print in dry-run)
    if dry_run:
        logger.info("\n--- DRY RUN: Not writing watchlist ---")
        symbols_by_sector: dict[str, list[str]] = {}
        for c in candidates:
            sector = c["sector"]
            if sector not in symbols_by_sector:
                symbols_by_sector[sector] = []
            symbols_by_sector[sector].append(c["symbol"])
        logger.info(f"Would write: {json.dumps(symbols_by_sector, indent=2)}")
    else:
        logger.info("\n--- Step 5: Updating watchlist ---")
        if not update_watchlist(candidates):
            msg = "Failed to update watchlist file"
            logger.error(msg)
            send_telegram(f"\U0001F6A8 <b>Screener Error</b>\n\n{msg}\n\nKeeping yesterday's watchlist.")
            return False

    # Step 6: Send Telegram notification
    logger.info("\n--- Step 6: Sending Telegram notification ---")
    tg_message = build_telegram_message(candidates, previous, dry_run=dry_run)
    send_telegram(tg_message)

    logger.info("\nScreener completed successfully.")
    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="IBKR Bot - Daily Instrument Screener")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without updating the watchlist",
    )
    args = parser.parse_args()

    # Set up logging with dated log file
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    log_file = LOG_DIR / f"screener_{today}.log"
    setup_logging(str(log_file))

    try:
        success = run_screener(dry_run=args.dry_run)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception(f"Screener crashed: {e}")
        send_telegram(f"\U0001F6A8 <b>Screener Crashed</b>\n\n{e}\n\nKeeping yesterday's watchlist.")
        sys.exit(1)


if __name__ == "__main__":
    main()
