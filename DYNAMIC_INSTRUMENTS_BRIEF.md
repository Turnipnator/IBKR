# IBKR Bot - Dynamic Instrument Selection Layer

## Brief for Claude Code

---

## What This Is

An upgrade to the existing IBKR trading bot to replace the static hardcoded instrument list
with a dynamic daily screener that automatically selects the best instruments to trade each day.

The existing bot strategy, risk management, and execution logic are **NOT to be changed**.
We are only adding a new layer that sits above it and populates the watchlist dynamically.

---

## Step 1 - Read Before Writing Anything

Before writing a single line of code, read and understand the following:

1. **The main bot file** - understand how the watchlist/instrument list is currently defined
2. **The config file** - understand the structure so the new screener writes to it correctly
3. **How the bot is started** - Docker container name, any startup scripts, cron jobs
4. **How symbols are currently referenced** - variable names, data structures, file paths

Do not assume generic paths or variable names. Use what actually exists in the codebase.

---

## What To Build

### Component 1: Morning Screener Script (`screener.py`)

A standalone Python script that runs every weekday at **14:00 UTC** (30 minutes before US market open at 14:30 UTC).

**What it does:**

1. Connects to **Yahoo Finance** (via `yfinance` library - free, no API key needed) to screen stocks
2. Applies the screening criteria below
3. Scores and ranks the candidates
4. Writes the top results to the bot's config/watchlist
5. Sends a Telegram notification confirming what instruments were selected and why
6. Logs everything to a dated log file

**Screening Criteria (in order of priority):**

```
1. Sector filter       - Technology, Semiconductors, AI-adjacent large caps preferred
                         (mirrors the bot's existing 63% win rate on tech/AI)

2. Market cap          - Minimum $10 billion (liquid large caps only, no small caps)

3. Relative volume     - Minimum 1.3x 20-day average volume
                         (something is happening - news, momentum, catalyst)

4. Price range         - Between $15 and $800
                         (avoids penny stocks and ultra-high priced stocks)

5. ATR%                - Minimum 1.5% daily ATR relative to price
                         (needs enough intraday movement to be worth trading)

6. Pre-market move     - Prefer stocks with >0.5% pre-market move (catalyst signal)

7. Not already in loss - Skip any symbol where an open losing position exists in the bot
```

**Universe to screen from:**

Use this base list of liquid large-cap candidates as the screening universe. This is deliberately
broader than the current static list so the screener has room to find the best ones each day:

```python
SCREENING_UNIVERSE = [
    # AI / Semiconductors
    "NVDA", "AMD", "AVGO", "TSM", "QCOM", "INTC", "MU", "ARM", "ANET",
    # Mega-cap Tech
    "MSFT", "GOOGL", "META", "AMZN", "AAPL", "TSLA",
    # Enterprise / Cloud
    "CRM", "NOW", "SNOW", "PLTR", "PANW", "CRWD", "NET",
    # Diversified / Other
    "V", "MA", "JPM", "GS", "NVO", "LLY", "XOM"
]
```

**Output - top N symbols:**

Select the top symbols that pass all mandatory criteria, ranked by composite score.
Max symbols to output: **match whatever the bot's current max watchlist size is**.
Minimum symbols to output: **4** (if fewer than 4 pass screening, relax the volume criterion first).

---

### Component 2: Watchlist Update Function

A function (within `screener.py` or as a shared utility) that:

1. Reads the bot's current config/watchlist file
2. Replaces the instrument list with the new screened symbols
3. Writes the file back cleanly (preserving all other config settings exactly)
4. Creates a timestamped backup of the previous watchlist before overwriting

```python
# Example of what this function should do - adapt to actual config structure
def update_watchlist(new_symbols: list[str]) -> bool:
    """
    Safely updates the bot watchlist with new symbols.
    Backs up current config before making changes.
    Returns True if successful, False if something went wrong.
    """
    # 1. Read current config
    # 2. Backup to /opt/bots/backups/watchlist_YYYYMMDD_HHMM.json (or equivalent)
    # 3. Replace only the symbol list - leave everything else untouched
    # 4. Write back
    # 5. Validate the file is valid JSON/YAML/whatever format it uses
    # 6. Return success/failure
```

---

### Component 3: Cron Job Setup

Add a cron entry that runs the screener Monday-Friday at 14:00 UTC:

```bash
# Dynamic instrument screener - runs 30 mins before US market open
0 14 * * 1-5 /usr/bin/python3 /opt/bots/screener.py >> /opt/bots/logs/screener.log 2>&1
```

Adjust the path to match wherever the bot actually lives on the VPS.

---

### Component 4: Telegram Notification

When the screener runs, send a Telegram message in this format:

```
📊 Daily Instrument Screen Complete

Selected: NVDA, AMD, CRWD, NET

Reasoning:
• NVDA - 1.8x rel vol, +1.2% pre-market, strong ATR
• AMD - 1.5x rel vol, sector momentum
• CRWD - Earnings catalyst, 2.1x rel vol
• NET - Breaking out, 1.6x rel vol

Dropped from yesterday: TSLA, META
Added vs yesterday: CRWD, NET

Market opens in 28 minutes.
```

Reuse the existing Telegram bot token and chat ID from the bot's config - do not hardcode new ones.

---

## Error Handling Requirements

The screener must be defensive - a failure here should **never** crash or affect the running bot.

- If Yahoo Finance is unreachable - keep yesterday's watchlist, send Telegram alert
- If fewer than 4 symbols pass screening - relax volume filter and try again before giving up
- If the config write fails - abort and keep original, send Telegram alert
- If the screener crashes entirely - the bot continues with whatever watchlist it had before
- Wrap the entire script in a try/except with Telegram error notification

---

## Logging

Write a log file for every run:

```
/opt/bots/logs/screener_YYYYMMDD.log
```

Each log should contain:
- Timestamp of run
- Full list of symbols screened and their scores
- Which symbols were selected and why
- Which symbols were rejected and why
- Whether the watchlist update succeeded
- Any errors or warnings

---

## What NOT To Change

- Do not touch the bot's signal logic (EMA, RSI, MACD, Volume checks)
- Do not touch risk management settings (stop loss, take profit, position sizing)
- Do not touch the SPY market filter logic
- Do not touch Docker configuration
- Do not touch Telegram notification logic that already exists in the bot
- Do not change any existing variable names or config keys the bot relies on

---

## Testing Before Going Live

1. Run the screener manually once: `python3 screener.py --dry-run`
   - `--dry-run` flag should print what it *would* update without actually writing the config
2. Verify the output symbols look sensible (liquid, large cap tech/AI names expected)
3. Verify the Telegram notification fires correctly
4. Verify the backup file is created
5. Verify the log file is created with full detail
6. Only after manual verification - set up the cron job

---

## Dependencies to Install if Not Present

```bash
pip install yfinance pandas --break-system-packages
```

Check if these are already installed before running pip.

---

## Summary

The end result should be:
- A `screener.py` script that runs daily at 14:00 UTC
- It screens a universe of ~30 liquid large caps
- Picks the best 4-12 for that day based on volume, momentum and sector
- Updates the bot's watchlist automatically
- Sends a Telegram summary of what was picked and why
- The bot then trades those instruments exactly as it does today
- If anything goes wrong, the bot carries on with the previous day's list

The bot becomes self-selecting on instruments while all trading logic stays exactly the same.
