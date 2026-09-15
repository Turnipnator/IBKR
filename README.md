# IBKR Trend-Following Bot

An automated, long-only **trend-following / momentum** trading bot for Interactive
Brokers. It trades a fixed universe of 24 **UCITS ETFs and ETCs listed on the London
Stock Exchange** across equities, bonds, commodities and property. Once a day it holds
the top 3 names by momentum score, and every position carries a server-side ATR
trailing stop.

**Status:** 🔴 **LIVE** since 2026-05-22 on a small GBP-based IBKR **cash account**. There
is no paper environment. See [Testing](#testing) for how changes are tested before
they're deployed.

> Built for a UK retail account. UK/EU retail investors can't buy US-listed ETFs
> (PRIIPs/KID rules, IBKR Error 201), which is why the universe is UCITS-on-LSE rather
> than SPY/GLD/TLT.

---

## Contents

- [Strategy](#strategy)
- [Execution & safety nets](#execution--safety-nets)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Telegram](#telegram)
- [Database](#database)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## Strategy

This is time-series and cross-sectional momentum after Moskowitz, Ooi & Pedersen
(2012), adapted for a small cash account where IBKR's fixed minimum commission is a
first-order cost.

### Universe (`data/watchlist.json` + `src/contracts.py`)

| Class | Instruments |
|-------|-------------|
| Equity | CSPX (S&P 500), EQQQ (Nasdaq-100), RTWO (Russell 2000), EIMU (EM IMI), VEUR (Developed Europe), CNYA (China A), IJPN (Japan) |
| Bond | DTLA (UST 20+y), IDTM (UST 7–10y), IBTA (UST 1–3y), LQDE (USD IG corp), IHYU (USD HY corp), JPEA (EM sovereign), IDTP (TIPS) |
| Commodity | IGLN (gold), ISLN (silver), CRUD (WTI), NGAS (nat gas), AIGA (agriculture), AIGI (industrial metals), CMOD / AIGS (broad commodities), COPA (copper) |
| Alt | IDUP (US property) |

All contracts use `SMART` routing with `primaryExchange='LSEETF'`. Most trade in USD
share classes. VEUR trades in GBP. **EQQQ and IJPN are quoted in pence** (IBKR
`priceMagnifier=100`), so the registry tags them `GBX` and the engine converts at
0.01 GBP per price unit. `data/phase1_ucits_mapping.csv` records which US ETF each
UCITS line replaced.

### Signal (daily bars, recomputed at every rebalance)

1. **TSMOM**: the weighted sign of each instrument's total return over three lookbacks:
   `0.3 × sign(21d) + 0.3 × sign(63d) + 0.4 × sign(252d)`, giving a score in [−1, +1].
2. **CSMOM**: the instrument's TSMOM percentile rank across the universe, mapped linearly to [−1, +1].
3. **Combined** = `0.6 × TSMOM + 0.4 × CSMOM`. A name is a candidate only if
   **combined ≥ 0.5** (long-only; shorting is disabled).

### Selection filters (applied before ranking, so empty slots backfill)

- **Re-entry cooldown (10 days):** a name whose stop fired recently is skipped for new
  entries. Names already held are never filtered, because that would force a sale.
- **Minimum volatility (8% annualised):** cash proxies such as T-bill ETFs look like
  perfect trends (a yield-accruing flat line is "up" over every lookback), but no stop
  can survive their bid-ask spread. These names get no new money. Held ones exit via
  their stop.
- **Affordability:** a name that can't fill at least 1 share within the per-position cap is skipped.

### Sizing

- **3 slots** (`max_open_positions`). Equal risk per slot: `risk_budget × capital / N`
  divided by the stop distance (`3 × ATR / price`), rounded to the nearest whole share
  (the API doesn't support fractional shares).
- **Max 30% of equity per name.** In practice this cap binds for every instrument, so it
  effectively sets position size.
- **Max 60% per asset class** (2 × the per-name cap), and **gross exposure ≤ 100%**
  (no leverage).
- Capital is **cash + gross position value**. It is *not* IBKR's `EquityWithLoanValue`,
  which on a cash account only counts settled cash.

Why so few, large positions: IBKR's $4 minimum per order makes a round trip cost about
$8 whatever the trade size. At small capital that fee was consuming a large share of
every trade's risk-to-stop. Cutting from 8 slots to 5 and then to 3 spread the fixed
cost over more capital. The reasoning and backtests are in `src/config.py` comments
and `research_notes.md`.

### Exits & risk limits

| Control | Setting | Behaviour |
|---------|---------|-----------|
| Trailing stop | 3 × ATR(20) | GTC `TRAIL` order held **at IBKR**. It ratchets up server-side and fires even if the bot is down |
| Daily loss halt | £200 (base GBP) | Blocks new entries for the rest of the day; stops stay active |
| Drawdown reduce | 10% from peak | Halves all targets |
| Drawdown halt | 20% from peak | Flattens the book and halts |
| Top-up threshold | < 70% of target | A held position is only topped up once it drifts more than 30% below target |

Nothing is force-sold at rebalance. A name that drops out of the top 3 stops being
topped up and exits on its own trailing stop.

---

## Execution & safety nets

The live order path is built around the constraints of an IBKR **cash account**:

- **Entry race:** the protective SELL stop is attached only after the market BUY
  reports `Filled`. Attaching it earlier opens a short, which a cash account rejects
  with Error 201.
- **Settled-cash sizing:** each BUY is trimmed to what `AvailableFunds` covers, with a
  6% buffer. A trimmed entry (or top-up) below 50% of what was wanted is skipped. Within
  one rebalance, a per-cycle committed-cash tally subtracts earlier BUYs, because
  ib_insync's cached account summary lags.
- **Top-ups:** BUY only the shortfall. Once it fills, swap the stop for one covering the
  full position. The swap **keeps the ratcheted trigger** if the fresh 3×ATR level would
  be lower, and restores the old stop if the replacement fails. Top-ups are skipped when
  price is within 1×ATR of the stop the position would carry.
- **Order parity (quantity-aware):** at startup, at every risk check and at every
  rebalance, the bot compares shares held with shares covered by working stops. It
  rebuilds any shortfall and alerts on orphaned stops.
- **Post-rebalance stop sweep:** reconciles again 30s and 120s after a rebalance with
  executions.
- **Fill notifier:** subscribes to `commissionReportEvent`. Stop fills (including ones
  that happened while the bot was offline) are logged, recorded with IBKR's realized
  P&L, sent to Telegram, and start the re-entry cooldown.
- **Trade ledger:** every order is written to the `trades` table as `SUBMITTED` and
  moved to `FILLED` / `CANCELLED` / `REJECTED` from `orderStatusEvent`.
- **NLV reconciliation:** compares IBKR NetLiquidation with the last equity snapshot and
  alerts at ≥ 2% drift.
- **Data-health probe:** every 5 minutes during market hours. On repeated failures it
  restarts the `ib-gateway` container through the Docker socket (capped at 3/day). It
  **won't restart a gateway that is mid-login or waiting for 2FA**, because that would
  kill the pending push.
- **Loop watchdog:** if the probe keeps failing and self-heal isn't working, the bot
  exits so Docker recreates it. It also holds off while a 2FA approval is outstanding.
- **Per-symbol isolation:** an unexpected exception on one instrument is logged and
  alerted, and the stop sweep still runs.

---

## Architecture

```
┌──────────────────────────── Docker host (network_mode: host) ────────────────────────────┐
│                                                                                           │
│  ┌──────────────────────┐  API 4001 (live)  ┌──────────────────────────────────────────┐ │
│  │  ib-gateway          │◄─────────────────►│  trading-bot  (python -m src.bot)        │ │
│  │  gnzsnz/ib-gateway   │                   │                                          │ │
│  │  IBC auto-login,     │                   │  bot.py ─ scheduler, live loop, parity,  │ │
│  │  nightly 23:55       │                   │           fill/ledger handlers           │ │
│  │  soft restart        │                   │  engine.py ─ signals, sizing, top-ups    │ │
│  └──────────▲───────────┘                   │  orders.py ─ orders & trailing stops     │ │
│             │ restart (docker.sock)         │  data_fetcher / indicators / contracts   │ │
│             └───────────────────────────────┤  data_health_checker / gateway_monitor   │ │
│                                             │  database.py ─ SQLite (data/trading.db)  │ │
│                                             └───────────────────┬──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┼────────────────────────┘
                                                                  ▼
                                                    Telegram (alerts + commands)
```

**Schedule (Europe/London, so BST/GMT is handled automatically):**

| When | What |
|------|------|
| Mon–Fri 08:00–16:30 | Market hours: data probe every 5 min, Telegram polling every 3s |
| 14:00 | Daily rebalance: fetch 1Y of daily bars → signals → targets → orders |
| Every 4h after the last check/rebalance | Intraday risk check: order parity, NLV reconcile, daily-loss check |
| 16:25 | Daily summary to Telegram |
| Outside hours | Answers Telegram commands only |

There is no exchange-holiday calendar. On a UK bank holiday the rebalance runs on the
previous session's bars, and its orders wait for the next open.

---

## Project structure

```
IBKR_Bot/
├── src/
│   ├── __main__.py            # python -m src → bot.main()
│   ├── bot.py                 # Scheduler, live execution loop, risk checks, parity,
│   │                          #   fill notifier + ledger handlers, Telegram wiring
│   ├── engine.py              # Signals → filtered/ranked targets → sizing, top-ups,
│   │                          #   settled-cash gate, drawdown brakes
│   ├── orders.py              # Order placement, trailing stops, stop replacement
│   ├── indicators.py          # TSMOM, CSMOM, ATR, volatility (pure NumPy/pandas)
│   ├── contracts.py           # UCITS contract registry (currency, GBX pence lines)
│   ├── config.py              # All strategy parameters, with rationale in comments
│   ├── connection.py          # IBKR connection manager + reconnect, FX rates
│   ├── data_fetcher.py        # Historical bars from IBKR
│   ├── data_health_checker.py # Market-data probe + self-heal
│   ├── gateway_monitor.py     # Gateway restart via Docker API, 2FA/mid-login guard
│   ├── database.py            # SQLite persistence
│   ├── telegram_bot.py        # Notifications + command handlers
│   ├── backtester.py          # Legacy (scalping-era) backtester
│   └── screener.py            # Legacy US-universe screener — DISABLED, do not re-enable
├── tests/                     # pytest suite (run in the Docker image — see Testing)
├── data/
│   ├── watchlist.json         # Live trading universe (loaded at startup)
│   └── phase1_ucits_mapping.csv
├── research/                  # Study scripts + results behind parameter decisions
├── RESEARCH.md                # Protocol for research / strategy studies
├── research_notes.md          # Study findings, newest first
├── docker-compose.yml         # ib-gateway + trading-bot
├── Dockerfile                 # python:3.12-slim, non-root, copies src/ only
├── requirements.txt           # Pinned direct dependencies
├── constraints.txt            # Full pinned freeze of the live image
├── .env.example               # Configuration template
├── .claude/skills/healthcheck # Operational health-check runbook
└── CLAUDE.md                  # Development guide / pre-flight rules
```

Legacy files from the original US momentum-scalping version, kept for reference and not
used by the live bot: `screener-cron.sh`, `scripts/`, the root-level `test_*.py` smoke
scripts, `DYNAMIC_INSTRUMENTS_BRIEF.md`, `CLAUDE_LOCAL.md` and `bot-health-check-prompt.md`.

---

## Configuration

### Environment (`.env`)

```bash
cp .env.example .env
```

| Variable | Used by | Description |
|----------|---------|-------------|
| `IBKR_USERNAME` / `IBKR_PASSWORD` | gateway | IBKR login |
| `IBKR_TRADING_MODE` | gateway | `paper` or `live`: which IBKR account the **gateway logs into** |
| `IBKR_PORT` | both | Gateway API port: `4001` live, `4002` paper |
| `IBKR_LIVE_CONFIRMED` | bot | **The only switch for real orders.** `true` = live; anything else = dry run |
| `IBKR_HOST` / `IBKR_CLIENT_ID` | bot | Default `127.0.0.1` / `1` |
| `IBKR_TIMEOUT` / `IBKR_READONLY` | bot | Connect timeout (s) / read-only API session |
| `VNC_PASSWORD` | gateway | VNC access to the gateway desktop (keep the port firewalled) |
| `DOCKER_GID` | bot | GID of the host `docker` group, so the bot can restart the gateway (default `988`) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | bot | Telegram alerts + commands (both required to enable) |
| `DB_PATH` / `LOG_PATH` | bot | Default `data/trading.db` / `logs/trading.log` |
| `WATCHLIST_PATH` | bot | Default `data/watchlist.json` |

`IBKR_TRADING_MODE` and `IBKR_LIVE_CONFIRMED` are independent. The first decides which
account the gateway logs into; the second decides whether the bot sends orders.

### Strategy parameters

All strategy parameters live in `TradingConfig` in `src/config.py`: slots, per-name
and class caps, ATR multiplier, volatility floor, cooldown, top-up gates, drawdown and
daily-loss limits, and rebalance time. The comments record why each value was chosen.
The risk limits scale with NLV **except `max_daily_loss`**, which is a fixed GBP amount
and needs re-bumping if capital changes.

### Adding an instrument

1. Add it to `CONTRACT_REGISTRY` in `src/contracts.py` with its currency and primary
   exchange. Check `ContractDetails.priceMagnifier` first: a GBP line with
   magnifier 100 must be registered as `GBX`.
2. Add the symbol to the right class in `data/watchlist.json`.
3. Recreate the bot container (see below). The universe is loaded at startup.

---

## Deployment

### Docker

```bash
docker compose up -d                          # start gateway + bot
docker compose logs -f trading-bot            # follow the bot
```

The bot container runs `python -m src.bot --interval 60`.

**Code or `.env` changes: rebuild only the bot.**

```bash
docker compose up -d --build trading-bot
```

This rebuilds the image and reloads `.env` while leaving `ib-gateway` logged in.

- `docker compose restart` does **not** reload `.env`.
- A full `docker compose down` also stops the gateway, which forces a fresh IBKR login
  and a **manual 2FA approval on IBKR Mobile**.

**2FA:** the gateway soft-restarts nightly at 23:55 and usually re-authenticates on its
own, but IBKR asks for 2FA at least weekly (Sunday night). There is no TOTP key
configured, so approve the push on IBKR Mobile. The bot won't restart the gateway while
that approval is pending.

**Server-side stops survive everything:** trailing stops are GTC orders at IBKR, so
positions stay protected through bot restarts and rebuilds.

### Deploy flow

commit → `git push` → on the server `git pull` → `docker compose up -d --build trading-bot`
→ verify the startup log shows:

- `Subscribed to commissionReportEvent` and `Subscribed to orderStatusEvent`
- `Order-parity: OK`
- `NLV reconcile … drift` < 1%
- `Daily P&L:` inside the cap

Back up `data/trading.db` before any change that writes to it.

### Local run

Needs Python **3.12** (ib_insync fails to import on 3.14) and a running TWS or IB Gateway.

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -c constraints.txt -r requirements.txt

python -m src.bot --once                 # one dry-run analysis cycle, then exit
python -m src.bot --interval 60          # scheduled loop (dry run)
```

| Option | Description |
|--------|-------------|
| `--once` | Connect, reconcile stops, run one rebalance cycle, exit |
| `--interval N` | Minutes between checks (default `60`) |
| `--live` | Request live mode. Still refuses to start unless `IBKR_LIVE_CONFIRMED=true` |
| `--log-file PATH` | Log file (default `logs/trading.log`) |

---

## Telegram

**Commands**

| Command | Description |
|---------|-------------|
| `/positions` (`/status`, `/pos`) | Open positions with live P&L and stop levels |
| `/balance` | Live IBKR account balance |
| `/markets` | Current signals across the watchlist |
| `/health` | Connection and bot health |
| `/pnl` | Today's realized + unrealized P&L (IBKR session figures) and % of the daily cap used |
| `/stats` (`/performance`) | Closed-trade statistics from the live ledger: win rate, payoff, best/worst |
| `/history` | Equity curve and recent stop-fill exits |
| `/help` (`/start`) | Command list |

**Alerts:** bot start/stop, trade opportunities, entries placed, protective stops filled
(with realized P&L), stops placed by reconciliation, orphan stops, NLV drift, daily-loss
and drawdown halts, gateway 2FA waiting for approval, connection failures, and a daily
summary.

---

## Database

SQLite at `data/trading.db` (gitignored).

| Table | Contents |
|-------|----------|
| `trades` | Order ledger: every placed order plus per-fill stop executions. `status` ∈ `SUBMITTED` / `FILLED` / `CANCELLED` / `REJECTED`; `pnl`, `commission`, `currency` on stop fills |
| `instrument_signals` | Per-day TSMOM / CSMOM / combined score, price, ATR, volatility for each instrument |
| `portfolio_snapshots` | Equity, peak equity, drawdown at each rebalance |
| `symbol_cooldowns` | Re-entry cooldown windows set by stop fills |
| `ohlcv` | Cached daily bars |
| `paper_trades` | Dry-run simulated trades only (empty in live mode) |

`SUBMITTED` rows in `trades` should match the stop orders currently working at IBKR.

---

## Monitoring

```bash
# Full log history (docker logs only covers the current container)
tail -f logs/trading.log

# Latest signals
sqlite3 data/trading.db "SELECT symbol, ROUND(combined_score,2), price, ROUND(atr_value,3), ROUND(volatility,3)
  FROM instrument_signals WHERE signal_date=(SELECT MAX(signal_date) FROM instrument_signals)
  ORDER BY combined_score DESC LIMIT 10;"

# Ledger state
sqlite3 data/trading.db "SELECT status, COUNT(*) FROM trades GROUP BY status;"

# Equity & drawdown
sqlite3 data/trading.db "SELECT created_at, equity, peak_equity, drawdown FROM portfolio_snapshots ORDER BY id DESC LIMIT 5;"
```

Useful log lines: `=== DAILY REBALANCE ===`, `Order-parity:`, `NLV reconcile`,
`Daily P&L:`, `Protective stop FILLED`, `Post-rebalance stop sweep`, `Ledger:`.

For live broker state, use a **separate read-only session**. Pick any clientId other
than `1`, which is the bot's, and connect with `readonly=True`.

The full operational runbook is `.claude/skills/healthcheck/SKILL.md`. It covers
process, logs, signals, positions and stop coverage, parity, NLV, P&L, deploy parity
and firewall.

---

## Testing

Because the bot trades live only, tests exercise the **real code paths**. Engine,
orders and bot objects are built with `__new__` plus `SimpleNamespace`/mock
dependencies, and many tests replay real production snapshots and IBKR messages. Run
the suite **inside the Docker image** (Python 3.12):

```bash
docker run --rm --user root -v "$PWD/tests:/app/tests:ro" ibkr_bot-trading-bot:latest \
  sh -c 'pip install -q pytest; cd /app && python -m pytest tests -q'
```

To test uncommitted source without rebuilding, also mount `src`: `-v "$PWD/src:/app/src:ro"`.

Every test is expected to pass; any failure is a regression. (`make test` still runs
the legacy root-level smoke scripts, not this suite.)

Research and parameter studies follow `RESEARCH.md`, and their results are recorded in
`research_notes.md` and `research/`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| BUY rejected **Error 201**, "PRIIPs/KID" | US-listed ETF. Only trade UCITS lines from `contracts.py` |
| BUY rejected **Error 201**, "Available settled cash" | Sale proceeds not settled yet. The bot trims or skips and retries next rebalance |
| SELL stop rejected **Error 201**, "short in cash account" | Stop placed before the BUY filled. The entry-race wait prevents this; reconcile heals it |
| **Error 10147** cancelling a stop | Orders can only be cancelled by the clientId that placed them (the bot's `1`) |
| **Error 10243** | Fractional shares aren't available through the API; sizes round to whole shares |
| A pence-quoted ETF looks unaffordable | It's a GBX line (priceMagnifier 100). Register it as `GBX` |
| Bot sees no data at the open on a Monday | Gateway waiting for weekly 2FA. Approve on IBKR Mobile |
| `.env` change had no effect | Recreate the container with `up -d --build trading-bot`, not `restart` |
| Log greps come back empty after a deploy | `docker logs` only spans the current container. Use `logs/trading.log` |

**Manually closing a bot-held position.** The GTC stop reserves the shares, so a
second SELL is rejected, and only clientId 1 can cancel that stop. Steps:

1. `docker stop trading-bot`. The other positions' stops stay live at IBKR.
2. In a temporary container on the gateway network, connect as clientId 1, cancel the
   stop, then send the market SELL.
3. `docker start trading-bot`. Startup reconciliation re-verifies the rest of the book.

---

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through use of this software.

At this account size, fixed commissions are a major cost, and the momentum signal has
not yet shown a statistically measurable edge on the live record (see
`research_notes.md`). Treat this as an engineering project, not an investment
recommendation.

## License

Private repository - not for public distribution.
