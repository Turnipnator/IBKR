---
name: healthcheck
description: Run a comprehensive health check on the IBKR trend-following trading bot
---

# IBKR Trend-Following Bot Health Check

Run a comprehensive health check on the IBKR trading bot. Work through each section systematically and provide a summary dashboard at the end.

## VPS Details
- Server: 149.102.144.190
- SSH Key: ~/.ssh/id_ed25519_vps
- Containers: ib-gateway, trading-bot
- Path: /root/IBKR_Bot

> Note: other containers (`ig-trading-bot`, `betfair-bot`, `horse-racing-bot`, …)
> also run on this VPS — those are **separate bots**, not part of this project.
> Ignore them here.

> **Log-source rule** (memory `bot-log-persistence`): `docker logs trading-bot`
> only spans the **current container** — after a rebuild it starts empty, so
> greps for rebalances/parity/P&L come back blank and look like failures.
> Persistent history lives in `/root/IBKR_Bot/logs/trading.log` (back to
> Jan 2026). Use `docker logs` only for "what happened since the last deploy"
> (e.g. startup lines); use `logs/trading.log` for everything else.

> **Calendar rule**: rebalances and risk checks are gated on LSE market days.
> On weekends and UK bank holidays, "no rebalance today" and "last risk check
> was Friday" are normal, not failures. The weekly IBC gateway restart fires
> **Sunday 23:55 UTC and needs manual 2FA** (no TOTP key) — on a Monday-morning
> check, verify the gateway re-login completed:
> `docker logs ib-gateway --since 40h 2>&1 | grep -iE 'second factor|Login has completed' | tail -5`

## 1. PROCESS STATUS
- Are both containers running? (ib-gateway, trading-bot)
- How long have they been running (uptime)?
- Any recent restarts or crashes?
- **When was the last intraday risk check?** (should fire every 4h; alarm if >5h ago during LSE hours ~07:00–15:30 UTC)

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'ib-gateway|trading-bot'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep 'Intraday risk check' /root/IBKR_Bot/logs/trading.log | tail -2"
```

## 2. LOG ANALYSIS
- Check the last 100 lines of bot logs for errors, warnings, or anomalies
- Check IB Gateway connection status
- **Unhandled errors in the last 24h** — gateway reconnects are auto-recovered and filtered out; anything that survives the filter is an actual problem (e.g. `Bot error:`, NameError, KeyError)

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot --tail 100 2>&1"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot --since 24h 2>&1 | grep -E 'ERROR|CRITICAL|Bot error:|Traceback' | grep -vE 'Connection timed out|Connection reset by peer|Reconnection attempt|API connection failed|Failed to reconnect|Disconnected from IBKR' | tail -20"
```

## 3. REBALANCE STATUS
- Did today's daily rebalance happen? (scheduled ~13:00 UTC, during LSE hours)
- When was the last rebalance?
- How many instruments were analyzed?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep -E 'DAILY REBALANCE|Analysis complete' /root/IBKR_Bot/logs/trading.log | tail -10"
```

## 4. SCREENER (retired) & WATCHLIST (live universe)

The **screener** container was retired when the bot migrated to the fixed
UCITS-on-LSE universe — it still pointed at the old US universe and would
clobber the watchlist (Error 201; see memory `project_screener_us_landmine`).
There is no `screener` container; nothing to check there.

`data/watchlist.json` however **is live** — it defines the trading universe the
engine loads at startup (24 names since IJPN was re-added 2026-08-28, commit
f5208d3). Sanity: the symbol count in the file should match the per-day
instrument count in section 5's signals query (universe changes only apply
after a bot restart).

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "python3 -c \"import json; s=json.load(open('/root/IBKR_Bot/data/watchlist.json'))['symbols']; print({k: len(v) for k,v in s.items()}, sum(len(v) for v in s.values()), 'total')\""
```

## 5. INSTRUMENT SIGNALS
- Check the latest TSMOM/CSMOM signals from the database
- Are signals being recorded each day?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 data/trading.db 'SELECT signal_date, COUNT(*) as instruments, ROUND(AVG(combined_score), 2) as avg_signal FROM instrument_signals GROUP BY signal_date ORDER BY signal_date DESC LIMIT 5;'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 data/trading.db 'SELECT symbol, combined_score, price, atr_value, volatility FROM instrument_signals WHERE signal_date = (SELECT MAX(signal_date) FROM instrument_signals) ORDER BY combined_score DESC LIMIT 10;'"
```

## 6. OPEN POSITIONS & TRAILING-STOP HEALTH

> **Do NOT use `paper_trades` for this in LIVE mode** — that table belongs to
> the retired paper/scalping era and was wiped at the 2026-05-22 cutover; a
> query against it silently returns zero rows and proves nothing. In LIVE the
> source of truth is IBKR itself: positions + working GTC stops via a
> **read-only probe** (`/root/probe3.py`, clientId 17, `readonly=True` — never
> clientId 1, that's the bot's).

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker run --rm --network container:ib-gateway -v /root/probe3.py:/probe.py:ro ibkr_bot-trading-bot:latest python /probe.py 2>&1 | grep -vE '^(Error|Warning) '"
```

The probe prints, per working stop, `STOP <sym> SELL <qty> TRAIL trailStop=<trigger>`,
then per position `held == covered` (`OK` / `GAP`), then totals + NLV. Read it as:

- **Coverage**: every `POS` row must be `OK` (held == covered by working stops)
  with `total_gap=0` and `orphans=[]`. A GAP is a naked position — the bot's
  reconcile should heal it; investigate immediately if it persists.
- **Ratchet sanity**: for each held symbol compare `trailStop` against
  `price − 3×ATR` from the latest `instrument_signals` row (section 5). The
  trigger should sit within ~1–2% of that level (it ratchets off intraday highs
  server-side, so small differences vs the close-based figure are normal). A
  trigger far BELOW it means the ratchet is broken — this is the check that
  would have caught the May 2026 incident where all 8 stops sat at entry-day
  initial values for 9 days.
- **Buffer**: `(price − trailStop)/price` per name — small % = close to
  stopping out (worth mentioning, not a fault).
- Stops are **server-side GTC** — they survive bot restarts and fire with the
  bot down; a firing stop shows up via the `commissionReportEvent` fill
  notifier (section 7).

## 7. ORDER PARITY (live-mode)

Every live position must have a working TRAIL/STP order, and every working stop must back a held position. The bot heals naked positions automatically (via the startup-reconcile reused at every risk check) and alerts on orphan stops.

- In DRY_RUN: parity is `n/a` — the bot doesn't place IBKR orders.
- In LIVE: any `healed` count > 0 means a BUY filled after the 5s wait window (entry-race deferral) or the bot crashed mid-rebalance; any `orphans` row means a stop exists for a symbol you're not holding (manual closure, stale order, etc.).
- Parity is **quantity-aware** since 420d520 (covered shares, not just "a stop exists").
- The **post-rebalance stop sweep** logs `Post-rebalance stop sweep (+Ns): N stop(s) placed` after any rebalance with executions — "0 placed" twice is the healthy case.
- Startup must show `Subscribed to commissionReportEvent for LIVE fill alerts` (the stop-fill notifier — without it stops fire silently).
- On any **top-up**, a `protective stop replaced` line should be preceded by `keeping ratcheted trigger …` whenever the fresh 3×ATR level would have lowered the stop (ratchet-preserving swap, f5208d3).

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep -E 'Order-parity:|Orphan protective|stop sweep|keeping ratcheted trigger' /root/IBKR_Bot/logs/trading.log | tail -8"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot 2>&1 | grep 'commissionReportEvent' | head -1"
```

## 8. NLV RECONCILIATION (live-mode)

Compares IBKR's live `NetLiquidation` against the most recent `portfolio_snapshots.equity`. Drift ≥ 2% raises a Telegram alert — usually means a stale feed, a manual trade, or a fee/transfer the bot didn't see.

- In DRY_RUN: still useful — paper account's NLV should track the equity snapshot the engine saves each rebalance.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep 'NLV reconcile' /root/IBKR_Bot/logs/trading.log | tail -5"
```

## 9. DAILY-LOSS HALT

Tracks today's realized + unrealized P&L. If `session_pnl <= -max_daily_loss` (read the current value from `max_daily_loss` in `src/config.py` — £200 since the £5k step-up, commit 641ee96), new entries are blocked for the rest of the day (existing positions remain — trail-stops still active). Auto-clears at midnight.

- The `Daily P&L:` log line fires at every risk check and at every rebalance — you'll always see today's running number.
- `DAILY LOSS HALT` only fires once per day on first breach (Telegram alert).
- `Skipped: daily-loss halt active` lines confirm the gate worked.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep -E 'Daily P&L:|DAILY LOSS HALT|daily-loss halt' /root/IBKR_Bot/logs/trading.log | tail -10"
```

## 10. PORTFOLIO & DRAWDOWN
- Check portfolio snapshots and drawdown tracking
- Peak equity, current drawdown level
- Any circuit breaker triggers?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 data/trading.db 'SELECT equity, drawdown, peak_equity, created_at FROM portfolio_snapshots ORDER BY id DESC LIMIT 5;'"
```

## 11. PERFORMANCE METRICS (live tape)

> `paper_trades` is dead in LIVE mode (see section 6). The live closed-trade
> tape is the `Protective stop FILLED` lines in `logs/trading.log` — each
> carries IBKR's own `realizedPnL` and `commission` (account-base GBP,
> written by the `commissionReportEvent` notifier since fbff002). Note: a fill
> occasionally logs twice (one line per commission report on partial fills) —
> dedupe by orderId when counting.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep 'Protective stop FILLED' /root/IBKR_Bot/logs/trading.log | tail -15"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 -header data/trading.db 'SELECT status, COUNT(*) FROM trades GROUP BY status;'"
```

Assess: win count vs loss count, avg win vs avg loss (payoff ratio), and
**commission as % of the average risk unit** — at this account size fee drag,
not signal quality, has been the dominant cost (memory
`commission-drag-small-capital`); check fee-to-risk before blaming signals.

## 12. SYSTEM RESOURCES
- RAM usage, disk space

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "free -h | head -3 && echo '---' && df -h / | tail -1"
```

## 13. DEPLOY PARITY

Deploy flow is commit → `git push` → VPS `git pull` → bot-only rebuild (memory
`github-push-works-deploy-via-pull`). Verify the three copies agree:

- **VPS tree** — clean and on the same commit as local/origin. (Known benign
  untracked files on the VPS: `.env.pre-*` backups, `data/backups/`,
  `data/trading.db.pre-*` — ignore those; anything else is drift.)
- **Running image** — built at/after the latest commit (a pulled-but-not-rebuilt
  bot silently runs old code).

```bash
git log --oneline -1
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && git log --oneline -1 && git status --porcelain | grep -vE '^\?\? (\.env\.pre-|data/)' ; docker inspect --format 'image built: {{.Created}}' ibkr_bot-trading-bot:latest ; docker inspect --format 'bot started: {{.State.StartedAt}}' trading-bot"
```

**Optional (after test changes or when suspicious): run the in-image suite** —
the local Mac venv is Python 3.14 where `ib_insync` fails to import; tests only
run in-image:

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker run --rm --user root -v /root/IBKR_Bot/tests:/app/tests:ro ibkr_bot-trading-bot:latest sh -c 'pip install -q pytest 2>/dev/null; cd /app && python -m pytest tests -q 2>&1 | tail -3'"
```

Expected: **all passing** (106 as of 2026-08-31). Any failure is a regression —
there is no longer a known-bad set to ignore.

## 14. NETWORK SECURITY

Verify the VPS firewall is in place. Locked down on 2026-05-13 after finding VNC (5900), IBKR Gateway API (4001), and socat (4003) all publicly listening with default-password VNC. Only port 22 (SSH) should be allowed inbound.

- **`Status: active`** is mandatory. If it shows `inactive`, the firewall was disabled and the IB Gateway desktop is exposed to the internet again.
- The allow list should contain **only** `22/tcp` (v4 and v6). Any extra `ALLOW IN` rule means another port has been opened — investigate why.
- Outbound default must remain `allow` (needed for IBKR API, Telegram long-polling, broker calls).

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "ufw status verbose | head -15"
```

**Optional external sanity check** (run from your local machine, not the VPS):

```bash
for p in 5900 4001 4003; do
  nc -z -w 3 149.102.144.190 $p && echo "$p OPEN (BAD)" || echo "$p blocked (good)"
done
nc -z -w 3 149.102.144.190 22 && echo "22 OPEN (good)" || echo "22 BLOCKED (BAD)"
```

## 15. STRATEGY ASSESSMENT
Read the current parameters from `src/config.py` first (slots, per-name cap,
class cap, ATR multiplier, vol floor, cooldown — they change; don't assess
against remembered values). Then assess:
- Are the TSMOM/CSMOM signals diverse across asset classes? (not concentrated)
- Is the trailing stop appropriate for current volatility?
- Are position sizes reasonable — and what is commission as % of risk-to-stop?
- Is the drawdown within acceptable limits?
- Any parameter tweaks recommended? (Parameter changes are the user's call —
  recommend, don't apply.)

## 16. RECOMMENDATIONS
Provide prioritised recommendations:
- P1 (Critical): Issues that need immediate attention
- P2 (Important): Should be addressed soon
- P3 (Nice to have): Optimisations for later

## 17. SUMMARY DASHBOARD
Present a quick status summary table:

| Check | Status | Notes |
|-------|--------|-------|
| IB Gateway | | (uptime + login state; Mondays: did the Sunday 23:55 2FA complete?) |
| Trading Bot | | |
| Signals | | (instruments recorded on last market day + concentration) |
| Rebalance | | |
| Last Risk Check | | (timestamp + age, vs market calendar) |
| Open Positions | | (probe: N positions, held == covered) |
| **Trailing Stops** | | (probe coverage gap + ratchet sanity vs price−3×ATR) |
| **Order Parity** | | (OK / healed N / N orphans / n/a) |
| **NLV Drift** | | (% drift from last snapshot) |
| **Daily Loss** | | (today's session P&L / halt state) |
| Drawdown | | |
| Logs (errors 24h) | | (count after gateway-reconnect filter) |
| **Deploy Parity** | | (local == origin == VPS == running image; tests green) |
| Resources | | |
| **Firewall (ufw)** | | (active + only 22 allowed / or flag) |
| Strategy Edge | | |
