---
name: healthcheck
description: Run a comprehensive health check on the IBKR trend-following trading bot
---

# IBKR Trend-Following Bot Health Check

Run a comprehensive health check on the IBKR trading bot. Work through each section systematically and provide a summary dashboard at the end.

## VPS Details
- Server: 149.102.144.190
- SSH Key: ~/.ssh/id_ed25519_vps
- Containers: ib-gateway, trading-bot, screener
- Path: /root/IBKR_Bot

## 1. PROCESS STATUS
- Are all THREE containers running? (ib-gateway, trading-bot, screener)
- How long have they been running (uptime)?
- Any recent restarts or crashes?
- **When was the last intraday risk check?** (should fire every 4h; alarm if >5h ago during US market hours 14:30–21:00 UTC)

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker ps --format '{{.Names}}\t{{.Status}}' | grep -E 'ib-gateway|trading-bot|screener'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot 2>&1 | grep 'Intraday risk check' | tail -1"
```

## 2. LOG ANALYSIS
- Check the last 100 lines of bot logs for errors, warnings, or anomalies
- Check screener logs for last run status
- Check IB Gateway connection status
- **Unhandled errors in the last 24h** — gateway reconnects are auto-recovered and filtered out; anything that survives the filter is an actual problem (e.g. `Bot error:`, NameError, KeyError)

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot --tail 100 2>&1"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs screener --tail 30 2>&1"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot --since 24h 2>&1 | grep -E 'ERROR|CRITICAL|Bot error:|Traceback' | grep -vE 'Connection timed out|Reconnection attempt|API connection failed|Failed to reconnect|Disconnected from IBKR' | tail -20"
```

## 3. REBALANCE STATUS
- Did today's daily rebalance happen? (scheduled for 15:30 ET / 19:30 UTC)
- When was the last rebalance?
- How many instruments were analyzed?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot 2>&1 | grep -E 'DAILY REBALANCE|Analysis complete|Rebalance:' | tail -10"
```

## 4. SCREENER & WATCHLIST
- When did the screener last run?
- What does the current watchlist look like?
- How many LONG / SHORT / FLAT signals?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cat /root/IBKR_Bot/data/watchlist.json | python3 -c \"
import json, sys
d = json.load(sys.stdin)
print('Strategy:', d.get('strategy', 'N/A'))
print('Updated:', d['updated_at'])
total = sum(len(v) for v in d['symbols'].values())
print('Instruments:', total)
sigs = d.get('signals', {})
longs = sum(1 for s in sigs.values() if s['tsmom_score'] > 0.3)
shorts = sum(1 for s in sigs.values() if s['tsmom_score'] < -0.3)
flat = len(sigs) - longs - shorts
print('Long:', longs, '| Short:', shorts, '| Flat:', flat)
\""
```

## 5. INSTRUMENT SIGNALS
- Check the latest TSMOM/CSMOM signals from the database
- Are signals being recorded each day?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 data/trading.db 'SELECT signal_date, COUNT(*) as instruments, ROUND(AVG(combined_score), 2) as avg_signal FROM instrument_signals GROUP BY signal_date ORDER BY signal_date DESC LIMIT 5;'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 data/trading.db 'SELECT symbol, combined_score, price, atr_value, volatility FROM instrument_signals WHERE signal_date = (SELECT MAX(signal_date) FROM instrument_signals) ORDER BY combined_score DESC LIMIT 10;'"
```

## 6. OPEN POSITIONS & TRAILING-STOP HEALTH

This single query gives you the full picture for each open position:
- `entry`, `best`, `price`, `stored_stop` — the live numbers
- `expected_stop` — what the trail SHOULD be right now (`best − 3×ATR` for longs, `best + 3×ATR` for shorts), using the most recent ATR from `instrument_signals`
- `gap` — how far behind the stored stop is vs expected; should be ~0 if the ratchet is healthy
- `pct_buffer` — % distance from current price to stop (smaller = closer to stopping out)
- `trail_status` — `OK` / `STALE` (gap > 1% of entry, indicating the ratchet hasn't fired) / `?` (no recent ATR signal)

**Action**: any `STALE` row means the trailing-stop ratchet is broken — investigate the risk-check logs for that symbol immediately. This is the check that would have caught the May 2026 incident where all 8 stops sat at entry-day initial values for 9 days.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 -header -column data/trading.db \"
WITH latest_signals AS (
  SELECT symbol, atr_value, price AS latest_price
  FROM instrument_signals
  WHERE signal_date = (SELECT MAX(signal_date) FROM instrument_signals)
)
SELECT
  p.id,
  p.symbol,
  p.action,
  ROUND(p.entry_price, 2) AS entry,
  ROUND(p.best_price, 2) AS best,
  ROUND(s.latest_price, 2) AS price,
  ROUND(p.stop_loss, 2) AS stored_stop,
  ROUND(CASE WHEN p.action='BUY' THEN p.best_price - 3*s.atr_value ELSE p.best_price + 3*s.atr_value END, 2) AS expected_stop,
  ROUND(CASE WHEN p.action='BUY' THEN (p.best_price - 3*s.atr_value) - p.stop_loss ELSE p.stop_loss - (p.best_price + 3*s.atr_value) END, 2) AS gap,
  ROUND(CASE WHEN p.action='BUY' THEN (s.latest_price - p.stop_loss)/s.latest_price*100 ELSE (p.stop_loss - s.latest_price)/s.latest_price*100 END, 1) AS pct_buffer,
  CASE
    WHEN s.atr_value IS NULL THEN '?'
    WHEN p.action='BUY'  AND (p.best_price - 3*s.atr_value) > p.stop_loss + (p.entry_price * 0.01) THEN 'STALE'
    WHEN p.action='SELL' AND (p.best_price + 3*s.atr_value) < p.stop_loss - (p.entry_price * 0.01) THEN 'STALE'
    ELSE 'OK'
  END AS trail_status,
  p.min_exit_date
FROM paper_trades p
LEFT JOIN latest_signals s ON p.symbol = s.symbol
WHERE p.status = 'OPEN'
ORDER BY p.id;
\""
```

## 7. ORDER PARITY (live-mode)

Every live position must have a working TRAIL/STP order, and every working stop must back a held position. The bot heals naked positions automatically (via the startup-reconcile reused at every risk check) and alerts on orphan stops.

- In DRY_RUN: parity is `n/a` — the bot doesn't place IBKR orders.
- In LIVE: any `healed` count > 0 means the bot recovered from a crash mid-rebalance; any `orphans` row means a stop exists for a symbol you're not holding (manual closure, stale order, etc.).

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot 2>&1 | grep -E 'Order-parity:|Orphan protective' | tail -5"
```

## 8. NLV RECONCILIATION (live-mode)

Compares IBKR's live `NetLiquidation` against the most recent `portfolio_snapshots.equity`. Drift ≥ 2% raises a Telegram alert — usually means a stale feed, a manual trade, or a fee/transfer the bot didn't see.

- In DRY_RUN: still useful — paper account's NLV should track the equity snapshot the engine saves each rebalance.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot 2>&1 | grep 'NLV reconcile' | tail -5"
```

## 9. DAILY-LOSS HALT

Tracks today's realized + unrealized P&L. If `session_pnl <= -max_daily_loss` (default −$300), new entries are blocked for the rest of the day (existing positions remain — trail-stops still active). Auto-clears at midnight.

- The `Daily P&L:` log line fires at every risk check and at every rebalance — you'll always see today's running number.
- `DAILY LOSS HALT` only fires once per day on first breach (Telegram alert).
- `Skipped: daily-loss halt active` lines confirm the gate worked.

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot 2>&1 | grep -E 'Daily P&L:|DAILY LOSS HALT|daily-loss halt' | tail -10"
```

## 10. PORTFOLIO & DRAWDOWN
- Check portfolio snapshots and drawdown tracking
- Peak equity, current drawdown level
- Any circuit breaker triggers?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 data/trading.db 'SELECT equity, drawdown, peak_equity, created_at FROM portfolio_snapshots ORDER BY id DESC LIMIT 5;'"
```

## 11. PERFORMANCE METRICS
- Overall paper trade stats (wins, losses, P&L)
- Average win vs average loss size
- Separate old scalping-era trades from new trend-following trades if possible

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 data/trading.db \"SELECT COUNT(*) as total, SUM(CASE WHEN status != 'OPEN' AND pnl_amount > 0 THEN 1 ELSE 0 END) as wins, SUM(CASE WHEN status != 'OPEN' AND pnl_amount <= 0 THEN 1 ELSE 0 END) as losses, SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_now, ROUND(COALESCE(SUM(CASE WHEN status != 'OPEN' THEN pnl_amount END), 0), 2) as total_pnl, ROUND(AVG(CASE WHEN status != 'OPEN' AND pnl_amount > 0 THEN pnl_amount END), 2) as avg_win, ROUND(AVG(CASE WHEN status != 'OPEN' AND pnl_amount < 0 THEN pnl_amount END), 2) as avg_loss FROM paper_trades;\""
```

## 12. SYSTEM RESOURCES
- RAM usage, disk space

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "free -h | head -3 && echo '---' && df -h / | tail -1"
```

## 13. STRATEGY ASSESSMENT
Based on the data gathered above, assess:
- Are the TSMOM/CSMOM signals diverse across asset classes? (not concentrated)
- Is the trailing stop (3x ATR) appropriate for current volatility?
- Are position sizes reasonable given the volatility-scaling?
- Is the drawdown within acceptable limits?
- Any parameter tweaks recommended?

## 14. RECOMMENDATIONS
Provide prioritised recommendations:
- P1 (Critical): Issues that need immediate attention
- P2 (Important): Should be addressed soon
- P3 (Nice to have): Optimisations for later

## 15. SUMMARY DASHBOARD
Present a quick status summary table:

| Check | Status | Notes |
|-------|--------|-------|
| IB Gateway | | |
| Trading Bot | | |
| Screener | | |
| Watchlist/Signals | | |
| Rebalance | | |
| Last Risk Check | | (timestamp + age) |
| Open Positions | | |
| **Trailing Stops** | | (count of OK / STALE / ?) |
| **Order Parity** | | (OK / healed N / N orphans / n/a) |
| **NLV Drift** | | (% drift from last snapshot) |
| **Daily Loss** | | (today's session P&L / halt state) |
| Drawdown | | |
| Logs (errors 24h) | | (count after gateway-reconnect filter) |
| Resources | | |
| Strategy Edge | | |
