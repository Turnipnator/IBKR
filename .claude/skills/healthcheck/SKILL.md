---
name: healthcheck
description: Run a comprehensive health check on the IBKR trading bot
---

# IBKR Trading Bot Health Check

Run a comprehensive health check on the IBKR trading-bot. Work through each section systematically and provide a summary dashboard at the end.

## VPS Details
- Server: 149.102.144.190
- SSH Key: ~/.ssh/id_ed25519_vps
- Containers: ib-gateway, trading-bot
- Path: /root/IBKR_Bot

## 1. PROCESS STATUS
- Are both containers running? (ib-gateway AND trading-bot)
- How long have they been running (uptime)?
- Any recent restarts or crashes?

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker ps --format '{{.Names}}\t{{.Status}}\t{{.RunningFor}}' | grep -E 'ib-gateway|trading-bot'"
```

## 2. LOG ANALYSIS
- Check the last 100 lines of logs for errors, warnings, or anomalies
- Identify any recurring error patterns
- Check IB Gateway connection status

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot --tail 100 2>&1"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs ib-gateway --tail 50 2>&1"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker logs trading-bot 2>&1 | grep -iE 'error|warn|fail|disconnect|connection' | tail -20"
```

## 3. SIGNAL GENERATION
- Is the bot actively producing trading signals?
- What was the last signal generated and when?
- Check data files and SQLite database

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "ls -la /root/IBKR_Bot/data/"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 data/trading.db 'SELECT * FROM paper_trades ORDER BY id DESC LIMIT 5;' 2>/dev/null || echo 'No trades DB'"
```

## 4. PERFORMANCE METRICS
- Check current trades/positions
- Review recent P&L from database

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "cd /root/IBKR_Bot && sqlite3 data/trading.db 'SELECT COUNT(*) as total, SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, SUM(pnl) as total_pnl FROM paper_trades;' 2>/dev/null || echo 'No trades'"
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "docker exec trading-bot cat /app/data/positions.json 2>/dev/null || echo 'No positions file'"
```

## 5. SYSTEM RESOURCES
- RAM usage, disk space, CPU usage

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "free -h && echo '---' && df -h / && echo '---' && top -bn1 | head -12"
```

## 6. CONFIGURATION REVIEW
- Check key environment variables are set correctly
- Verify IB Gateway settings

```bash
ssh -i ~/.ssh/id_ed25519_vps root@149.102.144.190 "grep -E 'ENABLE_|MODE|PAPER|TRADING' /root/IBKR_Bot/.env 2>/dev/null | head -15"
```

## 7. IBKR-SPECIFIC CHECKS
- IB Gateway connection status (is TWS/Gateway authenticated?)
- Market data subscription status
- Account margin usage
- Any order rejection issues

## 8. STRATEGY EDGE ASSESSMENT
- Calculate win rate from database
- Is the strategy performing as expected?
- Any parameter tweaks recommended?

## 9. RECOMMENDATIONS
Provide prioritised recommendations:
- P1 (Critical): Issues that need immediate attention
- P2 (Important): Should be addressed soon
- P3 (Nice to have): Optimisations for later

## 10. SUMMARY DASHBOARD
Present a quick status summary table:

| Check | Status | Notes |
|-------|--------|-------|
| IB Gateway Running | ?/? | |
| Trading Bot Running | ?/? | |
| Logs Healthy | ?/?/? | |
| Signals Active | ?/? | |
| Resources OK | ?/?/? | |
| Strategy Edge | ?/?/? | |

Traffic light summary: ? All good / ? Minor issues / ? Needs attention
