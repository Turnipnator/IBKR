# IBKR Trading Bot - Development Context

This file serves as persistent memory for Claude Code sessions working on this project.

## Project Overview

- **Purpose**: Automated trading bot for IBKR paper/live trading
- **Strategy**: Technical analysis (SMA, RSI, MACD) on Precious Metals, AI, and Tech sectors
- **Broker**: Interactive Brokers via ib_insync library
- **Notifications**: Telegram bot for trade alerts and status updates

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         VPS (Contabo)                       │
│  ┌─────────────────┐     ┌─────────────────────────────┐   │
│  │   IB Gateway    │◄───►│      Trading Bot            │   │
│  │  (Docker/host)  │     │  ┌─────────────────────┐    │   │
│  │                 │     │  │ Decision Engine     │    │   │
│  │  Port 4002      │     │  │ - Data Fetcher      │    │   │
│  │  (paper)        │     │  │ - Indicators        │    │   │
│  └─────────────────┘     │  │ - Order Manager     │    │   │
│                          │  └─────────────────────┘    │   │
│                          │           │                 │   │
│                          │           ▼                 │   │
│                          │  ┌─────────────────────┐    │   │
│                          │  │ Telegram Notifier   │────┼───┼──► User
│                          │  └─────────────────────┘    │   │
│                          └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `src/bot.py` | Main entry point, CLI interface, scheduling |
| `src/engine.py` | Decision engine, trade opportunity detection |
| `src/indicators.py` | Technical analysis (SMA, EMA, RSI, MACD, etc.) |
| `src/orders.py` | Order placement and position management |
| `src/data_fetcher.py` | Historical data from IBKR |
| `src/connection.py` | IBKR connection management with reconnect |
| `src/telegram_bot.py` | Telegram notifications (HTTP-based) |
| `src/database.py` | SQLite storage for OHLCV and trades |
| `src/config.py` | Configuration from environment variables |

## Trading Universe

```python
TRADING_UNIVERSE = {
    "Precious Metals": ["GLD", "SLV", "NEM"],
    "AI": ["NVDA", "AMD", "GOOGL", "MSFT"],
    "Tech": ["AAPL", "AMZN", "TSLA", "META"],
}
```

## Trading Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Max position size | 10% of portfolio | `src/config.py` |
| Max sector exposure | 40% | `src/config.py` |
| Stop loss | 10% | `src/engine.py` |
| Take profit | 25% | `src/engine.py` |
| SMA fast period | 50 days | `src/indicators.py` |
| SMA slow period | 200 days | `src/indicators.py` |
| RSI period | 14 days | `src/indicators.py` |
| RSI overbought | 70 | `src/indicators.py` |
| RSI oversold | 30 | `src/indicators.py` |

## Deployment

### VPS Details
- **Provider**: Contabo
- **IP**: Set in local environment (not committed)
- **SSH Key**: Set in local environment (not committed)
- **Project Path**: `/root/IBKR_Bot`
- **Network Mode**: Host (both containers share localhost)

### Docker Setup
- **IB Gateway**: `ghcr.io/gnzsnz/ib-gateway:stable` on host network
- **Trading Bot**: Custom build on host network
- **Connection**: Bot connects to `127.0.0.1:4002` (paper) or `4001` (live)

### Common Commands

```bash
# SSH to VPS (replace with your details)
ssh -i <YOUR_SSH_KEY> root@<YOUR_VPS_IP>

# On VPS - Container management
cd /root/IBKR_Bot
docker compose ps              # Check status
docker compose logs -f         # Stream all logs
docker compose logs trading-bot --tail=50  # Bot logs
docker compose restart trading-bot         # Restart bot only
docker compose down && docker compose up -d  # Full restart

# Sync code changes to VPS (replace with your details)
rsync -avz -e "ssh -i <YOUR_SSH_KEY>" \
  /path/to/IBKR_Bot/src/ \
  root@<YOUR_VPS_IP>:/root/IBKR_Bot/src/

# Rebuild after code changes
ssh -i <YOUR_SSH_KEY> root@<YOUR_VPS_IP> \
  "cd /root/IBKR_Bot && docker compose build trading-bot && docker compose up -d trading-bot"
```

## Local Development

### Prerequisites
- TWS or IB Gateway running locally on port 7497 (paper)
- Python 3.12+ with virtual environment

### Setup
```bash
cd /Users/paulturner/IBKR_Bot
source venv/bin/activate
pip install -r requirements.txt
```

### Testing
```bash
# Test IBKR connection
python test_connection.py

# Test data fetching
python test_data_layer.py

# Test indicators
python test_indicators.py

# Test Telegram (requires .env with credentials)
python test_telegram.py

# Run bot once (dry run)
python -m src.bot --once

# Run bot once (live mode - REAL TRADES)
python -m src.bot --once --live
```

## Known Issues & Fixes

### 1. IB Gateway TrustedIPs
**Problem**: IB Gateway only trusts 127.0.0.1 by default
**Solution**: Use `network_mode: host` in docker-compose.yml so both containers share localhost

### 2. Python 3.14 Event Loop
**Problem**: `RuntimeError: There is no current event loop in thread 'MainThread'`
**Solution**: Add at top of files that use ib_insync:
```python
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
```

### 3. Telegram Async Issues
**Problem**: python-telegram-bot library conflicts with ib_insync event loop
**Solution**: Use simple HTTP requests via urllib instead of async library

### 4. Insufficient Data for 200-day SMA
**Problem**: Default 6-month fetch doesn't provide enough data
**Solution**: Fetch 1 year of data (`durationStr='1 Y'`)

### 5. Docker Log Rotation
**Problem**: Logs can fill disk on VPS
**Solution**: Configure Docker daemon with log rotation:
```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

## Environment Variables

Required in `.env` file (create from `.env.example`):

```
# IBKR Connection
IBKR_HOST=127.0.0.1          # Local: 127.0.0.1, VPS: 127.0.0.1 (host network)
IBKR_PORT=4002               # 4002=paper, 4001=live
IBKR_CLIENT_ID=1

# IBKR Credentials (VPS only, for IB Gateway auto-login)
IBKR_USERNAME=<your_username>
IBKR_PASSWORD=<your_password>
IBKR_TRADING_MODE=paper      # paper or live

# Telegram
TELEGRAM_BOT_TOKEN=<from_botfather>
TELEGRAM_CHAT_ID=<your_chat_id>

# Paths
DB_PATH=data/trading.db
LOG_PATH=logs/trading.log
```

## Signal Logic

**BUY Signal** (all must be true):
- Fast SMA (50) > Slow SMA (200) - bullish trend
- RSI < 70 - not overbought
- MACD line > Signal line - bullish momentum

**SELL Signal** (any):
- RSI > 70 - overbought
- MACD line < Signal line with bearish crossover
- Price below stop loss

## Future Enhancements

- [ ] Add sentiment analysis from news APIs
- [ ] Implement trailing stop losses
- [ ] Add more sophisticated position sizing (Kelly criterion)
- [ ] Dashboard with Streamlit
- [ ] Backtesting framework
- [ ] Support for options trading
- [ ] Multi-account support

## Account Info

- **Paper Account**: DUO390812
- **Base Currency**: GBP
- **Market Hours**: US markets 9:30 AM - 4:00 PM EST
