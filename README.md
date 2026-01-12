# IBKR Trading Bot

An automated trading bot for Interactive Brokers (IBKR) that implements a momentum scalping strategy across Precious Metals, AI, and Tech sectors.

## Overview

This bot connects to Interactive Brokers via their API and executes trades based on technical analysis signals. It runs continuously during market hours, analyzing price action and executing trades when conditions align.

**Current Mode:** Paper trading (dry run) for strategy validation.

## Trading Strategy

### Momentum Scalping

The strategy is optimised for quick, high-probability trades:

- **Take Profit:** 1.5% (lock in gains quickly)
- **Stop Loss:** 3% (wider to avoid noise and false breakouts)
- **Timeframe:** 5-minute candles
- **Trade Direction:** Long positions only (BULLISH trend requirement)

### Asset Universe

| Sector | Symbols |
|--------|---------|
| Precious Metals | GLD, SLV |
| AI | NVDA, AMD, GOOGL, MSFT |
| Tech | AAPL, TSLA, META, AMZN |

### Entry Criteria

A trade is triggered when multiple indicators align (minimum 50% signal strength):

1. **EMA Alignment** - Fast EMA (9) > Slow EMA (21) > Trend EMA (50)
2. **RSI Confirmation** - Not overbought (RSI < 70)
3. **MACD Crossover** - MACD line above signal line with positive histogram
4. **Bollinger Bands** - Price position within bands
5. **Volume Confirmation** - Current volume >= average volume
6. **Trend Filter** - Overall trend must be BULLISH

### Technical Indicators

All indicators are calculated using pure Python/NumPy (no TA-Lib dependency):

- **EMA** (9, 21, 50) - Exponential Moving Averages for trend detection
- **RSI** (7-period) - Shorter period for faster scalping signals
- **MACD** (8, 17, 9) - Faster settings for momentum detection
- **Bollinger Bands** (10-period, 2 std) - Volatility and mean reversion
- **ATR** (7-period) - Volatility measurement
- **Stochastic** (5, 3) - Momentum oscillator

### Risk Management

- **Position Sizing:** Max 10% of portfolio per position
- **Sector Exposure:** Max 40% in any single sector
- **Cooldown Period:** 20 minutes after a stop loss is hit
- **Daily Trade Limit:** Max 3 trades per symbol per day
- **Daily Loss Limit:** Stops trading if daily losses exceed threshold
- **Volume Filter:** Minimum 100,000 volume requirement

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Environment                    │
│  ┌─────────────────┐      ┌─────────────────────────┐  │
│  │   IB Gateway    │◄────►│     Trading Bot         │  │
│  │   (Port 4002)   │      │                         │  │
│  │                 │      │  ┌─────────────────┐    │  │
│  │  - IBKR Auth    │      │  │ Decision Engine │    │  │
│  │  - API Proxy    │      │  └────────┬────────┘    │  │
│  └─────────────────┘      │           │             │  │
│                           │  ┌────────▼────────┐    │  │
│                           │  │ Technical       │    │  │
│                           │  │ Analysis        │    │  │
│                           │  └────────┬────────┘    │  │
│                           │           │             │  │
│                           │  ┌────────▼────────┐    │  │
│                           │  │ Order Manager   │    │  │
│                           │  └────────┬────────┘    │  │
│                           │           │             │  │
│                           │  ┌────────▼────────┐    │  │
│                           │  │ SQLite Database │    │  │
│                           │  └─────────────────┘    │  │
│                           └─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    Telegram     │
                    │  Notifications  │
                    └─────────────────┘
```

## Project Structure

```
IBKR_Bot/
├── src/
│   ├── __init__.py
│   ├── __main__.py      # Entry point
│   ├── bot.py           # Main bot loop and scheduling
│   ├── config.py        # Configuration management
│   ├── connection.py    # IBKR connection handling
│   ├── data_fetcher.py  # Historical data retrieval
│   ├── database.py      # SQLite persistence
│   ├── engine.py        # Decision engine and trade logic
│   ├── indicators.py    # Technical analysis indicators
│   ├── orders.py        # Order and position management
│   └── telegram_bot.py  # Telegram notifications and commands
├── data/                # SQLite database (gitignored)
├── logs/                # Log files (gitignored)
├── docker-compose.yml   # Docker orchestration
├── Dockerfile           # Bot container definition
├── requirements.txt     # Python dependencies
├── .env.example         # Configuration template
└── CLAUDE.md            # Development guide
```

## Prerequisites

- **IBKR Account** with API access enabled
- **Docker** and Docker Compose (for containerised deployment)
- **Python 3.12+** (for local development)
- **Telegram Bot** (optional, for notifications)

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your credentials:
   ```bash
   # IBKR Credentials
   IBKR_USERNAME=your_username
   IBKR_PASSWORD=your_password
   IBKR_TRADING_MODE=paper  # or 'live'

   # Telegram (optional)
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `IBKR_HOST` | IBKR Gateway host | `127.0.0.1` |
| `IBKR_PORT` | IBKR Gateway port | `4002` (paper) |
| `IBKR_CLIENT_ID` | API client ID | `1` |
| `IBKR_TRADING_MODE` | `paper` or `live` | `paper` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | - |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | - |
| `DB_PATH` | SQLite database path | `data/trading.db` |
| `LOG_PATH` | Log file path | `logs/trading.log` |

## Deployment

### Docker (Recommended)

```bash
# Start all services (IB Gateway + Trading Bot)
docker compose up -d

# View logs
docker compose logs -f trading-bot

# Stop services
docker compose down
```

**Important:** When changing `.env` values, you must rebuild:
```bash
docker compose down && docker compose up -d --build
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run bot (requires TWS or IB Gateway running locally)
python -m src.bot --interval 5
```

### Command Line Options

```bash
python -m src.bot [OPTIONS]

Options:
  --once          Run analysis once and exit
  --interval N    Run every N minutes (default: 5)
  --live          Enable live trading (default: dry run)
```

## Telegram Integration

The bot supports Telegram for notifications and commands.

### Notifications

- Trade opportunities detected
- Paper trades opened/closed
- Stop loss and take profit hits
- Daily performance summaries
- Connection failure alerts

### Commands

Send these commands to your Telegram bot:

| Command | Description |
|---------|-------------|
| `/positions` | Show open paper trades |
| `/stats` | Show trading statistics |
| `/help` | List available commands |

## Database

The bot uses SQLite for persistence:

- **OHLCV Data** - Historical price data cache
- **Trade Log** - All executed trades
- **Paper Trades** - Simulated trade tracking with P&L
- **Cooldowns** - Symbol-level trading cooldowns
- **Daily Counts** - Trade count limits per symbol

## Market Hours

The bot only operates during US market hours:
- **Open:** 9:30 AM Eastern
- **Close:** 4:00 PM Eastern
- **Days:** Monday to Friday (excludes holidays)

Outside these hours, the bot sleeps and does not attempt to connect to IBKR (prevents weekend login lockouts).

## Monitoring

### Logs

```bash
# Docker logs
docker compose logs -f trading-bot

# Log file (if running locally)
tail -f logs/trading.log
```

### Database Queries

```bash
# Recent paper trades
sqlite3 data/trading.db "SELECT * FROM paper_trades ORDER BY id DESC LIMIT 10;"

# Open positions
sqlite3 data/trading.db "SELECT * FROM paper_trades WHERE status = 'OPEN';"

# Trade statistics
sqlite3 data/trading.db "SELECT COUNT(*), SUM(pnl_amount) FROM paper_trades WHERE status != 'OPEN';"
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `ib_insync` | Interactive Brokers API wrapper |
| `pandas` | Data manipulation |
| `numpy` | Numerical computations |
| `python-telegram-bot` | Telegram integration |
| `python-dotenv` | Environment variable management |

## Troubleshooting

### Connection Issues

1. **"Too many failed login attempts"** - Wait 15 minutes, then restart. Often caused by weekend connection attempts.

2. **Gateway unhealthy** - Check IB Gateway logs:
   ```bash
   docker compose logs ib-gateway
   ```

3. **Connection refused** - Ensure IB Gateway is healthy before starting the bot.

### No Trades Executing

1. Check market hours (bot only trades 9:30 AM - 4:00 PM ET)
2. Verify signal strength meets threshold (currently 50%)
3. Check volume requirements (need >= average volume)
4. Review trend requirement (BULLISH only)

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through use of this software.

**Always test thoroughly with paper trading before considering live deployment.**

## License

Private repository - not for public distribution.
