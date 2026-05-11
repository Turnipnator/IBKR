"""
SQLite database layer for storing market data, trade history, and portfolio tracking.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

from .config import data_config

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database manager for market data, trades, and portfolio tracking.

    Usage:
        db = Database()
        db.save_ohlcv(df, "AAPL")
        df = db.load_ohlcv("AAPL", days=30)
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or data_config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        try:
            conn.executescript("""
                -- OHLCV price data
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    average REAL,
                    bar_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                );

                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date
                ON ohlcv(symbol, date);

                -- Trade log
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    order_id INTEGER,
                    status TEXT,
                    reason TEXT,
                    executed_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Account snapshots
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    net_liquidation REAL,
                    total_cash REAL,
                    buying_power REAL,
                    currency TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Paper trades for tracking dry run performance
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'OPEN',
                    exit_price REAL,
                    pnl_amount REAL,
                    pnl_percent REAL,
                    reasons TEXT,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_paper_trades_status
                ON paper_trades(status);

                -- Portfolio snapshots for drawdown tracking
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equity REAL NOT NULL,
                    drawdown REAL NOT NULL DEFAULT 0.0,
                    peak_equity REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Daily instrument signals (audit trail)
                CREATE TABLE IF NOT EXISTS instrument_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    tsmom_score REAL,
                    csmom_score REAL,
                    combined_score REAL,
                    price REAL,
                    atr_value REAL,
                    volatility REAL,
                    signal_date TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_instrument_signals_date
                ON instrument_signals(signal_date, symbol);
            """)
            conn.commit()

            # Migrations for existing databases
            for migration_sql in [
                "ALTER TABLE paper_trades ADD COLUMN best_price REAL",
                "ALTER TABLE paper_trades ADD COLUMN atr_stop REAL",
                "ALTER TABLE paper_trades ADD COLUMN min_exit_date TEXT",
                "ALTER TABLE paper_trades ADD COLUMN signal_score REAL",
            ]:
                try:
                    conn.execute(migration_sql)
                    conn.commit()
                except Exception:
                    pass  # Column already exists

            logger.info(f"Database initialized at {self.db_path}")
        finally:
            conn.close()

    # ==================== OHLCV Methods ====================

    def save_ohlcv(self, df: pd.DataFrame, symbol: Optional[str] = None):
        """Save OHLCV data to database."""
        if df is None or df.empty:
            return

        conn = self._get_connection()
        try:
            for _, row in df.iterrows():
                sym = symbol or row.get("symbol", "UNKNOWN")
                date_val = row["date"]
                if isinstance(date_val, pd.Timestamp):
                    date_val = date_val.isoformat()

                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, date, open, high, low, close, volume, average, bar_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sym, date_val,
                    row.get("open"), row.get("high"), row.get("low"),
                    row.get("close"), row.get("volume"),
                    row.get("average"), row.get("barCount"),
                ))
            conn.commit()
            logger.info(f"Saved {len(df)} bars for {symbol or 'multiple symbols'}")
        finally:
            conn.close()

    def load_ohlcv(
        self, symbol: str,
        days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from database."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM ohlcv WHERE symbol = ?"
            params = [symbol]
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            query += " ORDER BY date DESC"
            if days:
                query += f" LIMIT {days}"
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
            return df
        finally:
            conn.close()

    # ==================== Trade Methods ====================

    def save_trade(
        self, symbol: str, action: str, quantity: int, price: float,
        order_id: Optional[int] = None, status: str = "PENDING",
        reason: Optional[str] = None,
    ):
        """Log a trade to the database."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO trades
                (symbol, action, quantity, price, order_id, status, reason, executed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action, quantity, price, order_id, status, reason,
                  datetime.now().isoformat()))
            conn.commit()
        finally:
            conn.close()

    # ==================== Paper Trade Methods ====================

    def save_paper_trade(
        self, symbol: str, action: str, quantity: int, entry_price: float,
        stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
        reasons: Optional[list[str]] = None, signal_score: Optional[float] = None,
        min_hold_days: int = 0,
    ) -> int:
        """Save a new paper trade. Returns the trade ID."""
        import json
        now = datetime.now()
        min_exit = (now + timedelta(days=min_hold_days)).isoformat() if min_hold_days > 0 else None

        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO paper_trades
                (symbol, action, quantity, entry_price, stop_loss, take_profit,
                 reasons, entry_time, status, signal_score, min_exit_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            """, (
                symbol, action, quantity, entry_price, stop_loss, take_profit,
                json.dumps(reasons) if reasons else None,
                now.isoformat(), signal_score, min_exit,
            ))
            conn.commit()
            trade_id = cursor.lastrowid
            logger.info(f"Saved paper trade #{trade_id}: {action} {quantity} {symbol} @ ${entry_price:.2f}")
            return trade_id
        finally:
            conn.close()

    def get_open_paper_trades(self) -> list[dict]:
        """Get all open paper trades."""
        import json
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM paper_trades WHERE status = 'OPEN' ORDER BY entry_time DESC"
            )
            trades = []
            for row in cursor.fetchall():
                trade = dict(row)
                if trade.get('reasons'):
                    trade['reasons'] = json.loads(trade['reasons'])
                trades.append(trade)
            return trades
        finally:
            conn.close()

    def close_paper_trade(self, trade_id: int, exit_price: float, status: str) -> dict:
        """Close a paper trade and calculate P&L."""
        import json
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            if not row:
                return {}

            trade = dict(row)
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            action = trade['action']

            if action == 'BUY':
                pnl_amount = (exit_price - entry_price) * quantity
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            else:
                pnl_amount = (entry_price - exit_price) * quantity
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100

            conn.execute("""
                UPDATE paper_trades
                SET status = ?, exit_price = ?, pnl_amount = ?, pnl_percent = ?, exit_time = ?
                WHERE id = ?
            """, (status, exit_price, pnl_amount, pnl_percent, datetime.now().isoformat(), trade_id))
            conn.commit()

            trade.update({
                'status': status, 'exit_price': exit_price,
                'pnl_amount': pnl_amount, 'pnl_percent': pnl_percent,
            })
            if trade.get('reasons'):
                trade['reasons'] = json.loads(trade['reasons'])

            logger.info(f"Closed paper trade #{trade_id}: {status} @ ${exit_price:.2f} (P&L: ${pnl_amount:.2f})")
            return trade
        finally:
            conn.close()

    def update_paper_trade_stop(self, trade_id: int, new_stop_loss: float, best_price: float):
        """Update trailing stop loss and best price for an open paper trade."""
        conn = self._get_connection()
        try:
            conn.execute("""
                UPDATE paper_trades SET stop_loss = ?, best_price = ?
                WHERE id = ? AND status = 'OPEN'
            """, (new_stop_loss, best_price, trade_id))
            conn.commit()
        finally:
            conn.close()

    def has_open_paper_trade(self, symbol: str) -> bool:
        """Check if there's already an open paper trade for a symbol."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE symbol = ? AND status = 'OPEN'",
                (symbol,)
            )
            return cursor.fetchone()[0] > 0
        finally:
            conn.close()

    def get_paper_trade_stats(self) -> dict:
        """Get summary statistics for paper trades."""
        conn = self._get_connection()
        try:
            stats = {
                'total_trades': 0, 'open_trades': 0, 'closed_trades': 0,
                'winning_trades': 0, 'losing_trades': 0,
                'total_pnl': 0.0, 'win_rate': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0,
            }
            cursor = conn.execute("SELECT COUNT(*) FROM paper_trades")
            stats['total_trades'] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status = 'OPEN'")
            stats['open_trades'] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status != 'OPEN'")
            stats['closed_trades'] = cursor.fetchone()[0]

            cursor = conn.execute("""
                SELECT COUNT(*), SUM(pnl_amount), AVG(pnl_amount)
                FROM paper_trades WHERE status != 'OPEN' AND pnl_amount > 0
            """)
            row = cursor.fetchone()
            stats['winning_trades'] = row[0] or 0
            stats['avg_win'] = row[2] or 0.0

            cursor = conn.execute("""
                SELECT COUNT(*), SUM(pnl_amount), AVG(pnl_amount)
                FROM paper_trades WHERE status != 'OPEN' AND pnl_amount <= 0
            """)
            row = cursor.fetchone()
            stats['losing_trades'] = row[0] or 0
            stats['avg_loss'] = row[2] or 0.0

            cursor = conn.execute("SELECT SUM(pnl_amount) FROM paper_trades WHERE status != 'OPEN'")
            stats['total_pnl'] = cursor.fetchone()[0] or 0.0

            if stats['closed_trades'] > 0:
                stats['win_rate'] = (stats['winning_trades'] / stats['closed_trades']) * 100

            return stats
        finally:
            conn.close()

    def get_daily_pnl(self) -> float:
        """Get total P&L for trades closed today."""
        today = datetime.now().strftime('%Y-%m-%d')
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT COALESCE(SUM(pnl_amount), 0) FROM paper_trades
                WHERE status != 'OPEN' AND date(exit_time) = ?
            """, (today,))
            return cursor.fetchone()[0] or 0.0
        finally:
            conn.close()

    # ==================== Portfolio Tracking ====================

    def save_portfolio_snapshot(self, equity: float, drawdown: float, peak_equity: float):
        """Save a portfolio snapshot for drawdown tracking."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO portfolio_snapshots (equity, drawdown, peak_equity)
                VALUES (?, ?, ?)
            """, (equity, drawdown, peak_equity))
            conn.commit()
        finally:
            conn.close()

    def get_peak_equity(self) -> float:
        """Get the highest recorded equity (for drawdown calculation)."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT MAX(peak_equity) FROM portfolio_snapshots"
            )
            row = cursor.fetchone()
            return row[0] or 0.0
        finally:
            conn.close()

    def get_current_drawdown(self) -> float:
        """Get the most recent drawdown reading."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT drawdown FROM portfolio_snapshots ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return row[0] if row else 0.0
        finally:
            conn.close()

    def get_latest_portfolio_snapshot(self) -> Optional[dict]:
        """Get the most recent portfolio snapshot, or None if none exist."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT equity, drawdown, peak_equity, created_at "
                "FROM portfolio_snapshots ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "equity": row[0],
                "drawdown": row[1],
                "peak_equity": row[2],
                "created_at": row[3],
            }
        finally:
            conn.close()

    def get_initial_equity(self) -> Optional[float]:
        """Equity at the first recorded snapshot — proxy for starting capital."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT equity FROM portfolio_snapshots ORDER BY id ASC LIMIT 1"
            )
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    # ==================== Instrument Signals ====================

    def save_instrument_signal(
        self, symbol: str,
        tsmom_score: float, csmom_score: float, combined_score: float,
        price: float, atr_value: float, volatility: float,
    ):
        """Save daily instrument signal for audit trail."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO instrument_signals
                (symbol, tsmom_score, csmom_score, combined_score,
                 price, atr_value, volatility, signal_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, tsmom_score, csmom_score, combined_score,
                  price, atr_value, volatility, datetime.now().strftime('%Y-%m-%d')))
            conn.commit()
        finally:
            conn.close()

    def get_instrument_signals(self, date: Optional[str] = None) -> list[dict]:
        """Get instrument signals for a date (default: today)."""
        date = date or datetime.now().strftime('%Y-%m-%d')
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM instrument_signals WHERE signal_date = ? ORDER BY combined_score DESC",
                (date,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_latest_signal_prices(self) -> dict:
        """Symbol -> last cached signal price (most recent signal_date)."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT symbol, price FROM instrument_signals
                WHERE signal_date = (SELECT MAX(signal_date) FROM instrument_signals)
            """)
            return {row[0]: row[1] for row in cursor.fetchall()}
        finally:
            conn.close()

    # ==================== Legacy compatibility ====================

    def set_symbol_cooldown(self, symbol: str, minutes: int, reason: str = "stop_loss"):
        """Legacy: cooldowns not used in trend-following but kept for compatibility."""
        pass

    def is_symbol_in_cooldown(self, symbol: str) -> tuple[bool, Optional[str]]:
        """Legacy: always returns not in cooldown."""
        return (False, None)

    def increment_daily_trade_count(self, symbol: str) -> int:
        """Legacy: not used in daily rebalancing."""
        return 0

    def get_daily_trade_count(self, symbol: str) -> int:
        """Legacy: not used in daily rebalancing."""
        return 0

    def get_symbols_with_data(self) -> list[str]:
        """Get list of symbols that have stored data."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
