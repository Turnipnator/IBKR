"""
SQLite database layer for storing market data and trade history.
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from .config import data_config

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database manager for market data and trades.

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

                -- Index for faster queries
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date
                ON ohlcv(symbol, date);

                -- Trade log
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,  -- BUY, SELL
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    order_id INTEGER,
                    status TEXT,  -- FILLED, CANCELLED, PENDING
                    reason TEXT,  -- Why the trade was made
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
                    action TEXT NOT NULL,  -- BUY or SELL
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'OPEN',  -- OPEN, CLOSED_TP, CLOSED_SL, CLOSED_MANUAL
                    exit_price REAL,
                    pnl_amount REAL,
                    pnl_percent REAL,
                    reasons TEXT,  -- JSON list of reasons
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Index for faster queries on open paper trades
                CREATE INDEX IF NOT EXISTS idx_paper_trades_status
                ON paper_trades(status);
            """)
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        finally:
            conn.close()

    def save_ohlcv(self, df: pd.DataFrame, symbol: Optional[str] = None):
        """
        Save OHLCV data to database.

        Args:
            df: DataFrame with OHLCV columns
            symbol: Symbol to use (overrides df['symbol'] if present)
        """
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
                    sym,
                    date_val,
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("volume"),
                    row.get("average"),
                    row.get("barCount"),
                ))

            conn.commit()
            logger.info(f"Saved {len(df)} bars for {symbol or 'multiple symbols'}")
        finally:
            conn.close()

    def load_ohlcv(
        self,
        symbol: str,
        days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from database.

        Args:
            symbol: Stock ticker
            days: Number of days to load (from most recent)
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            DataFrame with OHLCV data
        """
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

    def save_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        order_id: Optional[int] = None,
        status: str = "PENDING",
        reason: Optional[str] = None,
    ):
        """Log a trade to the database."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO trades
                (symbol, action, quantity, price, order_id, status, reason, executed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                action,
                quantity,
                price,
                order_id,
                status,
                reason,
                datetime.now().isoformat(),
            ))
            conn.commit()
            logger.info(f"Logged trade: {action} {quantity} {symbol} @ {price}")
        finally:
            conn.close()

    def get_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get trade history."""
        conn = self._get_connection()
        try:
            query = "SELECT * FROM trades"
            params = []

            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()

    def save_account_snapshot(
        self,
        net_liquidation: float,
        total_cash: float,
        buying_power: float,
        currency: str = "GBP",
    ):
        """Save an account snapshot."""
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO account_snapshots
                (net_liquidation, total_cash, buying_power, currency)
                VALUES (?, ?, ?, ?)
            """, (net_liquidation, total_cash, buying_power, currency))
            conn.commit()
        finally:
            conn.close()

    def get_symbols_with_data(self) -> list[str]:
        """Get list of symbols that have stored data."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_data_range(self, symbol: str) -> tuple[Optional[str], Optional[str]]:
        """Get the date range of stored data for a symbol."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT MIN(date), MAX(date) FROM ohlcv WHERE symbol = ?
            """, (symbol,))
            row = cursor.fetchone()
            return (row[0], row[1]) if row else (None, None)
        finally:
            conn.close()

    # ==================== Paper Trade Methods ====================

    def save_paper_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reasons: Optional[list[str]] = None,
    ) -> int:
        """
        Save a new paper trade.

        Returns:
            The ID of the created paper trade
        """
        import json
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO paper_trades
                (symbol, action, quantity, entry_price, stop_loss, take_profit, reasons, entry_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            """, (
                symbol,
                action,
                quantity,
                entry_price,
                stop_loss,
                take_profit,
                json.dumps(reasons) if reasons else None,
                datetime.now().isoformat(),
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
            cursor = conn.execute("""
                SELECT * FROM paper_trades WHERE status = 'OPEN' ORDER BY entry_time DESC
            """)
            trades = []
            for row in cursor.fetchall():
                trade = dict(row)
                if trade.get('reasons'):
                    trade['reasons'] = json.loads(trade['reasons'])
                trades.append(trade)
            return trades
        finally:
            conn.close()

    def close_paper_trade(
        self,
        trade_id: int,
        exit_price: float,
        status: str,  # CLOSED_TP, CLOSED_SL, CLOSED_MANUAL
    ) -> dict:
        """
        Close a paper trade and calculate P&L.

        Returns:
            Dict with trade details including P&L
        """
        import json
        conn = self._get_connection()
        try:
            # Get the trade
            cursor = conn.execute("SELECT * FROM paper_trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            if not row:
                return {}

            trade = dict(row)
            entry_price = trade['entry_price']
            quantity = trade['quantity']
            action = trade['action']

            # Calculate P&L (for BUY: profit if exit > entry)
            if action == 'BUY':
                pnl_amount = (exit_price - entry_price) * quantity
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            else:  # SELL (short)
                pnl_amount = (entry_price - exit_price) * quantity
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100

            # Update the trade
            conn.execute("""
                UPDATE paper_trades
                SET status = ?, exit_price = ?, pnl_amount = ?, pnl_percent = ?, exit_time = ?
                WHERE id = ?
            """, (status, exit_price, pnl_amount, pnl_percent, datetime.now().isoformat(), trade_id))
            conn.commit()

            trade['status'] = status
            trade['exit_price'] = exit_price
            trade['pnl_amount'] = pnl_amount
            trade['pnl_percent'] = pnl_percent
            if trade.get('reasons'):
                trade['reasons'] = json.loads(trade['reasons'])

            logger.info(f"Closed paper trade #{trade_id}: {status} @ ${exit_price:.2f} (P&L: ${pnl_amount:.2f} / {pnl_percent:.1f}%)")
            return trade
        finally:
            conn.close()

    def get_paper_trade_stats(self) -> dict:
        """Get summary statistics for paper trades."""
        conn = self._get_connection()
        try:
            stats = {
                'total_trades': 0,
                'open_trades': 0,
                'closed_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
            }

            # Count trades
            cursor = conn.execute("SELECT COUNT(*) FROM paper_trades")
            stats['total_trades'] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status = 'OPEN'")
            stats['open_trades'] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status != 'OPEN'")
            stats['closed_trades'] = cursor.fetchone()[0]

            # P&L stats
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
