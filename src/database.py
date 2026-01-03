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
