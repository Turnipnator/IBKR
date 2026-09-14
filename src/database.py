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

                CREATE TABLE IF NOT EXISTS symbol_cooldowns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    cooldown_until TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

            # Migrations for existing databases
            for migration_sql in [
                "ALTER TABLE paper_trades ADD COLUMN best_price REAL",
                "ALTER TABLE paper_trades ADD COLUMN atr_stop REAL",
                "ALTER TABLE paper_trades ADD COLUMN min_exit_date TEXT",
                "ALTER TABLE paper_trades ADD COLUMN signal_score REAL",
                # 2026-09-14: stop-fill execution rows carry IBKR's realized
                # P&L and commission as numbers (were only in the reason text)
                "ALTER TABLE trades ADD COLUMN pnl REAL",
                "ALTER TABLE trades ADD COLUMN commission REAL",
                "ALTER TABLE trades ADD COLUMN currency TEXT",
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
        pnl: Optional[float] = None, commission: Optional[float] = None,
        currency: Optional[str] = None,
    ):
        """Log a trade to the database.

        `pnl` / `commission` / `currency` are set on stop-fill execution rows
        (IBKR's realizedPNL and commission in the instrument's currency, as
        the commission report gives them); placement rows leave them NULL.
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO trades
                (symbol, action, quantity, price, order_id, status, reason,
                 executed_at, pnl, commission, currency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, action, quantity, price, order_id, status, reason,
                  datetime.now().isoformat(), pnl, commission, currency))
            conn.commit()
        finally:
            conn.close()

    # ---- live closed-trade tape --------------------------------------------

    @staticmethod
    def _rate_to_base(ccy: Optional[str], fx_rates: dict, base: str) -> Optional[float]:
        """BASE per 1 unit of `ccy` from an IBKR ExchangeRate dict; None if unknown."""
        code = (ccy or base or "").upper()
        base = (base or "").upper()
        if not code or code == base or code == "BASE":
            return 1.0
        if code == "GBX":
            gbp = 1.0 if base == "GBP" else fx_rates.get("GBP")
            return None if gbp is None else gbp / 100.0
        rate = fx_rates.get(code)
        return float(rate) if rate else None

    def get_live_round_trips(self, fx_rates: Optional[dict] = None,
                             base: str = "GBP") -> list[dict]:
        """One row per closed live round-trip, newest first.

        Built from the stop-fill execution rows the fill notifier writes
        (status FILLED, reason '... stop fill ...'), merged per order_id so a
        stop that filled in partials is one trade. `pnl`/`commission` are in
        the instrument's currency; `pnl_base`/`commission_base` are converted
        with `fx_rates` (BASE per 1 CCY, as `ConnectionManager.get_fx_rates`
        returns) and are None when the rate is unknown. Rows whose pnl was
        never recorded (pre-2026-09-14 fills not backfilled) are skipped.
        """
        fx_rates = fx_rates or {}
        conn = self._get_connection()
        try:
            rows = conn.execute("""
                SELECT order_id, symbol, action,
                       SUM(quantity)                    AS quantity,
                       SUM(pnl)                         AS pnl,
                       COALESCE(SUM(commission), 0.0)   AS commission,
                       MAX(currency)                    AS currency,
                       MAX(price)                       AS price,
                       MAX(executed_at)                 AS executed_at,
                       COUNT(*)                         AS fills
                FROM trades
                WHERE status = 'FILLED'
                  AND reason LIKE '%stop fill%'
                  AND pnl IS NOT NULL
                GROUP BY order_id
                ORDER BY MAX(executed_at) DESC
            """).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            d = dict(r)
            rate = self._rate_to_base(d.get("currency"), fx_rates, base)
            d["fx_to_base"] = rate
            d["pnl_base"] = (d["pnl"] * rate) if rate is not None else None
            d["commission_base"] = (
                (d["commission"] * rate) if rate is not None else None
            )
            out.append(d)
        return out

    def get_live_trade_stats(self, fx_rates: Optional[dict] = None,
                             base: str = "GBP") -> dict:
        """Cumulative stats over the live closed-trade tape, in base currency.

        Wins/losses are judged on native-currency P&L (sign is FX-proof);
        sums use `pnl_base` and skip trips whose currency has no rate — those
        are reported in `fx_missing` so the caller can say so.
        """
        trips = self.get_live_round_trips(fx_rates, base)
        stats = {
            "closed": len(trips), "wins": 0, "losses": 0, "win_rate": 0.0,
            "total_pnl_base": 0.0, "total_commission_base": 0.0,
            "avg_win_base": 0.0, "avg_loss_base": 0.0, "payoff": 0.0,
            "best": None, "worst": None, "fx_missing": [],
            "priced": 0, "last_exit": trips[0]["executed_at"] if trips else None,
        }
        wins, losses = [], []
        for t in trips:
            if t["pnl"] > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
            if t["pnl_base"] is None:
                if t["currency"] not in stats["fx_missing"]:
                    stats["fx_missing"].append(t["currency"])
                continue
            stats["priced"] += 1
            stats["total_pnl_base"] += t["pnl_base"]
            stats["total_commission_base"] += t["commission_base"] or 0.0
            (wins if t["pnl_base"] > 0 else losses).append(t["pnl_base"])
            if stats["best"] is None or t["pnl_base"] > stats["best"]["pnl_base"]:
                stats["best"] = t
            if stats["worst"] is None or t["pnl_base"] < stats["worst"]["pnl_base"]:
                stats["worst"] = t
        if stats["closed"]:
            stats["win_rate"] = stats["wins"] / stats["closed"] * 100.0
        if wins:
            stats["avg_win_base"] = sum(wins) / len(wins)
        if losses:
            stats["avg_loss_base"] = sum(losses) / len(losses)
        if wins and losses and stats["avg_loss_base"] != 0:
            stats["payoff"] = stats["avg_win_base"] / abs(stats["avg_loss_base"])
        return stats

    def count_live_round_trips_today(self) -> int:
        """Stop fills (distinct orders) whose execution landed today."""
        today = datetime.now().strftime('%Y-%m-%d')
        conn = self._get_connection()
        try:
            row = conn.execute("""
                SELECT COUNT(DISTINCT order_id) FROM trades
                WHERE status = 'FILLED' AND reason LIKE '%stop fill%'
                  AND substr(executed_at, 1, 10) = ?
            """, (today,)).fetchone()
            return int(row[0] or 0)
        finally:
            conn.close()

    TERMINAL_TRADE_STATUSES = ("FILLED", "CANCELLED", "REJECTED")

    def update_trade_status(
        self, order_id: Optional[int], status: str,
        price: Optional[float] = None, note: Optional[str] = None,
    ) -> int:
        """Move the SUBMITTED ledger row(s) for `order_id` to a terminal status.

        Returns the number of rows changed. Only rows still at SUBMITTED are
        touched, so repeated events (IBKR re-sends orderStatus; a stop reports
        once per partial fill) are no-ops, an unknown order_id (probe/manual
        orders) changes nothing, and the execution rows the fill notifier
        inserts with status=FILLED are never rewritten.

        `price` only fills in a row whose recorded price is 0 — market orders
        are saved with price=0 because the fill price is unknown at placement.
        A stop's recorded initial trigger is never overwritten; its actual exit
        price lives on the execution row. `note` is appended to `reason`.
        """
        if status not in self.TERMINAL_TRADE_STATUSES:
            raise ValueError(f"not a terminal trade status: {status!r}")
        if order_id is None:
            return 0
        px = float(price or 0.0)
        note = (note or "").strip()
        conn = self._get_connection()
        try:
            cur = conn.execute("""
                UPDATE trades
                SET status = ?,
                    price = CASE WHEN price = 0 AND ? > 0 THEN ? ELSE price END,
                    reason = CASE WHEN ? != ''
                                  THEN COALESCE(reason, '') || ' | ' || ?
                                  ELSE reason END,
                    executed_at = ?
                WHERE order_id = ? AND status = 'SUBMITTED'
            """, (status, px, px, note, note,
                  datetime.now().isoformat(), int(order_id)))
            conn.commit()
            return cur.rowcount
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

    def get_today_stats(self) -> dict:
        """Today's activity only — opened, closed, won/lost, realized P&L."""
        today = datetime.now().strftime('%Y-%m-%d')
        conn = self._get_connection()
        try:
            stats = {
                'opened_today': 0,
                'closed_today': 0,
                'won_today': 0,
                'lost_today': 0,
                'realized_pnl_today': 0.0,
            }
            cursor = conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE date(entry_time) = ?",
                (today,),
            )
            stats['opened_today'] = cursor.fetchone()[0] or 0

            cursor = conn.execute("""
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN pnl_amount > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN pnl_amount <= 0 THEN 1 ELSE 0 END),
                    COALESCE(SUM(pnl_amount), 0)
                FROM paper_trades
                WHERE status != 'OPEN' AND date(exit_time) = ?
            """, (today,))
            row = cursor.fetchone()
            stats['closed_today'] = row[0] or 0
            stats['won_today'] = row[1] or 0
            stats['lost_today'] = row[2] or 0
            stats['realized_pnl_today'] = row[3] or 0.0

            return stats
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

    # ==================== Re-entry cooldowns ====================

    def set_symbol_cooldown(self, symbol: str, days: int, reason: str = "stop_loss"):
        """Block re-entry into ``symbol`` for ``days`` calendar days.

        Called when a protective stop fills. Upserts so a later stop-out always
        extends the window rather than being ignored.
        """
        if days <= 0:
            return
        until = (datetime.now() + timedelta(days=days)).isoformat()
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO symbol_cooldowns (symbol, cooldown_until, reason)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    cooldown_until = excluded.cooldown_until,
                    reason = excluded.reason
                """,
                (symbol, until, reason),
            )
            conn.commit()
        finally:
            conn.close()

    def is_symbol_in_cooldown(self, symbol: str) -> tuple[bool, Optional[str]]:
        """Return (in_cooldown, cooldown_until_iso) for ``symbol``."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT cooldown_until FROM symbol_cooldowns WHERE symbol = ?",
                (symbol,),
            )
            row = cursor.fetchone()
            if not row or not row[0]:
                return (False, None)
            try:
                until = datetime.fromisoformat(row[0])
            except (TypeError, ValueError):
                return (False, None)
            if datetime.now() < until:
                return (True, row[0])
            return (False, None)
        finally:
            conn.close()

    def get_active_cooldowns(self) -> dict:
        """Symbol -> cooldown_until ISO string, for symbols still cooling down."""
        now = datetime.now().isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT symbol, cooldown_until FROM symbol_cooldowns "
                "WHERE cooldown_until > ?",
                (now,),
            )
            return {row[0]: row[1] for row in cursor.fetchall()}
        finally:
            conn.close()

    # ==================== Legacy compatibility ====================

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
