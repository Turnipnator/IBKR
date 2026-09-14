"""
Tests for the 2026-09-14 live-mode /stats, /history and /pnl.

Before: all three Telegram commands read the paper_trades table, which was
wiped at the 2026-05-22 live cutover and is never written in LIVE mode, so
they reported zero trades and zero P&L against a book that had closed 31
round-trips. After: in live mode they read the trades ledger (stop-fill
execution rows, now carrying IBKR's realized P&L + commission + currency as
numbers), portfolio_snapshots for the equity curve, and IBKR's own session
P&L via the bot status dict. Paper mode is untouched.

These drive the REAL `Database` on a tmp file and the REAL handler methods
on a `TelegramNotifier` with a mocked config (no network). The seeded rows
replay real fills from the live tape.
"""

import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.database import Database
from src.telegram_bot import TelegramNotifier, check_telegram_commands


FX = {"USD": 0.7425, "GBP": 1.0, "EUR": 0.8567}


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _db(tmp_path):
    return Database(str(tmp_path / "t.db"))


def _seed_tape(db):
    """Four real round-trips (one in two partials) + noise rows."""
    # placement rows (must be ignored by the tape)
    db.save_trade("AIGA", "BUY", 126, 6.98, order_id=3552, status="FILLED",
                  reason="Top-up to target")
    db.save_trade("AIGA", "SELL", 126, 6.79, order_id=3554, status="FILLED",
                  reason="Top-up: re-cover full position of 126")
    db.save_trade("CMOD", "SELL", 52, 35.29, order_id=5820, status="SUBMITTED",
                  reason="Trailing stop 3.0xATR")
    # AIGS 09-03 +$52.87
    db.save_trade("AIGS", "SELL", 122, 7.49, order_id=3548, status="FILLED",
                  reason="TRAIL stop fill (realizedPnL=$+52.87)",
                  pnl=52.87, commission=4.0, currency="USD")
    # VEUR 09-01 -£14.88 (GBP line)
    db.save_trade("VEUR", "SELL", 18, 43.005, order_id=4382, status="FILLED",
                  reason="TRAIL stop fill (realizedPnL=£-14.88)",
                  pnl=-14.88, commission=3.0, currency="GBP")
    # AIGI 09-10 -$21.89
    db.save_trade("AIGI", "SELL", 62, 19.8306, order_id=5713, status="FILLED",
                  reason="TRAIL stop fill (realizedPnL=$-21.89)",
                  pnl=-21.89, commission=4.0, currency="USD")
    # AIGA 09-14 in two partials: -$2.88 + $45.81 = +$42.93 net
    db.save_trade("AIGA", "SELL", 3, 7.0325, order_id=3554, status="FILLED",
                  reason="TRAIL stop fill (realizedPnL=$-2.88)",
                  pnl=-2.88, commission=4.0, currency="USD")
    db.save_trade("AIGA", "SELL", 123, 7.0325, order_id=3554, status="FILLED",
                  reason="TRAIL stop fill (realizedPnL=$+45.81)",
                  pnl=45.81, commission=0.0, currency="USD")


def _snapshots(db):
    db.save_portfolio_snapshot(5008.24, 0.0, 5008.24)
    db.save_portfolio_snapshot(4724.45, 0.0567, 5008.24)


def _notifier():
    cfg = MagicMock()
    cfg.enabled = True
    cfg.bot_token = "t"
    cfg.chat_id = "1"
    return TelegramNotifier(config=cfg)


EXPECTED_TOTAL = (52.87 - 21.89 + 42.93) * FX["USD"] - 14.88     # £40.00


# --------------------------------------------------------------------------
# 1. schema + storage
# --------------------------------------------------------------------------

class TestLedgerColumns:

    def test_legacy_table_is_migrated_in_place(self, tmp_path):
        path = tmp_path / "legacy.db"
        con = sqlite3.connect(path)
        con.execute("""CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
            action TEXT NOT NULL, quantity INTEGER NOT NULL, price REAL NOT NULL,
            order_id INTEGER, status TEXT, reason TEXT, executed_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        con.execute("INSERT INTO trades (symbol, action, quantity, price, order_id, status, reason)"
                    " VALUES ('AIGS','SELL',122,7.49,3548,'FILLED','TRAIL stop fill (realizedPnL=$+52.87)')")
        con.commit()
        con.close()

        db = Database(str(path))          # runs the migration
        db2 = Database(str(path))         # and again: idempotent
        cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(trades)")}
        assert {"pnl", "commission", "currency"} <= cols
        # legacy row survives with NULLs and is skipped by the tape
        assert db2.get_live_round_trips(FX) == []

    def test_save_trade_stores_pnl_fields(self, tmp_path):
        db = _db(tmp_path)
        db.save_trade("AIGS", "SELL", 122, 7.49, order_id=3548, status="FILLED",
                      reason="TRAIL stop fill (realizedPnL=$+52.87)",
                      pnl=52.87, commission=4.0, currency="USD")
        row = sqlite3.connect(db.db_path).execute(
            "SELECT pnl, commission, currency FROM trades").fetchone()
        assert row == (52.87, 4.0, "USD")
        # placement rows leave them NULL
        db.save_trade("AIGS", "BUY", 122, 0.0, order_id=3546, status="SUBMITTED")
        row = sqlite3.connect(db.db_path).execute(
            "SELECT pnl, commission, currency FROM trades WHERE order_id=3546").fetchone()
        assert row == (None, None, None)


# --------------------------------------------------------------------------
# 2. round-trips + stats
# --------------------------------------------------------------------------

class TestLiveTape:

    def test_round_trips_merge_partials_and_convert(self, tmp_path):
        db = _db(tmp_path)
        _seed_tape(db)
        trips = db.get_live_round_trips(FX, "GBP")

        assert [t["symbol"] for t in trips] == ["AIGA", "AIGI", "VEUR", "AIGS"]   # newest first
        aiga = trips[0]
        assert aiga["order_id"] == 3554 and aiga["fills"] == 2
        assert aiga["quantity"] == 126
        assert aiga["pnl"] == pytest.approx(42.93)
        assert aiga["commission"] == pytest.approx(4.0)
        assert aiga["pnl_base"] == pytest.approx(42.93 * FX["USD"])
        veur = trips[2]
        assert veur["pnl_base"] == pytest.approx(-14.88)      # base currency, rate 1
        assert veur["fx_to_base"] == 1.0

    def test_stats_numbers(self, tmp_path):
        db = _db(tmp_path)
        _seed_tape(db)
        s = db.get_live_trade_stats(FX, "GBP")

        assert (s["closed"], s["wins"], s["losses"]) == (4, 2, 2)
        assert s["win_rate"] == pytest.approx(50.0)
        assert s["total_pnl_base"] == pytest.approx(EXPECTED_TOTAL, abs=0.01)
        assert s["total_commission_base"] == pytest.approx(12 * FX["USD"] + 3.0, abs=0.01)
        assert s["best"]["symbol"] == "AIGS"
        assert s["worst"]["symbol"] == "AIGI"          # -21.89 USD = -£16.25 < -£14.88
        assert s["avg_win_base"] == pytest.approx((52.87 + 42.93) / 2 * FX["USD"], abs=0.01)
        assert s["avg_loss_base"] == pytest.approx((-21.89 * FX["USD"] - 14.88) / 2, abs=0.01)
        assert s["payoff"] == pytest.approx(s["avg_win_base"] / abs(s["avg_loss_base"]))
        assert s["fx_missing"] == [] and s["priced"] == 4
        assert s["last_exit"].startswith("20")

    def test_missing_rate_is_reported_not_faked(self, tmp_path):
        db = _db(tmp_path)
        _seed_tape(db)
        s = db.get_live_trade_stats({"GBP": 1.0}, "GBP")     # no USD rate
        assert s["closed"] == 4 and s["wins"] == 2             # counts still right
        assert s["priced"] == 1                                # only VEUR priced
        assert s["total_pnl_base"] == pytest.approx(-14.88)
        assert s["fx_missing"] == ["USD"]

    def test_gbx_pence_lines_convert_via_gbp(self, tmp_path):
        db = _db(tmp_path)
        db.save_trade("IJPN", "SELL", 55, 1850.0, order_id=9001, status="FILLED",
                      reason="TRAIL stop fill (realizedPnL=p-2200.00)",
                      pnl=-2200.0, commission=300.0, currency="GBX")
        t = db.get_live_round_trips(FX, "GBP")[0]
        assert t["pnl_base"] == pytest.approx(-22.0)
        assert t["commission_base"] == pytest.approx(3.0)

    def test_count_today(self, tmp_path):
        db = _db(tmp_path)
        _seed_tape(db)                       # all executed_at = now -> today
        assert db.count_live_round_trips_today() == 4   # AIGA partials = 1 order


# --------------------------------------------------------------------------
# 3. Telegram handlers
# --------------------------------------------------------------------------

class TestStatsCommand:

    def test_live_stats_reads_the_ledger(self, tmp_path):
        db = _db(tmp_path)
        _seed_tape(db)
        _snapshots(db)
        n = _notifier()

        msg = n._handle_stats_command(
            db, lambda: "GBP", live_mode=True, fx_resolver=lambda: FX,
            positions_provider=lambda: [1, 2, 3, 4],
        )

        assert "LIVE" in msg
        assert "Closed round-trips:</b> 4" in msg
        assert "Winners:</b> 2 | <b>Losers:</b> 2" in msg
        assert "Win rate:</b> 50.0%" in msg
        assert "Realized:</b> +£40.00" in msg
        assert "Best:</b> AIGS +£39.26" in msg
        assert "Worst:</b> AIGI -£16.25" in msg
        assert "Open positions:</b> 4" in msg
        assert "Equity:</b> £4,724.45" in msg
        assert "Total P&L:</b> -£283.79 (-5.67%)" in msg
        assert "Drawdown:</b> 5.67%" in msg
        assert "paper" not in msg.lower()

    def test_live_stats_without_fx_or_positions_degrades_honestly(self, tmp_path):
        db = _db(tmp_path)
        _seed_tape(db)
        n = _notifier()
        msg = n._handle_stats_command(db, lambda: "GBP", live_mode=True)
        assert "Closed round-trips:</b> 4" in msg
        assert "No FX rate for USD" in msg
        assert "Open positions:</b> n/a" in msg
        assert "No portfolio snapshot yet" in msg

    def test_paper_mode_is_unchanged(self):
        n = _notifier()
        db = MagicMock()
        db.get_paper_trade_stats.return_value = {
            "total_trades": 3, "open_trades": 1, "closed_trades": 2,
            "winning_trades": 1, "losing_trades": 1, "total_pnl": 5.0,
            "win_rate": 50.0, "avg_win": 10.0, "avg_loss": -5.0,
        }
        db.get_latest_portfolio_snapshot.return_value = None
        msg = n._handle_stats_command(db, lambda: "USD", live_mode=False)
        db.get_paper_trade_stats.assert_called_once()
        db.get_live_trade_stats.assert_not_called()
        assert "Total Trades:</b> 3" in msg


class TestHistoryCommand:

    def test_live_history_lists_recent_exits(self, tmp_path):
        db = _db(tmp_path)
        _seed_tape(db)
        _snapshots(db)
        n = _notifier()

        msg = n._handle_history_command(
            db, lambda: "GBP", live_mode=True, fx_resolver=lambda: FX)

        assert "Starting:</b> £5,008.24" in msg
        assert "Current:</b>  £4,724.45" in msg
        assert "4 trades | 2W / 2L (50.0%)" in msg
        assert "P&L: +£40.00" in msg
        assert "Best: AIGS +£39.26" in msg
        assert "Worst: AIGI £-16.25" in msg
        assert "Recent exits:" in msg
        assert "AIGA SELL 126 @ $7.03 → +£31.88" in msg
        assert "VEUR SELL 18 @ £43.01 → -£14.88" in msg     # 43.005 formats as 43.01
        assert "paper" not in msg.lower()

    def test_paper_history_still_queries_paper_trades(self, tmp_path):
        db = _db(tmp_path)
        _snapshots(db)
        n = _notifier()
        msg = n._handle_history_command(db, lambda: "GBP", live_mode=False)
        assert "0 trades | 0W / 0L" in msg
        assert "Recent exits" not in msg


class TestPnlCommand:

    def test_live_pnl_uses_ibkr_session_figures(self, tmp_path):
        db = _db(tmp_path)
        _seed_tape(db)
        n = _notifier()
        status = lambda: {"connected": True, "dry_run": False,
                          "session_pnl": (31.85, 91.94)}

        msg = n._handle_pnl_command(db, None, lambda: "GBP",
                                    live_mode=True, status_fetcher=status)

        assert "LIVE" in msg
        assert "Realized:</b> +£31.85 (4 stop fills today)" in msg
        assert "Unrealized:</b> +£91.94" in msg
        assert "Session:</b> +£123.79" in msg
        assert "Daily-loss cap:</b> -£200 (0% used)" in msg

    def test_live_pnl_when_ibkr_unreadable(self, tmp_path):
        db = _db(tmp_path)
        n = _notifier()
        msg = n._handle_pnl_command(db, None, lambda: "GBP", live_mode=True,
                                    status_fetcher=lambda: {"session_pnl": None})
        assert "unavailable" in msg

    def test_paper_pnl_unchanged(self):
        n = _notifier()
        db = MagicMock()
        db.get_daily_pnl.return_value = 1.5
        db.get_open_paper_trades.return_value = []
        db._get_connection.side_effect = RuntimeError("no conn")
        msg = n._handle_pnl_command(db, None, lambda: "USD", live_mode=False)
        db.get_daily_pnl.assert_called_once()
        assert "Realized:</b> +$1.50" in msg


class TestRouting:

    def test_process_command_routes_live_flags(self, tmp_path):
        db = _db(tmp_path)
        _seed_tape(db)
        _snapshots(db)
        n = _notifier()
        status = lambda: {"session_pnl": (1.0, 2.0)}

        stats = n.process_command("/stats", db, None, lambda: "GBP", None, status,
                                  lambda: [], fx_resolver=lambda: FX, live_mode=True)
        hist = n.process_command("/history", db, None, lambda: "GBP", None, status,
                                 lambda: [], fx_resolver=lambda: FX, live_mode=True)
        pnl = n.process_command("/pnl", db, None, lambda: "GBP", None, status,
                                lambda: [], fx_resolver=lambda: FX, live_mode=True)

        assert "Closed round-trips:</b> 4" in stats
        assert "Recent exits:" in hist
        assert "Session:</b> +£3.00" in pnl

    def test_help_no_longer_says_paper(self):
        msg = _notifier()._handle_help_command()
        assert "paper" not in msg.lower()

    def test_check_telegram_commands_accepts_new_kwargs(self, monkeypatch):
        # Disabled notifier -> returns early; the point is the signature.
        import src.telegram_bot as tb
        cfg = MagicMock()
        cfg.enabled = False
        monkeypatch.setattr(tb, "_notifier", TelegramNotifier(config=cfg))
        check_telegram_commands(None, None, None, None, None, None,
                                fx_resolver=lambda: FX, live_mode=True)
