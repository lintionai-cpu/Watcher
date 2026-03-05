"""
Data Store — SQLite-backed persistent storage for ticks, signals, and model state.
Uses aiosqlite for async I/O to keep FastAPI non-blocking.
"""
import json
import time
import asyncio
import aiosqlite
from typing import Optional, List, Dict, Any
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "deriv_intel.db"


async def init_db() -> None:
    """Create all tables on startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
        CREATE TABLE IF NOT EXISTS ticks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            market      TEXT    NOT NULL,
            timestamp   REAL    NOT NULL,
            price       REAL    NOT NULL,
            digit       INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ticks_market_ts ON ticks(market, timestamp DESC);

        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            market          TEXT    NOT NULL,
            category        TEXT    NOT NULL,
            strategy        TEXT    NOT NULL,
            direction       TEXT    NOT NULL,
            barrier         INTEGER NOT NULL,
            confidence      REAL    NOT NULL,
            valid_window    INTEGER NOT NULL,
            duration_ticks  INTEGER NOT NULL,
            model_votes     TEXT    NOT NULL,
            timestamp       REAL    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market, timestamp DESC);

        CREATE TABLE IF NOT EXISTS model_accuracy (
            market          TEXT    NOT NULL,
            model_name      TEXT    NOT NULL,
            correct         INTEGER DEFAULT 0,
            total           INTEGER DEFAULT 0,
            weight          REAL    DEFAULT 1.0,
            updated_at      REAL    DEFAULT 0,
            PRIMARY KEY(market, model_name)
        );

        CREATE TABLE IF NOT EXISTS settings (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL
        );
        """)
        await db.commit()


async def insert_tick(market: str, timestamp: float, price: float, digit: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO ticks (market, timestamp, price, digit) VALUES (?,?,?,?)",
            (market, timestamp, price, digit)
        )
        # Keep only last 5000 ticks per market
        await db.execute(
            """DELETE FROM ticks WHERE market=? AND id NOT IN (
               SELECT id FROM ticks WHERE market=? ORDER BY timestamp DESC LIMIT 5000)""",
            (market, market)
        )
        await db.commit()


async def get_recent_ticks(market: str, limit: int = 500) -> List[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM ticks WHERE market=? ORDER BY timestamp DESC LIMIT ?",
            (market, limit)
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in reversed(rows)]


async def get_digit_history(market: str, limit: int = 500) -> List[int]:
    ticks = await get_recent_ticks(market, limit)
    return [t["digit"] for t in ticks]


async def insert_signal(
    market: str, category: str, strategy: str,
    direction: str, barrier: int, confidence: float,
    valid_window: int, duration_ticks: int,
    model_votes: Dict, timestamp: float
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO signals
               (market, category, strategy, direction, barrier, confidence,
                valid_window, duration_ticks, model_votes, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (market, category, strategy, direction, barrier, confidence,
             valid_window, duration_ticks, json.dumps(model_votes), timestamp)
        )
        await db.commit()


async def get_latest_signals(market: Optional[str] = None, limit: int = 50) -> List[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if market:
            async with db.execute(
                "SELECT * FROM signals WHERE market=? ORDER BY timestamp DESC LIMIT ?",
                (market, limit)
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["model_votes"] = json.loads(d["model_votes"])
        result.append(d)
    return result


async def get_model_weights(market: str) -> Dict[str, float]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT model_name, weight FROM model_accuracy WHERE market=?",
            (market,)
        ) as cursor:
            rows = await cursor.fetchall()
    if not rows:
        return {}
    return {r["model_name"]: r["weight"] for r in rows}


async def update_model_accuracy(
    market: str, model_name: str,
    correct: bool, new_weight: float
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO model_accuracy (market, model_name, correct, total, weight, updated_at)
               VALUES (?,?,?,1,?,?)
               ON CONFLICT(market,model_name) DO UPDATE SET
                 correct = correct + ?,
                 total   = total + 1,
                 weight  = ?,
                 updated_at = ?""",
            (market, model_name, int(correct), new_weight, time.time(),
             int(correct), new_weight, time.time())
        )
        await db.commit()


async def get_setting(key: str, default: Any = None) -> Any:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
    if row:
        return json.loads(row[0])
    return default


async def set_setting(key: str, value: Any) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=?",
            (key, json.dumps(value), json.dumps(value))
        )
        await db.commit()


async def get_digit_stats(market: str, window: int = 500) -> Dict:
    """Return digit frequency distribution and basic stats."""
    digits = await get_digit_history(market, window)
    if not digits:
        return {}
    freq = [0] * 10
    for d in digits:
        freq[d] += 1
    total = len(digits)
    return {
        "total": total,
        "frequency": {str(i): freq[i] for i in range(10)},
        "percentage": {str(i): round(freq[i] / total * 100, 2) for i in range(10)},
        "last_20": digits[-20:],
        "mean": round(sum(digits) / total, 3),
    }
