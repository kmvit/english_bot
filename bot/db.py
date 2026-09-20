"""Слой SQLite: профиль, история диалога, ошибки, баллы произношения, расходы."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    user_id           INTEGER PRIMARY KEY,
    level             TEXT,
    interests         TEXT,
    native_language   TEXT NOT NULL DEFAULT 'ru',
    voice_replies     INTEGER NOT NULL DEFAULT 1,
    messages_total    INTEGER NOT NULL DEFAULT 0,
    level_checked_at_message INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);

CREATE TABLE IF NOT EXISTS errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    type        TEXT NOT NULL,
    description TEXT NOT NULL,
    norm_key    TEXT NOT NULL,
    example     TEXT,
    count       INTEGER NOT NULL DEFAULT 1,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    UNIQUE(user_id, type, norm_key)
);

CREATE TABLE IF NOT EXISTS pronunciation (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    accuracy     REAL,
    fluency      REAL,
    prosody      REAL,
    completeness REAL,
    overall      REAL,
    duration_sec REAL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pron_user_date ON pronunciation(user_id, created_at);

CREATE TABLE IF NOT EXISTS usage_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    model         TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read    INTEGER NOT NULL DEFAULT 0,
    cache_write   INTEGER NOT NULL DEFAULT 0,
    audio_seconds REAL NOT NULL DEFAULT 0,
    chars         INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_user_date ON usage_log(user_id, created_at);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(moment: datetime | None = None) -> str:
    return (moment or utc_now()).isoformat()


def _norm_key(text: str) -> str:
    return " ".join(text.lower().split())[:120]


@dataclass
class Profile:
    user_id: int
    level: str | None = None
    interests: str | None = None
    native_language: str = "ru"
    voice_replies: bool = True
    messages_total: int = 0
    level_checked_at_message: int = 0

    @property
    def is_onboarded(self) -> bool:
        return bool(self.level)


@dataclass
class ErrorStat:
    type: str
    description: str
    example: str | None
    count: int
    last_seen: str


@dataclass
class ProgressStat:
    samples: int
    accuracy: float | None
    fluency: float | None
    prosody: float | None
    overall: float | None


class Database:
    """Тонкая обёртка над aiosqlite с одним постоянным соединением."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database.connect() не вызван")
        return self._conn

    # --- профиль -------------------------------------------------------

    async def ensure_profile(self, user_id: int) -> Profile:
        now = _ts()
        await self.conn.execute(
            "INSERT INTO profile (user_id, created_at, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO NOTHING",
            (user_id, now, now),
        )
        await self.conn.commit()
        profile = await self.get_profile(user_id)
        assert profile is not None
        return profile

    async def get_profile(self, user_id: int) -> Profile | None:
        async with self.conn.execute(
            "SELECT * FROM profile WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return Profile(
            user_id=row["user_id"],
            level=row["level"],
            interests=row["interests"],
            native_language=row["native_language"],
            voice_replies=bool(row["voice_replies"]),
            messages_total=row["messages_total"],
            level_checked_at_message=row["level_checked_at_message"],
        )

    async def update_profile(self, user_id: int, **fields: Any) -> None:
        allowed = {
            "level",
            "interests",
            "native_language",
            "voice_replies",
            "messages_total",
            "level_checked_at_message",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        if "voice_replies" in updates:
            updates["voice_replies"] = int(bool(updates["voice_replies"]))
        assignments = ", ".join(f"{key} = ?" for key in updates)
        await self.conn.execute(
            f"UPDATE profile SET {assignments}, updated_at = ? WHERE user_id = ?",
            (*updates.values(), _ts(), user_id),
        )
        await self.conn.commit()

    async def bump_messages_total(self, user_id: int) -> int:
        await self.conn.execute(
            "UPDATE profile SET messages_total = messages_total + 1, updated_at = ? "
            "WHERE user_id = ?",
            (_ts(), user_id),
        )
        await self.conn.commit()
        async with self.conn.execute(
            "SELECT messages_total FROM profile WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["messages_total"]) if row else 0

    # --- история диалога ------------------------------------------------

    async def add_message(self, user_id: int, role: str, content: str) -> None:
        await self.conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at) VALUES (?,?,?,?)",
            (user_id, role, content, _ts()),
        )
        await self.conn.commit()

    async def get_history(self, user_id: int, limit: int) -> list[dict[str, str]]:
        """Последние `limit` реплик в хронологическом порядке."""
        async with self.conn.execute(
            "SELECT role, content FROM messages WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    async def recent_user_messages(self, user_id: int, limit: int) -> list[str]:
        """Последние реплики ученика — для периодической оценки уровня."""
        async with self.conn.execute(
            "SELECT content FROM messages WHERE user_id = ? AND role = 'user' "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [row["content"] for row in reversed(rows)]

    async def clear_history(self, user_id: int) -> None:
        await self.conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        await self.conn.commit()

    # --- ошибки --------------------------------------------------------

    async def record_errors(
        self,
        user_id: int,
        errors: Iterable[tuple[str, str]],
        example: str | None = None,
    ) -> None:
        now = _ts()
        rows = [
            (user_id, error_type, description, _norm_key(description), example, now, now)
            for error_type, description in errors
            if description.strip()
        ]
        if not rows:
            return
        await self.conn.executemany(
            "INSERT INTO errors "
            "(user_id, type, description, norm_key, example, first_seen, last_seen) "
            "VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(user_id, type, norm_key) DO UPDATE SET "
            "count = count + 1, last_seen = excluded.last_seen, "
            "example = COALESCE(excluded.example, errors.example)",
            rows,
        )
        await self.conn.commit()

    async def top_errors(self, user_id: int, limit: int = 5) -> list[ErrorStat]:
        async with self.conn.execute(
            "SELECT type, description, example, count, last_seen FROM errors "
            "WHERE user_id = ? ORDER BY count DESC, last_seen DESC LIMIT ?",
            (user_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            ErrorStat(
                type=row["type"],
                description=row["description"],
                example=row["example"],
                count=row["count"],
                last_seen=row["last_seen"],
            )
            for row in rows
        ]

    # --- произношение --------------------------------------------------

    async def add_pronunciation(
        self, user_id: int, scores: dict[str, float], duration_sec: float
    ) -> None:
        await self.conn.execute(
            "INSERT INTO pronunciation "
            "(user_id, accuracy, fluency, prosody, completeness, overall, duration_sec, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                user_id,
                scores.get("accuracy"),
                scores.get("fluency"),
                scores.get("prosody"),
                scores.get("completeness"),
                scores.get("overall"),
                duration_sec,
                _ts(),
            ),
        )
        await self.conn.commit()

    async def pronunciation_progress(self, user_id: int, days: int) -> ProgressStat:
        cutoff = _ts(utc_now() - timedelta(days=days))
        async with self.conn.execute(
            "SELECT COUNT(*) AS n, AVG(accuracy) AS accuracy, AVG(fluency) AS fluency, "
            "AVG(prosody) AS prosody, AVG(overall) AS overall FROM pronunciation "
            "WHERE user_id = ? AND created_at >= ?",
            (user_id, cutoff),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or not row["n"]:
            return ProgressStat(0, None, None, None, None)
        return ProgressStat(
            samples=int(row["n"]),
            accuracy=row["accuracy"],
            fluency=row["fluency"],
            prosody=row["prosody"],
            overall=row["overall"],
        )

    # --- расходы -------------------------------------------------------

    async def add_usage(
        self,
        user_id: int,
        kind: str,
        cost_usd: float,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_write: int = 0,
        audio_seconds: float = 0.0,
        chars: int = 0,
    ) -> None:
        await self.conn.execute(
            "INSERT INTO usage_log (user_id, kind, model, input_tokens, output_tokens, "
            "cache_read, cache_write, audio_seconds, chars, cost_usd, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                user_id,
                kind,
                model,
                input_tokens,
                output_tokens,
                cache_read,
                cache_write,
                audio_seconds,
                chars,
                cost_usd,
                _ts(),
            ),
        )
        await self.conn.commit()

    async def usage_since(
        self, user_id: int, since: datetime
    ) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT kind, COUNT(*) AS calls, SUM(cost_usd) AS cost, "
            "SUM(input_tokens + cache_read + cache_write) AS tokens_in, "
            "SUM(output_tokens) AS tokens_out, SUM(audio_seconds) AS seconds, "
            "SUM(chars) AS chars FROM usage_log "
            "WHERE user_id = ? AND created_at >= ? GROUP BY kind",
            (user_id, _ts(since)),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]


def month_start(moment: datetime | None = None) -> datetime:
    now = moment or utc_now()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
