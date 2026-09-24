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

CREATE TABLE IF NOT EXISTS vocabulary (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    word       TEXT NOT NULL,
    norm_word  TEXT NOT NULL,
    meaning    TEXT,
    example    TEXT,
    added_at   TEXT NOT NULL,
    last_seen  TEXT,
    seen_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(user_id, norm_word)
);

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


# Интервалы повторения в днях по длине серии верных ответов: ошибся — завтра,
# дальше всё реже. Последнее значение держится для давно закрытых ошибок.
DRILL_INTERVALS = (1, 2, 4, 9, 21)

# Столбцы, появившиеся после первого запуска. Пересоздавать таблицы нельзя —
# в базе живой прогресс ученика, поэтому недостающее добавляется на месте.
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("profile", "daily_time", "TEXT"),
    ("profile", "daily_last_sent", "TEXT"),
    ("profile", "weekly_last_sent", "TEXT"),
    ("profile", "voice_only", "INTEGER NOT NULL DEFAULT 0"),
    ("profile", "translate_replies", "INTEGER NOT NULL DEFAULT 1"),
    ("errors", "drill_due", "TEXT"),
    ("errors", "drill_streak", "INTEGER NOT NULL DEFAULT 0"),
    ("pronunciation", "words", "INTEGER"),
    ("pronunciation", "wpm", "REAL"),
    ("pronunciation", "pauses", "INTEGER"),
    ("pronunciation", "pause_ratio", "REAL"),
)


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
    #: Только голос: разговорная часть уходит голосовым, без дублирования текстом.
    voice_only: bool = False
    #: Показывать русский перевод реплики бота сразу, без нажатия кнопки.
    translate_replies: bool = True
    messages_total: int = 0
    level_checked_at_message: int = 0
    #: Время ежедневного вопроса в местном часовом поясе, "HH:MM" или None.
    daily_time: str | None = None
    daily_last_sent: str | None = None
    weekly_last_sent: str | None = None

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
    id: int = 0
    #: Когда ошибку пора повторить; None — ещё ни разу не тренировали.
    drill_due: str | None = None
    #: Сколько раз подряд ученик справился. Чем больше, тем реже повтор.
    drill_streak: int = 0


@dataclass
class VocabWord:
    word: str
    meaning: str | None
    example: str | None
    seen_count: int
    added_at: str


@dataclass
class FluencyStat:
    samples: int
    wpm: float | None
    pauses: float | None
    pause_ratio: float | None


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
        await self._migrate()
        await self._conn.commit()

    async def _migrate(self) -> None:
        for table, column, ddl in MIGRATIONS:
            async with self.conn.execute(f"PRAGMA table_info({table})") as cursor:
                existing = {row["name"] for row in await cursor.fetchall()}
            if column not in existing:
                await self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"
                )

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
            voice_only=bool(row["voice_only"]),
            translate_replies=bool(row["translate_replies"]),
            messages_total=row["messages_total"],
            level_checked_at_message=row["level_checked_at_message"],
            daily_time=row["daily_time"],
            daily_last_sent=row["daily_last_sent"],
            weekly_last_sent=row["weekly_last_sent"],
        )

    async def update_profile(self, user_id: int, **fields: Any) -> None:
        allowed = {
            "level",
            "interests",
            "native_language",
            "voice_replies",
            "voice_only",
            "translate_replies",
            "messages_total",
            "level_checked_at_message",
            "daily_time",
            "daily_last_sent",
            "weekly_last_sent",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        for flag in ("voice_replies", "voice_only", "translate_replies"):
            if flag in updates:
                updates[flag] = int(bool(updates[flag]))
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

    @staticmethod
    def _error_stat(row: Any) -> ErrorStat:
        return ErrorStat(
            id=row["id"],
            type=row["type"],
            description=row["description"],
            example=row["example"],
            count=row["count"],
            last_seen=row["last_seen"],
            drill_due=row["drill_due"],
            drill_streak=row["drill_streak"] or 0,
        )

    async def top_errors(self, user_id: int, limit: int = 5) -> list[ErrorStat]:
        async with self.conn.execute(
            "SELECT * FROM errors WHERE user_id = ? "
            "ORDER BY count DESC, last_seen DESC LIMIT ?",
            (user_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._error_stat(row) for row in rows]

    async def errors_due_for_drill(self, user_id: int, limit: int = 3) -> list[ErrorStat]:
        """Ошибки, которые пора повторить: ни разу не тренированные или отлежавшие срок."""
        now = _ts()
        async with self.conn.execute(
            "SELECT * FROM errors WHERE user_id = ? "
            "AND (drill_due IS NULL OR drill_due <= ?) "
            "ORDER BY drill_streak ASC, count DESC, last_seen DESC LIMIT ?",
            (user_id, now, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._error_stat(row) for row in rows]

    async def record_drill_result(self, error_id: int, correct: bool) -> None:
        """Интервальное повторение: справился — следующий раз нескоро, нет — завтра."""
        async with self.conn.execute(
            "SELECT drill_streak FROM errors WHERE id = ?", (error_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return
        streak = (row["drill_streak"] or 0) + 1 if correct else 0
        due = utc_now() + timedelta(days=DRILL_INTERVALS[min(streak, len(DRILL_INTERVALS) - 1)])
        await self.conn.execute(
            "UPDATE errors SET drill_streak = ?, drill_due = ? WHERE id = ?",
            (streak, _ts(due), error_id),
        )
        await self.conn.commit()

    # --- произношение --------------------------------------------------

    async def add_pronunciation(
        self,
        user_id: int,
        scores: dict[str, float],
        duration_sec: float,
        fluency: dict[str, float] | None = None,
    ) -> None:
        fluency = fluency or {}
        await self.conn.execute(
            "INSERT INTO pronunciation "
            "(user_id, accuracy, fluency, prosody, completeness, overall, duration_sec, "
            "words, wpm, pauses, pause_ratio, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                user_id,
                scores.get("accuracy"),
                scores.get("fluency"),
                scores.get("prosody"),
                scores.get("completeness"),
                scores.get("overall"),
                duration_sec,
                fluency.get("words"),
                fluency.get("wpm"),
                fluency.get("pauses"),
                fluency.get("pause_ratio"),
                _ts(),
            ),
        )
        await self.conn.commit()

    async def fluency_progress(self, user_id: int, days: int) -> FluencyStat:
        """Темп речи и паузы за период: считаются локально, без облачных оценок."""
        cutoff = _ts(utc_now() - timedelta(days=days))
        async with self.conn.execute(
            "SELECT COUNT(wpm) AS n, AVG(wpm) AS wpm, AVG(pauses) AS pauses, "
            "AVG(pause_ratio) AS pause_ratio FROM pronunciation "
            "WHERE user_id = ? AND created_at >= ? AND wpm IS NOT NULL",
            (user_id, cutoff),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or not row["n"]:
            return FluencyStat(0, None, None, None)
        return FluencyStat(
            samples=int(row["n"]),
            wpm=row["wpm"],
            pauses=row["pauses"],
            pause_ratio=row["pause_ratio"],
        )

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

    # --- словарь --------------------------------------------------------

    async def add_words(
        self, user_id: int, words: Iterable[tuple[str, str, str]]
    ) -> None:
        """Слова, которые учитель ввёл в разговоре. Повтор не плодит дубликаты."""
        now = _ts()
        rows = [
            (user_id, word.strip(), _norm_key(word), meaning, example, now)
            for word, meaning, example in words
            if word.strip()
        ]
        if not rows:
            return
        await self.conn.executemany(
            "INSERT INTO vocabulary (user_id, word, norm_word, meaning, example, added_at) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(user_id, norm_word) DO UPDATE SET "
            "meaning = COALESCE(excluded.meaning, vocabulary.meaning), "
            "example = COALESCE(excluded.example, vocabulary.example)",
            rows,
        )
        await self.conn.commit()

    async def mark_words_seen(self, user_id: int, words: Iterable[str]) -> None:
        """Ученик снова встретил слово — отмечаем, чтобы не подсовывать без нужды."""
        keys = [_norm_key(word) for word in words if word.strip()]
        if not keys:
            return
        now = _ts()
        await self.conn.executemany(
            "UPDATE vocabulary SET seen_count = seen_count + 1, last_seen = ? "
            "WHERE user_id = ? AND norm_word = ?",
            [(now, user_id, key) for key in keys],
        )
        await self.conn.commit()

    async def recent_words(self, user_id: int, limit: int = 20) -> list[VocabWord]:
        async with self.conn.execute(
            "SELECT word, meaning, example, seen_count, added_at FROM vocabulary "
            "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            VocabWord(
                word=row["word"],
                meaning=row["meaning"],
                example=row["example"],
                seen_count=row["seen_count"],
                added_at=row["added_at"],
            )
            for row in rows
        ]

    async def words_total(self, user_id: int) -> int:
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM vocabulary WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["n"]) if row else 0

    async def activity_since(self, user_id: int, since: datetime) -> dict[str, int]:
        """Чем ученик занимался за период — для недельной сводки."""
        moment = _ts(since)
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE user_id = ? AND role = 'user' "
            "AND created_at >= ?",
            (user_id, moment),
        ) as cursor:
            messages = int((await cursor.fetchone())["n"])
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM pronunciation WHERE user_id = ? AND created_at >= ?",
            (user_id, moment),
        ) as cursor:
            voices = int((await cursor.fetchone())["n"])
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM vocabulary WHERE user_id = ? AND added_at >= ?",
            (user_id, moment),
        ) as cursor:
            words = int((await cursor.fetchone())["n"])
        async with self.conn.execute(
            "SELECT COUNT(*) AS n FROM errors WHERE user_id = ? AND last_seen >= ?",
            (user_id, moment),
        ) as cursor:
            errors = int((await cursor.fetchone())["n"])
        return {
            "messages": messages,
            "voices": voices,
            "words": words,
            "errors": errors,
        }

    async def recent_topics(self, user_id: int, limit: int = 6) -> list[str]:
        """Последние реплики ученика — чтобы утренний вопрос цеплялся за них."""
        return await self.recent_user_messages(user_id, limit)

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
