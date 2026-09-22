"""Тесты расписания: утренний вопрос и недельная сводка."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from bot.config import Config
from bot.db import Database
from bot.scheduler import Scheduler, parse_time
from bot.service import TeacherService

from .fakes import FakeSpeech, FakeTeacher

USER = 1
MSK = ZoneInfo("Europe/Moscow")


class FakeBot:
    def __init__(self):
        self.messages: list[tuple[int, str]] = []
        self.voices: list[int] = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text))

    async def send_voice(self, chat_id, voice, **kwargs):
        self.voices.append(chat_id)


@pytest.fixture
def parts(config: Config, db: Database):
    teacher = FakeTeacher()
    # Без озвучки: проверяем расписание, а не синтез.
    speech = FakeSpeech(can_speak=False)
    service = TeacherService(config, db, teacher, speech)
    bot = FakeBot()
    scheduler = Scheduler(replace(config, timezone="Europe/Moscow"), db, service, bot)
    return scheduler, bot, teacher


@pytest.mark.parametrize(
    "value,expected",
    [("09:00", (9, 0)), ("9:5", (9, 5)), ("23:59", (23, 59))],
)
def test_parse_time(value, expected):
    parsed = parse_time(value)
    assert (parsed.hour, parsed.minute) == expected


@pytest.mark.parametrize("value", [None, "", "утром", "25:00", "9-00"])
def test_parse_time_rejects_garbage(value):
    assert parse_time(value) is None


async def test_daily_question_is_sent_at_the_right_time(parts, db: Database):
    scheduler, bot, teacher = parts
    await db.ensure_profile(USER)
    await db.update_profile(USER, daily_time="09:00")

    await scheduler.tick(datetime(2026, 9, 21, 8, 59, tzinfo=MSK))
    assert bot.messages == []

    await scheduler.tick(datetime(2026, 9, 21, 9, 0, tzinfo=MSK))
    assert len(bot.messages) == 1
    assert teacher.plain_calls[0]["system"].startswith("You open the day")


async def test_daily_question_is_sent_once_a_day(parts, db: Database):
    scheduler, bot, _ = parts
    await db.ensure_profile(USER)
    await db.update_profile(USER, daily_time="09:00")

    for minute in (0, 1, 30):
        await scheduler.tick(datetime(2026, 9, 21, 9, minute, tzinfo=MSK))
    assert len(bot.messages) == 1

    await scheduler.tick(datetime(2026, 9, 22, 9, 0, tzinfo=MSK))
    assert len(bot.messages) == 2


async def test_daily_question_goes_into_history(parts, db: Database):
    """Ответ ученика должен попасть к модели вместе с вопросом, на который он отвечает."""
    scheduler, bot, teacher = parts
    await db.ensure_profile(USER)
    await db.update_profile(USER, daily_time="09:00")
    teacher.plain_value = "Morning! What are your plans for today?"

    await scheduler.tick(datetime(2026, 9, 21, 9, 0, tzinfo=MSK))

    history = await db.get_history(USER, 10)
    assert history[-1] == {
        "role": "assistant",
        "content": "Morning! What are your plans for today?",
    }


async def test_daily_question_off_by_default(parts, db: Database):
    scheduler, bot, _ = parts
    await db.ensure_profile(USER)

    await scheduler.tick(datetime(2026, 9, 21, 9, 0, tzinfo=MSK))

    assert bot.messages == []


async def test_model_failure_does_not_break_the_loop(parts, db: Database):
    scheduler, bot, teacher = parts
    await db.ensure_profile(USER)
    await db.update_profile(USER, daily_time="09:00")
    teacher.fail = True

    await scheduler.tick(datetime(2026, 9, 21, 9, 0, tzinfo=MSK))

    assert bot.messages == []  # молча пропустили день, без падения


async def test_weekly_digest_on_sunday_evening(parts, db: Database):
    scheduler, bot, _ = parts
    await db.ensure_profile(USER)
    await db.add_message(USER, "user", "Hello there")
    await db.record_errors(USER, [("grammar", "пропускает артикль")])

    # 21 сентября 2026 — понедельник, сводки быть не должно.
    await scheduler.tick(datetime(2026, 9, 21, 19, 0, tzinfo=MSK))
    assert bot.messages == []

    # 27 сентября — воскресенье.
    await scheduler.tick(datetime(2026, 9, 27, 19, 0, tzinfo=MSK))
    assert len(bot.messages) == 1
    assert "Итоги недели" in bot.messages[0][1]
    assert "пропускает артикль" in bot.messages[0][1]


async def test_weekly_digest_skipped_without_activity(parts, db: Database):
    scheduler, bot, _ = parts
    await db.ensure_profile(USER)

    await scheduler.tick(datetime(2026, 9, 27, 19, 0, tzinfo=MSK))

    assert bot.messages == []


async def test_unknown_timezone_falls_back_to_utc(config: Config, db: Database):
    service = TeacherService(config, db, FakeTeacher(), FakeSpeech())
    scheduler = Scheduler(replace(config, timezone="Мордор/Барад-Дур"), db, service, FakeBot())

    assert str(scheduler._tz) == "UTC"
