"""Фоновые задачи по расписанию: утренний вопрос и недельная сводка.

Бот, который только отвечает, тихо умирает: неделя без сообщений — и о нём
забыли. Здесь он пишет первым.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import Bot
from aiogram.types import FSInputFile

from .audio import AudioError
from .config import Config
from .db import Database
from .keyboards import reply_actions
from .llm import LLMError
from .service import TeacherService
from .speech import SpeechError

log = logging.getLogger(__name__)

#: Как часто проверяем расписание. Минуты достаточно: время задаётся в HH:MM.
TICK_SECONDS = 60


def parse_time(value: str | None) -> dt_time | None:
    """'09:30' -> time(9, 30). Мусор и None превращаются в None."""
    if not value:
        return None
    try:
        hours, _, minutes = value.strip().partition(":")
        return dt_time(hour=int(hours), minute=int(minutes))
    except (ValueError, TypeError):
        log.warning("Не разобрать время расписания: %r", value)
        return None


class Scheduler:
    def __init__(
        self, config: Config, db: Database, service: TeacherService, bot: Bot
    ) -> None:
        self._config = config
        self._db = db
        self._service = service
        self._bot = bot
        try:
            self._tz = ZoneInfo(config.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            log.warning("Неизвестный часовой пояс %r, беру UTC", config.timezone)
            self._tz = ZoneInfo("UTC")

    async def run(self) -> None:
        """Бесконечный цикл проверки расписания. Ошибки внутри не роняют его."""
        log.info("Расписание запущено, часовой пояс: %s", self._tz)
        while True:
            try:
                await self.tick(datetime.now(self._tz))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Сбой в расписании")
            await asyncio.sleep(TICK_SECONDS)

    async def tick(self, now: datetime) -> None:
        profile = await self._db.get_profile(self._config.allowed_user_id)
        if profile is None:
            return

        daily_at = parse_time(profile.daily_time)
        today = now.date().isoformat()
        if (
            daily_at is not None
            and now.time() >= daily_at
            and profile.daily_last_sent != today
        ):
            # Дату проставляем до отправки: если упадём на полпути, лучше
            # пропустить день, чем прислать десять вопросов подряд.
            await self._db.update_profile(profile.user_id, daily_last_sent=today)
            await self._send_daily(profile.user_id)

        weekly_at = parse_time(self._config.weekly_time)
        if (
            weekly_at is not None
            and now.weekday() == self._config.weekly_weekday
            and now.time() >= weekly_at
            and profile.weekly_last_sent != today
        ):
            await self._db.update_profile(profile.user_id, weekly_last_sent=today)
            await self._send_weekly(profile.user_id)

    async def _send_daily(self, user_id: int) -> None:
        try:
            question = await self._service.daily_question(user_id)
        except LLMError as exc:
            log.warning("Утренний вопрос не составлен: %s", exc)
            return
        if not question:
            return

        await self._bot.send_message(user_id, question, reply_markup=reply_actions())
        await self._send_voice(user_id, question)
        log.info("Отправлен утренний вопрос")

    async def _send_voice(self, user_id: int, text: str) -> None:
        profile = await self._db.get_profile(user_id)
        if profile is None or not profile.voice_replies:
            return
        import tempfile

        try:
            with tempfile.TemporaryDirectory(prefix="daily-") as tmp:
                path = await self._service.voice_answer(user_id, text, Path(tmp))
                if path is not None:
                    await self._bot.send_voice(user_id, FSInputFile(path))
        except (SpeechError, AudioError, OSError) as exc:
            log.warning("Озвучка утреннего вопроса не удалась: %s", exc)

    async def _send_weekly(self, user_id: int) -> None:
        digest = await self._service.weekly_digest(user_id)
        if digest is None:
            log.info("Недельная сводка пропущена: за неделю занятий не было")
            return
        await self._bot.send_message(user_id, digest)
        log.info("Отправлена недельная сводка")
