"""Точка входа: сборка зависимостей и long polling."""
from __future__ import annotations

import asyncio
import logging
import socket
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, ErrorEvent

from .config import Config, load_config
from .db import Database
from .handlers import build_router
from .llm import Teacher
from .middlewares import AllowedUserMiddleware
from .service import TeacherService
from .speech import build_speech

log = logging.getLogger("bot")

COMMANDS = [
    BotCommand(command="topic", description="начать разговор на тему"),
    BotCommand(command="settings", description="уровень, интересы, голос"),
    BotCommand(command="mistakes", description="мои частые ошибки"),
    BotCommand(command="progress", description="динамика произношения"),
    BotCommand(command="cost", description="расходы за месяц"),
    BotCommand(command="voice", description="голосовые ответы on/off"),
    BotCommand(command="reset", description="очистить историю диалога"),
    BotCommand(command="help", description="что я умею"),
]


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)


def build_session(config: Config) -> AiohttpSession | None:
    """Сессия с принудительным IPv4: на части сетей IPv6 до Telegram не проходит,
    и polling зависает без единой ошибки в логе."""
    if not config.telegram_force_ipv4:
        return None
    session = AiohttpSession()
    # Публичной точки для настройки коннектора у aiogram нет.
    session._connector_init["family"] = socket.AF_INET
    return session


async def run(config: Config) -> None:
    db = Database(config.db_path)
    await db.connect()
    teacher = Teacher(config)
    speech = build_speech(config)
    service = TeacherService(config, db, teacher, speech)

    bot = Bot(
        token=config.telegram_token,
        session=build_session(config),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher["config"] = config
    dispatcher["db"] = db
    dispatcher["service"] = service

    guard = AllowedUserMiddleware(config.allowed_user_id)
    dispatcher.message.outer_middleware(guard)
    dispatcher.callback_query.outer_middleware(guard)
    dispatcher.include_router(build_router())

    @dispatcher.errors()
    async def on_unexpected_error(event: ErrorEvent) -> bool:
        """Любая необработанная ошибка не должна ронять polling."""
        log.exception("Необработанная ошибка: %s", event.exception)
        message = getattr(event.update, "message", None)
        if message is not None:
            try:
                await message.answer("Что-то сломалось на моей стороне. Попробуй ещё раз.")
            except Exception:  # pragma: no cover - Telegram может отказать
                log.exception("Не удалось сообщить пользователю об ошибке")
        return True

    await bot.set_my_commands(COMMANDS)
    log.info(
        "Бот запущен. Модель: %s | распознавание: %s (произношение: %s) | озвучка: %s",
        config.llm_model,
        speech.recognizer_name,
        "да" if speech.assesses_pronunciation else "нет",
        speech.synthesizer_name,
    )
    # Прогрев в фоне: загрузка модели не должна задерживать старт polling.
    warmup = getattr(speech, "warmup", None)
    if warmup is not None:
        asyncio.create_task(warmup())
    try:
        await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())
    finally:
        await db.close()
        await teacher.close()
        await bot.session.close()


def main() -> int:
    try:
        config = load_config()
    except RuntimeError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 2
    setup_logging(config.log_level)
    try:
        asyncio.run(run(config))
    except (KeyboardInterrupt, SystemExit):
        log.info("Остановлен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
