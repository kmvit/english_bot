"""Разговор: /topic, текстовые сообщения, голосовые."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, FSInputFile, Message
from aiogram.utils.chat_action import ChatActionSender

from .. import keyboards as kb
from ..assessment import Assessment
from ..audio import AudioError, ogg_to_wav, wav_duration
from ..config import Config
from ..db import Database
from ..formatting import render_voice_too_long
from ..llm import LLMError
from ..prompts import (
    build_text_turn,
    build_topic_turn,
    build_voice_turn,
    is_short_answer,
)
from ..service import TeacherService
from ..speech import SpeechError

log = logging.getLogger(__name__)

LLM_DOWN = "Модель сейчас не отвечает. Попробуй ещё раз через минуту."
SPEECH_DOWN = "Не получилось разобрать голосовое. Попробуй записать ещё раз."
AUDIO_DOWN = "Не смог обработать аудио. Попробуй записать голосовое ещё раз."
NOT_RECOGNIZED = (
    "Я не разобрал ни слова. Проверь, что говоришь по-английски и не слишком тихо."
)
NOTHING_TO_ACT_ON = "Не вижу, что обрабатывать — напиши сообщение заново."


async def _reply_turn(
    message: Message,
    service: TeacherService,
    llm_turn: str,
    history_entry: str,
    assessment: Assessment | None = None,
    user_id: int | None = None,
) -> None:
    """Спросить модель, отправить текст и (если включено) голосовой ответ.

    `user_id` нужен для ответов на нажатие кнопки: там `message` принадлежит
    боту, и его `from_user` — сам бот, а не ученик.
    """
    user_id = user_id if user_id is not None else message.from_user.id
    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            result = await service.handle_turn(
                user_id, llm_turn, history_entry, assessment
            )
        except LLMError as exc:
            log.error("LLM недоступна: %s", exc)
            await message.answer(LLM_DOWN)
            return

    if result.text:
        await message.answer(
            result.text, reply_markup=kb.reply_actions(result.translated)
        )

    if result.hint:
        await message.answer(result.hint)

    if not (result.voice_enabled and result.spoken_text):
        return
    with tempfile.TemporaryDirectory(prefix="tts-") as tmp:
        async with ChatActionSender.record_voice(
            bot=message.bot, chat_id=message.chat.id
        ):
            voice_path = await service.voice_answer(
                user_id, result.spoken_text, Path(tmp)
            )
        if voice_path is not None:
            await message.answer_voice(FSInputFile(voice_path))


async def _start_topic(
    message: Message, service: TeacherService, topic: str, user_id: int | None = None
) -> None:
    await _reply_turn(
        message,
        service,
        build_topic_turn(topic),
        f"Let's talk about {topic}.",
        user_id=user_id,
    )


async def cmd_topic(
    message: Message, command: CommandObject, service: TeacherService, db: Database
) -> None:
    topic = (command.args or "").strip()
    if topic:
        await _start_topic(message, service, topic)
        return
    profile = await db.ensure_profile(message.from_user.id)
    await message.answer(
        "О чём поговорим? Выбери или просто напиши свою тему:",
        reply_markup=kb.topic_suggestions(profile.interests),
    )


async def btn_topic(message: Message, service: TeacherService, db: Database) -> None:
    profile = await db.ensure_profile(message.from_user.id)
    await message.answer(
        "О чём поговорим? Выбери или просто напиши свою тему:",
        reply_markup=kb.topic_suggestions(profile.interests),
    )


async def on_topic_pick(callback: CallbackQuery, service: TeacherService) -> None:
    topic = (callback.data or "").split(":", 1)[1]
    await callback.answer(topic)
    if callback.message is None:
        return
    await callback.message.edit_text(f"Тема: <b>{topic}</b>")
    await _start_topic(callback.message, service, topic, user_id=callback.from_user.id)


async def on_voice(
    message: Message, service: TeacherService, config: Config
) -> None:
    voice = message.voice
    if voice is None:
        return
    duration = voice.duration or 0
    if duration > config.max_voice_seconds:
        await message.answer(render_voice_too_long(duration, config.max_voice_seconds))
        return

    with tempfile.TemporaryDirectory(prefix="voice-") as tmp:
        workdir = Path(tmp)
        ogg_path = workdir / "voice.ogg"
        wav_path = workdir / "voice.wav"
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            try:
                await message.bot.download(voice, destination=ogg_path)
                await ogg_to_wav(ogg_path, wav_path, config.ffmpeg_bin)
            except (AudioError, OSError) as exc:
                log.error("Конвертация голосового не удалась: %s", exc)
                await message.answer(AUDIO_DOWN)
                return

            seconds = wav_duration(wav_path) or float(duration)
            try:
                assessment = await service.transcribe(wav_path, message.from_user.id, seconds)
            except SpeechError as exc:
                log.error("Azure Speech недоступен: %s", exc)
                await message.answer(SPEECH_DOWN)
                return

        if assessment.is_empty:
            await message.answer(NOT_RECOGNIZED)
            return

        short = is_short_answer(assessment.transcript, config.short_answer_words)
        await _reply_turn(
            message,
            service,
            build_voice_turn(assessment, short),
            assessment.transcript,
            assessment,
        )


async def on_text(message: Message, service: TeacherService, config: Config) -> None:
    text = (message.text or "").strip()
    if not text:
        return
    short = is_short_answer(text, config.short_answer_words)
    await _reply_turn(message, service, build_text_turn(text, short), text)


async def on_unknown_command(message: Message) -> None:
    await message.answer("Такой команды нет. Список команд — /help")


async def on_other(message: Message) -> None:
    await message.answer(
        "Я понимаю текст и голосовые. Пришли сообщение или посмотри /help."
    )


async def _act_on_reply(
    callback: CallbackQuery, service: TeacherService, action: str
) -> None:
    """Перевести или разобрать подробнее текст сообщения, под которым нажата кнопка."""
    source = (callback.message.text if callback.message else "") or ""
    if not source.strip():
        await callback.answer(NOTHING_TO_ACT_ON, show_alert=True)
        return
    await callback.answer("Секунду…")
    async with ChatActionSender.typing(
        bot=callback.bot, chat_id=callback.message.chat.id
    ):
        try:
            if action == "translate":
                text = await service.translate_reply(callback.from_user.id, source)
            else:
                text = await service.explain_reply(callback.from_user.id, source)
        except LLMError as exc:
            log.error("LLM недоступна (%s): %s", action, exc)
            await callback.message.answer(LLM_DOWN)
            return
    # Баг с кнопками плавающий и в базе не оседает: перевод и пояснение
    # нигде не хранятся. Без журнала поймать его можно только на словах.
    log.info(
        "Кнопка %s. Исходник: %r. Ответ: %r",
        action,
        source[:300],
        (text or "")[:300],
    )
    await callback.message.reply(text or LLM_DOWN)


async def on_translate(callback: CallbackQuery, service: TeacherService) -> None:
    await _act_on_reply(callback, service, "translate")


async def on_explain(callback: CallbackQuery, service: TeacherService) -> None:
    await _act_on_reply(callback, service, "explain")


def build() -> Router:
    router = Router(name="chat")
    router.callback_query.register(on_translate, F.data == kb.REPLY_TRANSLATE)
    router.callback_query.register(on_explain, F.data == kb.REPLY_EXPLAIN)
    router.callback_query.register(on_topic_pick, F.data.startswith(kb.TOPIC_PICK))
    router.message.register(btn_topic, F.text == kb.BTN_TOPIC)
    router.message.register(cmd_topic, Command("topic"))
    router.message.register(on_voice, F.voice)
    router.message.register(on_text, F.text & ~F.text.startswith("/"))
    router.message.register(on_unknown_command, F.text.startswith("/"))
    # Последний: всё, что не текст и не голос.
    router.message.register(on_other)
    return router
