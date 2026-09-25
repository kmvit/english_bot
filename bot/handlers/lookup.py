"""Разбор выделенного фрагмента: ученик цитирует кусок сообщения и спрашивает.

Выбор слова делается родным жестом Telegram: выделить текст в сообщении и
нажать «Ответить». Telegram присылает выделенное в `message.quote`, так что
никакого отдельного режима, команды или ввода слова руками не нужно.
"""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, Message
from aiogram.utils.chat_action import ChatActionSender

from .. import keyboards as kb
from ..formatting import (
    DROPPED_FOOTER,
    LOOKUP_FAILED,
    LOOKUP_TOO_LONG,
    render_lookup,
)
from ..llm import LLMError
from ..service import MAX_LOOKUP_CHARS, TeacherService

log = logging.getLogger(__name__)

LLM_DOWN = "Модель сейчас не отвечает. Попробуй ещё раз через минуту."


async def on_quote(message: Message, service: TeacherService) -> None:
    fragment = (message.quote.text if message.quote else "").strip()
    if not fragment:
        return
    if len(fragment) > MAX_LOOKUP_CHARS:
        await message.reply(LOOKUP_TOO_LONG)
        return

    # Слово живёт в предложении: без него не выбрать нужное значение у
    # многозначного глагола и не понять, что за фраза выделена.
    source = message.reply_to_message
    context = (source.text or source.caption or "") if source else ""
    question = (message.text or "").strip()

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            found = await service.lookup(
                message.from_user.id, fragment, context, question
            )
        except LLMError as exc:
            log.error("LLM недоступна (разбор слова): %s", exc)
            await message.reply(LLM_DOWN)
            return

    if found is None:
        await message.reply(LOOKUP_FAILED)
        return
    card, word_id = found
    await message.reply(render_lookup(card), reply_markup=kb.lookup_actions(word_id))


async def on_drop(callback: CallbackQuery, service: TeacherService) -> None:
    raw = (callback.data or "").removeprefix(kb.WORD_DROP)
    word = (
        await service.forget_word(callback.from_user.id, int(raw))
        if raw.isdigit()
        else None
    )
    await callback.answer(f"Убрал: {word}" if word else "Этого слова уже нет в словаре")
    if callback.message is None or not callback.message.text:
        return
    # Переписываем подпись под карточкой, а не убираем кнопку молча: иначе
    # сообщение останется с обещанием спросить слово на тренировке. Подпись —
    # всегда последняя строка, её и меняем: так замена не зависит от того,
    # как aiogram соберёт HTML обратно из сущностей сообщения.
    body = callback.message.html_text.rsplit("\n", 1)[0]
    try:
        await callback.message.edit_text(f"{body}\n{DROPPED_FOOTER}")
    except TelegramBadRequest as exc:
        log.warning("Не удалось переписать карточку слова: %s", exc)


def build() -> Router:
    router = Router(name="lookup")
    router.callback_query.register(on_drop, F.data.startswith(kb.WORD_DROP))
    # Цитата — всегда просьба разобрать фрагмент, в том числе посреди
    # тренировки: поэтому роутер стоит раньше остальных обработчиков текста.
    router.message.register(on_quote, F.quote)
    return router
