"""/drill — отработка собственных ошибок с интервальным повторением."""
from __future__ import annotations

import logging
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from aiogram.utils.chat_action import ChatActionSender

from ..formatting import (
    DRILL_NOTHING_TO_DO,
    render_drill_result,
    render_drill_summary,
    render_drill_task,
)
from ..llm import LLMError
from ..service import TeacherService

log = logging.getLogger(__name__)

DRILL_STOP = "drill:stop"
LLM_DOWN = "Модель сейчас не отвечает, тренировка отложена. Попробуй через минуту."
INTRO = (
    "Разберём то, в чём ты ошибаешься чаще всего. "
    "Пиши предложение целиком — так лучше запоминается."
)


class Drill(StatesGroup):
    answering = State()


def _stop_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Закончить тренировку", callback_data=DRILL_STOP)]
        ]
    )


async def _send_task(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    exercises: list[dict[str, Any]] = data["exercises"]
    index: int = data["index"]
    await message.answer(
        render_drill_task(index + 1, len(exercises), exercises[index]),
        reply_markup=_stop_keyboard(),
    )


async def _finish(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    await state.clear()
    await message.answer(
        render_drill_summary(data.get("correct", 0), data.get("answered", 0))
    )


async def cmd_drill(message: Message, service: TeacherService, state: FSMContext) -> None:
    await state.clear()
    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            exercises = await service.start_drill(message.from_user.id)
        except LLMError as exc:
            log.error("Не удалось составить тренировку: %s", exc)
            await message.answer(LLM_DOWN)
            return

    if not exercises:
        await message.answer(DRILL_NOTHING_TO_DO)
        return

    await state.set_state(Drill.answering)
    await state.update_data(exercises=exercises, index=0, correct=0, answered=0)
    await message.answer(INTRO)
    await _send_task(message, state)


async def on_answer(message: Message, service: TeacherService, state: FSMContext) -> None:
    data = await state.get_data()
    exercises: list[dict[str, Any]] = data["exercises"]
    index: int = data["index"]
    exercise = exercises[index]

    async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        try:
            correct, feedback = await service.check_drill_answer(
                message.from_user.id, exercise, message.text or ""
            )
        except LLMError as exc:
            log.error("Не удалось проверить ответ: %s", exc)
            # Состояние не трогаем: ученик сможет ответить ещё раз.
            await message.answer(LLM_DOWN)
            return

    await message.answer(
        render_drill_result(correct, feedback, exercise.get("answer", ""))
    )

    await state.update_data(
        index=index + 1,
        correct=data.get("correct", 0) + int(correct),
        answered=data.get("answered", 0) + 1,
    )
    if index + 1 >= len(exercises):
        await _finish(message, state)
        return
    await _send_task(message, state)


async def on_stop(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer("Тренировка закончена")
    if callback.message is not None:
        await _finish(callback.message, state)


def build() -> Router:
    router = Router(name="drill")
    router.message.register(cmd_drill, Command("drill"))
    router.callback_query.register(on_stop, F.data == DRILL_STOP)
    # Во время тренировки текст — это ответ на задание, а не реплика диалога.
    router.message.register(
        on_answer, Drill.answering, F.text & ~F.text.startswith("/")
    )
    return router
