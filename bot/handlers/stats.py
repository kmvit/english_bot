"""/mistakes, /progress, /cost."""
from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from ..config import Config
from ..costs import LLM, STT, TTS, format_usd
from ..db import Database, month_start
from ..formatting import render_mistakes, render_progress, render_words
from ..service import TeacherService

KIND_RU = {
    LLM: "модель",
    STT: "распознавание речи",
    TTS: "озвучка",
}


async def cmd_mistakes(message: Message, db: Database) -> None:
    errors = await db.top_errors(message.from_user.id, limit=10)
    await message.answer(render_mistakes(errors))


async def cmd_progress(
    message: Message, db: Database, service: TeacherService
) -> None:
    user_id = message.from_user.id
    await message.answer(
        render_progress(
            week=await db.pronunciation_progress(user_id, days=7),
            month=await db.pronunciation_progress(user_id, days=30),
            week_fluency=await db.fluency_progress(user_id, days=7),
            month_fluency=await db.fluency_progress(user_id, days=30),
            pronunciation_enabled=service.assesses_pronunciation,
        )
    )


async def cmd_words(message: Message, db: Database) -> None:
    user_id = message.from_user.id
    words = await db.recent_words(user_id, limit=20)
    await message.answer(
        render_words(
            words,
            await db.words_total(user_id),
            due=await db.words_due_total(user_id),
        )
    )


async def cmd_cost(message: Message, db: Database, config: Config) -> None:
    rows = await db.usage_since(message.from_user.id, month_start())
    if not rows:
        await message.answer("В этом месяце расходов пока нет.")
        return

    lines = ["<b>Расходы с начала месяца</b>"]
    total = 0.0
    for row in rows:
        cost = float(row.get("cost") or 0.0)
        total += cost
        name = KIND_RU.get(row["kind"], row["kind"])
        detail = ""
        if row["kind"] == LLM:
            tokens_in = int(row.get("tokens_in") or 0)
            tokens_out = int(row.get("tokens_out") or 0)
            detail = f", {tokens_in} вх. / {tokens_out} исх. токенов"
        elif row["kind"] == STT:
            detail = f", {float(row.get('seconds') or 0):.0f} с аудио"
        elif row["kind"] == TTS:
            detail = f", {int(row.get('chars') or 0)} символов"
        lines.append(
            f"{name}: {format_usd(cost)} ({int(row.get('calls') or 0)} запросов{detail})"
        )
    lines.append(f"\n<b>Итого: {format_usd(total)}</b>")
    lines.append(f"<i>Модель: {config.llm_model}</i>")
    await message.answer("\n".join(lines))


def build() -> Router:
    router = Router(name="stats")
    router.message.register(cmd_mistakes, Command("mistakes"))
    router.message.register(cmd_progress, Command("progress"))
    router.message.register(cmd_words, Command("words"))
    router.message.register(cmd_cost, Command("cost"))
    return router
