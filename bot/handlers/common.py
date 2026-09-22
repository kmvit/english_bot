"""Онбординг, настройки, нижняя клавиатура, /voice, /reset, /help."""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message

from .. import keyboards as kb
from ..db import Database, Profile
from ..formatting import render_settings
from ..service import TeacherService
from .drill import cmd_drill
from .stats import cmd_mistakes, cmd_progress, cmd_words

log = logging.getLogger(__name__)

SKIP_WORDS = {"-", "—", "skip", "пропустить", "потом"}

HELP_TEXT = (
    "<b>Что я умею</b>\n"
    "Пиши или присылай голосовые на английском — я отвечаю, исправляю ошибки "
    "и разбираю произношение.\n\n"
    "Под каждым моим ответом есть кнопки «Перевести» и «Подробнее».\n\n"
    "/topic &lt;тема&gt; — начать разговор на тему\n"
    "/settings — уровень, интересы, голосовые ответы\n"
    "/mistakes — мои частые ошибки\n"
    "/progress — динамика произношения\n"
    "/cost — расходы за месяц\n"
    "/voice on|off — голосовые ответы\n"
    "/reset — очистить историю диалога"
)

ASK_INTERESTS = (
    "Напиши через запятую, что тебе интересно (например: travel, IT, football) — "
    "буду подбирать темы.\nИли отправь «-», чтобы пропустить."
)


class Onboarding(StatesGroup):
    interests = State()


class Settings(StatesGroup):
    interests = State()


async def _edit(callback: CallbackQuery, text: str, markup: InlineKeyboardMarkup) -> None:
    """Перерисовать сообщение, не падая на «message is not modified»."""
    if callback.message is None:
        return
    try:
        await callback.message.edit_text(text, reply_markup=markup)
    except TelegramBadRequest as exc:
        if "message is not modified" not in str(exc):
            raise


async def _show_settings(callback: CallbackQuery, db: Database) -> None:
    profile = await db.ensure_profile(callback.from_user.id)
    await _edit(callback, render_settings(profile), kb.settings_menu(profile))


# --- онбординг ---------------------------------------------------------


async def cmd_start(message: Message, db: Database, state: FSMContext) -> None:
    await state.clear()
    await db.ensure_profile(message.from_user.id)
    await message.answer(
        "Hi! I'm your English tutor. Let's talk — I'll fix your mistakes as we go.",
        reply_markup=kb.main_keyboard(),
    )
    await message.answer("Сначала выбери свой уровень:", reply_markup=kb.onboarding_levels())


async def set_level(callback: CallbackQuery, db: Database, state: FSMContext) -> None:
    level = (callback.data or "").split(":", 1)[1]
    if level not in kb.LEVELS:
        await callback.answer("Неизвестный уровень")
        return
    await db.ensure_profile(callback.from_user.id)
    await db.update_profile(callback.from_user.id, level=level)
    await callback.answer(f"Уровень {level}")
    if callback.message is not None:
        await callback.message.edit_text(f"Уровень: <b>{level}</b>\n\n{ASK_INTERESTS}")
    await state.set_state(Onboarding.interests)


async def set_interests(message: Message, db: Database, state: FSMContext) -> None:
    interests = (message.text or "").strip()
    await state.clear()
    await db.ensure_profile(message.from_user.id)
    if interests.lower() in SKIP_WORDS:
        await message.answer(
            "Хорошо, спрошу про интересы по ходу разговора. Пиши или присылай голосовое.",
            reply_markup=kb.main_keyboard(),
        )
        return
    await db.update_profile(message.from_user.id, interests=interests[:500])
    await message.answer(
        "Готово. Пиши или присылай голосовое — начинаем.\n"
        "Подсказка: /topic travel, чтобы я задал тему сам.",
        reply_markup=kb.main_keyboard(),
    )


# --- настройки ---------------------------------------------------------


async def cmd_settings(message: Message, db: Database, state: FSMContext) -> None:
    await state.clear()
    profile = await db.ensure_profile(message.from_user.id)
    await message.answer(render_settings(profile), reply_markup=kb.settings_menu(profile))


async def cfg_root(callback: CallbackQuery, db: Database, state: FSMContext) -> None:
    await state.clear()
    await _show_settings(callback, db)
    await callback.answer()


async def cfg_level(callback: CallbackQuery, db: Database) -> None:
    profile = await db.ensure_profile(callback.from_user.id)
    await _edit(callback, "Выбери уровень:", kb.settings_levels(profile.level))
    await callback.answer()


async def cfg_level_set(callback: CallbackQuery, db: Database) -> None:
    level = (callback.data or "").rsplit(":", 1)[-1]
    if level not in kb.LEVELS:
        await callback.answer("Неизвестный уровень")
        return
    await db.ensure_profile(callback.from_user.id)
    await db.update_profile(callback.from_user.id, level=level)
    await callback.answer(f"Уровень {level}")
    await _show_settings(callback, db)


async def cfg_interests(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(Settings.interests)
    if callback.message is not None:
        await callback.message.answer(ASK_INTERESTS)
    await callback.answer()


async def edit_interests(message: Message, db: Database, state: FSMContext) -> None:
    await state.clear()
    profile = await db.ensure_profile(message.from_user.id)
    text = (message.text or "").strip()
    if text.lower() not in SKIP_WORDS:
        await db.update_profile(message.from_user.id, interests=text[:500])
        profile = await db.ensure_profile(message.from_user.id)
    await message.answer(render_settings(profile), reply_markup=kb.settings_menu(profile))


async def cfg_daily(callback: CallbackQuery, db: Database) -> None:
    profile = await db.ensure_profile(callback.from_user.id)
    await _edit(
        callback,
        "Во сколько писать утренний вопрос?\n"
        "Бот напишет первым и предложит тему — так занятия не забываются.",
        kb.daily_times(profile.daily_time),
    )
    await callback.answer()


async def cfg_daily_set(callback: CallbackQuery, db: Database) -> None:
    value = (callback.data or "").rsplit(":", 1)[-1]
    await db.ensure_profile(callback.from_user.id)
    if value == "off":
        await db.update_profile(callback.from_user.id, daily_time=None)
        await callback.answer("По утрам не пишу")
    elif value in kb.DAILY_PRESETS:
        # Сбрасываем отметку об отправке: иначе при смене времени в тот же
        # день вопрос не придёт до завтра.
        await db.update_profile(
            callback.from_user.id, daily_time=value, daily_last_sent=None
        )
        await callback.answer(f"Буду писать в {value}")
    else:
        await callback.answer("Не понял время")
        return
    await _show_settings(callback, db)


async def cfg_voice_only(callback: CallbackQuery, db: Database) -> None:
    profile = await db.ensure_profile(callback.from_user.id)
    await db.update_profile(callback.from_user.id, voice_only=not profile.voice_only)
    await callback.answer(
        "Текст вернул" if profile.voice_only else "Теперь только голос"
    )
    await _show_settings(callback, db)


async def cfg_voice(callback: CallbackQuery, db: Database) -> None:
    profile = await db.ensure_profile(callback.from_user.id)
    await db.update_profile(callback.from_user.id, voice_replies=not profile.voice_replies)
    await callback.answer("Голос включён" if not profile.voice_replies else "Голос выключен")
    await _show_settings(callback, db)


# --- команды -----------------------------------------------------------


async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT, reply_markup=kb.main_keyboard())


async def cmd_voice(message: Message, command: CommandObject, db: Database) -> None:
    arg = (command.args or "").strip().lower()
    profile: Profile = await db.ensure_profile(message.from_user.id)
    if arg not in ("on", "off"):
        state = "включены" if profile.voice_replies else "выключены"
        await message.answer(
            f"Голосовые ответы {state}. Переключить: /voice on или /voice off"
        )
        return
    await db.update_profile(message.from_user.id, voice_replies=(arg == "on"))
    await message.answer(
        "Буду присылать голосовые ответы." if arg == "on" else "Только текст, без голоса."
    )


async def cmd_reset(message: Message, db: Database, state: FSMContext) -> None:
    await state.clear()
    await db.clear_history(message.from_user.id)
    await message.answer("История диалога очищена. Профиль, ошибки и баллы на месте.")


# --- нижняя клавиатура -------------------------------------------------


async def btn_mistakes(message: Message, db: Database) -> None:
    await cmd_mistakes(message, db)


async def btn_progress(message: Message, db: Database, service: TeacherService) -> None:
    await cmd_progress(message, db, service)


async def btn_settings(message: Message, db: Database, state: FSMContext) -> None:
    await cmd_settings(message, db, state)


async def btn_words(message: Message, db: Database) -> None:
    await cmd_words(message, db)


async def btn_drill(message: Message, service, state: FSMContext) -> None:
    await cmd_drill(message, service, state)


def build() -> Router:
    """Свежий роутер на каждый вызов — так его можно поднимать в тестах."""
    router = Router(name="common")
    router.message.register(cmd_start, CommandStart())
    router.callback_query.register(set_level, F.data.startswith(kb.ONBOARD_LEVEL))

    router.callback_query.register(cfg_root, F.data == kb.CFG_ROOT)
    router.callback_query.register(cfg_level, F.data == kb.CFG_LEVEL)
    router.callback_query.register(cfg_level_set, F.data.startswith(kb.CFG_LEVEL_SET))
    router.callback_query.register(cfg_interests, F.data == kb.CFG_INTERESTS)
    router.callback_query.register(cfg_voice, F.data == kb.CFG_VOICE)
    router.callback_query.register(cfg_voice_only, F.data == kb.CFG_VOICE_ONLY)
    router.callback_query.register(cfg_daily, F.data == kb.CFG_DAILY)
    router.callback_query.register(cfg_daily_set, F.data.startswith(kb.CFG_DAILY_SET))

    # Ввод интересов: команды и кнопки не должны попадать в профиль как текст.
    free_text = F.text & ~F.text.startswith("/") & ~F.text.in_(kb.BOTTOM_BUTTONS)
    router.message.register(set_interests, Onboarding.interests, free_text)
    router.message.register(edit_interests, Settings.interests, free_text)

    router.message.register(btn_drill, F.text == kb.BTN_DRILL)
    router.message.register(btn_mistakes, F.text == kb.BTN_MISTAKES)
    router.message.register(btn_words, F.text == kb.BTN_WORDS)
    router.message.register(btn_progress, F.text == kb.BTN_PROGRESS)
    router.message.register(btn_settings, F.text == kb.BTN_SETTINGS)

    router.message.register(cmd_settings, Command("settings"))
    router.message.register(cmd_help, Command("help"))
    router.message.register(cmd_voice, Command("voice"))
    router.message.register(cmd_reset, Command("reset"))
    return router
