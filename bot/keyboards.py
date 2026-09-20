"""Клавиатуры Telegram: онбординг, настройки, действия под ответом учителя."""
from __future__ import annotations

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from .db import Profile

LEVELS = ("A1", "A2", "B1", "B2", "C1")
LEVEL_HINTS = {
    "A1": "только начал",
    "A2": "простые фразы",
    "B1": "говорю о знакомом",
    "B2": "говорю свободно",
    "C1": "почти как носитель",
}

# --- callback_data -----------------------------------------------------
ONBOARD_LEVEL = "level:"          # выбор уровня при первом запуске
CFG_ROOT = "cfg:root"
CFG_LEVEL = "cfg:level"
CFG_LEVEL_SET = "cfg:level:"
CFG_INTERESTS = "cfg:interests"
CFG_VOICE = "cfg:voice"
REPLY_TRANSLATE = "reply:translate"
REPLY_EXPLAIN = "reply:explain"

# --- подписи нижней клавиатуры ----------------------------------------
BTN_MISTAKES = "📌 Ошибки"
BTN_PROGRESS = "📈 Прогресс"
BTN_SETTINGS = "⚙️ Настройки"
BOTTOM_BUTTONS = frozenset({BTN_MISTAKES, BTN_PROGRESS, BTN_SETTINGS})


def onboarding_levels() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"{level} — {LEVEL_HINTS[level]}",
                    callback_data=f"{ONBOARD_LEVEL}{level}",
                )
            ]
            for level in LEVELS
        ]
    )


def settings_menu(profile: Profile) -> InlineKeyboardMarkup:
    voice_label = "🔇 Выключить голос" if profile.voice_replies else "🔊 Включить голос"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"Уровень: {profile.level or 'не задан'}",
                    callback_data=CFG_LEVEL,
                ),
            ],
            [InlineKeyboardButton(text="Интересы", callback_data=CFG_INTERESTS)],
            [InlineKeyboardButton(text=voice_label, callback_data=CFG_VOICE)],
        ]
    )


def settings_levels(current: str | None) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=("✅ " if level == current else "") + f"{level} — {LEVEL_HINTS[level]}",
                callback_data=f"{CFG_LEVEL_SET}{level}",
            )
        ]
        for level in LEVELS
    ]
    rows.append([InlineKeyboardButton(text="← Назад", callback_data=CFG_ROOT)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def reply_actions() -> InlineKeyboardMarkup:
    """Кнопки под ответом учителя. Каждое нажатие — отдельный запрос к модели."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🇷🇺 Перевести", callback_data=REPLY_TRANSLATE),
                InlineKeyboardButton(text="💡 Подробнее", callback_data=REPLY_EXPLAIN),
            ]
        ]
    )


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_MISTAKES), KeyboardButton(text=BTN_PROGRESS)],
            [KeyboardButton(text=BTN_SETTINGS)],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Пиши по-английски или запиши голосовое",
    )
