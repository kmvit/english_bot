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
CFG_VOICE_ONLY = "cfg:voiceonly"
CFG_TRANSLATE = "cfg:translate"
CFG_STYLE = "cfg:style"
CFG_DAILY = "cfg:daily"
CFG_DAILY_SET = "cfg:daily:"
TOPIC_PICK = "topic:"
REPLY_TRANSLATE = "reply:translate"
REPLY_EXPLAIN = "reply:explain"
WORD_DROP = "word:drop:"          # убрать слово, сохранённое по разбору

# --- подписи нижней клавиатуры ----------------------------------------
BTN_DRILL = "🎯 Тренировка"
BTN_MISTAKES = "📌 Ошибки"
BTN_WORDS = "📖 Словарь"
BTN_PROGRESS = "📈 Прогресс"
BTN_TOPIC = "💬 Тема"
BTN_SETTINGS = "⚙️ Настройки"
BOTTOM_BUTTONS = frozenset(
    {BTN_DRILL, BTN_MISTAKES, BTN_WORDS, BTN_PROGRESS, BTN_TOPIC, BTN_SETTINGS}
)

# Время утреннего вопроса: готовые варианты вместо ручного ввода.
DAILY_PRESETS = ("08:00", "09:00", "12:00", "19:00", "21:00")

# Манера учителя: по кругу нажатием одной кнопки.
STYLES = ("neutral", "casual", "savage")
STYLE_LABELS = {
    "neutral": "ровная",
    "casual": "свободная, с матом",
    "savage": "жёсткая, может и обругать",
}

# Темы на случай, когда интересы ещё не заданы.
DEFAULT_TOPICS = ("travel", "food", "work", "movies", "sport", "plans for the weekend")


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
    only_label = "📝 Вернуть текст" if profile.voice_only else "🎧 Только голос"
    translate_label = (
        "🇷🇺 Перевод: включён" if profile.translate_replies else "🇷🇺 Перевод: выключен"
    )
    daily_label = (
        f"⏰ Вопрос по утрам: {profile.daily_time}"
        if profile.daily_time
        else "⏰ Вопрос по утрам: выкл"
    )
    rows = [
        [
            InlineKeyboardButton(
                text=f"Уровень: {profile.level or 'не задан'}", callback_data=CFG_LEVEL
            )
        ],
        [InlineKeyboardButton(text="Интересы", callback_data=CFG_INTERESTS)],
        [InlineKeyboardButton(text=daily_label, callback_data=CFG_DAILY)],
        [InlineKeyboardButton(text=translate_label, callback_data=CFG_TRANSLATE)],
        [
            InlineKeyboardButton(
                text=f"🎭 Манера: {STYLE_LABELS.get(profile.tutor_style, 'ровная')}",
                callback_data=CFG_STYLE,
            )
        ],
        [InlineKeyboardButton(text=voice_label, callback_data=CFG_VOICE)],
    ]
    # Режим «только голос» имеет смысл лишь когда голосовые вообще включены.
    if profile.voice_replies:
        rows.append(
            [InlineKeyboardButton(text=only_label, callback_data=CFG_VOICE_ONLY)]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def daily_times(current: str | None) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=("✅ " if preset == current else "") + preset,
                callback_data=f"{CFG_DAILY_SET}{preset}",
            )
            for preset in DAILY_PRESETS[:3]
        ],
        [
            InlineKeyboardButton(
                text=("✅ " if preset == current else "") + preset,
                callback_data=f"{CFG_DAILY_SET}{preset}",
            )
            for preset in DAILY_PRESETS[3:]
        ],
        [InlineKeyboardButton(text="Не писать по утрам", callback_data=f"{CFG_DAILY_SET}off")],
        [InlineKeyboardButton(text="← Назад", callback_data=CFG_ROOT)],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def topic_suggestions(interests: str | None) -> InlineKeyboardMarkup:
    """Темы из интересов ученика, добитые общими до шести."""
    picked = [part.strip() for part in (interests or "").split(",") if part.strip()][:4]
    for topic in DEFAULT_TOPICS:
        if len(picked) >= 6:
            break
        if topic.lower() not in {p.lower() for p in picked}:
            picked.append(topic)
    rows = [
        [InlineKeyboardButton(text=topic, callback_data=f"{TOPIC_PICK}{topic}"[:64])]
        for topic in picked
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


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


def reply_actions(translated: bool = False) -> InlineKeyboardMarkup:
    """Кнопки под ответом учителя. Каждое нажатие — отдельный запрос к модели.

    Кнопку перевода прячем, когда перевод уже показан: нажимать её незачем,
    а место она занимает.
    """
    buttons = []
    if not translated:
        buttons.append(
            InlineKeyboardButton(text="🇷🇺 Перевести", callback_data=REPLY_TRANSLATE)
        )
    buttons.append(InlineKeyboardButton(text="💡 Подробнее", callback_data=REPLY_EXPLAIN))
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def lookup_actions(word_id: int) -> InlineKeyboardMarkup:
    """Единственное действие под карточкой слова: передумать и убрать его."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🗑 Не сохранять", callback_data=f"{WORD_DROP}{word_id}"
                )
            ]
        ]
    )


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_DRILL), KeyboardButton(text=BTN_MISTAKES)],
            [KeyboardButton(text=BTN_WORDS), KeyboardButton(text=BTN_PROGRESS)],
            [KeyboardButton(text=BTN_TOPIC), KeyboardButton(text=BTN_SETTINGS)],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Пиши по-английски или запиши голосовое",
    )
