"""Сборка сообщений Telegram (HTML) из структур данных."""
from __future__ import annotations

from html import escape
from typing import Sequence

from .db import ErrorStat, FluencyStat, ProgressStat, Profile, VocabWord
from .parsing import Lookup, TeacherReply

TELEGRAM_LIMIT = 4096

ERROR_TYPE_RU = {
    "grammar": "грамматика",
    "vocabulary": "лексика",
    "pronunciation": "произношение",
}

#: Подсказка про жест: без неё выделение текста так и останется незамеченным.
LOOKUP_HINT = (
    "💡 Непонятное слово или фраза в моём сообщении? Выдели её и нажми "
    "«Ответить» — разберу и добавлю в словарь. Можно сразу дописать вопрос."
)

SAVED_FOOTER = "➕ <i>Сохранил в словарь — спрошу на тренировке</i>"
DROPPED_FOOTER = "🗑 <i>Убрал из словаря</i>"

LOOKUP_FAILED = (
    "Не разобрал этот фрагмент. Попробуй выделить слово или фразу покороче."
)
LOOKUP_TOO_LONG = (
    "Это слишком длинный кусок. Выдели слово или фразу — а весь ответ целиком "
    "разберёт кнопка «Подробнее»."
)


def _e(text: str) -> str:
    return escape(text or "", quote=False)


def render_reply(reply: TeacherReply, include_conversation: bool = True) -> str:
    """Разговорная часть, затем блок исправлений, затем произношение.

    В режиме «только голос» разговорная часть уходит голосовым, а текстом
    остаются лишь исправления: их нужно видеть глазами.
    """
    parts: list[str] = []
    if reply.reply and include_conversation:
        head = _e(reply.reply)
        if reply.reply_ru:
            head += f"\n\n🇷🇺 <i>{_e(reply.reply_ru)}</i>"
        parts.append(head)
    # Подсказка идёт сразу за репликой, до разбора ошибок: это приглашение
    # продолжить разговор, а не замечание.
    if reply.expand:
        parts.append(f"💬 <i>{_e(reply.expand)}</i>")

    if reply.corrections:
        # Кому принадлежат зачёркнутые слова, должно быть видно из заголовка:
        # бот часто пересказывает фразу ученика, и без подписи разбор читается
        # так, будто он правит сам себя.
        lines = ["<b>Что поправить в твоей фразе</b>"]
        for correction in reply.corrections:
            if correction.original:
                lines.append(f"❌ {_e(correction.original)}")
            if correction.corrected:
                lines.append(f"✅ {_e(correction.corrected)}")
            if correction.note:
                lines.append(f"<i>{_e(correction.note)}</i>")
            lines.append("")
        parts.append("\n".join(lines).strip())

    if reply.pronunciation:
        lines = ["<b>Твоё произношение</b>"]
        for tip in reply.pronunciation:
            head = f"🗣 <b>{_e(tip.word)}</b>"
            if tip.phoneme:
                head += f" — /{_e(tip.phoneme)}/"
            lines.append(head)
            if tip.tip:
                lines.append(_e(tip.tip))
            if tip.example:
                lines.append(f"Тот же звук: <i>{_e(tip.example)}</i>")
            lines.append("")
        parts.append("\n".join(lines).strip())

    text = "\n\n".join(part for part in parts if part).strip()
    if not text:
        if not include_conversation:
            return ""  # нечего исправлять — в голосовом режиме текст не нужен
        text = "Не получилось составить ответ. Попробуй написать ещё раз."
    return _truncate(text)


def _truncate(text: str, limit: int = TELEGRAM_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def render_mistakes(errors: Sequence[ErrorStat]) -> str:
    if not errors:
        return "Пока ничего не накопилось — поговори со мной, и я начну вести список."
    lines = ["<b>Твои частые ошибки</b>"]
    for index, error in enumerate(errors, 1):
        kind = ERROR_TYPE_RU.get(error.type, error.type)
        lines.append(f"{index}. [{_e(kind)}] {_e(error.description)} — ×{error.count}")
        if error.example:
            lines.append(f"   <i>{_e(error.example)}</i>")
    return _truncate("\n".join(lines))


def _score_line(label: str, value: float | None) -> str:
    return f"{label}: {value:.0f}" if value is not None else f"{label}: —"


PRONUNCIATION_OFF = (
    "<i>Баллы произношения по фонемам появятся, когда подключим Azure — "
    "локальный Whisper их не даёт.</i>"
)


def _fluency_lines(stat: FluencyStat) -> list[str]:
    lines = []
    if stat.wpm is not None:
        lines.append(f"темп: {stat.wpm:.0f} слов/мин")
    if stat.pauses is not None:
        pause_part = f"паузы: {stat.pauses:.1f} за запись"
        if stat.pause_ratio is not None:
            pause_part += f" ({stat.pause_ratio * 100:.0f}% времени)"
        lines.append(pause_part)
    return lines


def render_progress(
    week: ProgressStat,
    month: ProgressStat,
    week_fluency: FluencyStat | None = None,
    month_fluency: FluencyStat | None = None,
    pronunciation_enabled: bool = True,
) -> str:
    if not month.samples:
        text = "Голосовых пока не было — пришли голосовое, и я начну считать динамику."
        return text if pronunciation_enabled else f"{text}\n\n{PRONUNCIATION_OFF}"

    def block(title: str, stat: ProgressStat, fluency: FluencyStat | None) -> str:
        if not stat.samples:
            return f"<b>{title}</b>\nнет голосовых"
        lines = [f"<b>{title}</b> ({stat.samples} голосовых)"]
        if fluency is not None:
            lines.extend(_fluency_lines(fluency))
        if pronunciation_enabled:
            lines.append(_score_line("общий", stat.overall))
            lines.append(_score_line("точность", stat.accuracy))
            lines.append(_score_line("интонация", stat.prosody))
        return "\n".join(lines)

    parts = [
        block("За 7 дней", week, week_fluency),
        block("За 30 дней", month, month_fluency),
    ]

    if pronunciation_enabled and week.overall and month.overall:
        delta = week.overall - month.overall
        arrow = "▲" if delta > 1 else ("▼" if delta < -1 else "▬")
        parts.append(f"{arrow} за неделю относительно месяца: {delta:+.0f}")
    elif week_fluency and month_fluency and week_fluency.wpm and month_fluency.wpm:
        delta = week_fluency.wpm - month_fluency.wpm
        arrow = "▲" if delta > 3 else ("▼" if delta < -3 else "▬")
        parts.append(f"{arrow} темп за неделю относительно месяца: {delta:+.0f} слов/мин")

    if not pronunciation_enabled:
        parts.append(PRONUNCIATION_OFF)
    return _truncate("\n\n".join(parts))


def render_words(words: Sequence[VocabWord], total: int, due: int = 0) -> str:
    if not words:
        return (
            "Словарь пока пуст. Сюда попадают слова, которые я ввожу в разговоре, "
            "и те, что ты сам просишь разобрать.\n\n" + LOOKUP_HINT
        )
    lines = [f"<b>Твой словарь</b> — {total} слов, последние {len(words)}:"]
    for word in words:
        line = f"• <b>{_e(word.word)}</b>"
        if word.meaning:
            line += f" — {_e(word.meaning)}"
        lines.append(line)
        if word.example:
            lines.append(f"   <i>{_e(word.example)}</i>")
    if due:
        lines.append("")
        lines.append(f"К повторению готово {due} — жми «🎯 Тренировка».")
    return _truncate("\n".join(lines))


def render_lookup(card: Lookup) -> str:
    """Карточка выделенного слова: перевод, звучание, пример, грабли."""
    head = f"📖 <b>{_e(card.term)}</b>"
    if card.translation:
        head += f" — {_e(card.translation)}"
    lines = [head]
    if card.ipa:
        lines.append(f"🗣 /{_e(card.ipa)}/")
    if card.meaning:
        lines.append("")
        lines.append(_e(card.meaning))
    if card.example:
        lines.append("")
        lines.append(f"<i>{_e(card.example)}</i>")
        if card.example_ru:
            lines.append(f"<i>{_e(card.example_ru)}</i>")
    if card.note:
        lines.append("")
        lines.append(f"💡 {_e(card.note)}")
    lines.append("")
    lines.append(SAVED_FOOTER)
    return _truncate("\n".join(lines))


def render_settings(profile: Profile) -> str:
    voice = "включены" if profile.voice_replies else "выключены"
    return (
        "<b>Настройки</b>\n"
        f"Уровень: {_e(profile.level or 'не задан')}\n"
        f"Интересы: {_e(profile.interests or 'не заданы')}\n"
        f"Голосовые ответы: {voice}"
    )


DRILL_NOTHING_TO_DO = (
    "Повторять нечего — либо ошибок пока не накопилось, либо все уже отработаны "
    "и ждут своего срока. Поговори со мной, и материал появится."
)


#: Что делать с предложением — зависит от вида задания.
DRILL_TASK_PROMPTS = {
    "error": "Найди ошибку и напиши предложение правильно:",
    "word": "Вставь пропущенное слово и напиши предложение целиком:",
}


def render_drill_task(index: int, total: int, exercise: dict) -> str:
    focus = exercise.get("focus") or ""
    head = f"<b>Задание {index} из {total}</b>"
    if focus:
        head += f" — {_e(focus)}"
    prompt = DRILL_TASK_PROMPTS.get(exercise.get("kind", "error"), DRILL_TASK_PROMPTS["error"])
    return (
        f"{head}\n\n{prompt}\n\n"
        f"<code>{_e(exercise.get('sentence', ''))}</code>"
    )


def render_drill_result(correct: bool, feedback: str, answer: str) -> str:
    lines = ["✅ Верно" if correct else "❌ Не совсем"]
    if feedback:
        lines.append(_e(feedback))
    if not correct and answer:
        lines.append(f"Правильно так: <code>{_e(answer)}</code>")
    return "\n".join(lines)


def render_drill_summary(correct: int, total: int) -> str:
    if not total:
        return DRILL_NOTHING_TO_DO
    if correct == total:
        tail = "Все верно — эти ошибки вернутся нескоро."
    elif correct:
        tail = "То, где ошибся, спрошу снова завтра."
    else:
        tail = "Ничего страшного: повторим завтра, пока не закрепится."
    return f"<b>Итог: {correct} из {total}</b>\n{tail}"


def render_weekly_digest(
    activity: dict,
    errors: Sequence[ErrorStat],
    week: FluencyStat,
    previous: FluencyStat,
    words_total: int,
) -> str:
    lines = ["<b>Итоги недели</b>"]
    lines.append(
        f"Сообщений: {activity['messages']}, из них голосовых: {activity['voices']}"
    )
    if activity["words"]:
        lines.append(f"Новых слов: {activity['words']} (всего в словаре {words_total})")

    if week.wpm is not None:
        line = f"Темп речи: {week.wpm:.0f} слов/мин"
        if previous.wpm is not None:
            delta = week.wpm - previous.wpm
            if abs(delta) >= 3:
                line += f" ({delta:+.0f} к месячному)"
        lines.append(line)

    if errors:
        lines.append("")
        lines.append("<b>Держатся крепче всего</b>")
        for error in errors:
            lines.append(f"• {_e(error.description)} — ×{error.count}")
        lines.append("")
        lines.append("Разобрать их: /drill")
    else:
        lines.append("")
        lines.append("Повторяющихся ошибок не набралось — хорошая неделя.")
    return _truncate("\n".join(lines))


def render_voice_too_long(duration: int, limit: int) -> str:
    return (
        f"Голосовое {duration} с — это слишком долго, я разбираю до {limit} с. "
        "Запиши, пожалуйста, короче."
    )
