"""Сборка сообщений Telegram (HTML) из структур данных."""
from __future__ import annotations

from html import escape
from typing import Sequence

from .db import ErrorStat, FluencyStat, ProgressStat, Profile, VocabWord
from .parsing import TeacherReply

TELEGRAM_LIMIT = 4096

ERROR_TYPE_RU = {
    "grammar": "грамматика",
    "vocabulary": "лексика",
    "pronunciation": "произношение",
}


def _e(text: str) -> str:
    return escape(text or "", quote=False)


def render_reply(reply: TeacherReply) -> str:
    """Разговорная часть, затем блок исправлений, затем произношение."""
    parts: list[str] = []
    if reply.reply:
        parts.append(_e(reply.reply))

    if reply.corrections:
        lines = ["<b>Исправления</b>"]
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
        lines = ["<b>Произношение</b>"]
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


def render_words(words: Sequence[VocabWord], total: int) -> str:
    if not words:
        return (
            "Словарь пока пуст. Я добавляю сюда слова, которые ввожу в разговоре — "
            "поговори со мной, и он начнёт наполняться."
        )
    lines = [f"<b>Твой словарь</b> — {total} слов, последние {len(words)}:"]
    for word in words:
        line = f"• <b>{_e(word.word)}</b>"
        if word.meaning:
            line += f" — {_e(word.meaning)}"
        lines.append(line)
        if word.example:
            lines.append(f"   <i>{_e(word.example)}</i>")
    return _truncate("\n".join(lines))


def render_settings(profile: Profile) -> str:
    voice = "включены" if profile.voice_replies else "выключены"
    return (
        "<b>Настройки</b>\n"
        f"Уровень: {_e(profile.level or 'не задан')}\n"
        f"Интересы: {_e(profile.interests or 'не заданы')}\n"
        f"Голосовые ответы: {voice}"
    )


def render_voice_too_long(duration: int, limit: int) -> str:
    return (
        f"Голосовое {duration} с — это слишком долго, я разбираю до {limit} с. "
        "Запиши, пожалуйста, короче."
    )
