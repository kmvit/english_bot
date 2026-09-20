"""Тесты сборки сообщений Telegram."""
from __future__ import annotations

from bot.db import ErrorStat, ProgressStat
from bot.formatting import (
    TELEGRAM_LIMIT,
    render_mistakes,
    render_progress,
    render_reply,
    render_voice_too_long,
)
from bot.parsing import Correction, LoggedError, PronunciationTip, TeacherReply


def test_reply_has_conversation_first_then_corrections():
    reply = TeacherReply(
        reply="Sounds great! Where next?",
        corrections=[Correction("I go there", "I went there", "Past Simple")],
        pronunciation=[PronunciationTip("think", "th", "Язык между зубами", "thanks")],
        new_errors=[LoggedError("grammar", "времена")],
    )
    text = render_reply(reply)

    assert text.index("Sounds great!") < text.index("Исправления")
    assert text.index("Исправления") < text.index("Произношение")
    assert "❌ I go there" in text
    assert "✅ I went there" in text
    assert "/th/" in text
    assert "thanks" in text
    # new_errors — только для журнала, в сообщении их нет.
    assert "времена" not in text


def test_reply_without_mistakes_is_just_conversation():
    text = render_reply(TeacherReply(reply="Nice one! What about you?"))
    assert text == "Nice one! What about you?"


def test_html_is_escaped():
    reply = TeacherReply(
        reply="use <b>bold</b> & co",
        corrections=[Correction("a < b", "a > b", "note & note")],
    )
    text = render_reply(reply)
    assert "&lt;b&gt;bold&lt;/b&gt;" in text
    assert "&amp; co" in text
    assert "a &lt; b" in text


def test_empty_reply_gets_placeholder():
    assert render_reply(TeacherReply(reply="")).startswith("Не получилось")


def test_long_reply_is_truncated():
    text = render_reply(TeacherReply(reply="x" * (TELEGRAM_LIMIT + 500)))
    assert len(text) <= TELEGRAM_LIMIT
    assert text.endswith("…")


def test_mistakes_rendering():
    errors = [ErrorStat("grammar", "пропускает артикль", "I have cat", 4, "2026-01-01")]
    text = render_mistakes(errors)
    assert "грамматика" in text
    assert "×4" in text
    assert "I have cat" in text


def test_mistakes_empty():
    assert "Пока ничего" in render_mistakes([])


def test_progress_rendering():
    week = ProgressStat(3, 80.4, 85.0, 70.0, 79.6)
    month = ProgressStat(12, 70.0, 75.0, 65.0, 70.2)
    text = render_progress(week, month)

    assert "За 7 дней" in text and "За 30 дней" in text
    assert "3 голосовых" in text
    assert "▲" in text


def test_progress_without_data():
    empty = ProgressStat(0, None, None, None, None)
    assert "Голосовых пока не было" in render_progress(empty, empty)


def test_progress_week_empty_month_present():
    week = ProgressStat(0, None, None, None, None)
    month = ProgressStat(5, 70.0, None, None, 70.0)
    text = render_progress(week, month)
    assert "нет голосовых" in text
    assert "—" in text


def test_voice_too_long_message():
    text = render_voice_too_long(95, 60)
    assert "95" in text and "60" in text
