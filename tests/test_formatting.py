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

    assert text.index("Sounds great!") < text.index("Что поправить в твоей фразе")
    assert text.index("Что поправить в твоей фразе") < text.index("Твоё произношение")
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


def test_progress_shows_fluency_without_pronunciation_scores():
    from bot.db import FluencyStat

    week = ProgressStat(2, None, None, None, None)
    month = ProgressStat(5, None, None, None, None)
    text = render_progress(
        week,
        month,
        week_fluency=FluencyStat(2, 118.0, 2.5, 0.12),
        month_fluency=FluencyStat(5, 104.0, 4.0, 0.2),
        pronunciation_enabled=False,
    )

    assert "118 слов/мин" in text
    assert "2.5 за запись" in text and "12% времени" in text
    assert "▲" in text          # темп за неделю выше месячного
    assert "Баллы произношения" in text
    assert "общий" not in text  # баллов нет — строк с ними быть не должно


def test_words_rendering():
    from bot.db import VocabWord
    from bot.formatting import render_words

    words = [VocabWord("commute", "поездка на работу", "My commute is long.", 2, "2026-01-01")]
    text = render_words(words, total=7)

    assert "7 слов" in text
    assert "commute" in text and "поездка на работу" in text
    assert "My commute is long." in text


def test_words_empty():
    from bot.formatting import render_words

    assert "Словарь пока пуст" in render_words([], 0)


def test_correction_heading_says_whose_phrase_it_is():
    """Бот пересказывает фразу ученика — без подписи разбор читается как самокритика."""
    reply = TeacherReply(
        reply="You like tomato and something else — did you mean coconut?",
        corrections=[Correction("I like tomato and coco roach", "I like tomato and coconut", "")],
    )
    text = render_reply(reply)

    assert "твоей фразе" in text


def test_expand_hint_goes_right_after_the_reply():
    """Подсказка — приглашение продолжить разговор, а не замечание в конце."""
    reply = TeacherReply(
        reply="Good to hear!",
        expand="Скажи, почему именно так — что тебе в этом нравится?",
        corrections=[Correction("I go", "I went", "Past")],
    )
    text = render_reply(reply)

    assert text.index("Good to hear!") < text.index("Скажи, почему")
    assert text.index("Скажи, почему") < text.index("Что поправить")


def test_no_hint_when_answer_was_enough():
    text = render_reply(TeacherReply(reply="Good to hear!", expand=""))
    assert text == "Good to hear!"


def test_translation_goes_under_the_reply():
    reply = TeacherReply(
        reply="Good to hear!",
        reply_ru="Рад слышать!",
        corrections=[Correction("I go", "I went", "Past")],
    )
    text = render_reply(reply)

    assert text.index("Good to hear!") < text.index("Рад слышать!")
    assert text.index("Рад слышать!") < text.index("Что поправить")


def test_no_translation_block_when_empty():
    text = render_reply(TeacherReply(reply="Good to hear!", reply_ru=""))
    assert text == "Good to hear!"


def test_translation_is_skipped_in_voice_only_mode():
    """В голосовом режиме перевод не нужен: смысл в том, чтобы слушать."""
    reply = TeacherReply(reply="Good to hear!", reply_ru="Рад слышать!")
    text = render_reply(reply, include_conversation=False)

    assert "Рад слышать!" not in text


def test_lookup_card_shows_translation_sound_and_example():
    from bot.parsing import Lookup
    from bot.formatting import render_lookup

    text = render_lookup(
        Lookup(
            term="grab",
            translation="схватить",
            meaning="взять быстро",
            ipa="ɡræb",
            example="Let me grab my jacket.",
            example_ru="Дай схвачу куртку.",
            note="Дополнение без предлога.",
        )
    )

    assert "<b>grab</b> — схватить" in text
    assert "/ɡræb/" in text
    assert "Let me grab my jacket." in text
    assert "Дай схвачу куртку." in text
    assert "Дополнение без предлога." in text
    # Подпись — последняя строка: по ней кнопка «не сохранять» её и заменяет.
    assert text.splitlines()[-1] == "➕ <i>Сохранил в словарь — спрошу на тренировке</i>"


def test_lookup_card_survives_empty_fields():
    from bot.parsing import Lookup
    from bot.formatting import render_lookup

    text = render_lookup(Lookup(term="grab"))
    assert "grab" in text
    assert "//" not in text


def test_lookup_card_escapes_html():
    from bot.parsing import Lookup
    from bot.formatting import render_lookup

    text = render_lookup(Lookup(term="<b>", meaning="a < b"))
    assert "&lt;b&gt;" in text
    assert "a &lt; b" in text


def test_word_task_asks_to_fill_the_gap():
    from bot.formatting import render_drill_task

    text = render_drill_task(
        2, 3, {"kind": "word", "sentence": "I want to ___ it.", "focus": "grab — схватить"}
    )
    assert "Вставь пропущенное слово" in text
    assert "grab — схватить" in text


def test_error_task_keeps_its_wording():
    from bot.formatting import render_drill_task

    text = render_drill_task(1, 3, {"sentence": "I have cat.", "focus": "артикль"})
    assert "Найди ошибку" in text


def test_empty_dictionary_explains_how_to_fill_it():
    from bot.formatting import render_words

    assert "Ответить" in render_words([], 0)


def test_dictionary_mentions_words_due_for_drill():
    from bot.db import VocabWord
    from bot.formatting import render_words

    words = [VocabWord("grab", "схватить", None, 0, "2026-01-01")]
    assert "К повторению готово 2" in render_words(words, total=5, due=2)
    assert "К повторению" not in render_words(words, total=5, due=0)
