"""Тесты слоя SQLite."""
from __future__ import annotations

from datetime import timedelta

from bot.costs import LLM, STT
from bot.db import Database, month_start, utc_now

USER = 1


async def test_profile_created_once_with_defaults(db: Database):
    first = await db.ensure_profile(USER)
    await db.update_profile(USER, level="B1", interests="travel")
    second = await db.ensure_profile(USER)

    assert first.native_language == "ru"
    assert first.voice_replies is True
    assert first.is_onboarded is False
    assert second.level == "B1"
    assert second.interests == "travel"
    assert second.is_onboarded is True


async def test_unknown_field_is_ignored(db: Database):
    await db.ensure_profile(USER)
    await db.update_profile(USER, nonsense="x", level="A2")
    profile = await db.get_profile(USER)
    assert profile is not None and profile.level == "A2"


async def test_voice_replies_toggle(db: Database):
    await db.ensure_profile(USER)
    await db.update_profile(USER, voice_replies=False)
    profile = await db.get_profile(USER)
    assert profile is not None and profile.voice_replies is False


async def test_history_is_limited_and_chronological(db: Database):
    await db.ensure_profile(USER)
    for index in range(5):
        await db.add_message(USER, "user", f"u{index}")
        await db.add_message(USER, "assistant", f"a{index}")

    history = await db.get_history(USER, limit=4)
    assert [item["content"] for item in history] == ["u3", "a3", "u4", "a4"]
    assert history[0]["role"] == "user"


async def test_clear_history_keeps_profile_and_errors(db: Database):
    await db.ensure_profile(USER)
    await db.update_profile(USER, level="B2")
    await db.add_message(USER, "user", "hello")
    await db.record_errors(USER, [("grammar", "артикли")])

    await db.clear_history(USER)

    assert await db.get_history(USER, 10) == []
    profile = await db.get_profile(USER)
    assert profile is not None and profile.level == "B2"
    assert len(await db.top_errors(USER)) == 1


async def test_recent_user_messages_only_user_role(db: Database):
    await db.ensure_profile(USER)
    await db.add_message(USER, "user", "one")
    await db.add_message(USER, "assistant", "reply")
    await db.add_message(USER, "user", "two")

    assert await db.recent_user_messages(USER, 10) == ["one", "two"]


async def test_repeated_error_increments_counter(db: Database):
    await db.ensure_profile(USER)
    await db.record_errors(USER, [("grammar", "Пропускает   артикль")], example="I have cat")
    await db.record_errors(USER, [("grammar", "пропускает артикль")])
    await db.record_errors(USER, [("vocabulary", "путает make и do")])

    errors = await db.top_errors(USER)
    assert len(errors) == 2
    assert errors[0].count == 2
    assert errors[0].example == "I have cat"
    assert errors[1].type == "vocabulary"


async def test_top_errors_is_limited_and_ordered(db: Database):
    await db.ensure_profile(USER)
    for index in range(7):
        for _ in range(index + 1):
            await db.record_errors(USER, [("grammar", f"ошибка {index}")])

    errors = await db.top_errors(USER, limit=5)
    assert [error.count for error in errors] == [7, 6, 5, 4, 3]


async def test_empty_error_description_skipped(db: Database):
    await db.ensure_profile(USER)
    await db.record_errors(USER, [("grammar", "   ")])
    assert await db.top_errors(USER) == []


async def test_pronunciation_progress(db: Database):
    await db.ensure_profile(USER)
    await db.add_pronunciation(USER, {"accuracy": 60.0, "overall": 62.0}, 10.0)
    await db.add_pronunciation(USER, {"accuracy": 80.0, "overall": 78.0}, 12.0)

    week = await db.pronunciation_progress(USER, days=7)
    assert week.samples == 2
    assert week.accuracy == 70.0
    assert week.overall == 70.0
    assert week.fluency is None


async def test_progress_ignores_old_records(db: Database):
    await db.ensure_profile(USER)
    old = (utc_now() - timedelta(days=40)).isoformat()
    await db.conn.execute(
        "INSERT INTO pronunciation (user_id, accuracy, overall, duration_sec, created_at) "
        "VALUES (?,?,?,?,?)",
        (USER, 20.0, 20.0, 5.0, old),
    )
    await db.conn.commit()
    await db.add_pronunciation(USER, {"accuracy": 90.0, "overall": 90.0}, 5.0)

    week = await db.pronunciation_progress(USER, days=7)
    month = await db.pronunciation_progress(USER, days=30)
    assert week.samples == 1 and week.accuracy == 90.0
    assert month.samples == 1


async def test_progress_without_data(db: Database):
    await db.ensure_profile(USER)
    stat = await db.pronunciation_progress(USER, days=7)
    assert stat.samples == 0 and stat.overall is None


async def test_usage_grouped_by_kind(db: Database):
    await db.ensure_profile(USER)
    await db.add_usage(USER, LLM, 0.01, model="m", input_tokens=100, output_tokens=20, cache_read=50)
    await db.add_usage(USER, LLM, 0.02, model="m", input_tokens=200, output_tokens=40)
    await db.add_usage(USER, STT, 0.005, audio_seconds=18.0)

    rows = {row["kind"]: row for row in await db.usage_since(USER, month_start())}
    assert rows[LLM]["calls"] == 2
    assert rows[LLM]["cost"] == 0.03
    assert rows[LLM]["tokens_in"] == 350
    assert rows[LLM]["tokens_out"] == 60
    assert rows[STT]["seconds"] == 18.0


async def test_messages_counter(db: Database):
    await db.ensure_profile(USER)
    assert await db.bump_messages_total(USER) == 1
    assert await db.bump_messages_total(USER) == 2


async def test_users_are_isolated(db: Database):
    await db.ensure_profile(1)
    await db.ensure_profile(2)
    await db.add_message(1, "user", "mine")
    await db.record_errors(1, [("grammar", "x")])

    assert await db.get_history(2, 10) == []
    assert await db.top_errors(2) == []


# --- словарь ---


async def test_words_are_saved_without_duplicates(db: Database):
    await db.ensure_profile(USER)
    await db.add_words(USER, [("resilient", "стойкий", "She is resilient.")])
    await db.add_words(USER, [("Resilient", "стойкий", "Another example.")])
    await db.add_words(USER, [("commute", "поездка на работу", "My commute is long.")])

    words = await db.recent_words(USER)
    assert await db.words_total(USER) == 2
    assert {w.word for w in words} == {"resilient", "commute"}


async def test_empty_words_are_skipped(db: Database):
    await db.ensure_profile(USER)
    await db.add_words(USER, [("   ", "", "")])
    assert await db.words_total(USER) == 0


async def test_seen_counter_grows(db: Database):
    await db.ensure_profile(USER)
    await db.add_words(USER, [("commute", "поездка", "example")])

    await db.mark_words_seen(USER, ["commute"])
    await db.mark_words_seen(USER, ["Commute"])

    assert (await db.recent_words(USER))[0].seen_count == 2


# --- тренировка ошибок ---


async def test_untrained_errors_are_due(db: Database):
    await db.ensure_profile(USER)
    await db.record_errors(USER, [("grammar", "артикли"), ("vocabulary", "make/do")])

    due = await db.errors_due_for_drill(USER)
    assert {e.description for e in due} == {"артикли", "make/do"}
    assert all(e.drill_streak == 0 for e in due)


async def test_correct_answer_postpones_the_error(db: Database):
    await db.ensure_profile(USER)
    await db.record_errors(USER, [("grammar", "артикли")])
    error = (await db.top_errors(USER))[0]

    await db.record_drill_result(error.id, correct=True)

    assert await db.errors_due_for_drill(USER) == []
    updated = (await db.top_errors(USER))[0]
    assert updated.drill_streak == 1
    assert updated.drill_due is not None


async def test_wrong_answer_resets_streak(db: Database):
    await db.ensure_profile(USER)
    await db.record_errors(USER, [("grammar", "артикли")])
    error = (await db.top_errors(USER))[0]

    await db.record_drill_result(error.id, correct=True)
    await db.record_drill_result(error.id, correct=True)
    await db.record_drill_result(error.id, correct=False)

    assert (await db.top_errors(USER))[0].drill_streak == 0


async def test_drill_result_for_missing_error_is_ignored(db: Database):
    await db.ensure_profile(USER)
    await db.record_drill_result(9999, correct=True)  # не должно бросить


# --- беглость ---


async def test_fluency_progress(db: Database):
    await db.ensure_profile(USER)
    await db.add_pronunciation(USER, {}, 10.0, {"words": 20, "wpm": 100.0, "pauses": 2, "pause_ratio": 0.1})
    await db.add_pronunciation(USER, {}, 10.0, {"words": 30, "wpm": 140.0, "pauses": 4, "pause_ratio": 0.2})

    stat = await db.fluency_progress(USER, days=7)
    assert stat.samples == 2
    assert stat.wpm == 120.0
    assert stat.pauses == 3.0


async def test_fluency_ignores_records_without_metrics(db: Database):
    await db.ensure_profile(USER)
    await db.add_pronunciation(USER, {"overall": 70.0}, 5.0)

    assert (await db.fluency_progress(USER, days=7)).samples == 0


# --- словарь: сохранение и повторение -----------------------------------


async def test_saved_word_is_due_right_away(db: Database):
    """Слово, о котором спросил ученик, спрашиваем на ближайшей тренировке."""
    word_id = await db.save_word(USER, "grab", "схватить", "Let me grab it.", "asked")

    due = await db.words_due_for_drill(USER)
    assert [(w.id, w.word, w.source) for w in due] == [(word_id, "grab", "asked")]
    assert await db.words_due_total(USER) == 1


async def test_saving_same_word_twice_keeps_one_entry(db: Database):
    first = await db.save_word(USER, "grab", "схватить", None, "asked")
    second = await db.save_word(USER, "Grab", None, "New example.", "asked")

    assert first == second
    words = await db.recent_words(USER)
    assert len(words) == 1
    # Пустые поля не затирают то, что уже было разобрано.
    assert (words[0].meaning, words[0].example) == ("схватить", "New example.")


async def test_deleted_word_leaves_the_dictionary(db: Database):
    word_id = await db.save_word(USER, "grab", "схватить", None, "asked")

    assert await db.delete_word(USER, word_id) == "grab"
    assert await db.recent_words(USER) == []
    assert await db.delete_word(USER, word_id) is None


async def test_word_is_not_deleted_for_another_user(db: Database):
    word_id = await db.save_word(USER, "grab", "схватить", None, "asked")

    assert await db.delete_word(USER + 1, word_id) is None
    assert len(await db.recent_words(USER)) == 1


async def test_recalled_word_comes_back_later(db: Database):
    word_id = await db.save_word(USER, "grab", "схватить", None, "asked")

    await db.record_word_drill_result(word_id, correct=True)

    word = (await db.recent_words(USER))[0]
    assert word.drill_streak == 1
    assert word.seen_count == 1
    assert await db.words_due_for_drill(USER) == []


async def test_forgotten_word_resets_the_streak(db: Database):
    word_id = await db.save_word(USER, "grab", "схватить", None, "asked")
    await db.record_word_drill_result(word_id, correct=True)

    await db.record_word_drill_result(word_id, correct=False)

    assert (await db.recent_words(USER))[0].drill_streak == 0


async def test_never_drilled_words_go_first(db: Database):
    old = await db.save_word(USER, "commute", "поездка", None, "tutor")
    await db.record_word_drill_result(old, correct=False)
    await db.save_word(USER, "grab", "схватить", None, "asked")

    due = await db.words_due_for_drill(USER, limit=1)
    assert [w.word for w in due] == ["grab"]
