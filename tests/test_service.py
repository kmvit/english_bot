"""Интеграционные тесты хода диалога: реальная БД, моки LLM и Azure."""
from __future__ import annotations

from pathlib import Path

import pytest

from bot.assessment import Assessment, WordScore
from bot.config import Config
from bot.costs import LLM, STT, TTS
from bot.db import Database, month_start
from bot.llm import LLMError
from bot.parsing import Correction, LoggedError, PronunciationTip, TeacherReply
from bot.service import TeacherService

from .fakes import FakeSpeech, FakeTeacher

USER = 1


@pytest.fixture
def parts(config: Config, db: Database):
    teacher = FakeTeacher()
    speech = FakeSpeech()
    service = TeacherService(config, db, teacher, speech)
    return service, teacher, speech


async def test_text_turn_stores_history_and_usage(parts, db: Database):
    service, teacher, _ = parts

    result = await service.handle_turn(USER, "I go to Rome", "I go to Rome")

    assert "Nice! And you?" in result.text
    assert result.spoken_text == "Nice! And you?"
    assert result.voice_enabled is True

    history = await db.get_history(USER, 10)
    assert [item["content"] for item in history] == ["I go to Rome", "Nice! And you?"]

    rows = {row["kind"]: row for row in await db.usage_since(USER, month_start())}
    assert rows[LLM]["calls"] == 1
    assert rows[LLM]["tokens_in"] == 110
    assert rows[LLM]["tokens_out"] == 40


async def test_history_and_recurring_errors_are_passed_to_model(parts, db: Database):
    service, teacher, _ = parts
    await db.ensure_profile(USER)
    await db.record_errors(USER, [("grammar", "пропускает артикль")])
    await db.add_message(USER, "user", "earlier")

    await service.handle_turn(USER, "now", "now")

    call = teacher.calls[0]
    assert call["history"] == [{"role": "user", "content": "earlier"}]
    assert call["recurring"][0].description == "пропускает артикль"
    assert call["turn"] == "now"


async def test_history_respects_limit(config: Config, db: Database):
    service = TeacherService(config, db, FakeTeacher(), FakeSpeech())
    await db.ensure_profile(USER)
    for index in range(20):
        await db.add_message(USER, "user", f"m{index}")

    await service.handle_turn(USER, "now", "now")

    assert len(service._teacher.calls[0]["history"]) == config.history_limit


async def test_new_errors_are_recorded_with_example(config: Config, db: Database):
    reply = TeacherReply(
        reply="ok",
        corrections=[Correction("I go", "I went", "Past")],
        new_errors=[
            LoggedError("grammar", "времена"),
            LoggedError("vocabulary", "make/do"),
        ],
    )
    service = TeacherService(config, db, FakeTeacher(reply), FakeSpeech())

    await service.handle_turn(USER, "turn", "I go to Rome yesterday")

    errors = await db.top_errors(USER)
    assert {error.description for error in errors} == {"времена", "make/do"}
    assert errors[0].example == "I go to Rome yesterday"


async def test_voice_turn_stores_pronunciation_scores(config: Config, db: Database):
    assessment = Assessment(
        transcript="I think so",
        scores={"accuracy": 70.0, "fluency": 80.0, "prosody": 65.0, "overall": 72.0},
        worst_words=[WordScore("think", 41.0, "Mispronunciation")],
        duration_sec=8.0,
    )
    reply = TeacherReply(
        reply="Interesting!",
        pronunciation=[PronunciationTip("think", "th", "язык между зубами", "thanks")],
    )
    service = TeacherService(config, db, FakeTeacher(reply), FakeSpeech())

    result = await service.handle_turn(USER, "voice turn", assessment.transcript, assessment)

    assert "Произношение" in result.text
    progress = await db.pronunciation_progress(USER, days=7)
    assert progress.samples == 1
    assert progress.overall == 72.0
    assert progress.prosody == 65.0


async def test_assessment_without_scores_is_not_stored(config: Config, db: Database):
    service = TeacherService(config, db, FakeTeacher(), FakeSpeech())
    await service.handle_turn(USER, "t", "t", Assessment(transcript="hi"))
    assert (await db.pronunciation_progress(USER, days=7)).samples == 0


async def test_llm_failure_propagates_and_writes_nothing(config: Config, db: Database):
    teacher = FakeTeacher()
    teacher.fail = True
    service = TeacherService(config, db, teacher, FakeSpeech())

    with pytest.raises(LLMError):
        await service.handle_turn(USER, "turn", "turn")

    assert await db.get_history(USER, 10) == []
    assert await db.usage_since(USER, month_start()) == []


async def test_voice_replies_flag_is_reported(config: Config, db: Database):
    service = TeacherService(config, db, FakeTeacher(), FakeSpeech())
    await db.ensure_profile(USER)
    await db.update_profile(USER, voice_replies=False)

    result = await service.handle_turn(USER, "turn", "turn")
    assert result.voice_enabled is False


async def test_transcribe_logs_audio_seconds(config: Config, db: Database):
    service = TeacherService(config, db, FakeTeacher(), FakeSpeech())
    await db.ensure_profile(USER)

    assessment = await service.transcribe(Path("/tmp/none.wav"), USER, 12.5)

    assert assessment.duration_sec == 12.5
    rows = {row["kind"]: row for row in await db.usage_since(USER, month_start())}
    assert rows[STT]["seconds"] == 12.5
    assert rows[STT]["cost"] > 0


async def test_voice_answer_converts_and_logs_chars(
    config: Config, db: Database, tmp_path, monkeypatch
):
    speech = FakeSpeech()
    service = TeacherService(config, db, FakeTeacher(), speech)
    await db.ensure_profile(USER)

    async def fake_convert(src: Path, dst: Path, ffmpeg_bin: str = "ffmpeg") -> Path:
        dst.write_bytes(b"fake-ogg")
        return dst

    monkeypatch.setattr("bot.service.to_ogg_opus", fake_convert)

    path = await service.voice_answer(USER, "Hello there", tmp_path)

    assert path is not None and path.exists()
    assert speech.synthesized == ["Hello there"]
    rows = {row["kind"]: row for row in await db.usage_since(USER, month_start())}
    assert rows[TTS]["chars"] == len("Hello there")


async def test_voice_answer_skips_empty_text(config: Config, db: Database, tmp_path):
    service = TeacherService(config, db, FakeTeacher(), FakeSpeech())
    assert await service.voice_answer(USER, "   ", tmp_path) is None


async def test_voice_answer_survives_tts_failure(
    config: Config, db: Database, tmp_path, monkeypatch
):
    from bot.speech import SpeechError

    class BrokenSpeech(FakeSpeech):
        async def synthesize(self, text, mp3_path):
            raise SpeechError("Azure упал")

    service = TeacherService(config, db, BrokenSpeech(), FakeSpeech())
    await db.ensure_profile(USER)

    assert await service.voice_answer(USER, "Hello", tmp_path) is None
    assert await db.usage_since(USER, month_start()) == []


async def test_level_check_runs_after_n_messages(config: Config, db: Database):
    """config.level_check_every == 3: третий ход должен запустить проверку."""
    teacher = FakeTeacher(level="B2")
    service = TeacherService(config, db, teacher, FakeSpeech())

    for index in range(config.level_check_every):
        await service.handle_turn(USER, f"turn {index}", f"turn {index}")
    await service.wait_background()

    assert len(teacher.level_calls) == 1
    profile = await db.get_profile(USER)
    assert profile is not None
    assert profile.level == "B2"
    assert profile.level_checked_at_message == config.level_check_every

    # Следующий ход не должен запускать проверку снова.
    await service.handle_turn(USER, "one more", "one more")
    await service.wait_background()
    assert len(teacher.level_calls) == 1


async def test_level_check_failure_does_not_break_turn(config: Config, db: Database):
    class BrokenLevel(FakeTeacher):
        async def assess_level(self, messages):
            raise LLMError("нет связи")

    service = TeacherService(config, db, BrokenLevel(), FakeSpeech())
    for index in range(config.level_check_every):
        result = await service.handle_turn(USER, f"t{index}", f"t{index}")
        assert result.text

    await service.wait_background()
    profile = await db.get_profile(USER)
    assert profile is not None and profile.level is None


# --- режим Whisper: без оценки произношения и без оплаты распознавания ---


async def test_local_recognition_is_free(config: Config, db: Database):
    speech = FakeSpeech(assesses_pronunciation=False, billable=False)
    service = TeacherService(config, db, FakeTeacher(), speech)
    await db.ensure_profile(USER)

    await service.transcribe(Path("/tmp/none.wav"), USER, 12.5)

    rows = {row["kind"]: row for row in await db.usage_since(USER, month_start())}
    assert rows[STT]["seconds"] == 12.5
    assert rows[STT]["cost"] == 0.0
    assert service.assesses_pronunciation is False


async def test_voice_answer_skipped_without_tts(
    config: Config, db: Database, tmp_path
):
    speech = FakeSpeech(can_speak=False)
    service = TeacherService(config, db, FakeTeacher(), speech)
    await db.ensure_profile(USER)

    assert await service.voice_answer(USER, "Hello there", tmp_path) is None
    assert speech.synthesized == []
    assert await db.usage_since(USER, month_start()) == []


async def test_local_tts_is_free(config: Config, db: Database, tmp_path, monkeypatch):
    speech = FakeSpeech(billable=False)
    service = TeacherService(config, db, FakeTeacher(), speech)
    await db.ensure_profile(USER)

    async def fake_convert(src: Path, dst: Path, ffmpeg_bin: str = "ffmpeg") -> Path:
        dst.write_bytes(b"fake-ogg")
        return dst

    monkeypatch.setattr("bot.service.to_ogg_opus", fake_convert)

    await service.voice_answer(USER, "Hello", tmp_path)

    rows = {row["kind"]: row for row in await db.usage_since(USER, month_start())}
    assert rows[TTS]["chars"] == 5
    assert rows[TTS]["cost"] == 0.0
