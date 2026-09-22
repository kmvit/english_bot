"""Интеграционные тесты обработчиков: реальный Dispatcher, поддельная сессия Telegram."""
from __future__ import annotations

from typing import Any, AsyncGenerator

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.base import BaseSession
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.methods import TelegramMethod

from bot.assessment import Assessment, WordScore
from bot.config import Config
from bot.db import Database, month_start
from bot.handlers import build_router
from bot.middlewares import DENIED
from bot.parsing import TeacherReply
from bot.service import TeacherService

from .fakes import FakeSpeech, FakeTeacher

USER = 1
STRANGER = 999


class FakeSession(BaseSession):
    """Перехватывает вызовы Telegram API вместо сети."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[TelegramMethod[Any]] = []

    async def close(self) -> None:  # pragma: no cover - вызывается при закрытии бота
        pass

    async def make_request(self, bot, method, timeout=None):
        self.requests.append(method)
        return True

    async def stream_content(  # pragma: no cover - загрузки в тестах не используются
        self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True
    ) -> AsyncGenerator[bytes, None]:
        yield b""

    def messages(self) -> list[Any]:
        """Только отправленные и отредактированные сообщения, без chat action."""
        return [
            request
            for request in self.requests
            if type(request).__name__ in ("SendMessage", "EditMessageText")
        ]

    def texts(self) -> list[str]:
        return [
            getattr(request, "text", "")
            for request in self.requests
            if type(request).__name__ in ("SendMessage", "EditMessageText")
        ]

    def method_names(self) -> list[str]:
        return [type(request).__name__ for request in self.requests]


@pytest.fixture(autouse=True)
def no_ffmpeg(monkeypatch):
    """Озвучка проверяется отдельно: здесь ffmpeg подменён заглушкой."""

    async def fake_to_ogg(src, dst, ffmpeg_bin="ffmpeg"):
        dst.write_bytes(b"fake-ogg")
        return dst

    monkeypatch.setattr("bot.service.to_ogg_opus", fake_to_ogg)


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def telegram_bot(session: FakeSession) -> Bot:
    return Bot(
        token="42:TESTTOKENTESTTOKENTESTTOKENTESTTOKEN",
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


@pytest.fixture
def dispatcher(config: Config, db: Database):
    from bot.middlewares import AllowedUserMiddleware

    teacher = FakeTeacher()
    speech = FakeSpeech()
    service = TeacherService(config, db, teacher, speech)

    dp = Dispatcher(storage=MemoryStorage())
    dp["config"] = config
    dp["db"] = db
    dp["service"] = service
    guard = AllowedUserMiddleware(config.allowed_user_id)
    dp.message.outer_middleware(guard)
    dp.callback_query.outer_middleware(guard)
    dp.include_router(build_router())
    dp["_teacher"] = teacher
    dp["_speech"] = speech
    return dp


def text_update(text: str, user_id: int = USER, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": 1_700_000_000,
            "chat": {"id": user_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "first_name": "Student"},
            "text": text,
        },
    }


def voice_update(duration: int, user_id: int = USER, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": 1_700_000_000,
            "chat": {"id": user_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "first_name": "Student"},
            "voice": {
                "file_id": "voice-file-id",
                "file_unique_id": "uniq",
                "duration": duration,
            },
        },
    }


def callback_update(
    data: str,
    user_id: int = USER,
    update_id: int = 1,
    message_text: str = "Выбери уровень",
) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": "cb-1",
            "from": {"id": user_id, "is_bot": False, "first_name": "Student"},
            "chat_instance": "instance",
            "data": data,
            "message": {
                "message_id": 1,
                "date": 1_700_000_000,
                "chat": {"id": user_id, "type": "private"},
                "from": {"id": 42, "is_bot": True, "first_name": "Bot"},
                "text": message_text,
            },
        },
    }


async def test_stranger_is_denied_and_handler_not_called(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("hello", user_id=STRANGER))

    assert session.texts() == [DENIED]
    assert dispatcher["_teacher"].calls == []
    assert await db.get_profile(STRANGER) is None


async def test_stranger_callback_is_denied(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await dispatcher.feed_raw_update(
        telegram_bot, callback_update("level:B1", user_id=STRANGER)
    )
    assert session.method_names() == ["AnswerCallbackQuery"]
    assert await db.get_profile(STRANGER) is None


async def test_start_creates_profile_and_offers_levels(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("/start"))

    assert await db.get_profile(USER) is not None
    greeting, question = session.messages()[0], session.messages()[1]
    # Приветствие приносит постоянную клавиатуру, второе сообщение — выбор уровня.
    assert [b.text for row in greeting.reply_markup.keyboard for b in row] == [
        "📌 Ошибки", "📈 Прогресс", "⚙️ Настройки"
    ]
    assert "уровень" in question.text
    buttons = [
        button.callback_data
        for row in question.reply_markup.inline_keyboard
        for button in row
    ]
    assert buttons == ["level:A1", "level:A2", "level:B1", "level:B2", "level:C1"]


async def test_onboarding_flow(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("/start"))
    await dispatcher.feed_raw_update(telegram_bot, callback_update("level:B1", update_id=2))
    await dispatcher.feed_raw_update(
        telegram_bot, text_update("travel, IT, football", update_id=3)
    )

    profile = await db.get_profile(USER)
    assert profile is not None
    assert profile.level == "B1"
    assert profile.interests == "travel, IT, football"
    # Интересы не должны были уйти в модель как реплика диалога.
    assert dispatcher["_teacher"].calls == []


async def test_command_during_onboarding_is_not_saved_as_interests(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("/start"))
    await dispatcher.feed_raw_update(telegram_bot, callback_update("level:A2", update_id=2))
    await dispatcher.feed_raw_update(telegram_bot, text_update("/help", update_id=3))

    profile = await db.get_profile(USER)
    assert profile is not None and profile.interests is None
    assert any("Что я умею" in text for text in session.texts())


async def test_interests_can_be_skipped(
    dispatcher: Dispatcher, telegram_bot: Bot, db: Database
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("/start"))
    await dispatcher.feed_raw_update(telegram_bot, callback_update("level:A2", update_id=2))
    await dispatcher.feed_raw_update(telegram_bot, text_update("-", update_id=3))

    profile = await db.get_profile(USER)
    assert profile is not None and profile.interests is None


async def test_bad_level_callback_is_rejected(
    dispatcher: Dispatcher, telegram_bot: Bot, db: Database
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("/start"))
    await dispatcher.feed_raw_update(telegram_bot, callback_update("level:Z9", update_id=2))

    profile = await db.get_profile(USER)
    assert profile is not None and profile.level is None


async def test_text_message_goes_through_the_model_and_answers(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("I go to Rome"))

    assert dispatcher["_teacher"].calls[0]["turn"] == "I go to Rome"
    assert "Nice! And you?" in session.texts()[0]
    # Голосовые ответы включены по умолчанию — озвучка тоже уходит.
    assert "SendVoice" in session.method_names()
    assert dispatcher["_speech"].synthesized == ["Nice! And you?"]


async def test_voice_replies_can_be_turned_off(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("/voice off"))
    await dispatcher.feed_raw_update(telegram_bot, text_update("hello", update_id=2))

    assert "SendVoice" not in session.method_names()
    assert dispatcher["_speech"].synthesized == []


async def test_voice_command_without_argument_reports_state(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("/voice"))
    assert "включены" in session.texts()[0]


async def test_llm_failure_is_reported_to_user(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession
):
    dispatcher["_teacher"].fail = True

    await dispatcher.feed_raw_update(telegram_bot, text_update("hello"))

    assert "не отвечает" in session.texts()[0]
    assert "SendVoice" not in session.method_names()


async def test_topic_command_requires_argument(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("/topic"))
    assert "/topic travel" in session.texts()[0]
    assert dispatcher["_teacher"].calls == []


async def test_topic_command_starts_conversation(
    dispatcher: Dispatcher, telegram_bot: Bot, db: Database
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("/topic space travel"))

    turn = dispatcher["_teacher"].calls[0]["turn"]
    assert "space travel" in turn
    history = await db.get_history(USER, 10)
    assert history[0]["content"] == "Let's talk about space travel."


async def test_long_voice_is_rejected_before_azure(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, config: Config
):
    await dispatcher.feed_raw_update(
        telegram_bot, voice_update(config.max_voice_seconds + 5)
    )

    assert "слишком долго" in session.texts()[0]
    assert dispatcher["_teacher"].calls == []


async def test_voice_is_transcribed_assessed_and_answered(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database,
    monkeypatch, tmp_path,
):
    dispatcher["_speech"].assessment = Assessment(
        transcript="I think so",
        scores={"accuracy": 70.0, "overall": 72.0},
        worst_words=[WordScore("think", 41.0, "Mispronunciation")],
    )

    async def fake_download(self, file, destination, **kwargs):
        destination.write_bytes(b"fake-ogg")

    async def fake_ogg_to_wav(src, dst, ffmpeg_bin="ffmpeg"):
        dst.write_bytes(b"fake-wav")
        return dst

    monkeypatch.setattr(Bot, "download", fake_download, raising=False)
    monkeypatch.setattr("bot.handlers.chat.ogg_to_wav", fake_ogg_to_wav)
    monkeypatch.setattr("bot.handlers.chat.wav_duration", lambda path: 9.0)

    await dispatcher.feed_raw_update(telegram_bot, voice_update(9))

    turn = dispatcher["_teacher"].calls[0]["turn"]
    assert "I think so" in turn
    assert '"worst_words"' in turn
    history = await db.get_history(USER, 10)
    assert history[0]["content"] == "I think so"
    assert (await db.pronunciation_progress(USER, days=7)).samples == 1


async def test_unrecognized_voice_gets_hint(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, monkeypatch
):
    dispatcher["_speech"].assessment = Assessment(transcript="")

    async def fake_download(self, file, destination, **kwargs):
        destination.write_bytes(b"fake-ogg")

    async def fake_ogg_to_wav(src, dst, ffmpeg_bin="ffmpeg"):
        dst.write_bytes(b"fake-wav")
        return dst

    monkeypatch.setattr(Bot, "download", fake_download, raising=False)
    monkeypatch.setattr("bot.handlers.chat.ogg_to_wav", fake_ogg_to_wav)
    monkeypatch.setattr("bot.handlers.chat.wav_duration", lambda path: 3.0)

    await dispatcher.feed_raw_update(telegram_bot, voice_update(3))

    assert "не разобрал ни слова" in session.texts()[0]
    assert dispatcher["_teacher"].calls == []


async def test_azure_failure_is_reported(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, monkeypatch
):
    dispatcher["_speech"].fail = True

    async def fake_download(self, file, destination, **kwargs):
        destination.write_bytes(b"fake-ogg")

    async def fake_ogg_to_wav(src, dst, ffmpeg_bin="ffmpeg"):
        dst.write_bytes(b"fake-wav")
        return dst

    monkeypatch.setattr(Bot, "download", fake_download, raising=False)
    monkeypatch.setattr("bot.handlers.chat.ogg_to_wav", fake_ogg_to_wav)
    monkeypatch.setattr("bot.handlers.chat.wav_duration", lambda path: 3.0)

    await dispatcher.feed_raw_update(telegram_bot, voice_update(3))

    assert "Не получилось разобрать" in session.texts()[0]


async def test_reset_clears_history_only(
    dispatcher: Dispatcher, telegram_bot: Bot, db: Database
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("hello"))
    await db.update_profile(USER, level="B1")

    await dispatcher.feed_raw_update(telegram_bot, text_update("/reset", update_id=2))

    assert await db.get_history(USER, 10) == []
    profile = await db.get_profile(USER)
    assert profile is not None and profile.level == "B1"


async def test_mistakes_progress_and_cost_commands(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    dispatcher["_teacher"].reply_value = TeacherReply(
        reply="Careful!",
        new_errors=[__import__("bot.parsing", fromlist=["LoggedError"]).LoggedError(
            "grammar", "пропускает артикль"
        )],
    )
    await dispatcher.feed_raw_update(telegram_bot, text_update("I have cat"))

    await dispatcher.feed_raw_update(telegram_bot, text_update("/mistakes", update_id=2))
    await dispatcher.feed_raw_update(telegram_bot, text_update("/progress", update_id=3))
    await dispatcher.feed_raw_update(telegram_bot, text_update("/cost", update_id=4))

    texts = session.texts()
    assert any("пропускает артикль" in text for text in texts)
    assert any("Голосовых пока не было" in text for text in texts)
    assert any("Итого" in text for text in texts)


async def test_unknown_command_gets_hint(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("/nonsense"))

    assert "Такой команды нет" in session.texts()[0]
    assert dispatcher["_teacher"].calls == []


async def test_unsupported_content_gets_hint(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession
):
    update = text_update("x")
    del update["message"]["text"]
    update["message"]["sticker"] = {
        "file_id": "s",
        "file_unique_id": "su",
        "width": 1,
        "height": 1,
        "is_animated": False,
        "is_video": False,
        "type": "regular",
    }

    await dispatcher.feed_raw_update(telegram_bot, update)

    assert "текст и голосовые" in session.texts()[0]


# --- настройки и кнопки ------------------------------------------------


async def test_settings_shows_profile(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await db.ensure_profile(USER)
    await db.update_profile(USER, level="B1", interests="travel")

    await dispatcher.feed_raw_update(telegram_bot, text_update("/settings"))

    sent = session.messages()[0]
    assert "Уровень: B1" in sent.text
    assert "travel" in sent.text
    assert "Голосовые ответы: включены" in sent.text
    labels = [b.text for row in sent.reply_markup.inline_keyboard for b in row]
    assert labels == ["Уровень: B1", "Интересы", "🔇 Выключить голос"]


async def test_settings_level_can_be_changed_later(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await db.ensure_profile(USER)
    await db.update_profile(USER, level="A2")

    await dispatcher.feed_raw_update(telegram_bot, callback_update("cfg:level"))
    picker = session.messages()[-1]
    assert "Выбери уровень" in picker.text
    assert any("✅ A2" in b.text for row in picker.reply_markup.inline_keyboard for b in row)

    await dispatcher.feed_raw_update(telegram_bot, callback_update("cfg:level:C1", update_id=2))

    profile = await db.get_profile(USER)
    assert profile is not None and profile.level == "C1"
    assert "Уровень: C1" in session.messages()[-1].text


async def test_settings_voice_toggle(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await db.ensure_profile(USER)

    await dispatcher.feed_raw_update(telegram_bot, callback_update("cfg:voice"))
    profile = await db.get_profile(USER)
    assert profile is not None and profile.voice_replies is False
    assert "Голосовые ответы: выключены" in session.messages()[-1].text

    await dispatcher.feed_raw_update(telegram_bot, callback_update("cfg:voice", update_id=2))
    profile = await db.get_profile(USER)
    assert profile is not None and profile.voice_replies is True


async def test_settings_interests_edit(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await db.ensure_profile(USER)

    await dispatcher.feed_raw_update(telegram_bot, callback_update("cfg:interests"))
    assert "интересно" in session.texts()[-1]

    await dispatcher.feed_raw_update(telegram_bot, text_update("cooking, jazz", update_id=2))

    profile = await db.get_profile(USER)
    assert profile is not None and profile.interests == "cooking, jazz"
    assert "cooking, jazz" in session.messages()[-1].text
    # Это редактирование профиля, а не реплика диалога.
    assert dispatcher["_teacher"].calls == []


async def test_bottom_buttons_work_like_commands(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    await db.ensure_profile(USER)
    await db.record_errors(USER, [("grammar", "пропускает артикль")])

    await dispatcher.feed_raw_update(telegram_bot, text_update("📌 Ошибки"))
    await dispatcher.feed_raw_update(telegram_bot, text_update("📈 Прогресс", update_id=2))
    await dispatcher.feed_raw_update(telegram_bot, text_update("⚙️ Настройки", update_id=3))

    texts = session.texts()
    assert "пропускает артикль" in texts[0]
    assert "Голосовых пока не было" in texts[1]
    assert "Настройки" in texts[2]
    # Нажатия кнопок не должны уходить в модель как реплики ученика.
    assert dispatcher["_teacher"].calls == []


async def test_teacher_reply_carries_action_buttons(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession
):
    await dispatcher.feed_raw_update(telegram_bot, text_update("hello"))

    answer = session.messages()[0]
    assert [b.callback_data for row in answer.reply_markup.inline_keyboard for b in row] == [
        "reply:translate",
        "reply:explain",
    ]


async def test_translate_button(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession, db: Database
):
    dispatcher["_teacher"].plain_value = "Отличная мысль! А ты куда?"

    await dispatcher.feed_raw_update(
        telegram_bot, callback_update("reply:translate", message_text="Nice! Where to?")
    )

    call = dispatcher["_teacher"].plain_calls[0]
    assert "translate" in call["system"].lower()
    assert call["turn"] == "Nice! Where to?"
    assert "Отличная мысль" in session.texts()[-1]
    # Нажатие кнопки — платный запрос, он должен попасть в журнал расходов.
    rows = {row["kind"]: row for row in await db.usage_since(USER, month_start())}
    assert rows["llm"]["calls"] == 1


async def test_explain_button_uses_student_level(
    dispatcher: Dispatcher, telegram_bot: Bot, db: Database
):
    await db.ensure_profile(USER)
    await db.update_profile(USER, level="A2")

    await dispatcher.feed_raw_update(
        telegram_bot, callback_update("reply:explain", message_text="❌ I go → ✅ I went")
    )

    call = dispatcher["_teacher"].plain_calls[0]
    assert "explain" in call["system"].lower()
    assert "Russian" in call["turn"]
    assert "I went" in call["turn"]


async def test_explain_button_switches_to_english_from_b1(
    dispatcher: Dispatcher, telegram_bot: Bot, db: Database
):
    await db.ensure_profile(USER)
    await db.update_profile(USER, level="B2")

    await dispatcher.feed_raw_update(
        telegram_bot, callback_update("reply:explain", message_text="some correction")
    )

    assert "English" in dispatcher["_teacher"].plain_calls[0]["turn"]


async def test_action_button_survives_model_failure(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession
):
    dispatcher["_teacher"].fail = True

    await dispatcher.feed_raw_update(
        telegram_bot, callback_update("reply:translate", message_text="Nice!")
    )

    assert "не отвечает" in session.texts()[-1]


async def test_action_button_without_text(
    dispatcher: Dispatcher, telegram_bot: Bot, session: FakeSession
):
    await dispatcher.feed_raw_update(
        telegram_bot, callback_update("reply:translate", message_text="")
    )

    assert dispatcher["_teacher"].plain_calls == []
    assert session.method_names() == ["AnswerCallbackQuery"]


async def test_progress_explains_disabled_pronunciation(
    config: Config, db: Database, telegram_bot: Bot, session: FakeSession
):
    """В режиме Whisper баллов нет — /progress должен объяснить почему."""
    from aiogram.fsm.storage.memory import MemoryStorage

    from bot.middlewares import AllowedUserMiddleware
    from bot.service import TeacherService

    speech = FakeSpeech(assesses_pronunciation=False)
    dp = Dispatcher(storage=MemoryStorage())
    dp["config"] = config
    dp["db"] = db
    dp["service"] = TeacherService(config, db, FakeTeacher(), speech)
    dp.message.outer_middleware(AllowedUserMiddleware(config.allowed_user_id))
    dp.include_router(build_router())

    await dp.feed_raw_update(telegram_bot, text_update("/progress"))

    assert "Баллы произношения по фонемам появятся" in session.texts()[0]
