"""Тесты речевого слоя: выбор бэкенда, Whisper на поддельной модели, синтез."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot.config import Config
from bot.speech import (
    AZURE,
    KOKORO,
    NONE,
    PIPER,
    WHISPER,
    Speech,
    SpeechError,
    TTSUnavailable,
    build_speech,
)
from bot.speech.kokoro import KokoroSynthesizer
from bot.speech.piper import PiperSynthesizer, voice_url
from bot.speech.whisper import NO_SPEECH_LIMIT, WhisperRecognizer


class FakeModel:
    """Заменяет faster_whisper.WhisperModel: отдаёт заданные сегменты."""

    def __init__(self, segments, info=None):
        self.segments = segments
        self.info = info
        self.calls: list[dict] = []

    def transcribe(self, path, **kwargs):
        self.calls.append({"path": path, **kwargs})
        return iter(self.segments), self.info


def segment(text: str, no_speech_prob: float = 0.0):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob, avg_logprob=-0.3)


def recognizer(config: Config, segments) -> tuple[WhisperRecognizer, FakeModel]:
    model = FakeModel(segments)
    return WhisperRecognizer(config, model_factory=lambda: model), model


# --- Whisper -----------------------------------------------------------


async def test_transcript_is_joined_and_trimmed(config: Config):
    whisper, _ = recognizer(config, [segment("  Hello there. "), segment("How are you?")])

    result = await whisper.assess(Path("voice.wav"), duration_sec=7.5)

    assert result.transcript == "Hello there. How are you?"
    assert result.duration_sec == 7.5
    assert not result.is_empty


async def test_no_pronunciation_scores(config: Config):
    """Whisper даёт только текст — блока произношения в ответе быть не должно."""
    whisper, _ = recognizer(config, [segment("I have cat")])

    result = await whisper.assess(Path("voice.wav"))

    assert result.scores == {}
    assert result.worst_words == []
    assert result.to_compact_dict() == {"transcript": "I have cat"}
    assert whisper.assesses_pronunciation is False


async def test_silence_segments_are_dropped(config: Config):
    """На тишине Whisper выдумывает слова — отсекаем их по no_speech_prob."""
    whisper, _ = recognizer(
        config,
        [segment("You", no_speech_prob=NO_SPEECH_LIMIT + 0.1), segment("Real speech")],
    )

    result = await whisper.assess(Path("voice.wav"))

    assert result.transcript == "Real speech"


async def test_only_silence_gives_empty_assessment(config: Config):
    whisper, _ = recognizer(config, [segment("Thank you.", no_speech_prob=0.95)])

    result = await whisper.assess(Path("voice.wav"))

    assert result.is_empty


async def test_transcribe_parameters(config: Config):
    whisper, model = recognizer(config, [segment("hi")])

    await whisper.assess(Path("voice.wav"))

    call = model.calls[0]
    assert call["language"] == "en"          # из en-US
    assert call["temperature"] == 0
    assert call["condition_on_previous_text"] is False
    assert call["vad_filter"] is True


async def test_model_is_loaded_once(config: Config):
    loads = []

    def factory():
        loads.append(1)
        return FakeModel([segment("hi")])

    whisper = WhisperRecognizer(config, model_factory=factory)
    await whisper.assess(Path("a.wav"))
    await whisper.assess(Path("b.wav"))
    await whisper.warmup()

    assert len(loads) == 1


async def test_model_failure_becomes_speech_error(config: Config):
    class Broken:
        def transcribe(self, *a, **k):
            raise RuntimeError("модель упала")

    whisper = WhisperRecognizer(config, model_factory=Broken)
    with pytest.raises(SpeechError):
        await whisper.assess(Path("voice.wav"))


async def test_warmup_does_not_raise(config: Config):
    def broken_factory():
        raise OSError("нет места на диске")

    whisper = WhisperRecognizer(config, model_factory=broken_factory)
    await whisper.warmup()  # не должно бросить — иначе упадёт запуск бота


# --- фасад и выбор бэкенда ---------------------------------------------


def test_default_backend_is_whisper(config: Config):
    speech = build_speech(replace(config, tts_backend=NONE))

    assert speech.recognizer_name == WHISPER
    assert speech.assesses_pronunciation is False
    assert speech.stt_is_billable is False
    assert speech.can_speak is False


def test_azure_tts_picked_when_keys_present(config: Config):
    speech = build_speech(
        replace(config, azure_speech_key="k", azure_speech_region="westeurope")
    )

    assert speech.synthesizer_name == AZURE
    assert speech.tts_is_billable is True
    assert speech.tts_suffix == ".mp3"


def test_auto_tts_falls_back_to_piper_without_azure_keys(config: Config):
    speech = build_speech(config)

    assert speech.synthesizer_name == PIPER
    assert speech.can_speak is True
    assert speech.tts_is_billable is False
    assert speech.tts_suffix == ".wav"


def test_tts_can_be_disabled(config: Config):
    speech = build_speech(replace(config, tts_backend=NONE))

    assert speech.synthesizer_name == NONE
    assert speech.can_speak is False


def test_azure_tts_requested_without_keys_degrades(config: Config):
    speech = build_speech(replace(config, tts_backend=AZURE))

    # Явно просили Azure — молча подменять его на Piper нельзя.
    assert speech.can_speak is False


async def test_synthesize_without_backend_raises(config: Config):
    whisper, _ = recognizer(config, [segment("hi")])
    speech = Speech(whisper, None)

    with pytest.raises(TTSUnavailable):
        await speech.synthesize("hello", Path("out.mp3"))


# --- Piper ------------------------------------------------------------


class FakeVoice:
    """Заменяет piper.PiperVoice: пишет в wav-файл тишину."""

    def __init__(self):
        self.calls: list[str] = []
        self.configs: list = []

    def synthesize_wav(self, text, wav_file, syn_config=None):
        self.calls.append(text)
        self.configs.append(syn_config)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 2205)


@pytest.fixture
def voice_files(config: Config, tmp_path) -> Config:
    """Голос уже на диске — скачивание не должно даже начинаться."""
    model_dir = tmp_path / "voices"
    model_dir.mkdir()
    (model_dir / f"{config.piper_voice}.onnx").write_bytes(b"weights")
    (model_dir / f"{config.piper_voice}.onnx.json").write_text("{}")
    return replace(config, piper_model_dir=model_dir)


@pytest.mark.parametrize(
    "voice,expected_tail",
    [
        ("en_US-lessac-medium", "en/en_US/lessac/medium/en_US-lessac-medium.onnx"),
        ("en_GB-alba-low", "en/en_GB/alba/low/en_GB-alba-low.onnx"),
        ("ru_RU-irina-medium", "ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx"),
    ],
)
def test_voice_url_layout(voice, expected_tail):
    assert voice_url(voice).endswith(expected_tail)


def test_voice_url_rejects_garbage():
    with pytest.raises(SpeechError):
        voice_url("нечто-непонятное")


async def test_synthesize_writes_wav(voice_files: Config, tmp_path):
    fake = FakeVoice()
    piper = PiperSynthesizer(voice_files, voice_factory=lambda path: fake)
    out = tmp_path / "answer.wav"

    chars = await piper.synthesize("Hello there", out)

    assert chars == len("Hello there")
    assert fake.calls == ["Hello there"]
    assert out.exists() and out.stat().st_size > 44  # заголовок wav + данные


async def test_voice_loaded_once(voice_files: Config, tmp_path):
    loads = []

    def factory(path):
        loads.append(path)
        return FakeVoice()

    piper = PiperSynthesizer(voice_files, voice_factory=factory)
    await piper.synthesize("one", tmp_path / "a.wav")
    await piper.synthesize("two", tmp_path / "b.wav")
    await piper.warmup()

    assert len(loads) == 1


async def test_existing_files_are_not_downloaded(voice_files: Config, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("скачивание не должно запускаться")

    monkeypatch.setattr("bot.speech.downloads.urllib.request.urlopen", forbidden)
    piper = PiperSynthesizer(voice_files, voice_factory=lambda path: FakeVoice())

    await piper.synthesize("hi", tmp_path / "a.wav")


async def test_download_failure_is_reported(config: Config, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("сеть недоступна")

    monkeypatch.setattr("bot.speech.downloads.urllib.request.urlopen", broken)
    piper = PiperSynthesizer(
        replace(config, piper_model_dir=tmp_path / "voices"),
        voice_factory=lambda path: FakeVoice(),
    )

    with pytest.raises(SpeechError, match="скачать"):
        await piper.synthesize("hi", tmp_path / "a.wav")


async def test_warmup_survives_download_failure(config: Config, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("сеть недоступна")

    monkeypatch.setattr("bot.speech.downloads.urllib.request.urlopen", broken)
    piper = PiperSynthesizer(
        replace(config, piper_model_dir=tmp_path / "voices"),
        voice_factory=lambda path: FakeVoice(),
    )

    await piper.warmup()  # не должно бросить — иначе упадёт запуск бота


async def test_partial_download_is_not_left_behind(config: Config, tmp_path, monkeypatch):
    model_dir = tmp_path / "voices"

    def broken(*args, **kwargs):
        raise OSError("обрыв связи")

    monkeypatch.setattr("bot.speech.downloads.urllib.request.urlopen", broken)
    piper = PiperSynthesizer(
        replace(config, piper_model_dir=model_dir), voice_factory=lambda path: FakeVoice()
    )

    with pytest.raises(SpeechError):
        await piper.synthesize("hi", tmp_path / "a.wav")

    # Недокачанный файл не должен сойти за готовый голос при следующем запуске.
    assert list(model_dir.glob("*.onnx")) == []


async def test_tempo_is_passed_to_piper(voice_files: Config, tmp_path):
    """Темп речи задаётся настройкой: ученику важнее разборчивость."""
    fake = FakeVoice()
    piper = PiperSynthesizer(
        replace(voice_files, tts_length_scale=1.6), voice_factory=lambda path: fake
    )

    await piper.synthesize("Hello", tmp_path / "a.wav")

    assert fake.configs[0].length_scale == 1.6


async def test_default_tempo_is_slower_than_native(config: Config):
    """По умолчанию голос замедлен: это бот для изучающих язык."""
    assert config.tts_length_scale > 1.0


# --- Kokoro ------------------------------------------------------------


class FakeEngine:
    """Заменяет kokoro_onnx.Kokoro: отдаёт тишину нужной длины."""

    def __init__(self):
        self.calls: list[dict] = []

    def create(self, text, voice, speed, lang):
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        return [0.0] * 2400, 24000


@pytest.fixture
def kokoro_files(config: Config, tmp_path) -> Config:
    model_dir = tmp_path / "kokoro"
    model_dir.mkdir()
    (model_dir / "kokoro-v1.0.onnx").write_bytes(b"weights")
    (model_dir / "voices-v1.0.bin").write_bytes(b"voices")
    return replace(config, kokoro_model_dir=model_dir, tts_backend="kokoro")


def test_kokoro_is_picked_when_asked(kokoro_files: Config):
    speech = build_speech(kokoro_files)

    assert speech.synthesizer_name == KOKORO
    assert speech.tts_suffix == ".wav"
    assert speech.tts_is_billable is False


async def test_kokoro_synthesizes(kokoro_files: Config, tmp_path):
    engine = FakeEngine()
    kokoro = KokoroSynthesizer(kokoro_files, engine_factory=lambda m, v: engine)

    chars = await kokoro.synthesize("Hello there", tmp_path / "a.wav")

    assert chars == len("Hello there")
    assert engine.calls[0]["text"] == "Hello there"
    assert engine.calls[0]["voice"] == "af_bella"
    assert (tmp_path / "a.wav").exists()


@pytest.mark.parametrize(
    "length_scale,expected_speed",
    [(1.0, 1.0), (1.15, pytest.approx(0.87, abs=0.01)), (1.3, pytest.approx(0.77, abs=0.01))],
)
def test_slowdown_is_converted_to_kokoro_speed(
    kokoro_files: Config, length_scale, expected_speed
):
    """Настройка везде значит одно: 1.15 — на 15% медленнее, на любом движке."""
    kokoro = KokoroSynthesizer(replace(kokoro_files, tts_length_scale=length_scale))

    assert kokoro.speed == expected_speed


def test_zero_scale_does_not_divide_by_zero(kokoro_files: Config):
    kokoro = KokoroSynthesizer(replace(kokoro_files, tts_length_scale=0.0))

    assert kokoro.speed == 1.0


async def test_kokoro_engine_loaded_once(kokoro_files: Config, tmp_path):
    loads = []

    def factory(model, voices):
        loads.append((model, voices))
        return FakeEngine()

    kokoro = KokoroSynthesizer(kokoro_files, engine_factory=factory)
    await kokoro.synthesize("one", tmp_path / "a.wav")
    await kokoro.synthesize("two", tmp_path / "b.wav")

    assert len(loads) == 1


async def test_kokoro_download_failure_is_reported(config: Config, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("сеть недоступна")

    monkeypatch.setattr("bot.speech.downloads.urllib.request.urlopen", broken)
    kokoro = KokoroSynthesizer(
        replace(config, kokoro_model_dir=tmp_path / "kokoro"),
        engine_factory=lambda m, v: FakeEngine(),
    )

    with pytest.raises(SpeechError, match="скачать"):
        await kokoro.synthesize("hi", tmp_path / "a.wav")


async def test_kokoro_synthesis_failure_becomes_speech_error(kokoro_files: Config, tmp_path):
    class Broken:
        def create(self, *a, **k):
            raise RuntimeError("модель упала")

    kokoro = KokoroSynthesizer(kokoro_files, engine_factory=lambda m, v: Broken())

    with pytest.raises(SpeechError):
        await kokoro.synthesize("hi", tmp_path / "a.wav")


# --- определение короткого ответа --------------------------------------


@pytest.mark.parametrize(
    "text,limit,expected",
    [
        ("Fine", 4, True),
        ("Yes, I do", 4, True),
        ("Отлично, спасибо", 4, True),
        ("I am doing great today because the weather is warm", 4, False),
        ("", 4, False),          # пустое — не повод для подсказки
        ("   ", 4, False),
        ("one two three four five", 4, False),
    ],
)
def test_is_short_answer(text, limit, expected):
    from bot.prompts import is_short_answer

    assert is_short_answer(text, limit) is expected
