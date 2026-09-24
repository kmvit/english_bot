"""Речевой слой: распознавание и синтез за общим интерфейсом.

Бэкенды подключаются лениво — если выбран Whisper, SDK Azure не импортируется
вовсе, и наоборот. Так один отсутствующий пакет не роняет запуск.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ..assessment import Assessment
from ..config import Config

log = logging.getLogger(__name__)

WHISPER = "whisper"
AZURE = "azure"
PIPER = "piper"
KOKORO = "kokoro"
NONE = "none"
AUTO = "auto"


class SpeechError(RuntimeError):
    """Распознавание или синтез не удались."""


class TTSUnavailable(SpeechError):
    """Синтез речи не настроен — озвучивать нечем."""


class Recognizer(Protocol):
    name: str
    #: Умеет ли бэкенд оценивать произношение, а не только расшифровывать речь.
    assesses_pronunciation: bool

    async def assess(self, wav_path: Path, duration_sec: float = 0.0) -> Assessment: ...


class Synthesizer(Protocol):
    name: str
    #: Расширение файла, который пишет бэкенд (Azure — mp3, Piper — wav).
    suffix: str

    async def synthesize(self, text: str, out_path: Path) -> int: ...


class Speech:
    """Фасад: скрывает от сервиса, какой именно бэкенд подключён."""

    def __init__(self, recognizer: Recognizer, synthesizer: Synthesizer | None) -> None:
        self._recognizer = recognizer
        self._synthesizer = synthesizer

    @property
    def recognizer_name(self) -> str:
        return self._recognizer.name

    @property
    def synthesizer_name(self) -> str:
        return self._synthesizer.name if self._synthesizer else NONE

    @property
    def assesses_pronunciation(self) -> bool:
        return self._recognizer.assesses_pronunciation

    @property
    def can_speak(self) -> bool:
        return self._synthesizer is not None

    @property
    def tts_suffix(self) -> str:
        return self._synthesizer.suffix if self._synthesizer else ".wav"

    #: Локальное распознавание не тарифицируется — незачем писать в расходы.
    @property
    def stt_is_billable(self) -> bool:
        return self._recognizer.name == AZURE

    @property
    def tts_is_billable(self) -> bool:
        return self.synthesizer_name == AZURE

    async def warmup(self) -> None:
        """Прогреть бэкенды, которые это умеют: модель распознавания и голос."""
        for backend in (self._recognizer, self._synthesizer):
            warmup = getattr(backend, "warmup", None)
            if warmup is not None:
                await warmup()

    async def assess(self, wav_path: Path, duration_sec: float = 0.0) -> Assessment:
        return await self._recognizer.assess(wav_path, duration_sec)

    async def synthesize(self, text: str, out_path: Path) -> int:
        if self._synthesizer is None:
            raise TTSUnavailable("синтез речи не настроен")
        return await self._synthesizer.synthesize(text, out_path)


def build_speech(config: Config) -> Speech:
    return Speech(_build_recognizer(config), _build_synthesizer(config))


def _build_recognizer(config: Config) -> Recognizer:
    if config.speech_backend == AZURE:
        from .azure import AzureRecognizer

        return AzureRecognizer(config)
    from .whisper import WhisperRecognizer

    return WhisperRecognizer(config)


def _build_synthesizer(config: Config) -> Synthesizer | None:
    choice = config.tts_backend
    if choice == NONE:
        return None
    if choice in (AZURE, AUTO) and config.has_azure_credentials:
        from .azure import AzureSynthesizer

        return AzureSynthesizer(config)
    if choice == AZURE:
        log.warning("TTS_BACKEND=azure, но ключи Azure не заданы — голосовые ответы выключены")
        return None
    if choice == KOKORO:
        from .kokoro import KokoroSynthesizer

        return KokoroSynthesizer(config)
    if choice in (PIPER, AUTO):
        from .piper import PiperSynthesizer

        return PiperSynthesizer(config)
    log.warning("Неизвестный TTS_BACKEND=%s — голосовые ответы выключены", choice)
    return None
