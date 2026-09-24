"""Синтез речи через Piper: нейронные голоса локально, без ключей и без сети.

Голос — два файла (.onnx с весами и .onnx.json с конфигом). Если их нет на
диске, они скачиваются один раз при первом обращении и остаются в каталоге
данных рядом с базой, чтобы переживать перезапуск контейнера.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from . import PIPER, SpeechError
from .downloads import download

log = logging.getLogger(__name__)

VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def voice_url(voice: str) -> str:
    """`en_US-lessac-medium` -> путь в репозитории голосов Piper."""
    try:
        locale, name_quality = voice.split("-", 1)
        name, quality = name_quality.rsplit("-", 1)
        language = locale.split("_")[0]
    except ValueError as exc:
        raise SpeechError(
            f"Не разобрать имя голоса {voice!r}, ожидается вид en_US-lessac-medium"
        ) from exc
    return f"{VOICES_URL}/{language}/{locale}/{name}/{quality}/{voice}.onnx"


class PiperSynthesizer:
    name = PIPER
    #: Piper пишет wav, Azure — mp3; фасаду нужно знать расширение заранее.
    suffix = ".wav"

    def __init__(
        self,
        config: Config,
        voice_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self._config = config
        self._voice: Any | None = None
        self._lock = asyncio.Lock()
        self._voice_factory = voice_factory or self._load_voice

    @property
    def model_path(self) -> Path:
        return self._config.piper_model_dir / f"{self._config.piper_voice}.onnx"

    def _load_voice(self, model_path: Path) -> Any:
        from piper import PiperVoice

        log.info("Загружаю голос Piper %s", self._config.piper_voice)
        return PiperVoice.load(model_path)

    def _ensure_files(self) -> Path:
        """Скачать веса и конфиг голоса, если их ещё нет."""
        base = voice_url(self._config.piper_voice)
        download(base, self.model_path)
        download(f"{base}.json", self.model_path.with_suffix(".onnx.json"))
        return self.model_path

    async def _voice_ready(self) -> Any:
        if self._voice is None:
            async with self._lock:
                if self._voice is None:
                    model_path = await asyncio.to_thread(self._ensure_files)
                    self._voice = await asyncio.to_thread(self._voice_factory, model_path)
        return self._voice

    async def warmup(self) -> None:
        """Скачать и загрузить голос заранее, чтобы первый ответ не ждал."""
        try:
            await self._voice_ready()
        except Exception as exc:  # pragma: no cover - зависит от окружения
            log.error("Не удалось подготовить голос Piper: %s", exc)

    async def synthesize(self, text: str, out_path: Path) -> int:
        voice = await self._voice_ready()
        await asyncio.to_thread(self._write_wav, voice, text, out_path)
        return len(text)

    def _write_wav(self, voice: Any, text: str, out_path: Path) -> None:
        import wave

        from piper import SynthesisConfig

        try:
            with wave.open(str(out_path), "wb") as wav_file:
                voice.synthesize_wav(
                    text,
                    wav_file,
                    SynthesisConfig(length_scale=self._config.tts_length_scale),
                )
        except Exception as exc:
            raise SpeechError(f"Piper не смог озвучить текст: {exc}") from exc
