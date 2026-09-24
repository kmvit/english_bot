"""Синтез речи через Kokoro: живее Piper по интонации, тоже локально.

Модель одна на все голоса (их больше полусотни), голос выбирается по имени
при каждом вызове — отдельный файл под каждый, как у Piper, не нужен.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from . import KOKORO, SpeechError
from .downloads import download

log = logging.getLogger(__name__)

RELEASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"


class KokoroSynthesizer:
    name = KOKORO
    suffix = ".wav"

    def __init__(
        self,
        config: Config,
        engine_factory: Callable[[Path, Path], Any] | None = None,
    ) -> None:
        self._config = config
        self._engine: Any | None = None
        self._lock = asyncio.Lock()
        self._engine_factory = engine_factory or self._load_engine

    @property
    def _model_path(self) -> Path:
        return self._config.kokoro_model_dir / MODEL_FILE

    @property
    def _voices_path(self) -> Path:
        return self._config.kokoro_model_dir / VOICES_FILE

    @property
    def speed(self) -> float:
        """Наша настройка — растяжение (1.15 = на 15% медленнее), у Kokoro —
        множитель скорости. Пересчитываем, чтобы число значило одно и то же
        на любом движке."""
        scale = self._config.tts_length_scale
        return 1.0 / scale if scale > 0 else 1.0

    def _load_engine(self, model_path: Path, voices_path: Path) -> Any:
        from kokoro_onnx import Kokoro

        log.info("Загружаю Kokoro, голос %s", self._config.kokoro_voice)
        return Kokoro(str(model_path), str(voices_path))

    def _ensure_files(self) -> tuple[Path, Path]:
        download(f"{RELEASE_URL}/{MODEL_FILE}", self._model_path)
        download(f"{RELEASE_URL}/{VOICES_FILE}", self._voices_path)
        return self._model_path, self._voices_path

    async def _engine_ready(self) -> Any:
        if self._engine is None:
            async with self._lock:
                if self._engine is None:
                    paths = await asyncio.to_thread(self._ensure_files)
                    self._engine = await asyncio.to_thread(self._engine_factory, *paths)
        return self._engine

    async def warmup(self) -> None:
        """Скачать и загрузить модель заранее: она весит 340 МБ."""
        try:
            await self._engine_ready()
        except Exception as exc:  # pragma: no cover - зависит от окружения
            log.error("Не удалось подготовить Kokoro: %s", exc)

    async def synthesize(self, text: str, out_path: Path) -> int:
        engine = await self._engine_ready()
        await asyncio.to_thread(self._write_wav, engine, text, out_path)
        return len(text)

    def _write_wav(self, engine: Any, text: str, out_path: Path) -> None:
        import soundfile

        try:
            samples, rate = engine.create(
                text,
                voice=self._config.kokoro_voice,
                speed=self.speed,
                lang="en-us",
            )
            soundfile.write(str(out_path), samples, rate)
        except Exception as exc:
            raise SpeechError(f"Kokoro не смог озвучить текст: {exc}") from exc
