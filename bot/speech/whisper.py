"""Локальное распознавание через faster-whisper.

Оценки произношения не даёт — только транскрипт. Модель загружается один раз
при первом голосовом и живёт в памяти процесса; сама загрузка и распознавание
блокирующие, поэтому уходят в поток.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from ..assessment import Assessment, compute_fluency
from ..config import Config
from . import WHISPER, SpeechError

log = logging.getLogger(__name__)

# Сегменты, которые модель сама считает тишиной, — источник галлюцинаций
# вроде «You» или «Thank you» на пустой записи.
NO_SPEECH_LIMIT = 0.7


class WhisperRecognizer:
    name = WHISPER
    assesses_pronunciation = False

    def __init__(
        self,
        config: Config,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._config = config
        self._model: Any | None = None
        self._lock = asyncio.Lock()
        self._model_factory = model_factory or self._load_model

    def _load_model(self) -> Any:
        from faster_whisper import WhisperModel

        log.info(
            "Загружаю модель Whisper %s (%s, %s)",
            self._config.whisper_model,
            self._config.whisper_device,
            self._config.whisper_compute_type,
        )
        return WhisperModel(
            self._config.whisper_model,
            device=self._config.whisper_device,
            compute_type=self._config.whisper_compute_type,
            cpu_threads=self._config.whisper_cpu_threads,
        )

    async def _model_ready(self) -> Any:
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._model_factory)
        return self._model

    async def warmup(self) -> None:
        """Прогреть модель заранее, чтобы первое голосовое не ждало загрузку."""
        try:
            await self._model_ready()
        except Exception as exc:  # pragma: no cover - зависит от окружения
            log.error("Не удалось загрузить модель Whisper: %s", exc)

    async def assess(self, wav_path: Path, duration_sec: float = 0.0) -> Assessment:
        model = await self._model_ready()
        transcript, words = await asyncio.to_thread(self._transcribe, model, wav_path)
        # Баллов произношения нет — блок в ответе учителя просто не появится.
        # Зато тайминги слов дают темп и паузы, и это считается локально.
        return Assessment(
            transcript=transcript,
            duration_sec=round(duration_sec, 2),
            fluency=compute_fluency(words, duration_sec),
        )

    def _transcribe(
        self, model: Any, wav_path: Path
    ) -> tuple[str, list[tuple[str, float, float]]]:
        try:
            segments, _ = model.transcribe(
                str(wav_path),
                language=self._language,
                beam_size=5,
                temperature=0,
                # Каждое голосовое самостоятельно: контекст прошлого только мешает.
                condition_on_previous_text=False,
                vad_filter=True,
                # Нужны ради темпа речи и пауз: оценки произношения Whisper не даёт.
                word_timestamps=True,
            )
            kept: list[str] = []
            words: list[tuple[str, float, float]] = []
            for segment in segments:
                if getattr(segment, "no_speech_prob", 0.0) > NO_SPEECH_LIMIT:
                    log.debug("Пропускаю сегмент как тишину: %r", segment.text)
                    continue
                text = (segment.text or "").strip()
                if text:
                    kept.append(text)
                for word in getattr(segment, "words", None) or []:
                    label = (getattr(word, "word", "") or "").strip()
                    if label:
                        words.append(
                            (label, getattr(word, "start", 0.0), getattr(word, "end", 0.0))
                        )
        except SpeechError:
            raise
        except Exception as exc:
            raise SpeechError(f"Whisper не справился: {exc}") from exc
        return " ".join(kept).strip(), words

    @property
    def _language(self) -> str:
        """`en-US` из конфига -> `en`, как ждёт Whisper."""
        return (self._config.recognition_language or "en").split("-")[0].lower()
