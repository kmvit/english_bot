"""Azure AI Speech: распознавание с оценкой произношения и нейронный синтез.

SDK от Azure синхронный и блокирующий, поэтому каждый вызов уходит в поток
через `asyncio.to_thread` — иначе он застопорит весь event loop бота.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk

from ..assessment import Assessment, compact_assessment
from ..config import Config
from . import AZURE, SpeechError

log = logging.getLogger(__name__)


def _speech_config(config: Config) -> speechsdk.SpeechConfig:
    return speechsdk.SpeechConfig(
        subscription=config.azure_speech_key, region=config.azure_speech_region
    )


class AzureRecognizer:
    name = AZURE
    assesses_pronunciation = True

    def __init__(self, config: Config) -> None:
        self._config = config

    async def assess(self, wav_path: Path, duration_sec: float = 0.0) -> Assessment:
        timeout = self._config.max_voice_seconds * 2 + 30
        segments = await asyncio.to_thread(self._assess_blocking, wav_path, timeout)
        return compact_assessment(segments, duration_sec)

    def _assess_blocking(self, wav_path: Path, timeout: float) -> list[str]:
        speech_config = _speech_config(self._config)
        speech_config.speech_recognition_language = self._config.recognition_language
        audio_config = speechsdk.audio.AudioConfig(filename=str(wav_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        # Unscripted: эталонного текста нет, ученик говорит что хочет.
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text="",
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=False,
        )
        try:
            pronunciation_config.enable_prosody_assessment()
        except AttributeError:  # pragma: no cover - зависит от версии SDK
            log.warning("SDK Azure без prosody assessment — оценка интонации выключена")
        pronunciation_config.apply_to(recognizer)

        segments: list[str] = []
        errors: list[str] = []
        finished = threading.Event()

        def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
            if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
                return
            payload = evt.result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )
            if payload:
                segments.append(payload)

        def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
            if evt.reason == speechsdk.CancellationReason.Error:
                errors.append(f"{evt.error_code}: {evt.error_details}")
            finished.set()

        def on_stopped(evt: object) -> None:
            finished.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(on_stopped)

        recognizer.start_continuous_recognition()
        try:
            completed = finished.wait(timeout)
        finally:
            try:
                recognizer.stop_continuous_recognition()
            except Exception as exc:  # pragma: no cover - защитный код
                log.warning("Не удалось остановить распознавание: %s", exc)

        if errors:
            raise SpeechError("; ".join(errors))
        if not completed:
            raise SpeechError(f"Azure не ответил за {timeout:.0f} с")
        return segments


class AzureSynthesizer:
    name = AZURE
    suffix = ".mp3"

    def __init__(self, config: Config) -> None:
        self._config = config

    async def synthesize(self, text: str, out_path: Path) -> int:
        """Озвучить текст в mp3. Возвращает число символов для учёта стоимости."""
        audio = await asyncio.to_thread(self._synthesize_blocking, text)
        out_path.write_bytes(audio)
        return len(text)

    def _synthesize_blocking(self, text: str) -> bytes:
        speech_config = _speech_config(self._config)
        speech_config.speech_synthesis_voice_name = self._config.tts_voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3
        )
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return bytes(result.audio_data)
        details = ""
        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            details = f"{cancellation.reason}: {cancellation.error_details}"
        raise SpeechError(details or f"синтез не удался ({result.reason})")
