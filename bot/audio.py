"""Конвертация аудио через ffmpeg.

Telegram присылает голосовые в ogg/opus, распознавание принимает wav 16 kHz
mono PCM; синтез отдаёт wav (Piper) или mp3 (Azure), а Telegram ждёт ogg/opus,
чтобы показать сообщение как voice, а не как файл.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import wave
from pathlib import Path

log = logging.getLogger(__name__)


class AudioError(RuntimeError):
    """ffmpeg не смог сконвертировать файл."""


async def _run_ffmpeg(ffmpeg_bin: str, args: list[str]) -> None:
    try:
        process = await asyncio.create_subprocess_exec(
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise AudioError(f"ffmpeg не найден: {ffmpeg_bin}") from exc
    _, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", "replace").strip()
        raise AudioError(f"ffmpeg завершился с кодом {process.returncode}: {message}")


async def ogg_to_wav(src: Path, dst: Path, ffmpeg_bin: str = "ffmpeg") -> Path:
    """ogg/opus -> wav 16 kHz mono PCM16 (формат, который ждёт Azure Speech)."""
    await _run_ffmpeg(
        ffmpeg_bin,
        ["-i", str(src), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
    )
    return dst


async def to_ogg_opus(src: Path, dst: Path, ffmpeg_bin: str = "ffmpeg") -> Path:
    """Любой аудиофайл -> ogg/opus 48 kHz mono, как требует Telegram для voice."""
    await _run_ffmpeg(
        ffmpeg_bin,
        [
            "-i",
            str(src),
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(dst),
        ],
    )
    return dst


def wav_duration(path: Path) -> float:
    """Длительность wav в секундах — для журнала стоимости Azure."""
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as handle:
            rate = handle.getframerate()
            if not rate:
                return 0.0
            return handle.getnframes() / float(rate)
    except (OSError, wave.Error, EOFError) as exc:
        log.warning("Не удалось прочитать длительность %s: %s", path, exc)
        return 0.0
