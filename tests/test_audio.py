"""Тесты конвертации через ffmpeg. Пропускаются, если ffmpeg не установлен."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from bot.audio import AudioError, ogg_to_wav, to_ogg_opus, wav_duration

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg не установлен"
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True
    )


def _probe(path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_name,sample_rate,channels", "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    path = tmp_path / "src.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(path))
    return path


async def test_telegram_ogg_becomes_azure_ready_wav(tmp_path: Path, sine_wav: Path):
    telegram_voice = tmp_path / "voice.ogg"
    _ffmpeg("-i", str(sine_wav), "-c:a", "libopus", str(telegram_voice))

    wav = await ogg_to_wav(telegram_voice, tmp_path / "out.wav")

    assert _probe(wav) == "pcm_s16le,16000,1"
    assert wav_duration(wav) == pytest.approx(2.0, abs=0.1)


async def test_tts_mp3_becomes_telegram_voice(tmp_path: Path, sine_wav: Path):
    mp3 = tmp_path / "answer.mp3"
    _ffmpeg("-i", str(sine_wav), str(mp3))

    ogg = await to_ogg_opus(mp3, tmp_path / "answer.ogg")

    assert _probe(ogg).startswith("opus")


async def test_tts_wav_becomes_telegram_voice(tmp_path: Path, sine_wav: Path):
    """Piper отдаёт wav — он тоже должен превращаться в voice."""
    ogg = await to_ogg_opus(sine_wav, tmp_path / "answer.ogg")

    assert _probe(ogg).startswith("opus")


async def test_broken_input_raises_audio_error(tmp_path: Path):
    broken = tmp_path / "broken.ogg"
    broken.write_bytes(b"not audio at all")

    with pytest.raises(AudioError):
        await ogg_to_wav(broken, tmp_path / "out.wav")


async def test_missing_ffmpeg_binary(tmp_path: Path, sine_wav: Path):
    with pytest.raises(AudioError, match="не найден"):
        await ogg_to_wav(sine_wav, tmp_path / "out.wav", ffmpeg_bin="ffmpeg-does-not-exist")


def test_duration_of_non_wav_is_zero(tmp_path: Path):
    path = tmp_path / "x.wav"
    path.write_bytes(b"garbage")
    assert wav_duration(path) == 0.0
