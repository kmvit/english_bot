"""Конфигурация из переменных окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _req(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Не задана обязательная переменная окружения {name}")
    return value


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on", "да")


def _opt(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Prices:
    """Тарифы для журнала стоимости.

    Используются, только если OpenRouter не вернул фактическую стоимость
    запроса (он отдаёт её в `usage.cost`, и тогда берём её). Значения по
    умолчанию приблизительные — правятся через .env под выбранную модель.
    """

    input_per_mtok: float = 2.0
    output_per_mtok: float = 8.0
    cache_read_per_mtok: float = 0.5
    stt_per_hour: float = 1.0
    tts_per_1m_chars: float = 16.0


@dataclass(frozen=True)
class Config:
    telegram_token: str
    openrouter_api_key: str
    allowed_user_id: int

    # --- речь ---
    speech_backend: str = "whisper"      # whisper | azure
    tts_backend: str = "auto"            # auto | azure | piper | none
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_cpu_threads: int = 0         # 0 — на усмотрение ctranslate2
    piper_voice: str = "en_US-lessac-medium"
    piper_model_dir: Path = Path("data/voices")
    # Темп озвучки: 1.0 — как записан голос, меньше — быстрее, больше —
    # медленнее. Ученику ниже B1 разборчивость важнее естественности.
    tts_length_scale: float = 1.3
    azure_speech_key: str = ""
    azure_speech_region: str = ""

    llm_base_url: str = DEFAULT_BASE_URL
    llm_model: str = "openai/gpt-4.1"
    llm_fallback_model: str = "openai/gpt-4.1-mini"
    # Только для reasoning-моделей OpenAI (o-серия, gpt-5): low | medium | high.
    llm_reasoning_effort: str | None = None
    # Необязательные заголовки OpenRouter для атрибуции трафика.
    openrouter_referer: str | None = None
    openrouter_title: str | None = None

    tts_voice: str = "en-US-JennyNeural"
    recognition_language: str = "en-US"

    db_path: Path = Path("data/bot.db")
    history_limit: int = 20
    max_voice_seconds: int = 60
    level_check_every: int = 50
    ffmpeg_bin: str = "ffmpeg"
    log_level: str = "INFO"
    # Учить живой речи с матом: слова не вымарываются, а разбирается уместность.
    # По умолчанию выключено — включается осознанно самим учеником.
    teach_profanity: bool = False
    # Часовой пояс ученика: сервер может стоять где угодно, а «девять утра»
    # означает девять утра у него.
    timezone: str = "Europe/Moscow"
    # День недели (0 — понедельник) и время недельной сводки.
    weekly_weekday: int = 6
    weekly_time: str = "19:00"
    # На части сетей маршрут IPv6 до Telegram не работает и polling виснет молча.
    telegram_force_ipv4: bool = False
    # SOCKS5/HTTP-прокси для Telegram и OpenRouter: на сервере в России оба
    # недоступны напрямую, трафик идёт через ssh-туннель до зарубежного хоста.
    proxy_url: str | None = None

    prices: Prices = field(default_factory=Prices)

    @property
    def has_azure_credentials(self) -> bool:
        return bool(self.azure_speech_key and self.azure_speech_region)


def load_config() -> Config:
    speech_backend = os.getenv("SPEECH_BACKEND", "whisper").strip().lower() or "whisper"
    azure_key = os.getenv("AZURE_SPEECH_KEY", "").strip()
    azure_region = os.getenv("AZURE_SPEECH_REGION", "").strip()
    # Ключи Azure нужны, только если он действительно выбран бэкендом.
    if speech_backend == "azure" and not (azure_key and azure_region):
        raise RuntimeError(
            "SPEECH_BACKEND=azure требует AZURE_SPEECH_KEY и AZURE_SPEECH_REGION"
        )

    return Config(
        telegram_token=_req("TELEGRAM_TOKEN"),
        openrouter_api_key=_req("OPENROUTER_API_KEY"),
        allowed_user_id=int(_req("ALLOWED_USER_ID")),
        speech_backend=speech_backend,
        tts_backend=os.getenv("TTS_BACKEND", "auto").strip().lower() or "auto",
        whisper_model=os.getenv("WHISPER_MODEL", "small"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        whisper_cpu_threads=_int("WHISPER_CPU_THREADS", 0),
        piper_voice=os.getenv("PIPER_VOICE", "en_US-lessac-medium"),
        piper_model_dir=Path(os.getenv("PIPER_MODEL_DIR", "data/voices")),
        tts_length_scale=_float("TTS_LENGTH_SCALE", 1.3),
        azure_speech_key=azure_key,
        azure_speech_region=azure_region,
        llm_base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        llm_model=os.getenv("LLM_MODEL", "openai/gpt-4.1"),
        llm_fallback_model=os.getenv("LLM_FALLBACK_MODEL", "openai/gpt-4.1-mini"),
        llm_reasoning_effort=_opt("LLM_REASONING_EFFORT"),
        openrouter_referer=_opt("OPENROUTER_REFERER"),
        openrouter_title=_opt("OPENROUTER_TITLE"),
        tts_voice=os.getenv("TTS_VOICE", "en-US-JennyNeural"),
        recognition_language=os.getenv("SPEECH_RECOGNITION_LANGUAGE", "en-US"),
        db_path=Path(os.getenv("DB_PATH", "data/bot.db")),
        history_limit=_int("HISTORY_LIMIT", 20),
        max_voice_seconds=_int("MAX_VOICE_SECONDS", 60),
        level_check_every=_int("LEVEL_CHECK_EVERY", 50),
        ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        teach_profanity=_bool("TEACH_PROFANITY"),
        timezone=os.getenv("TIMEZONE", "Europe/Moscow"),
        weekly_weekday=_int("WEEKLY_WEEKDAY", 6),
        weekly_time=os.getenv("WEEKLY_TIME", "19:00"),
        telegram_force_ipv4=_bool("TELEGRAM_FORCE_IPV4"),
        proxy_url=_opt("PROXY_URL"),
        prices=Prices(
            input_per_mtok=_float("PRICE_INPUT_PER_MTOK", 2.0),
            output_per_mtok=_float("PRICE_OUTPUT_PER_MTOK", 8.0),
            cache_read_per_mtok=_float("PRICE_CACHE_READ_PER_MTOK", 0.5),
            stt_per_hour=_float("AZURE_STT_USD_PER_HOUR", 1.0),
            tts_per_1m_chars=_float("AZURE_TTS_USD_PER_1M_CHARS", 16.0),
        ),
    )
