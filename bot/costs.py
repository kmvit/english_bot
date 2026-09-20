"""Подсчёт стоимости: токены LLM и секунды/символы Azure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Prices

LLM = "llm"
# Метки не привязаны к поставщику: бэкенд переключается, история расходов — нет.
STT = "stt"
TTS = "tts"


def _int_attr(source: Any, name: str) -> int:
    value = getattr(source, name, None)
    if value is None and isinstance(source, dict):
        value = source.get(name)
    return int(value) if isinstance(value, (int, float)) else 0


@dataclass
class TokenUsage:
    """Токены одного запроса.

    `input_tokens` — только незакешированный ввод: OpenAI и OpenRouter
    включают кешированные токены в `prompt_tokens`, поэтому их вычитаем,
    чтобы не считать дважды.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reported_cost: float | None = None

    @classmethod
    def from_response(cls, usage: Any) -> "TokenUsage":
        if usage is None:
            return cls()
        prompt = _int_attr(usage, "prompt_tokens")
        completion = _int_attr(usage, "completion_tokens")
        details = getattr(usage, "prompt_tokens_details", None)
        if details is None and isinstance(usage, dict):
            details = usage.get("prompt_tokens_details")
        cached = _int_attr(details, "cached_tokens") if details is not None else 0
        cached = min(cached, prompt)

        # OpenRouter возвращает фактическую стоимость запроса в usage.cost.
        raw_cost = getattr(usage, "cost", None)
        if raw_cost is None and isinstance(usage, dict):
            raw_cost = usage.get("cost")
        cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else None

        return cls(
            input_tokens=prompt - cached,
            output_tokens=completion,
            cached_tokens=cached,
            reported_cost=cost,
        )

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cached_tokens

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.output_tokens


def llm_cost(usage: TokenUsage, prices: Prices) -> float:
    """Стоимость от провайдера, иначе — расчёт по тарифам из .env."""
    if usage.reported_cost is not None:
        return max(usage.reported_cost, 0.0)
    return (
        usage.input_tokens * prices.input_per_mtok
        + usage.output_tokens * prices.output_per_mtok
        + usage.cached_tokens * prices.cache_read_per_mtok
    ) / 1_000_000.0


def stt_cost(audio_seconds: float, prices: Prices) -> float:
    return max(audio_seconds, 0.0) / 3600.0 * prices.stt_per_hour


def tts_cost(chars: int, prices: Prices) -> float:
    return max(chars, 0) / 1_000_000.0 * prices.tts_per_1m_chars


def format_usd(amount: float) -> str:
    if amount and amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:.2f}"
