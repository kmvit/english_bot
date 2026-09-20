"""Юнит-тесты подсчёта стоимости."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.config import Prices
from bot.costs import TokenUsage, format_usd, llm_cost, stt_cost, tts_cost

PRICES = Prices(
    input_per_mtok=2.0,
    output_per_mtok=10.0,
    cache_read_per_mtok=1.0,
    stt_per_hour=1.0,
    tts_per_1m_chars=16.0,
)


def usage_obj(prompt, completion, cached=None, cost=None):
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    payload = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=details,
    )
    if cost is not None:
        payload.cost = cost
    return payload


def test_cached_tokens_are_not_double_counted():
    usage = TokenUsage.from_response(usage_obj(1000, 200, cached=800))

    assert usage.input_tokens == 200
    assert usage.cached_tokens == 800
    assert usage.prompt_tokens == 1000
    assert usage.total == 1200


def test_usage_without_details():
    usage = TokenUsage.from_response(usage_obj(500, 100))
    assert usage.input_tokens == 500
    assert usage.cached_tokens == 0


def test_usage_from_dict():
    usage = TokenUsage.from_response(
        {"prompt_tokens": 300, "completion_tokens": 50,
         "prompt_tokens_details": {"cached_tokens": 100}, "cost": 0.002}
    )
    assert usage.input_tokens == 200
    assert usage.cached_tokens == 100
    assert usage.reported_cost == pytest.approx(0.002)


def test_missing_usage():
    usage = TokenUsage.from_response(None)
    assert usage.total == 0
    assert llm_cost(usage, PRICES) == 0.0


def test_computed_cost():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=100_000, cached_tokens=500_000)
    # 2.0 + 1.0 + 0.5
    assert llm_cost(usage, PRICES) == pytest.approx(3.5)


def test_reported_cost_wins_over_tariffs():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, reported_cost=0.42)
    assert llm_cost(usage, PRICES) == pytest.approx(0.42)


def test_azure_costs():
    assert stt_cost(3600, PRICES) == pytest.approx(1.0)
    assert stt_cost(-5, PRICES) == 0.0
    assert tts_cost(1_000_000, PRICES) == pytest.approx(16.0)


@pytest.mark.parametrize(
    "amount,expected",
    [(0.0, "$0.00"), (0.0001234, "$0.0001"), (0.004, "$0.0040"), (1.5, "$1.50")],
)
def test_format_usd(amount, expected):
    assert format_usd(amount) == expected
