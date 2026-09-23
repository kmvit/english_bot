"""Интеграционные тесты LLM-слоя на моке OpenAI-клиента."""
from __future__ import annotations

import json
from types import SimpleNamespace

import openai
import pytest

try:  # openai 3.x перешёл на httpx2
    import httpx2 as httpx
except ModuleNotFoundError:  # pragma: no cover - openai 1.x/2.x
    import httpx

from bot.config import Config
from bot.db import Profile
from bot.llm import LLMError, Teacher

PROFILE = Profile(user_id=1, level="A2", interests="travel")
PROFANITY_PROFILE = Profile(user_id=1, level="B1", interests="travel")
REPLY_JSON = json.dumps(
    {
        "reply": "Nice! Where to?",
        "corrections": [{"original": "I go", "corrected": "I went", "note": "Past"}],
        "pronunciation": [],
        "new_errors": [{"type": "grammar", "description": "времена"}],
    }
)


def completion(content: str, prompt=120, completion_tokens=30, cached=20, cost=0.001):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=None))],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            cost=cost,
        ),
    )


def api_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, request=request)
    if status == 429:
        return openai.RateLimitError("rate limited", response=response, body=None)
    if status >= 500:
        return openai.InternalServerError("boom", response=response, body=None)
    return openai.BadRequestError("bad request", response=response, body=None)


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else completion(REPLY_JSON)
        if isinstance(item, Exception):
            raise item
        return item


def build(config: Config, script) -> tuple[Teacher, FakeCompletions]:
    fake = FakeCompletions(script)
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return Teacher(config, client=client), fake


async def test_reply_parses_json_and_usage(config: Config):
    teacher, fake = build(config, [completion(REPLY_JSON)])

    result = await teacher.reply(PROFILE, [], [], "I go to Rome")

    assert result.model == config.llm_model
    assert result.reply.reply == "Nice! Where to?"
    assert result.reply.corrections[0].corrected == "I went"
    assert result.usage.input_tokens == 100
    assert result.usage.cached_tokens == 20
    assert result.usage.reported_cost == pytest.approx(0.001)


async def test_request_shape(config: Config):
    teacher, fake = build(config, [completion(REPLY_JSON)])
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]

    await teacher.reply(PROFILE, [], history, "I go to Rome")

    kwargs = fake.calls[0]
    messages = kwargs["messages"]
    assert kwargs["model"] == config.llm_model
    # Стабильная инструкция первая — от этого зависит попадание в кеш префикса.
    assert messages[0]["role"] == "system"
    assert "English tutor" in messages[0]["content"]
    assert messages[1]["role"] == "system" and "level: A2" in messages[1]["content"]
    assert messages[2:4] == history
    assert messages[-1] == {"role": "user", "content": "I go to Rome"}
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["extra_body"]["usage"] == {"include": True}
    assert "reasoning" not in kwargs["extra_body"]


async def test_reasoning_effort_passed_when_set(config: Config):
    teacher, fake = build(
        __import__("dataclasses").replace(config, llm_reasoning_effort="low"),
        [completion(REPLY_JSON)],
    )
    await teacher.reply(PROFILE, [], [], "hi")
    assert fake.calls[0]["extra_body"]["reasoning"] == {"effort": "low"}


async def test_profile_block_carries_recurring_errors(config: Config):
    from bot.db import ErrorStat

    teacher, fake = build(config, [completion(REPLY_JSON)])
    errors = [ErrorStat("grammar", "пропускает артикль", None, 4, "2026-01-01")]

    await teacher.reply(PROFILE, errors, [], "hi")

    assert "пропускает артикль" in fake.calls[0]["messages"][1]["content"]
    assert "x4" in fake.calls[0]["messages"][1]["content"]


async def test_falls_back_to_cheap_model_on_rate_limit(config: Config):
    teacher, fake = build(config, [api_error(429), completion(REPLY_JSON)])

    result = await teacher.reply(PROFILE, [], [], "hi")

    assert result.model == config.llm_fallback_model
    assert [call["model"] for call in fake.calls] == [
        config.llm_model,
        config.llm_fallback_model,
    ]


async def test_server_error_also_falls_back(config: Config):
    teacher, _ = build(config, [api_error(500), completion(REPLY_JSON)])
    result = await teacher.reply(PROFILE, [], [], "hi")
    assert result.model == config.llm_fallback_model


async def test_both_models_down_raises_llm_error(config: Config):
    teacher, fake = build(config, [api_error(429), api_error(500)])

    with pytest.raises(LLMError):
        await teacher.reply(PROFILE, [], [], "hi")
    assert len(fake.calls) == 2


async def test_timeout_is_retryable(config: Config):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    teacher, _ = build(
        config, [openai.APITimeoutError(request=request), completion(REPLY_JSON)]
    )
    result = await teacher.reply(PROFILE, [], [], "hi")
    assert result.model == config.llm_fallback_model


async def test_json_schema_rejected_downgrades_to_json_object(config: Config):
    teacher, fake = build(config, [api_error(400), completion(REPLY_JSON)])

    result = await teacher.reply(PROFILE, [], [], "hi")

    assert result.reply.reply == "Nice! Where to?"
    assert fake.calls[0]["response_format"]["type"] == "json_schema"
    assert fake.calls[1]["response_format"] == {"type": "json_object"}
    # Понижение запоминается: следующий запрос уже без json_schema.
    await teacher.reply(PROFILE, [], [], "hi again")
    assert fake.calls[2]["response_format"] == {"type": "json_object"}


async def test_response_format_dropped_entirely_after_second_refusal(config: Config):
    teacher, fake = build(config, [api_error(400), api_error(400), completion(REPLY_JSON)])

    await teacher.reply(PROFILE, [], [], "hi")

    assert "response_format" not in fake.calls[2]


async def test_prose_answer_still_reaches_user(config: Config):
    teacher, _ = build(config, [completion("Just talking, no JSON here.")])
    result = await teacher.reply(PROFILE, [], [], "hi")
    assert result.reply.raw_json_ok is False
    assert result.reply.reply == "Just talking, no JSON here."


async def test_refusal_gives_empty_text(config: Config):
    refused = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="no"))],
        usage=None,
    )
    teacher, _ = build(config, [refused])
    result = await teacher.reply(PROFILE, [], [], "hi")
    assert result.reply.reply == ""
    assert result.usage.total == 0


async def test_assess_level(config: Config):
    teacher, fake = build(config, [completion(json.dumps({"level": "B2"}))])

    level, usage, model = await teacher.assess_level(["I went to Rome", "It was great"])

    assert level == "B2"
    assert model == config.llm_model
    assert usage.output_tokens == 30
    assert "Student messages" in fake.calls[0]["messages"][1]["content"]


async def test_assess_level_from_prose(config: Config):
    teacher, _ = build(config, [completion("I'd say B1.")])
    level, _, _ = await teacher.assess_level(["hello"])
    assert level == "B1"


async def test_assess_level_without_messages_makes_no_request(config: Config):
    teacher, fake = build(config, [])
    level, usage, _ = await teacher.assess_level(["  ", ""])
    assert level is None and usage.total == 0 and fake.calls == []


async def test_profanity_block_is_off_by_default(config: Config):
    teacher, fake = build(config, [completion(REPLY_JSON)])

    await teacher.reply(PROFANITY_PROFILE, [], [], "he fucking girl")

    system = fake.calls[0]["messages"][0]["content"]
    assert "Strong language" not in system


async def test_profanity_block_is_added_when_enabled(config: Config):
    from dataclasses import replace

    teacher, fake = build(replace(config, teach_profanity=True), [completion(REPLY_JSON)])

    await teacher.reply(PROFANITY_PROFILE, [], [], "he fucking girl")

    system = fake.calls[0]["messages"][0]["content"]
    assert "Strong language" in system
    # Мат — не ошибка: это главное, ради чего блок и добавляется.
    assert "Swearing is not a mistake" in system
    # Блок внутри первого сообщения, иначе он ломал бы кеш префикса.
    assert len(fake.calls[0]["messages"]) == 3
