"""Обёртка над OpenAI SDK (через OpenRouter): реплика учителя и оценка уровня."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

import openai

from .config import Config
from .costs import TokenUsage
from .db import ErrorStat, Profile
from .parsing import REPLY_SCHEMA, TeacherReply, parse_level, parse_teacher_reply
from .prompts import (
    DRILL_CHECK_SYSTEM,
    DRILL_SYSTEM,
    LEVEL_CHECK_SYSTEM,
    build_drill_check_turn,
    build_drill_turn,
    build_system_messages,
)

log = logging.getLogger(__name__)

MAX_REPLY_TOKENS = 2048
MAX_LEVEL_TOKENS = 64
MAX_PLAIN_TOKENS = 1024

DRILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "exercises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "string"},
                    "answer": {"type": "string"},
                    "focus": {"type": "string"},
                },
                "required": ["sentence", "answer", "focus"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["exercises"],
    "additionalProperties": False,
}

DRILL_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "feedback": {"type": "string"},
    },
    "required": ["correct", "feedback"],
    "additionalProperties": False,
}

LEVEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"level": {"type": "string", "enum": ["A1", "A2", "B1", "B2", "C1"]}},
    "required": ["level"],
    "additionalProperties": False,
}

# Ошибки, при которых имеет смысл повторить запрос на резервной модели.
RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

# Режимы структурированного вывода — от строгого к отсутствующему.
FORMAT_SCHEMA = "json_schema"
FORMAT_OBJECT = "json_object"
FORMAT_NONE = "none"


class LLMError(RuntimeError):
    """Запрос не удался ни на основной, ни на резервной модели."""


@dataclass
class LLMResult:
    reply: TeacherReply
    usage: TokenUsage
    model: str


class Teacher:
    def __init__(self, config: Config, client: Any | None = None) -> None:
        self._config = config
        headers: dict[str, str] = {}
        if config.openrouter_referer:
            headers["HTTP-Referer"] = config.openrouter_referer
        if config.openrouter_title:
            headers["X-Title"] = config.openrouter_title
        if client is None:
            http_client = (
                openai.DefaultAsyncHttpxClient(proxy=config.proxy_url)
                if config.proxy_url
                else None
            )
            client = openai.AsyncOpenAI(
                api_key=config.openrouter_api_key,
                base_url=config.llm_base_url,
                timeout=90.0,
                default_headers=headers or None,
                http_client=http_client,
            )
        self._client = client
        # Не все модели на OpenRouter принимают json_schema; понижаем режим
        # один раз при первом отказе, чтобы не платить за повторы каждый ход.
        self._format_mode = FORMAT_SCHEMA

    async def close(self) -> None:
        await self._client.close()

    async def reply(
        self,
        profile: Profile,
        recurring_errors: Sequence[ErrorStat],
        history: Sequence[dict[str, str]],
        user_turn: str,
        known_words: Sequence[str] = (),
    ) -> LLMResult:
        messages = [
            *build_system_messages(
                profile,
                recurring_errors,
                known_words,
                self._config.teach_profanity,
            ),
            *history,
            {"role": "user", "content": user_turn},
        ]
        response, model = await self._create(
            messages=messages,
            max_tokens=MAX_REPLY_TOKENS,
            schema=REPLY_SCHEMA,
            schema_name="teacher_reply",
        )
        text = _message_text(response)
        parsed = parse_teacher_reply(text)
        if not parsed.raw_json_ok:
            log.warning("Ответ модели не разобран как JSON, показываю как текст")
        return LLMResult(
            reply=parsed,
            usage=TokenUsage.from_response(getattr(response, "usage", None)),
            model=model,
        )

    async def assess_level(
        self, student_messages: Sequence[str]
    ) -> tuple[str | None, TokenUsage, str]:
        """Определить уровень по последним репликам ученика."""
        transcript = "\n".join(f"- {m}" for m in student_messages if m.strip())
        if not transcript:
            return None, TokenUsage(), self._config.llm_model
        response, model = await self._create(
            messages=[
                {"role": "system", "content": LEVEL_CHECK_SYSTEM},
                {
                    "role": "user",
                    "content": f"Student messages:\n{transcript}\n\nCEFR level?",
                },
            ],
            max_tokens=MAX_LEVEL_TOKENS,
            schema=LEVEL_SCHEMA,
            schema_name="cefr_level",
        )
        text = _message_text(response)
        level = None
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                level = parse_level(str(payload.get("level", "")))
        except (ValueError, TypeError):
            level = None
        if level is None:
            level = parse_level(text)
        return level, TokenUsage.from_response(getattr(response, "usage", None)), model

    async def plain(
        self, system: str, user_turn: str, max_tokens: int = MAX_PLAIN_TOKENS
    ) -> tuple[str, TokenUsage, str]:
        """Свободный текстовый запрос без JSON-схемы: перевод, пояснение."""
        response, model = await self._create(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_turn},
            ],
            max_tokens=max_tokens,
            schema=None,
            schema_name="plain",
        )
        return (
            _message_text(response).strip(),
            TokenUsage.from_response(getattr(response, "usage", None)),
            model,
        )

    async def make_drill(
        self, profile: Profile, mistakes: Sequence[str]
    ) -> tuple[list[dict[str, str]], TokenUsage, str]:
        """Составить по одному заданию на каждую ошибку."""
        response, model = await self._create(
            messages=[
                {"role": "system", "content": DRILL_SYSTEM},
                {
                    "role": "user",
                    "content": build_drill_turn(profile.level, profile.interests, mistakes),
                },
            ],
            max_tokens=MAX_REPLY_TOKENS,
            schema=DRILL_SCHEMA,
            schema_name="drill",
        )
        usage = TokenUsage.from_response(getattr(response, "usage", None))
        payload = _loads_object(_message_text(response))
        raw = payload.get("exercises") if isinstance(payload, dict) else None
        exercises = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            sentence = str(item.get("sentence", "")).strip()
            if not sentence:
                continue
            exercises.append(
                {
                    "sentence": sentence,
                    "answer": str(item.get("answer", "")).strip(),
                    "focus": str(item.get("focus", "")).strip(),
                }
            )
        return exercises, usage, model

    async def check_drill(
        self, sentence: str, answer: str, student: str, focus: str
    ) -> tuple[bool, str, TokenUsage, str]:
        """Проверить ответ ученика. При сбое разбора считаем ответ неверным."""
        response, model = await self._create(
            messages=[
                {"role": "system", "content": DRILL_CHECK_SYSTEM},
                {
                    "role": "user",
                    "content": build_drill_check_turn(sentence, answer, student, focus),
                },
            ],
            max_tokens=MAX_PLAIN_TOKENS,
            schema=DRILL_CHECK_SCHEMA,
            schema_name="drill_check",
        )
        usage = TokenUsage.from_response(getattr(response, "usage", None))
        payload = _loads_object(_message_text(response))
        correct = bool(payload.get("correct")) if isinstance(payload, dict) else False
        feedback = str(payload.get("feedback", "")).strip() if isinstance(payload, dict) else ""
        return correct, feedback, usage, model

    # --- низкий уровень -------------------------------------------------

    async def _create(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        schema: dict[str, Any] | None,
        schema_name: str,
    ) -> tuple[Any, str]:
        models = [self._config.llm_model]
        fallback = self._config.llm_fallback_model
        if fallback and fallback != self._config.llm_model:
            models.append(fallback)

        last_error: Exception | None = None
        for model in models:
            try:
                response = await self._request(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    schema=schema,
                    schema_name=schema_name,
                )
            except RETRYABLE as exc:
                last_error = exc
                log.warning("Модель %s недоступна: %s", model, exc)
                continue
            except openai.APIStatusError as exc:
                last_error = exc
                log.error("Модель %s вернула ошибку %s: %s", model, exc.status_code, exc)
                continue
            return response, model

        raise LLMError(str(last_error) if last_error else "неизвестная ошибка")

    async def _request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        schema: dict[str, Any] | None,
        schema_name: str,
    ) -> Any:
        # Просим OpenRouter вернуть фактическую стоимость запроса в usage.cost.
        extra_body: dict[str, Any] = {"usage": {"include": True}}
        if self._config.llm_reasoning_effort:
            extra_body["reasoning"] = {"effort": self._config.llm_reasoning_effort}

        while True:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "extra_body": extra_body,
            }
            response_format = self._response_format(schema, schema_name)
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                return await self._client.chat.completions.create(**kwargs)
            except openai.BadRequestError as exc:
                # Понижать нечего, если схему и не просили.
                if response_format is None or not self._downgrade_format(exc):
                    raise

    def _response_format(
        self, schema: dict[str, Any] | None, schema_name: str
    ) -> dict[str, Any] | None:
        if schema is None:
            return None
        if self._format_mode == FORMAT_SCHEMA:
            return {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        if self._format_mode == FORMAT_OBJECT:
            return {"type": "json_object"}
        return None

    def _downgrade_format(self, exc: Exception) -> bool:
        """Понизить режим структурированного вывода. False — если уже некуда."""
        if self._format_mode == FORMAT_SCHEMA:
            log.warning("json_schema не принят моделью, перехожу на json_object: %s", exc)
            self._format_mode = FORMAT_OBJECT
            return True
        if self._format_mode == FORMAT_OBJECT:
            log.warning("json_object не принят моделью, отключаю response_format: %s", exc)
            self._format_mode = FORMAT_NONE
            return True
        return False


def _loads_object(text: str) -> dict[str, Any]:
    """Разобрать JSON-ответ. Пустой словарь означает, что доверять нечему."""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _message_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    refusal = getattr(message, "refusal", None)
    if refusal:
        log.warning("Модель отказалась отвечать: %s", refusal)
        return ""
    return getattr(message, "content", "") or ""
