"""Разбор JSON-ответа Claude в структуру ответа учителя.

Модель обычно возвращает валидный JSON (мы просим структурированный вывод),
но парсер терпим: снимает ```-обёртки, вытаскивает объект из окружающего
текста и в самом плохом случае отдаёт весь текст как разговорную часть.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)

REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
            "description": "Разговорная реплика на английском, 1-3 предложения со встречным вопросом.",
        },
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "corrected": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["original", "corrected", "note"],
                "additionalProperties": False,
            },
        },
        "pronunciation": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "phoneme": {"type": "string"},
                    "tip": {"type": "string"},
                    "example": {"type": "string"},
                },
                "required": ["word", "phoneme", "tip", "example"],
                "additionalProperties": False,
            },
        },
        "new_errors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["grammar", "vocabulary", "pronunciation"],
                    },
                    "description": {"type": "string"},
                },
                "required": ["type", "description"],
                "additionalProperties": False,
            },
        },
        "new_words": {
            "type": "array",
            "description": "Слова, которые учитель ввёл в этой реплике впервые.",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "meaning": {"type": "string"},
                    "example": {"type": "string"},
                },
                "required": ["word", "meaning", "example"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reply", "corrections", "pronunciation", "new_errors", "new_words"],
    "additionalProperties": False,
}

_CORRECTION_ALIASES = {
    "original": ("original", "wrong", "said", "before", "student"),
    "corrected": ("corrected", "correct", "right", "after", "fixed"),
    "note": ("note", "why", "explanation", "comment", "reason"),
}
_PRONUNCIATION_ALIASES = {
    "word": ("word",),
    "phoneme": ("phoneme", "sound", "phonemes"),
    "tip": ("tip", "advice", "hint", "note"),
    "example": ("example", "sample", "similar"),
}
_ERROR_ALIASES = {
    "type": ("type", "kind", "category"),
    "description": ("description", "text", "detail", "note"),
}
_WORD_ALIASES = {
    "word": ("word", "term", "lexeme"),
    "meaning": ("meaning", "translation", "definition", "gloss"),
    "example": ("example", "sample", "sentence"),
}

VALID_ERROR_TYPES = ("grammar", "vocabulary", "pronunciation")


@dataclass
class Correction:
    original: str
    corrected: str
    note: str = ""


@dataclass
class PronunciationTip:
    word: str
    phoneme: str = ""
    tip: str = ""
    example: str = ""


@dataclass
class LoggedError:
    type: str
    description: str


@dataclass
class NewWord:
    word: str
    meaning: str = ""
    example: str = ""


@dataclass
class TeacherReply:
    reply: str
    corrections: list[Correction] = field(default_factory=list)
    pronunciation: list[PronunciationTip] = field(default_factory=list)
    new_errors: list[LoggedError] = field(default_factory=list)
    new_words: list[NewWord] = field(default_factory=list)
    raw_json_ok: bool = True


def _strip_fence(text: str) -> str:
    match = FENCE_RE.match(text)
    return match.group(1) if match else text.strip()


def _extract_object(text: str) -> str | None:
    """Найти первый сбалансированный JSON-объект, игнорируя скобки в строках."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _loads(text: str) -> dict[str, Any] | None:
    candidates = [text]
    extracted = _extract_object(text)
    if extracted and extracted != text:
        candidates.append(extracted)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _pick(item: dict[str, Any], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        value = item.get(alias)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list) and value:
            return ", ".join(str(v) for v in value)
    return ""


def _items(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, str) and item.strip():
            result.append({"description": item.strip(), "note": item.strip()})
    return result


def parse_teacher_reply(text: str) -> TeacherReply:
    """Разобрать ответ модели. Никогда не бросает исключение."""
    text = (text or "").strip()
    if not text:
        return TeacherReply(reply="", raw_json_ok=False)

    payload = _loads(_strip_fence(text))
    if payload is None:
        # Модель ответила прозой — показываем её как есть, чтобы не терять ответ.
        return TeacherReply(reply=text, raw_json_ok=False)

    reply = payload.get("reply")
    if not isinstance(reply, str):
        reply = "" if reply is None else str(reply)

    corrections = []
    for item in _items(payload, "corrections"):
        original = _pick(item, _CORRECTION_ALIASES["original"])
        corrected = _pick(item, _CORRECTION_ALIASES["corrected"])
        if not original and not corrected:
            continue
        corrections.append(
            Correction(
                original=original,
                corrected=corrected,
                note=_pick(item, _CORRECTION_ALIASES["note"]),
            )
        )

    pronunciation = []
    for item in _items(payload, "pronunciation"):
        word = _pick(item, _PRONUNCIATION_ALIASES["word"])
        if not word:
            continue
        pronunciation.append(
            PronunciationTip(
                word=word,
                phoneme=_pick(item, _PRONUNCIATION_ALIASES["phoneme"]),
                tip=_pick(item, _PRONUNCIATION_ALIASES["tip"]),
                example=_pick(item, _PRONUNCIATION_ALIASES["example"]),
            )
        )

    new_errors = []
    for item in _items(payload, "new_errors"):
        description = _pick(item, _ERROR_ALIASES["description"])
        if not description:
            continue
        error_type = _pick(item, _ERROR_ALIASES["type"]).lower()
        if error_type not in VALID_ERROR_TYPES:
            error_type = "grammar"
        new_errors.append(LoggedError(type=error_type, description=description))

    new_words = []
    for item in _items(payload, "new_words"):
        word = _pick(item, _WORD_ALIASES["word"])
        if not word:
            continue
        new_words.append(
            NewWord(
                word=word,
                meaning=_pick(item, _WORD_ALIASES["meaning"]),
                example=_pick(item, _WORD_ALIASES["example"]),
            )
        )

    return TeacherReply(
        reply=reply.strip(),
        corrections=corrections,
        pronunciation=pronunciation,
        new_errors=new_errors,
        new_words=new_words,
        raw_json_ok=True,
    )


def parse_level(text: str) -> str | None:
    """Вытащить уровень (A1-C1) из короткого ответа модели."""
    match = re.search(r"\b([ABC][12])\b", (text or "").upper())
    if not match:
        return None
    level = match.group(1)
    return level if level in ("A1", "A2", "B1", "B2", "C1") else None
