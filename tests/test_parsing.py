"""Юнит-тесты разбора JSON-ответа модели."""
from __future__ import annotations

import json

import pytest

from bot.parsing import parse_level, parse_teacher_reply

FULL = {
    "reply": "That sounds fun! Where did you go last summer?",
    "corrections": [
        {
            "original": "I go to London last year",
            "corrected": "I went to London last year",
            "note": "Past Simple: go -> went",
        }
    ],
    "pronunciation": [
        {
            "word": "think",
            "phoneme": "th",
            "tip": "Кончик языка между зубами, без голоса.",
            "example": "thanks",
        }
    ],
    "new_errors": [
        {"type": "grammar", "description": "использует Present вместо Past Simple"}
    ],
}


def test_clean_json():
    result = parse_teacher_reply(json.dumps(FULL, ensure_ascii=False))

    assert result.raw_json_ok
    assert result.reply.startswith("That sounds fun!")
    assert result.corrections[0].original == "I go to London last year"
    assert result.corrections[0].corrected == "I went to London last year"
    assert result.pronunciation[0].word == "think"
    assert result.pronunciation[0].example == "thanks"
    assert result.new_errors[0].type == "grammar"


def test_fenced_json():
    payload = "```json\n" + json.dumps(FULL) + "\n```"
    result = parse_teacher_reply(payload)
    assert result.raw_json_ok
    assert result.corrections[0].note.startswith("Past Simple")


def test_bare_fence_without_language():
    payload = "```\n" + json.dumps(FULL) + "\n```"
    assert parse_teacher_reply(payload).raw_json_ok


def test_json_with_surrounding_prose():
    payload = f"Sure, here you go:\n{json.dumps(FULL)}\nHope that helps!"
    result = parse_teacher_reply(payload)
    assert result.raw_json_ok
    assert result.reply.startswith("That sounds fun!")


def test_braces_inside_strings_do_not_break_extraction():
    payload = 'Answer: {"reply": "use {braces} carefully", "corrections": [], ' \
              '"pronunciation": [], "new_errors": []}'
    result = parse_teacher_reply(payload)
    assert result.raw_json_ok
    assert result.reply == "use {braces} carefully"


def test_empty_arrays_are_valid():
    result = parse_teacher_reply(
        json.dumps({"reply": "Nice!", "corrections": [], "pronunciation": [], "new_errors": []})
    )
    assert result.raw_json_ok
    assert result.corrections == []
    assert result.pronunciation == []
    assert result.new_errors == []


def test_missing_keys_are_tolerated():
    result = parse_teacher_reply('{"reply": "Hi there!"}')
    assert result.raw_json_ok
    assert result.reply == "Hi there!"
    assert result.corrections == []


def test_plain_prose_falls_back_to_reply():
    result = parse_teacher_reply("Nice to meet you! What do you do?")
    assert not result.raw_json_ok
    assert result.reply == "Nice to meet you! What do you do?"


def test_truncated_json_falls_back_without_raising():
    result = parse_teacher_reply('{"reply": "Hello", "corrections": [{"original"')
    assert not result.raw_json_ok
    assert result.reply.startswith("{")


def test_empty_input():
    result = parse_teacher_reply("")
    assert not result.raw_json_ok
    assert result.reply == ""


def test_key_aliases_are_accepted():
    payload = {
        "reply": "Good one.",
        "corrections": [{"wrong": "I am agree", "right": "I agree", "why": "без be"}],
        "pronunciation": [{"word": "world", "sound": "r", "advice": "не смягчай"}],
        "new_errors": [{"kind": "GRAMMAR", "text": "лишний be перед agree"}],
    }
    result = parse_teacher_reply(json.dumps(payload, ensure_ascii=False))

    assert result.corrections[0].original == "I am agree"
    assert result.corrections[0].corrected == "I agree"
    assert result.corrections[0].note == "без be"
    assert result.pronunciation[0].phoneme == "r"
    assert result.pronunciation[0].tip == "не смягчай"
    assert result.new_errors[0].type == "grammar"


def test_unknown_error_type_becomes_grammar():
    result = parse_teacher_reply(
        json.dumps({"reply": "ok", "new_errors": [{"type": "spelling", "description": "x"}]})
    )
    assert result.new_errors[0].type == "grammar"


def test_single_object_instead_of_array():
    result = parse_teacher_reply(
        json.dumps(
            {
                "reply": "ok",
                "corrections": {"original": "a", "corrected": "b", "note": "c"},
            }
        )
    )
    assert len(result.corrections) == 1
    assert result.corrections[0].corrected == "b"


def test_incomplete_items_are_dropped():
    result = parse_teacher_reply(
        json.dumps(
            {
                "reply": "ok",
                "corrections": [{"note": "нет ни одной фразы"}],
                "pronunciation": [{"tip": "нет слова"}],
                "new_errors": [{"type": "grammar"}],
            }
        )
    )
    assert result.corrections == []
    assert result.pronunciation == []
    assert result.new_errors == []


def test_non_string_reply_is_stringified():
    result = parse_teacher_reply('{"reply": 42}')
    assert result.reply == "42"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("B1", "B1"),
        ("  a2 ", "A2"),
        ('{"level": "C1"}', "C1"),
        ("The student is around B2 level.", "B2"),
        ("C2", None),
        ("unknown", None),
        ("", None),
    ],
)
def test_parse_level(text, expected):
    assert parse_level(text) == expected
