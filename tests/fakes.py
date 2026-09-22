"""Моки внешних API для интеграционных тестов."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from bot.assessment import Assessment
from bot.costs import TokenUsage
from bot.llm import LLMError, LLMResult
from bot.parsing import TeacherReply


class FakeTeacher:
    """Вместо OpenAI-клиента: отдаёт заранее заданный ответ и пишет вызовы."""

    def __init__(self, reply: TeacherReply | None = None, level: str | None = "B1"):
        self.reply_value = reply or TeacherReply(reply="Nice! And you?")
        self.level = level
        self.calls: list[dict] = []
        self.level_calls: list[list[str]] = []
        self.plain_calls: list[dict] = []
        self.plain_value = "готовый текст от модели"
        self.drill_calls: list[list[str]] = []
        self.check_calls: list[dict] = []
        self.drill_correct = True
        self.fail = False

    async def reply(self, profile, recurring_errors, history, user_turn, known_words=()):
        self.calls.append(
            {
                "profile": profile,
                "recurring": list(recurring_errors),
                "history": list(history),
                "turn": user_turn,
                "known_words": list(known_words),
            }
        )
        if self.fail:
            raise LLMError("модель недоступна")
        return LLMResult(
            reply=self.reply_value,
            usage=TokenUsage(input_tokens=100, output_tokens=40, cached_tokens=10),
            model="openai/test-model",
        )

    async def plain(self, system, user_turn, max_tokens=1024):
        self.plain_calls.append({"system": system, "turn": user_turn})
        if self.fail:
            raise LLMError("модель недоступна")
        return self.plain_value, TokenUsage(input_tokens=30, output_tokens=15), "openai/test-model"

    async def make_drill(self, profile, mistakes):
        self.drill_calls.append(list(mistakes))
        if self.fail:
            raise LLMError("модель недоступна")
        exercises = [
            {
                "sentence": f"Sentence with {m}.",
                "answer": f"Fixed {m}.",
                "focus": m[:20],
            }
            for m in mistakes
        ]
        return exercises, TokenUsage(input_tokens=40, output_tokens=20), "openai/test-model"

    async def check_drill(self, sentence, answer, student, focus):
        self.check_calls.append(
            {"sentence": sentence, "answer": answer, "student": student, "focus": focus}
        )
        if self.fail:
            raise LLMError("модель недоступна")
        feedback = "Артикль нужен перед исчисляемым." if not self.drill_correct else ""
        return (
            self.drill_correct,
            feedback,
            TokenUsage(input_tokens=30, output_tokens=10),
            "openai/test-model",
        )

    async def assess_level(self, messages):
        self.level_calls.append(list(messages))
        return self.level, TokenUsage(input_tokens=50, output_tokens=5), "openai/test-mini"


class FakeSpeech:
    """Вместо речевого фасада: отдаёт готовый Assessment и пишет «озвученное»."""

    def __init__(
        self,
        assessment: Assessment | None = None,
        assesses_pronunciation: bool = True,
        can_speak: bool = True,
        billable: bool = True,
    ):
        self.assessment = assessment or Assessment(transcript="hello there")
        self.synthesized: list[str] = []
        self.fail = False
        self.assesses_pronunciation = assesses_pronunciation
        self.can_speak = can_speak
        self.stt_is_billable = billable
        self.tts_is_billable = billable
        self.recognizer_name = "fake"
        self.synthesizer_name = "fake"
        self.tts_suffix = ".wav"

    async def warmup(self) -> None:
        return None

    async def assess(self, wav_path, duration_sec=0.0):
        if self.fail:
            from bot.speech import SpeechError

            raise SpeechError("Azure недоступен")
        return replace(self.assessment, duration_sec=duration_sec)

    async def synthesize(self, text: str, out_path: Path) -> int:
        self.synthesized.append(text)
        out_path.write_bytes(b"fake-audio")
        return len(text)
