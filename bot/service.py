"""Ход диалога: собрать контекст, спросить модель, записать результат, озвучить."""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import timedelta
from dataclasses import dataclass
from typing import Any
from pathlib import Path

from .assessment import Assessment
from .config import Config
from .costs import LLM, STT, TTS, TokenUsage, llm_cost, stt_cost, tts_cost
from .db import Database, utc_now
from .formatting import LOOKUP_HINT, render_reply, render_weekly_digest
from .parsing import Lookup
from .prompts import (
    DAILY_QUESTION_SYSTEM,
    EXPLAIN_SYSTEM,
    TRANSLATE_SYSTEM,
    build_daily_turn,
    build_explain_turn,
)
from .llm import LLMError, Teacher
from .speech import Speech, SpeechError
from .audio import AudioError, to_ogg_opus

log = logging.getLogger(__name__)


def _interleave(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Перемешать два списка заданий через одно, начиная с первого.

    Вперемешку, а не блоками: пять однотипных заданий подряд читаются как
    список, и ученик перестаёт вдумываться уже на третьем.
    """
    mixed: list[dict[str, Any]] = []
    for index in range(max(len(first), len(second))):
        if index < len(first):
            mixed.append(first[index])
        if index < len(second):
            mixed.append(second[index])
    return mixed


def _mentions(text: str, word: str) -> bool:
    """Слово встретилось в тексте как отдельное слово, а не частью другого."""
    if not word:
        return False
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None

LEVEL_CHECK_SAMPLE = 15
#: Сколько ранее введённых слов показывать модели, чтобы она их переиспользовала.
KNOWN_WORDS_IN_PROMPT = 25
#: Сколько слов из словаря попадает в одну тренировку рядом с ошибками.
WORDS_PER_DRILL = 2
#: На каком по счёту сообщении подсказать про разбор выделенного фрагмента.
#: Не на первом: сначала нужно увидеть, как бот вообще отвечает.
HINT_AT_MESSAGE = 3
#: Длиннее этого выделенный фрагмент разбирать бессмысленно — это уже абзац.
MAX_LOOKUP_CHARS = 200


@dataclass
class TurnResult:
    text: str
    spoken_text: str
    voice_enabled: bool
    #: Разговорная часть ушла в голосовое, текстом остались только исправления.
    voice_only: bool = False
    #: Перевод уже показан — кнопка «Перевести» не нужна.
    translated: bool = False
    #: Разовая подсказка вдогонку ответу. Пусто, когда подсказывать нечего.
    hint: str = ""


class TeacherService:
    def __init__(
        self,
        config: Config,
        db: Database,
        teacher: Teacher,
        speech: Speech,
    ) -> None:
        self._config = config
        self._db = db
        self._teacher = teacher
        self._speech = speech
        # Ссылки на фоновые задачи: иначе сборщик мусора может их отменить.
        self._background: set[asyncio.Task] = set()

    async def handle_turn(
        self,
        user_id: int,
        llm_turn: str,
        history_entry: str,
        assessment: Assessment | None = None,
    ) -> TurnResult:
        """Один ход диалога. Бросает LLMError, если модель недоступна."""
        profile = await self._db.ensure_profile(user_id)
        recurring = await self._db.top_errors(user_id, limit=5)
        history = await self._db.get_history(user_id, self._config.history_limit)
        known = await self._db.recent_words(user_id, limit=KNOWN_WORDS_IN_PROMPT)

        result = await self._teacher.reply(
            profile, recurring, history, llm_turn, [w.word for w in known]
        )
        await self._log_llm(user_id, result.usage, result.model)

        reply = result.reply
        await self._db.add_message(user_id, "user", history_entry)
        if reply.reply:
            await self._db.add_message(user_id, "assistant", reply.reply)

        if reply.new_errors:
            await self._db.record_errors(
                user_id,
                [(error.type, error.description) for error in reply.new_errors],
                example=history_entry[:300],
            )

        if reply.new_words:
            await self._db.add_words(
                user_id,
                [(w.word, w.meaning, w.example) for w in reply.new_words],
            )
        # Слово, которое ученик употребил сам, засчитываем как встреченное:
        # по этому счётчику видно, что прижилось, а что осталось в списке.
        reused = [w.word for w in known if _mentions(history_entry, w.word)]
        if reused:
            await self._db.mark_words_seen(user_id, reused)

        # Баллов произношения может не быть (локальный Whisper), а темп речи
        # есть всегда — запись нужна и ради одной беглости.
        if assessment is not None and (assessment.scores or assessment.fluency):
            await self._db.add_pronunciation(
                user_id,
                assessment.scores,
                assessment.duration_sec,
                assessment.fluency,
            )

        total = await self._db.bump_messages_total(user_id)
        self._maybe_schedule_level_check(user_id, profile.level_checked_at_message, total)

        # Без работающей озвучки режим «только голос» оставил бы ученика
        # вообще без ответа, поэтому он включается лишь вместе с ней.
        voice_enabled = profile.voice_replies and self._speech.can_speak
        voice_only = profile.voice_only and voice_enabled and bool(reply.reply)
        return TurnResult(
            text=render_reply(reply, include_conversation=not voice_only),
            spoken_text=reply.reply,
            voice_enabled=profile.voice_replies,
            voice_only=voice_only,
            translated=bool(reply.reply_ru) and not voice_only,
            hint=LOOKUP_HINT if total == HINT_AT_MESSAGE else "",
        )

    async def _log_llm(self, user_id: int, usage: TokenUsage, model: str) -> None:
        await self._db.add_usage(
            user_id,
            kind=LLM,
            model=model,
            cost_usd=llm_cost(usage, self._config.prices),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read=usage.cached_tokens,
        )

    # --- действия по кнопкам под ответом --------------------------------

    async def translate_reply(self, user_id: int, reply_text: str) -> str:
        """Перевести ответ учителя на русский. Бросает LLMError."""
        text, usage, model = await self._teacher.plain(TRANSLATE_SYSTEM, reply_text)
        await self._log_llm(user_id, usage, model)
        return text

    async def explain_reply(self, user_id: int, reply_text: str) -> str:
        """Разобрать исправления подробнее. Бросает LLMError."""
        profile = await self._db.ensure_profile(user_id)
        text, usage, model = await self._teacher.plain(
            EXPLAIN_SYSTEM, build_explain_turn(reply_text, profile.level)
        )
        await self._log_llm(user_id, usage, model)
        return text

    # --- разбор выделенного фрагмента -----------------------------------

    async def lookup(
        self, user_id: int, fragment: str, context: str = "", question: str = ""
    ) -> tuple[Lookup, int] | None:
        """Разобрать фрагмент и сразу положить его в словарь.

        Возвращает карточку и id сохранённого слова — по нему ученик может
        передумать и убрать слово. None — если модель ответила не по делу.
        """
        card, usage, model = await self._teacher.lookup(fragment, context, question)
        await self._log_llm(user_id, usage, model)
        if card is None:
            return None
        word_id = await self._db.save_word(
            user_id,
            card.term,
            meaning=card.translation or None,
            example=card.example or None,
            source="asked",
        )
        return card, word_id

    async def forget_word(self, user_id: int, word_id: int) -> str | None:
        """Убрать слово, сохранённое по ошибке. Возвращает само слово."""
        return await self._db.delete_word(user_id, word_id)

    # --- инициатива бота -------------------------------------------------

    async def daily_question(self, user_id: int) -> str:
        """Сгенерировать утренний вопрос с опорой на последние разговоры."""
        profile = await self._db.ensure_profile(user_id)
        recent = await self._db.recent_topics(user_id, limit=6)
        text, usage, model = await self._teacher.plain(
            DAILY_QUESTION_SYSTEM,
            build_daily_turn(profile.level, profile.interests, recent),
            max_tokens=256,
        )
        await self._log_llm(user_id, usage, model)
        text = text.strip()
        if text:
            # Вопрос — часть беседы: без записи в историю ответ ученика
            # прилетит модели без контекста, на который он отвечает.
            await self._db.add_message(user_id, "assistant", text)
        return text

    async def weekly_digest(self, user_id: int) -> str | None:
        """Итоги недели. None — если заниматься было нечем, сводка бы только мешала."""
        since = utc_now() - timedelta(days=7)
        activity = await self._db.activity_since(user_id, since)
        if not activity["messages"]:
            return None
        return render_weekly_digest(
            activity=activity,
            errors=await self._db.top_errors(user_id, limit=3),
            week=await self._db.fluency_progress(user_id, days=7),
            previous=await self._db.fluency_progress(user_id, days=30),
            words_total=await self._db.words_total(user_id),
        )

    # --- тренировка ошибок ----------------------------------------------

    async def start_drill(self, user_id: int, limit: int = 3) -> list[dict[str, Any]]:
        """Составить задания: ошибки, которые пора повторить, и слова из словаря.

        Пустой список означает, что повторять нечего: либо материала ещё нет,
        либо всё отлёживает срок после верных ответов.
        """
        errors = await self._db.errors_due_for_drill(user_id, limit=limit)
        words = await self._db.words_due_for_drill(user_id, limit=WORDS_PER_DRILL)
        if not errors and not words:
            return []
        profile = await self._db.ensure_profile(user_id)

        error_tasks: list[dict[str, Any]] = []
        if errors:
            exercises, usage, model = await self._teacher.make_drill(
                profile, [error.description for error in errors]
            )
            await self._log_llm(user_id, usage, model)
            error_tasks = [
                {
                    **exercise,
                    "kind": "error",
                    "error_id": error.id,
                    "description": error.description,
                }
                for exercise, error in zip(exercises, errors)
            ]

        word_tasks: list[dict[str, Any]] = []
        if words:
            try:
                exercises, usage, model = await self._teacher.make_word_drill(
                    profile, [(w.word, w.meaning or "") for w in words]
                )
            except LLMError:
                # Слова — добавка к разбору ошибок. Если задания на них не
                # составились, тренировка всё равно должна состояться.
                if not error_tasks:
                    raise
                log.warning("Задания на слова не составились, оставляю только ошибки")
            else:
                await self._log_llm(user_id, usage, model)
                word_tasks = [
                    {**exercise, "kind": "word", "word_id": word.id, "word": word.word}
                    for exercise, word in zip(exercises, words)
                ]

        return _interleave(error_tasks, word_tasks)

    async def check_drill_answer(
        self, user_id: int, exercise: dict[str, Any], student_answer: str
    ) -> tuple[bool, str]:
        correct, feedback, usage, model = await self._teacher.check_drill(
            exercise.get("sentence", ""),
            exercise.get("answer", ""),
            student_answer,
            exercise.get("focus", ""),
        )
        await self._log_llm(user_id, usage, model)
        if exercise.get("word_id"):
            await self._db.record_word_drill_result(int(exercise["word_id"]), correct)
        elif exercise.get("error_id"):
            await self._db.record_drill_result(int(exercise["error_id"]), correct)
        return correct, feedback

    # --- голос ----------------------------------------------------------

    @property
    def assesses_pronunciation(self) -> bool:
        return self._speech.assesses_pronunciation

    async def transcribe(self, wav_path: Path, user_id: int, duration: float) -> Assessment:
        assessment = await self._speech.assess(wav_path, duration)
        # Локальный Whisper бесплатен — пишем секунды, но не стоимость.
        cost = (
            stt_cost(duration, self._config.prices)
            if self._speech.stt_is_billable
            else 0.0
        )
        await self._db.add_usage(
            user_id, kind=STT, cost_usd=cost, audio_seconds=duration
        )
        return assessment

    async def voice_answer(
        self, user_id: int, text: str, workdir: Path
    ) -> Path | None:
        """Озвучить разговорную часть ответа. None — если озвучка не удалась."""
        text = (text or "").strip()
        if not text or not self._speech.can_speak:
            return None
        raw_path = workdir / f"answer{self._speech.tts_suffix}"
        ogg_path = workdir / "answer.ogg"
        try:
            chars = await self._speech.synthesize(text, raw_path)
            await to_ogg_opus(raw_path, ogg_path, self._config.ffmpeg_bin)
        except (SpeechError, AudioError, OSError) as exc:
            log.error("Озвучка не удалась: %s", exc)
            return None
        cost = (
            tts_cost(chars, self._config.prices) if self._speech.tts_is_billable else 0.0
        )
        await self._db.add_usage(user_id, kind=TTS, cost_usd=cost, chars=chars)
        return ogg_path

    # --- периодическая оценка уровня ------------------------------------

    def _maybe_schedule_level_check(
        self, user_id: int, checked_at: int, total: int
    ) -> None:
        if total - checked_at < self._config.level_check_every:
            return
        # В фоне: ученик не должен ждать лишний запрос к модели.
        task = asyncio.create_task(self._update_level(user_id, total))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def wait_background(self) -> None:
        """Дождаться фоновых задач (оценка уровня). Используется в тестах."""
        while self._background:
            await asyncio.gather(*tuple(self._background), return_exceptions=True)

    async def _update_level(self, user_id: int, total: int) -> None:
        try:
            samples = await self._db.recent_user_messages(user_id, LEVEL_CHECK_SAMPLE)
            level, usage, model = await self._teacher.assess_level(samples)
            await self._log_llm(user_id, usage, model)
            fields = {"level_checked_at_message": total}
            if level:
                fields["level"] = level
            await self._db.update_profile(user_id, **fields)
            log.info("Уровень ученика %s обновлён: %s", user_id, level)
        except (LLMError, OSError) as exc:
            log.warning("Оценка уровня не удалась: %s", exc)
        except Exception:  # pragma: no cover - фон не должен ронять бота
            log.exception("Непредвиденная ошибка при оценке уровня")
