"""Ход диалога: собрать контекст, спросить модель, записать результат, озвучить."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from .assessment import Assessment
from .config import Config
from .costs import LLM, STT, TTS, TokenUsage, llm_cost, stt_cost, tts_cost
from .db import Database
from .formatting import render_reply
from .prompts import EXPLAIN_SYSTEM, TRANSLATE_SYSTEM, build_explain_turn
from .llm import LLMError, Teacher
from .speech import Speech, SpeechError
from .audio import AudioError, to_ogg_opus

log = logging.getLogger(__name__)

LEVEL_CHECK_SAMPLE = 15


@dataclass
class TurnResult:
    text: str
    spoken_text: str
    voice_enabled: bool


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

        result = await self._teacher.reply(profile, recurring, history, llm_turn)
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

        if assessment is not None and assessment.scores:
            await self._db.add_pronunciation(
                user_id, assessment.scores, assessment.duration_sec
            )

        total = await self._db.bump_messages_total(user_id)
        self._maybe_schedule_level_check(user_id, profile.level_checked_at_message, total)

        return TurnResult(
            text=render_reply(reply),
            spoken_text=reply.reply,
            voice_enabled=profile.voice_replies,
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
