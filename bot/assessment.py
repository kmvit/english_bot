"""Сжатие отчёта Azure Pronunciation Assessment до компактного JSON для LLM.

Полный отчёт Azure содержит по каждому слову слоги и фонемы — это десятки
килобайт на 20 секунд речи. В промпт уходит только транскрипт, общие баллы и
несколько худших слов: остальное не влияет на совет учителя, но стоит токенов.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Баллы выше этого порога считаем нормой и не показываем ученику.
WORD_SCORE_THRESHOLD = 80.0
# Фонему упоминаем, только если она заметно просажена.
PHONEME_SCORE_THRESHOLD = 70.0
WORST_WORDS_LIMIT = 5
PHONEMES_PER_WORD = 3

_SCORE_KEYS = {
    "accuracy": "AccuracyScore",
    "fluency": "FluencyScore",
    "prosody": "ProsodyScore",
    "completeness": "CompletenessScore",
    "overall": "PronScore",
}


@dataclass
class WordScore:
    word: str
    score: float
    error_type: str = "None"
    phonemes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {"word": self.word, "score": round(self.score)}
        if self.error_type and self.error_type != "None":
            item["error"] = self.error_type
        if self.phonemes:
            item["phonemes"] = self.phonemes
        return item


@dataclass
class Assessment:
    """Результат распознавания и оценки произношения одного голосового."""

    transcript: str
    scores: dict[str, float] = field(default_factory=dict)
    worst_words: list[WordScore] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.transcript.strip()

    def to_compact_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"transcript": self.transcript}
        if self.scores:
            payload["scores"] = {k: round(v) for k, v in self.scores.items()}
        if self.worst_words:
            payload["worst_words"] = [w.to_dict() for w in self.worst_words]
        return payload

    def to_compact_json(self) -> str:
        return json.dumps(self.to_compact_dict(), ensure_ascii=False, sort_keys=True)


def _as_dict(raw: Any) -> dict[str, Any]:
    """Azure отдаёт JSON строкой; тесты и кеш могут передать уже разобранный dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes)):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _best_hypothesis(segment: dict[str, Any]) -> dict[str, Any]:
    nbest = segment.get("NBest")
    if isinstance(nbest, list) and nbest and isinstance(nbest[0], dict):
        return nbest[0]
    return {}


def _segment_transcript(segment: dict[str, Any], hypothesis: dict[str, Any]) -> str:
    for key in ("Display", "DisplayText", "Lexical"):
        value = hypothesis.get(key) or segment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _extract_phonemes(word: dict[str, Any]) -> list[dict[str, Any]]:
    phonemes: list[tuple[float, str]] = []
    for phoneme in word.get("Phonemes") or []:
        if not isinstance(phoneme, dict):
            continue
        label = phoneme.get("Phoneme")
        score = _float_or_none(
            (phoneme.get("PronunciationAssessment") or {}).get("AccuracyScore")
        )
        if not isinstance(label, str) or score is None:
            continue
        if score < PHONEME_SCORE_THRESHOLD:
            phonemes.append((score, label))
    phonemes.sort(key=lambda item: item[0])
    return [
        {"p": label, "score": round(score)}
        for score, label in phonemes[:PHONEMES_PER_WORD]
    ]


def _extract_words(hypothesis: dict[str, Any]) -> list[WordScore]:
    words: list[WordScore] = []
    for word in hypothesis.get("Words") or []:
        if not isinstance(word, dict):
            continue
        label = word.get("Word")
        if not isinstance(label, str) or not label:
            continue
        pa = word.get("PronunciationAssessment") or {}
        score = _float_or_none(pa.get("AccuracyScore"))
        error_type = pa.get("ErrorType") or "None"
        if score is None:
            # Пропуски и вставки приходят без балла — считаем их нулём.
            if error_type in ("Omission", "Insertion"):
                score = 0.0
            else:
                continue
        words.append(
            WordScore(
                word=label,
                score=score,
                error_type=str(error_type),
                phonemes=_extract_phonemes(word),
            )
        )
    return words


def _merge_scores(
    per_segment: Sequence[tuple[dict[str, float], int]],
) -> dict[str, float]:
    """Средневзвешенное по числу слов: длинный отрезок весит больше короткого."""
    merged: dict[str, float] = {}
    for key in _SCORE_KEYS:
        total = 0.0
        weight_sum = 0.0
        for scores, weight in per_segment:
            if key not in scores:
                continue
            w = float(max(weight, 1))
            total += scores[key] * w
            weight_sum += w
        if weight_sum:
            merged[key] = total / weight_sum
    return merged


def compact_assessment(
    raw_segments: Iterable[Any],
    duration_sec: float = 0.0,
) -> Assessment:
    """Свести один или несколько отчётов Azure в компактный результат.

    `raw_segments` — JSON-строки (или dict) из свойства
    `SpeechServiceResponse_JsonResult`, по одной на распознанный отрезок.
    """
    transcripts: list[str] = []
    scores_with_weight: list[tuple[dict[str, float], int]] = []
    all_words: list[WordScore] = []

    for raw in raw_segments:
        segment = _as_dict(raw)
        if not segment:
            continue
        hypothesis = _best_hypothesis(segment)
        transcript = _segment_transcript(segment, hypothesis)
        if transcript:
            transcripts.append(transcript)

        words = _extract_words(hypothesis)
        all_words.extend(words)

        raw_scores = hypothesis.get("PronunciationAssessment") or {}
        segment_scores: dict[str, float] = {}
        for name, azure_key in _SCORE_KEYS.items():
            value = _float_or_none(raw_scores.get(azure_key))
            if value is not None:
                segment_scores[name] = value
        if segment_scores:
            scores_with_weight.append((segment_scores, len(words)))

    worst = [w for w in all_words if w.score < WORD_SCORE_THRESHOLD]
    worst.sort(key=lambda w: w.score)

    return Assessment(
        transcript=" ".join(transcripts).strip(),
        scores=_merge_scores(scores_with_weight),
        worst_words=worst[:WORST_WORDS_LIMIT],
        duration_sec=round(float(duration_sec), 2),
    )
