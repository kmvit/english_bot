"""Юнит-тесты сжатия отчёта Azure Pronunciation Assessment."""
from __future__ import annotations

import json

import pytest

from bot.assessment import (
    PHONEMES_PER_WORD,
    WORD_SCORE_THRESHOLD,
    WORST_WORDS_LIMIT,
    compact_assessment,
)


def word(name, score, error="None", phonemes=None):
    payload = {
        "Word": name,
        "PronunciationAssessment": {"AccuracyScore": score, "ErrorType": error},
    }
    if phonemes is not None:
        payload["Phonemes"] = [
            {"Phoneme": p, "PronunciationAssessment": {"AccuracyScore": s}}
            for p, s in phonemes
        ]
    return payload


def report(display, scores, words):
    return {
        "RecognitionStatus": "Success",
        "DisplayText": display,
        "NBest": [
            {
                "Confidence": 0.93,
                "Lexical": display.lower(),
                "Display": display,
                "PronunciationAssessment": scores,
                "Words": words,
            }
        ],
    }


SCORES = {
    "AccuracyScore": 72.4,
    "FluencyScore": 85.0,
    "ProsodyScore": 68.2,
    "CompletenessScore": 100.0,
    "PronScore": 74.5,
}

SAMPLE = report(
    "I think the weather is nice today.",
    SCORES,
    [
        word("I", 95.0),
        word(
            "think",
            41.0,
            "Mispronunciation",
            [("th", 18.0), ("ih", 88.0), ("ng", 55.0), ("k", 91.0)],
        ),
        word("the", 88.0),
        word("weather", 62.0, "Mispronunciation", [("w", 95.0), ("eh", 44.0)]),
        word("is", 99.0),
        word("nice", 79.0),
        word("today", 91.0),
    ],
)


def test_transcript_and_scores():
    result = compact_assessment([SAMPLE], duration_sec=7.25)

    assert result.transcript == "I think the weather is nice today."
    assert result.duration_sec == 7.25
    assert result.scores["accuracy"] == pytest.approx(72.4)
    assert result.scores["fluency"] == pytest.approx(85.0)
    assert result.scores["prosody"] == pytest.approx(68.2)
    assert result.scores["completeness"] == pytest.approx(100.0)
    assert result.scores["overall"] == pytest.approx(74.5)


def test_accepts_json_string_from_azure():
    """Azure отдаёт отчёт строкой — она должна разбираться так же."""
    result = compact_assessment([json.dumps(SAMPLE)])
    assert result.transcript == "I think the weather is nice today."


def test_worst_words_sorted_and_thresholded():
    result = compact_assessment([SAMPLE])

    words = [w.word for w in result.worst_words]
    assert words == ["think", "weather", "nice"]
    assert all(w.score < WORD_SCORE_THRESHOLD for w in result.worst_words)
    assert "I" not in words and "today" not in words


def test_only_weak_phonemes_kept_and_sorted():
    result = compact_assessment([SAMPLE])
    think = result.worst_words[0]

    assert think.word == "think"
    assert [p["p"] for p in think.phonemes] == ["th", "ng"]
    assert [p["score"] for p in think.phonemes] == [18, 55]
    assert len(think.phonemes) <= PHONEMES_PER_WORD


def test_compact_dict_is_small_and_has_no_noise():
    compact = compact_assessment([SAMPLE]).to_compact_dict()

    assert set(compact) == {"transcript", "scores", "worst_words"}
    assert compact["scores"] == {
        "accuracy": 72,
        "fluency": 85,
        "prosody": 68,
        "completeness": 100,
        "overall": 74,  # round(74.5) -> 74 (банковское округление)
    }
    # У слова без просаженных фонем ключа phonemes нет.
    assert "phonemes" not in compact["worst_words"][-1]
    # Слово с ошибкой несёт её тип, слово без ошибки — нет.
    assert compact["worst_words"][0]["error"] == "Mispronunciation"
    assert "error" not in compact["worst_words"][-1]
    # Компактность: полный отчёт заметно больше сжатого.
    assert len(json.dumps(compact)) < len(json.dumps(SAMPLE)) / 2


def test_worst_words_limited():
    many = report(
        "one two three four five six seven",
        SCORES,
        [word(str(index), float(index)) for index in range(1, 8)],
    )
    result = compact_assessment([many])
    assert len(result.worst_words) == WORST_WORDS_LIMIT
    assert [w.word for w in result.worst_words] == ["1", "2", "3", "4", "5"]


def test_multiple_segments_merge_weighted_by_word_count():
    long_segment = report(
        "First part here.",
        {"AccuracyScore": 90.0, "PronScore": 90.0},
        [word(f"w{i}", 90.0) for i in range(9)],
    )
    short_segment = report(
        "Second.",
        {"AccuracyScore": 50.0, "PronScore": 50.0},
        [word("bad", 50.0)],
    )
    result = compact_assessment([long_segment, short_segment])

    assert result.transcript == "First part here. Second."
    # (90*9 + 50*1) / 10 = 86
    assert result.scores["accuracy"] == pytest.approx(86.0)
    assert [w.word for w in result.worst_words] == ["bad"]


def test_omission_without_score_counts_as_zero():
    with_omission = report(
        "I have cat.",
        SCORES,
        [
            word("I", 95.0),
            word("have", 92.0),
            {"Word": "a", "PronunciationAssessment": {"ErrorType": "Omission"}},
            word("cat", 90.0),
        ],
    )
    result = compact_assessment([with_omission])
    assert [(w.word, w.score, w.error_type) for w in result.worst_words] == [
        ("a", 0.0, "Omission")
    ]


@pytest.mark.parametrize(
    "garbage",
    [
        None,
        "",
        "not json at all",
        {},
        {"RecognitionStatus": "NoMatch", "DisplayText": ""},
        {"NBest": []},
        {"NBest": [{"Display": None, "Words": None}]},
        [1, 2, 3],
    ],
)
def test_broken_input_gives_empty_assessment(garbage):
    result = compact_assessment([garbage])
    assert result.is_empty
    assert result.worst_words == []
    assert result.to_compact_dict() == {"transcript": ""}


def test_no_segments_at_all():
    result = compact_assessment([])
    assert result.is_empty
    assert result.scores == {}


def test_non_numeric_scores_are_skipped():
    weird = report(
        "Hello there.",
        {"AccuracyScore": "high", "PronScore": None, "FluencyScore": 70},
        [word("hello", 60.0, phonemes=[("h", "bad"), ("eh", 30.0)])],
    )
    result = compact_assessment([weird])
    assert result.scores == {"fluency": 70.0}
    assert [p["p"] for p in result.worst_words[0].phonemes] == ["eh"]


# --- беглость по таймингам слов ---


def test_fluency_basic_metrics():
    # Пять слов за 10 секунд, одна пауза в 2 секунды между 3-м и 4-м.
    words = [
        ("I", 0.0, 0.3),
        ("went", 0.35, 0.8),
        ("there", 0.85, 1.2),
        ("yesterday", 3.2, 4.0),
        ("morning", 4.05, 4.6),
    ]
    from bot.assessment import compute_fluency

    fluency = compute_fluency(words, duration_sec=10.0)

    assert fluency["words"] == 5
    assert fluency["wpm"] == pytest.approx(30.0)      # 5 слов за 10 с
    assert fluency["pauses"] == 1                      # только разрыв в 2 с
    assert fluency["pause_ratio"] == pytest.approx(0.2, abs=0.01)


def test_short_gaps_are_not_pauses():
    from bot.assessment import compute_fluency

    words = [(f"w{i}", i * 0.5, i * 0.5 + 0.3) for i in range(6)]
    fluency = compute_fluency(words, duration_sec=3.0)

    assert fluency["pauses"] == 0
    assert fluency["pause_ratio"] == 0.0


def test_fluency_without_words():
    from bot.assessment import compute_fluency

    assert compute_fluency([], 10.0) == {}
    assert compute_fluency([("hi", None, None)], 10.0) == {}


def test_fluency_uses_speech_span_when_duration_unknown():
    """Если длительность не передали, берём промежуток от первого слова до последнего."""
    from bot.assessment import compute_fluency

    words = [("one", 0.0, 0.5), ("two", 5.5, 6.0)]
    fluency = compute_fluency(words, duration_sec=0.0)

    assert fluency["wpm"] == pytest.approx(20.0)  # 2 слова за 6 с
