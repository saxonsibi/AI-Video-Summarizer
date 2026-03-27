"""
Deterministic transcript and summary quality metrics used across the pipeline.

These metrics are lightweight and model-agnostic. They are not WER/CER, but
they provide stable signals for routing, QA gates, and observability.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from django.conf import settings


DEFAULT_NOISE_PHRASES = (
    "ask ai about this moment",
    "screen recording",
    "subscribe to the channel",
    "click the bell icon",
)


def _sentence_split(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _word_tokens(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\b[\w']+\b", text)


def _unicode_alpha_tokens(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)?", text, flags=re.UNICODE)


def _normalize_tokens(text: str) -> List[str]:
    return [tok.lower() for tok in _word_tokens(text)]


def _phrase_blacklist() -> List[str]:
    raw = getattr(settings, "TRANSCRIPT_PHRASE_BLACKLIST", "")
    configured = [item.strip().lower() for item in str(raw).split("|") if item.strip()]
    ordered: List[str] = []
    for phrase in [*DEFAULT_NOISE_PHRASES, *configured]:
        if phrase and phrase not in ordered:
            ordered.append(phrase)
    return ordered


def _regex_hooks() -> List[re.Pattern]:
    raw = getattr(settings, "TRANSCRIPT_REGEX_CLEANUP_PATTERNS", "")
    hooks: List[re.Pattern] = []
    for pattern in str(raw).split("||"):
        pattern = pattern.strip()
        if not pattern:
            continue
        try:
            hooks.append(re.compile(pattern, flags=re.IGNORECASE))
        except re.error:
            continue
    return hooks


def evaluate_transcript_quality(text: str, segments: List[Dict] | None = None) -> Dict:
    """
    Return deterministic proxy metrics used for quality gates and observability.
    """
    text = (text or "").strip()
    segments = segments or []

    words = _word_tokens(text)
    lower_words = [w.lower() for w in words]
    sentences = _sentence_split(text)

    sentence_lengths = [len(_word_tokens(s)) for s in sentences] or [0]
    seg_lengths = [len(_word_tokens((s or {}).get("text", ""))) for s in segments if isinstance(s, dict)]

    long_sentences_gt35 = sum(1 for n in sentence_lengths if n > 35)
    avg_sentence_words = round(sum(sentence_lengths) / max(len(sentence_lengths), 1), 2)
    avg_segment_words = round(sum(seg_lengths) / max(len(seg_lengths), 1), 2) if seg_lengths else 0.0
    max_segment_words = max(seg_lengths) if seg_lengths else 0
    avg_segment_length = avg_segment_words
    very_short_segment_ratio = round(
        (sum(1 for n in seg_lengths if n <= 2) / max(len(seg_lengths), 1)) if seg_lengths else 0.0,
        4,
    )

    period_cap_artifacts = len(re.findall(r"[a-z]\.[A-Z]", text))
    question_dot_artifacts = len(re.findall(r"\?\.", text))
    malformed_question_starts = len(
        re.findall(r"\b(?:Does|Do|Is|Are|What|Why|How)\s+[A-Z][a-z]+\s+[A-Z][a-z]+[?](?!\s+[a-z])", text)
    )

    mid_caps = 0
    for sentence in sentences:
        toks = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", sentence)
        for idx, tok in enumerate(toks[1:], start=1):
            if tok[0].isupper():
                prev = toks[idx - 1]
                if prev.lower() not in {"i", "mr", "mrs", "ms", "dr", "sr", "jr"}:
                    mid_caps += 1

    duplicate_word_loops = len(re.findall(r"\b(\w+)\s+\1\b", text, flags=re.IGNORECASE))
    unique_words = len(set(lower_words))
    lexical_diversity = round(unique_words / max(len(words), 1), 4)

    cap_tokens = re.findall(r"\b[A-Z][a-zA-Z]+\b", text)
    cap_counts = Counter(t.lower() for t in cap_tokens)
    repeated_caps = sum(1 for _, count in cap_counts.items() if count >= 3)

    blacklist = _phrase_blacklist()
    regex_hooks = _regex_hooks()
    repeated_noise_phrase_count = sum(text.lower().count(phrase) for phrase in blacklist)
    regex_cleanup_hits = sum(len(pattern.findall(text)) for pattern in regex_hooks)

    malformed_fragment_score = round(
        min(
            1.0,
            (
                (duplicate_word_loops * 0.08)
                + (period_cap_artifacts * 0.04)
                + (question_dot_artifacts * 0.04)
                + (mid_caps * 0.01)
                + (very_short_segment_ratio * 0.45)
            ),
        ),
        4,
    )
    punctuation_score = round(
        max(
            0.0,
            min(
                1.0,
                1.0 - (
                    (period_cap_artifacts * 0.05)
                    + (question_dot_artifacts * 0.05)
                    + (malformed_question_starts * 0.03)
                ),
            ),
        ),
        4,
    )
    transcript_leak_score = round(
        min(
            1.0,
            (repeated_noise_phrase_count * 0.12) + (regex_cleanup_hits * 0.08) + (duplicate_word_loops * 0.05),
        ),
        4,
    )
    entity_normalization_count = int(
        sum(1 for token in lower_words if token in {"lovable", "pinterest", "mux", "figma", "upwork", "twitter", "landing", "cta"})
    )
    final_quality_score = round(
        max(
            0.0,
            min(
                1.0,
                0.55
                + min(0.25, lexical_diversity * 0.3)
                + min(0.15, (1.0 - very_short_segment_ratio) * 0.15)
                + (punctuation_score * 0.1)
                - (repeated_noise_phrase_count * 0.04)
                - (regex_cleanup_hits * 0.03)
                - (malformed_fragment_score * 0.22)
                - (transcript_leak_score * 0.18),
            ),
        ),
        4,
    )

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_words": avg_sentence_words,
        "long_sentences_gt35": long_sentences_gt35,
        "period_cap_artifacts": period_cap_artifacts,
        "question_dot_artifacts": question_dot_artifacts,
        "malformed_question_starts": malformed_question_starts,
        "mid_sentence_caps": mid_caps,
        "duplicate_word_loops": duplicate_word_loops,
        "lexical_diversity": lexical_diversity,
        "segment_count": len(seg_lengths),
        "avg_segment_words": avg_segment_words,
        "max_segment_words": max_segment_words,
        "avg_segment_length": avg_segment_length,
        "very_short_segment_ratio": very_short_segment_ratio,
        "repeated_cap_entities": repeated_caps,
        "repeated_noise_phrase_count": repeated_noise_phrase_count,
        "entity_normalization_count": entity_normalization_count,
        "punctuation_score": punctuation_score,
        "malformed_fragment_score": malformed_fragment_score,
        "transcript_leak_score": transcript_leak_score,
        "regex_cleanup_hits": regex_cleanup_hits,
        "final_quality_score": final_quality_score,
    }


def evaluate_summary_quality(summary_payload: Dict, transcript_text: str) -> Dict:
    """
    Deterministic quality proxy for structured summaries.
    """
    payload = summary_payload or {}
    tldr = (payload.get("tldr") or "").strip()
    key_points = [str(item).strip() for item in payload.get("key_points", []) if str(item).strip()]
    chapters = [item for item in payload.get("chapters", []) if isinstance(item, dict)]
    transcript_sentences = _sentence_split(transcript_text)

    malformed_key_points = sum(
        1
        for point in key_points
        if len(_word_tokens(point)) < 5
        or re.search(r"\b(?:this tutorial explains|the interview covers)\b", point.lower()) and len(_word_tokens(point)) < 7
    )
    transcript_leak_points = 0
    for point in key_points:
        point_tokens = set(_normalize_tokens(point))
        if not point_tokens:
            continue
        if any(
            len(point_tokens & set(_normalize_tokens(sentence))) / max(len(point_tokens), 1) >= 0.78
            for sentence in transcript_sentences
        ):
            transcript_leak_points += 1

    malformed_chapters = sum(
        1
        for chapter in chapters
        if len(_word_tokens(str(chapter.get("title", "")))) < 2
        or re.search(r"\bdiscussion on\b", str(chapter.get("title", "")).lower())
    )
    repeated_template_phrases = max(0, len(key_points) - len({point.lower() for point in key_points}))
    topic_entity_coverage = len(
        {
            token
            for token in _normalize_tokens(" ".join([tldr, *key_points]))
            if len(token) > 3 and token not in {"this", "that", "with", "from", "they", "have", "about"}
        }
    )
    final_quality_score = round(
        max(
            0.0,
            min(
                1.0,
                0.72
                + (0.04 if tldr else -0.2)
                + (0.05 if 4 <= len(key_points) <= 7 else -0.1)
                + (0.04 if chapters else -0.08)
                + min(0.08, topic_entity_coverage * 0.01)
                - (malformed_key_points * 0.08)
                - (transcript_leak_points * 0.08)
                - (malformed_chapters * 0.05)
                - (repeated_template_phrases * 0.04),
            ),
        ),
        4,
    )

    return {
        "tldr_present": bool(tldr),
        "key_point_count": len(key_points),
        "chapter_count": len(chapters),
        "malformed_key_points": malformed_key_points,
        "transcript_leak_points": transcript_leak_points,
        "malformed_chapters": malformed_chapters,
        "repeated_template_phrases": repeated_template_phrases,
        "topic_entity_coverage": topic_entity_coverage,
        "final_quality_score": final_quality_score,
    }


def evaluate_summary_faithfulness(summary_text: str, transcript_text: str) -> Dict:
    """
    Lightweight anti-hallucination proxy for summary outputs.

    This does not prove factual correctness, but it gives a stable signal for:
    - unsupported token drift
    - unsupported capitalized entities
    - extractive leakage
    """
    summary_text = (summary_text or "").strip()
    transcript_text = (transcript_text or "").strip()

    summary_tokens = [tok.lower() for tok in _unicode_alpha_tokens(summary_text) if len(tok) > 2]
    transcript_tokens = {tok.lower() for tok in _unicode_alpha_tokens(transcript_text) if len(tok) > 2}
    generic = {
        "this", "that", "with", "from", "into", "about", "video", "speaker", "speakers",
        "summary", "explains", "discusses", "covers", "shows", "using", "their", "there",
        "where", "which", "while", "because", "through", "around", "across", "overall",
    }
    informative_summary_tokens = [tok for tok in summary_tokens if tok not in generic]
    unsupported_tokens = [tok for tok in informative_summary_tokens if tok not in transcript_tokens]
    unsupported_token_ratio = round(
        (len(unsupported_tokens) / max(len(informative_summary_tokens), 1)),
        4,
    )

    capitalized_entities = {
        ent.strip()
        for ent in re.findall(r"\b[A-Z][A-Za-z0-9.+/-]{2,}(?:\s+[A-Z][A-Za-z0-9.+/-]{2,}){0,3}\b", summary_text)
        if ent.strip().lower() not in {"the video", "the speaker", "key points"}
    }
    transcript_lower = transcript_text.lower()
    unsupported_entities = [
        ent for ent in capitalized_entities
        if ent.lower() not in transcript_lower
    ]

    transcript_sentences = _sentence_split(transcript_text)
    transcript_leak_ratio = 0.0
    if summary_text and transcript_sentences:
        summary_sentences = _sentence_split(summary_text)
        copied = 0
        for summary_sentence in summary_sentences:
            summary_norm = set(_normalize_tokens(summary_sentence))
            if not summary_norm:
                continue
            if any(
                len(summary_norm & set(_normalize_tokens(transcript_sentence))) / max(len(summary_norm), 1) >= 0.82
                for transcript_sentence in transcript_sentences
            ):
                copied += 1
        transcript_leak_ratio = round(copied / max(len(summary_sentences), 1), 4)

    faithfulness_score = round(
        max(
            0.0,
            min(
                1.0,
                0.88
                - (unsupported_token_ratio * 0.55)
                - (min(len(unsupported_entities), 4) * 0.12)
                - (transcript_leak_ratio * 0.18),
            ),
        ),
        4,
    )

    return {
        "unsupported_token_ratio": unsupported_token_ratio,
        "unsupported_token_count": len(unsupported_tokens),
        "unsupported_entities": unsupported_entities,
        "unsupported_entity_count": len(unsupported_entities),
        "transcript_leak_ratio": transcript_leak_ratio,
        "faithfulness_score": faithfulness_score,
    }
