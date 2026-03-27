"""
Benchmark helpers for real processed videos.

This module measures stored pipeline quality plus a small live chatbot probe set
without changing the main processing pipeline.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from chatbot.rag_engine import ChatbotEngine
from videos.models import Summary, Transcript, Video
from videos.processing_metadata import build_processing_metadata
from videos.serializers import get_or_build_structured_summary
from videos.utils_metrics import evaluate_summary_quality, evaluate_transcript_quality


NOISY_PATTERNS = (
    r"\bask ai about this moment\b",
    r"\bdiscussion on\b",
    r"\bit covers the main themes that come up\b",
    r"\buphold these two workflow\b",
    r"\bmade out with a few times\b",
)

LOW_INFO_PATTERNS = (
    r"\bthis is about the main themes\b",
    r"\bit covers the main themes\b",
    r"\bgeneral discussion\b",
    r"\bnot clearly stated\b",
)


EVALUATION_CHECKLIST = {
    "transcript_cleanliness": [
        "No UI overlay phrases or repeated junk text",
        "Sentences are readable and punctuation is normalized",
        "Timestamps and segment boundaries still make sense",
    ],
    "summary_usefulness": [
        "TLDR captures the real video topic",
        "Key points are semantic sentences, not transcript fragments",
        "Action items appear only when genuinely actionable",
    ],
    "chapter_readability": [
        "Chapter titles are human-readable topic labels",
        "No malformed fragment-like chapter names",
        "Chapters reflect real topic shifts",
    ],
    "chatbot_answer_correctness": [
        "Answer directly addresses the question",
        "Answer is grounded in cited evidence",
        "No transcript garbage leaks into the final wording",
    ],
    "source_card_readability": [
        "Source titles/previews are readable and concise",
        "Timestamps are present and plausible",
        "Sources reflect the actual answer topic",
    ],
    "moment_explanation_quality": [
        "Moment answer stays close to the clicked timestamp window",
        "Moment answer explains the local step or topic clearly",
        "Moment sources remain readable and timestamp-grounded",
    ],
}


LANGUAGE_QUESTION_SETS = {
    "en": [
        "What is this video about?",
        "What are the key takeaways?",
        "What is discussed at the beginning of the video?",
    ],
    "hi": [
        "यह वीडियो किस बारे में है?",
        "मुख्य बातें क्या हैं?",
        "वीडियो की शुरुआत में क्या चर्चा होती है?",
    ],
    "ml": [
        "ഈ വീഡിയോ എന്തിനെക്കുറിച്ചാണ്?",
        "പ്രധാന കാര്യങ്ങൾ എന്തൊക്കെയാണ്?",
        "വീഡിയോയുടെ തുടക്കത്തിൽ എന്താണ് സംസാരിക്കുന്നത്?",
    ],
}


@dataclass
class BenchmarkRow:
    video_id: str
    title: str
    language: str
    duration: float
    asr_engine: str
    asr_model: str
    transcription_seconds: float
    cleanup_seconds: float
    summary_seconds: float
    indexing_seconds: float
    total_seconds: float
    transcript_quality_score: float
    summary_quality_score: float
    fallback_triggered: bool
    fallback_reason: str
    chatbot_avg_latency_seconds: float
    chatbot_answer_quality_signals: str
    source_label_quality_signals: str


def _normalize_tokens(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", (text or "").lower())


def _sentence_split(text: str) -> List[str]:
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _matches_any(patterns, text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(_normalize_tokens(left))
    right_tokens = set(_normalize_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), 1)


def _transcript_leak_score(answer_text: str, transcript_text: str) -> float:
    answer_sentences = _sentence_split(answer_text)
    transcript_sentences = _sentence_split(transcript_text)
    if not answer_sentences or not transcript_sentences:
        return 0.0
    best = 0.0
    for answer_sentence in answer_sentences:
        for transcript_sentence in transcript_sentences[:120]:
            best = max(best, _overlap_ratio(answer_sentence, transcript_sentence))
    return round(best, 4)


def _latest_summary(video: Video, summary_type: str) -> Optional[Summary]:
    return video.summaries.filter(summary_type=summary_type).order_by("-created_at").first()


def _question_set(language: str) -> List[str]:
    return LANGUAGE_QUESTION_SETS.get((language or "en").lower(), LANGUAGE_QUESTION_SETS["en"])


def _moment_probe_targets(segments: List[Dict]) -> List[Dict]:
    normalized = [seg for seg in segments or [] if str(seg.get("text", "")).strip()]
    if not normalized:
        return []
    indices = sorted({0, len(normalized) // 2})
    targets = []
    for idx in indices:
        seg = normalized[min(idx, len(normalized) - 1)]
        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", start) or start)
        targets.append(
            {
                "timestamp": start,
                "label": f"{int(start // 60):02d}:{int(start % 60):02d}",
                "excerpt": str(seg.get("text", "")).strip(),
                "segments": [seg],
                "question": "What is happening at this moment?",
                "window_seconds": None,
            }
        )
    return targets


def _load_asr_model(transcript: Transcript) -> str:
    if isinstance(transcript.json_data, dict):
        meta = transcript.json_data.get("asr_metadata", {})
        if isinstance(meta, dict):
            return str(
                meta.get("actual_local_model_name")
                or meta.get("selected_model")
                or meta.get("asr_model")
                or ""
            ).strip()
    return ""


def _source_quality_signals(cards: List[Dict]) -> Dict:
    texts = [str(card.get("text", "")).strip() for card in cards if str(card.get("text", "")).strip()]
    readable = sum(
        1
        for text in texts
        if len(text) >= 18 and not _matches_any(NOISY_PATTERNS, text)
    )
    malformed = sum(1 for text in texts if _matches_any(NOISY_PATTERNS, text))
    return {
        "count": len(texts),
        "readable_count": readable,
        "malformed_count": malformed,
        "readability_rate": round(readable / max(len(texts), 1), 4),
        "avg_chars": round(sum(len(text) for text in texts) / max(len(texts), 1), 2) if texts else 0.0,
    }


def _answer_quality_signals(results: List[Dict], transcript_text: str) -> Dict:
    malformed = 0
    low_info = 0
    transcript_leaks = 0
    latencies = []
    source_counts = []
    for result in results:
        answer = str(result.get("answer", "")).strip()
        latencies.append(float(result.get("latency_seconds", 0.0) or 0.0))
        source_counts.append(int(result.get("source_count", 0) or 0))
        if _matches_any(NOISY_PATTERNS, answer):
            malformed += 1
        if _matches_any(LOW_INFO_PATTERNS, answer) or len(_normalize_tokens(answer)) < 8:
            low_info += 1
        if _transcript_leak_score(answer, transcript_text) >= 0.78:
            transcript_leaks += 1
    return {
        "question_count": len(results),
        "malformed_answer_count": malformed,
        "low_information_answer_count": low_info,
        "transcript_leak_answer_count": transcript_leaks,
        "avg_latency_seconds": round(sum(latencies) / max(len(latencies), 1), 4) if latencies else 0.0,
        "avg_source_count": round(sum(source_counts) / max(len(source_counts), 1), 4) if source_counts else 0.0,
    }


def benchmark_video(
    video: Video,
    use_llm: bool = False,
    response_language: str = "auto",
    include_moment_probes: bool = False,
) -> Dict:
    transcript = video.transcripts.order_by("-created_at").first()
    if not transcript:
        raise ValueError(f"Video {video.id} has no transcript")

    processing_meta = build_processing_metadata(video, transcript)
    structured_summary = get_or_build_structured_summary(video, transcript)
    transcript_text = (
        transcript.transcript_original_text
        or transcript.full_text
        or ""
    )
    transcript_segments = []
    if isinstance(transcript.json_data, dict):
        raw_segments = transcript.json_data.get("segments", [])
        transcript_segments = raw_segments if isinstance(raw_segments, list) else []
    elif isinstance(transcript.json_data, list):
        transcript_segments = transcript.json_data

    transcript_quality = (
        transcript.json_data.get("quality_metrics")
        if isinstance(transcript.json_data, dict) and isinstance(transcript.json_data.get("quality_metrics"), dict)
        else evaluate_transcript_quality(transcript_text, transcript_segments)
    )
    summary_quality = (
        transcript.json_data.get("structured_summary_cache", {}).get("quality_metrics")
        if isinstance(transcript.json_data, dict)
        and isinstance(transcript.json_data.get("structured_summary_cache"), dict)
        and isinstance(transcript.json_data.get("structured_summary_cache", {}).get("quality_metrics"), dict)
        else evaluate_summary_quality(structured_summary, transcript_text)
    )

    processing_metrics = processing_meta.get("processing_metrics", {}) if isinstance(processing_meta, dict) else {}
    asr_meta = transcript.json_data.get("asr_metadata", {}) if isinstance(transcript.json_data, dict) else {}
    engine = ChatbotEngine(str(video.id))
    per_question: List[Dict] = []
    all_source_cards: List[Dict] = []
    moment_probe_results: List[Dict] = []

    for question in _question_set(transcript.transcript_language or transcript.language or "en"):
        start = time.perf_counter()
        response = engine.ask(
            question,
            use_llm=use_llm,
            strict_mode=True,
            response_language=response_language if response_language != "auto" else (transcript.transcript_language or transcript.language or "en"),
        )
        latency = max(0.0, time.perf_counter() - start)
        answer_text = str(response.get("answer", "") or "").strip()
        sources = response.get("sources") or []
        all_source_cards.extend(sources)
        per_question.append(
            {
                "question": question,
                "answer": answer_text,
                "latency_seconds": round(latency, 4),
                "source_count": len(sources),
                "source_preview_texts": [str(item.get("text", "")).strip() for item in sources],
            }
        )

    if include_moment_probes:
        for probe in _moment_probe_targets(transcript_segments):
            start = time.perf_counter()
            response = engine.ask(
                probe["question"],
                use_llm=use_llm,
                strict_mode=True,
                response_language=response_language if response_language != "auto" else (transcript.transcript_language or transcript.language or "en"),
                moment_segments=probe["segments"],
                context_timestamp=probe["timestamp"],
                context_window_seconds=probe["window_seconds"],
            )
            latency = max(0.0, time.perf_counter() - start)
            sources = response.get("sources") or []
            all_source_cards.extend(sources)
            moment_probe_results.append(
                {
                    "question": probe["question"],
                    "timestamp": probe["label"],
                    "answer": str(response.get("answer", "") or "").strip(),
                    "latency_seconds": round(latency, 4),
                    "source_count": len(sources),
                    "source_preview_texts": [str(item.get("text", "")).strip() for item in sources],
                    "timestamp_context": response.get("timestamp_context"),
                }
            )

    answer_quality = _answer_quality_signals(per_question, transcript_text)
    source_quality = _source_quality_signals(all_source_cards)
    moment_answer_quality = _answer_quality_signals(moment_probe_results, transcript_text) if moment_probe_results else {}
    moment_source_quality = _source_quality_signals(
        [
            {"text": preview}
            for result in moment_probe_results
            for preview in result.get("source_preview_texts", [])
        ]
    ) if moment_probe_results else {}

    row = BenchmarkRow(
        video_id=str(video.id),
        title=video.title,
        language=(transcript.transcript_language or transcript.language or "").strip(),
        duration=float(video.duration or 0.0),
        asr_engine=(processing_meta.get("asr_engine") or transcript.asr_engine_used or transcript.asr_engine or "").strip(),
        asr_model=_load_asr_model(transcript),
        transcription_seconds=float(processing_metrics.get("transcription_seconds", 0.0) or 0.0),
        cleanup_seconds=float(processing_metrics.get("cleanup_seconds", 0.0) or 0.0),
        summary_seconds=float(processing_metrics.get("summary_seconds", 0.0) or 0.0),
        indexing_seconds=float(processing_metrics.get("indexing_seconds", 0.0) or 0.0),
        total_seconds=float(processing_metrics.get("total_seconds", processing_meta.get("processing_time_seconds", 0.0)) or 0.0),
        transcript_quality_score=float(processing_meta.get("transcript_quality_score", transcript_quality.get("final_quality_score", 0.0)) or 0.0),
        summary_quality_score=float(processing_meta.get("summary_quality_score", summary_quality.get("final_quality_score", 0.0)) or 0.0),
        fallback_triggered=bool(asr_meta.get("fallback_triggered", False)),
        fallback_reason=str(asr_meta.get("fallback_reason", "") or ""),
        chatbot_avg_latency_seconds=float(answer_quality.get("avg_latency_seconds", 0.0) or 0.0),
        chatbot_answer_quality_signals=json.dumps(answer_quality, ensure_ascii=False),
        source_label_quality_signals=json.dumps(source_quality, ensure_ascii=False),
    )

    return {
        "row": asdict(row),
        "processing_metadata": processing_meta,
        "transcript_quality_metrics": transcript_quality,
        "summary_quality_metrics": summary_quality,
        "structured_summary": structured_summary,
        "chatbot_probes": per_question,
        "moment_probes": moment_probe_results,
        "chatbot_answer_quality_signals": answer_quality,
        "source_label_quality_signals": source_quality,
        "moment_answer_quality_signals": moment_answer_quality,
        "moment_source_label_quality_signals": moment_source_quality,
        "evaluation_checklist": EVALUATION_CHECKLIST,
    }


def write_benchmark_outputs(results: List[Dict], output_dir: Path, report_name: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report_name}.json"
    csv_path = output_dir / f"{report_name}.csv"
    checklist_path = output_dir / f"{report_name}_checklist.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_count": len(results),
                "evaluation_checklist": EVALUATION_CHECKLIST,
                "videos": results,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    rows = [item["row"] for item in results]
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    checklist_lines = ["# VideoIQ Benchmark Checklist", ""]
    for section, items in EVALUATION_CHECKLIST.items():
        checklist_lines.append(f"## {section.replace('_', ' ').title()}")
        for item in items:
            checklist_lines.append(f"- [ ] {item}")
        checklist_lines.append("")
    checklist_path.write_text("\n".join(checklist_lines).strip() + "\n", encoding="utf-8")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "checklist": str(checklist_path),
    }
