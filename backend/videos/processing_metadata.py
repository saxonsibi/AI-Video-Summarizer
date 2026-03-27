"""
Helpers for stable processing metadata payloads.
"""

from __future__ import annotations

from typing import Optional

from .models import Transcript, Video


def build_processing_metadata(video: Video, transcript: Optional[Transcript] = None) -> dict:
    """
    Build additive processing diagnostics for API responses.
    Always returns a stable schema and safe defaults.
    """
    t = transcript
    if t is None and video is not None:
        t = video.transcripts.order_by("-created_at").first()

    asr_engine = ""
    language = ""
    transcript_quality_score = 0.0
    transcript_state = ""
    summary_quality_score = 0.0
    processing_metrics = {}
    malayalam_observability = {}
    transcript_warning_message = ""
    downstream_suppressed = False
    downstream_suppression_reason = ""
    trusted_visible_word_count = 0
    trusted_display_unit_count = 0
    low_evidence_malayalam = False
    detected_language = ""
    detected_language_confidence = 0.0
    is_multilingual_content = False
    english_view_available = False
    current_available_views = ["original"]
    translation_state = ""
    translation_blocked_reason = ""
    if t is not None:
        asr_engine = (t.asr_engine_used or t.asr_engine or "").strip()
        language = (t.transcript_language or t.language or "").strip()
        transcript_quality_score = float(t.transcript_quality_score or 0.0)
        if isinstance(t.json_data, dict):
            transcript_state = str(t.json_data.get("transcript_state", "") or "").strip()
            transcript_warning_message = str(t.json_data.get("transcript_warning_message", "") or "").strip()
            processing_metrics = t.json_data.get("processing_metrics", {}) if isinstance(t.json_data.get("processing_metrics", {}), dict) else {}
            malayalam_observability = processing_metrics.get("malayalam_observability", {}) if isinstance(processing_metrics.get("malayalam_observability", {}), dict) else {}
            structured_cache = t.json_data.get("structured_summary_cache", {}) if isinstance(t.json_data.get("structured_summary_cache", {}), dict) else {}
            summary_quality_score = float(structured_cache.get("quality_score", processing_metrics.get("summary_quality_score", 0.0)) or 0.0)
            downstream_suppressed = bool(t.json_data.get("downstream_suppressed", processing_metrics.get("downstream_suppressed", False)))
            downstream_suppression_reason = str(t.json_data.get("downstream_suppression_reason", processing_metrics.get("downstream_suppression_reason", "")) or "").strip()
            trusted_visible_word_count = int(t.json_data.get("trusted_visible_word_count", processing_metrics.get("trusted_visible_word_count", 0)) or 0)
            trusted_display_unit_count = int(t.json_data.get("trusted_display_unit_count", processing_metrics.get("trusted_display_unit_count", 0)) or 0)
            low_evidence_malayalam = bool(t.json_data.get("low_evidence_malayalam", processing_metrics.get("low_evidence_malayalam", False)))
            detected_language = str(t.json_data.get("detected_language", language) or language).strip()
            detected_language_confidence = float(t.json_data.get("detected_language_confidence", 0.0) or 0.0)
            is_multilingual_content = bool(t.json_data.get("is_multilingual_content", False))
            english_view_available = bool(t.json_data.get("english_view_available", False))
            current_available_views = list(t.json_data.get("current_available_views", ["original"]) or ["original"])
            translation_state = str(t.json_data.get("translation_state", "") or "").strip()
            translation_blocked_reason = str(t.json_data.get("translation_blocked_reason", "") or "").strip()
    if t is not None and language and language != "en" and not english_view_available:
        transcript_state_value = (transcript_state or "").strip().lower()
        visible_text = ""
        canonical_text = ""
        if isinstance(getattr(t, "json_data", None), dict):
            visible_text = str(t.json_data.get("display_readable_transcript") or t.json_data.get("readable_transcript") or "").strip()
        canonical_text = str(getattr(t, "transcript_canonical_en_text", "") or getattr(t, "transcript_canonical_text", "") or "").strip()
        if transcript_state_value == "cleaned" and (visible_text or canonical_text):
            english_view_available = True
            current_available_views = ["original", "english"]
            translation_state = translation_state or ("same_as_original" if language == "en" else "translated")
        elif transcript_state_value == "degraded":
            translation_blocked_reason = translation_blocked_reason or "degraded_safe_translation_blocked"

    processing_time_seconds = 0.0
    if video is not None and getattr(video, "created_at", None):
        end_ts = getattr(video, "processed_at", None) or getattr(video, "updated_at", None)
        if end_ts is not None:
            try:
                processing_time_seconds = max(0.0, float((end_ts - video.created_at).total_seconds()))
            except Exception:
                processing_time_seconds = 0.0

    return {
        "asr_engine": asr_engine,
        "language": language,
        "processing_time_seconds": round(processing_time_seconds, 3),
        "transcript_quality_score": round(float(transcript_quality_score), 4),
        "summary_quality_score": round(float(summary_quality_score), 4),
        "transcript_state": transcript_state,
        "transcript_warning_message": transcript_warning_message,
        "downstream_suppressed": downstream_suppressed,
        "downstream_suppression_reason": downstream_suppression_reason,
        "trusted_visible_word_count": trusted_visible_word_count,
        "trusted_display_unit_count": trusted_display_unit_count,
        "low_evidence_malayalam": low_evidence_malayalam,
        "detected_language": detected_language or language,
        "detected_language_confidence": round(float(detected_language_confidence), 4),
        "is_multilingual_content": is_multilingual_content,
        "english_view_available": english_view_available,
        "current_available_views": current_available_views if english_view_available else ["original"],
        "translation_state": translation_state,
        "translation_blocked_reason": translation_blocked_reason,
        "summary_ready": bool(video and video.summaries.exists()) if video else False,
        "chat_ready": bool(video and getattr(video, "status", "") == "completed") if video else False,
        "malayalam_observability": malayalam_observability,
        "processing_metrics": processing_metrics,
    }
