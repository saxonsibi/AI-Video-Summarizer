"""
Canonical transcript builder for multilingual retrieval.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Any

from django.conf import settings

from .language import normalize_language_code
from .translation import translate_text, translate_segments

logger = logging.getLogger(__name__)


def _normalize_segments(payload: Any) -> List[Dict]:
    if isinstance(payload, dict):
        segs = payload.get("segments", [])
        return segs if isinstance(segs, list) else []
    if isinstance(payload, list):
        return payload
    return []


def build_canonical_text(
    transcript_text: str,
    transcript_segments: Any,
    transcript_language: Optional[str],
    canonical_language: Optional[str] = None
) -> Dict:
    """
    Build canonical transcript representation for embeddings/retrieval.
    Returns:
      {
        canonical_text,
        canonical_segments,
        canonical_language,
        translation_used
      }
    """
    start = time.time()
    src_lang = normalize_language_code(transcript_language, default="en", allow_auto=False)
    target = normalize_language_code(
        canonical_language or getattr(settings, "EMBED_CANONICAL_LANGUAGE", "en"),
        default="en",
        allow_auto=False
    )
    translation_provider = str(getattr(settings, "TRANSLATION_PROVIDER", "none") or "none").strip().lower()

    segments = _normalize_segments(transcript_segments)
    translation_used = False

    # No translation needed.
    if src_lang == target:
        canonical_segments = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            canonical_segments.append({
                "id": seg.get("id"),
                "start": float(seg.get("start", 0) or 0),
                "end": float(seg.get("end", 0) or 0),
                "text": text,
                "original_text": text,
            })
        canonical_text = (transcript_text or "").strip()
        result = {
            "canonical_text": canonical_text,
            "canonical_segments": canonical_segments,
            "canonical_language": target,
            "translation_used": translation_used,
        }
        logger.info(
            "[CANON] translated_to=%s chunks=%s time=%.2fs",
            target,
            len(canonical_segments),
            time.time() - start,
        )
        return result

    # If translation is disabled, preserve native text for multilingual retrieval.
    if translation_provider == "none":
        canonical_segments = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            canonical_segments.append({
                "id": seg.get("id"),
                "start": float(seg.get("start", 0) or 0),
                "end": float(seg.get("end", 0) or 0),
                "text": text,
                "original_text": text,
            })
        canonical_text = (transcript_text or "").strip()
        logger.info(
            "[CANON] translated_to=%s chunks=%s time=%.2fs",
            src_lang,
            len(canonical_segments),
            time.time() - start,
        )
        return {
            "canonical_text": canonical_text,
            "canonical_segments": canonical_segments,
            "canonical_language": src_lang,
            "translation_used": False,
        }

    # Default: translate full transcript text in larger chunks for speed/rate-limit safety.
    # Segment-wise translation can trigger many model calls on long videos.
    translate_segments_enabled = bool(getattr(settings, "CANONICAL_TRANSLATE_SEGMENTS", False))
    if translate_segments_enabled:
        canonical_segments = translate_segments(segments, src_lang, target)
        if canonical_segments:
            translation_used = True
            canonical_text = " ".join((s.get("text") or "").strip() for s in canonical_segments if s.get("text")).strip()
        else:
            canonical_text = translate_text(
                transcript_text or "",
                source_language=src_lang,
                target_language=target,
                preserve_format=False,
            ).strip()
            translation_used = canonical_text != (transcript_text or "").strip()
            canonical_segments = []
    else:
        canonical_text = translate_text(
            transcript_text or "",
            source_language=src_lang,
            target_language=target,
            preserve_format=False,
        ).strip()
        translation_used = canonical_text != (transcript_text or "").strip()
        canonical_segments = []

    logger.info(
        "[CANON] translated_to=%s chunks=%s time=%.2fs",
        target,
        len(canonical_segments),
        time.time() - start,
    )
    return {
        "canonical_text": canonical_text,
        "canonical_segments": canonical_segments,
        "canonical_language": target,
        "translation_used": translation_used,
    }
