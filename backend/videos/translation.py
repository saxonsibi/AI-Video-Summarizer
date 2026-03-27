"""
Translation utilities with caching and retry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Optional, List, Dict

from django.conf import settings
from django.utils import timezone

from .language import normalize_language_code, detect_text_language

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_TRANSLATION_CACHE: Dict[str, str] = {}


def _cache_key(text: str, src: str, dst: str, preserve_format: bool) -> str:
    payload = f"{src}|{dst}|{int(preserve_format)}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


def stable_content_hash(value) -> str:
    try:
        normalized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        normalized = str(value)
    digest = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()
    logger.info("[EN_VIEW_SOURCE_HASH] hash=%s", digest)
    return digest


def build_english_view_source_hash(content_kind: str, source_payload) -> str:
    return stable_content_hash({
        "kind": str(content_kind or "").strip().lower(),
        "source": source_payload,
    })


def evaluate_english_view_policy(
    *,
    content_kind: str,
    source_language: str,
    source_state: str = "",
    has_grounded_text: bool = True,
    low_evidence_malayalam: bool = False,
    blocked_reason: str = "",
    degraded_low_evidence_reason: str = "low_evidence_source_language",
) -> Dict[str, object]:
    src = normalize_language_code(source_language, default="en", allow_auto=False)
    state = str(source_state or "").strip().lower()
    kind = str(content_kind or "content").strip().lower()
    policy = {
        "content_kind": kind,
        "source_language": src,
        "source_state": state,
        "has_grounded_text": bool(has_grounded_text),
        "low_evidence_malayalam": bool(low_evidence_malayalam),
        "allow_translation": True,
        "translation_state": "translated",
        "blocked_reason": "",
        "policy_reason": "allowed_grounded_translation",
        "current_available_views": ["original", "english"],
    }
    logger.info(
        "[EN_VIEW_POLICY] kind=%s language=%s state=%s grounded=%s low_evidence=%s requested_block_reason=%s",
        kind,
        src,
        state,
        bool(has_grounded_text),
        bool(low_evidence_malayalam),
        str(blocked_reason or ""),
    )
    if src == "en":
        policy.update({
            "allow_translation": False,
            "translation_state": "same_as_original",
            "blocked_reason": "already_english",
            "policy_reason": "already_english",
            "current_available_views": ["original"],
        })
    elif blocked_reason:
        policy.update({
            "allow_translation": False,
            "translation_state": "blocked",
            "blocked_reason": str(blocked_reason or ""),
            "policy_reason": "explicit_block",
            "current_available_views": ["original"],
        })
    elif not has_grounded_text or state == "failed":
        policy.update({
            "allow_translation": False,
            "translation_state": "blocked",
            "blocked_reason": "insufficient_grounded_text",
            "policy_reason": "insufficient_grounded_text",
            "current_available_views": ["original"],
        })
    elif src == "ml" and state == "degraded" and low_evidence_malayalam:
        policy.update({
            "allow_translation": False,
            "translation_state": "blocked",
            "blocked_reason": str(degraded_low_evidence_reason or "low_evidence_source_language"),
            "policy_reason": "degraded_low_evidence_malayalam",
            "current_available_views": ["original"],
        })
    logger.info(
        "[EN_VIEW_POLICY_DECISION] kind=%s language=%s allow=%s state=%s reason=%s",
        kind,
        src,
        policy["allow_translation"],
        policy["translation_state"],
        policy["policy_reason"],
    )
    return policy


def is_english_view_cache_valid(cache_entry, source_hash: str) -> bool:
    if not isinstance(cache_entry, dict):
        return False
    cache_hash = str(cache_entry.get("english_view_source_hash", "") or "")
    cache_valid = bool(cache_entry.get("english_view_valid", False))
    logger.info(
        "[EN_VIEW_HASH_COMPARE] cache_hash=%s source_hash=%s cache_hit=%s",
        cache_hash,
        source_hash,
        bool(cache_valid and cache_hash == source_hash),
    )
    return bool(cache_valid and cache_hash == source_hash and isinstance(cache_entry.get("payload"), dict))


def build_english_view_cache_entry(
    payload: Dict[str, object],
    *,
    source_hash: str,
    build_reason: str,
    source_language: str,
    policy: Dict[str, object],
) -> Dict[str, object]:
    return {
        "payload": dict(payload or {}),
        "english_view_source_hash": str(source_hash or ""),
        "english_view_valid": True,
        "english_view_last_built_at": timezone.now().isoformat(),
        "english_view_build_reason": str(build_reason or ""),
        "original_language": normalize_language_code(source_language, default="en", allow_auto=False),
        "translation_state": str(payload.get("translation_state", policy.get("translation_state", "")) or ""),
        "translation_blocked_reason": str(payload.get("translation_blocked_reason", policy.get("blocked_reason", "")) or ""),
        "english_view_policy_mode": str(policy.get("translation_state", "") or ""),
        "english_view_policy_reason": str(policy.get("policy_reason", "") or ""),
        "current_available_views": list(payload.get("current_available_views") or policy.get("current_available_views") or ["original"]),
    }


def _split_for_translation(text: str, max_words: int) -> List[str]:
    words = (text or "").split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        start = end
    return chunks


def translate_text(
    text: str,
    source_language: str,
    target_language: str,
    preserve_format: bool = False
) -> str:
    """
    Translate text with Groq (if configured). Returns input text on failure.
    """
    if not text:
        return text

    src = normalize_language_code(source_language, default="en", allow_auto=False)
    dst = normalize_language_code(target_language, default="en", allow_auto=False)
    if src == dst:
        return text

    key = _cache_key(text, src, dst, preserve_format)
    with _CACHE_LOCK:
        cached = _TRANSLATION_CACHE.get(key)
        if cached is not None:
            return cached

    provider = str(getattr(settings, "TRANSLATION_PROVIDER", "none") or "none").strip().lower()
    if provider not in {"none", "local"}:
        provider = "none"
    if provider == "none":
        return text

    max_words = int(getattr(settings, "TRANSLATION_MAX_WORDS_PER_CALL", 1200))
    attempts = int(getattr(settings, "TRANSLATION_RETRY_ATTEMPTS", 2))
    base_delay = float(getattr(settings, "TRANSLATION_RETRY_BASE_DELAY_SEC", 1.5))
    model_name = str(getattr(settings, "LOCAL_TRANSLATION_MODEL", "") or "").strip()

    chunks = _split_for_translation(text, max_words=max_words)
    out_chunks: List[str] = []

    try:
        # Local translation backend (optional).
        # If no local model is configured/available, fail-soft to source text.
        translator = None
        if model_name:
            try:
                from transformers import pipeline  # type: ignore
                translator = pipeline("translation", model=model_name)
            except Exception as e:
                logger.warning("Local translation model unavailable (%s), using passthrough", e)
                translator = None

        for chunk in chunks:
            translated_chunk = None
            for attempt in range(1, max(1, attempts) + 1):
                try:
                    if translator is None:
                        translated_chunk = chunk
                        break
                    # HuggingFace translation pipeline returns list[{"translation_text": "..."}]
                    result = translator(chunk, max_length=1024)
                    if isinstance(result, list) and result:
                        translated_chunk = (result[0].get("translation_text") or "").strip()
                    if translated_chunk:
                        break
                except Exception as e:
                    if attempt >= attempts:
                        logger.warning(f"Translation failed after retries: {e}")
                    else:
                        delay = base_delay * attempt
                        logger.warning(f"Translation retry in {delay:.1f}s (attempt {attempt}/{attempts})")
                        time.sleep(delay)
            out_chunks.append(translated_chunk or chunk)

        translated = "\n".join(out_chunks).strip()
        if translated:
            with _CACHE_LOCK:
                _TRANSLATION_CACHE[key] = translated
            return translated
        return text
    except Exception as e:
        logger.warning(f"Translation unavailable: {e}")
        return text


def translate_segments(
    segments: List[Dict],
    source_language: str,
    target_language: str
) -> List[Dict]:
    """Translate a list of transcript segments while preserving timestamps."""
    if not segments:
        return []
    out = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        raw_text = (seg.get("text") or "").strip()
        if not raw_text:
            continue
        translated_text = translate_text(
            raw_text,
            source_language=source_language,
            target_language=target_language,
            preserve_format=False,
        )
        updated = dict(seg)
        updated["text"] = translated_text
        updated["original_text"] = raw_text
        out.append(updated)
    return out


def build_safe_english_view_text(
    text: str,
    source_language: str,
    *,
    allow_translation: bool = True,
    blocked_reason: str = "",
    warning: str = "",
    preserve_format: bool = False,
    translated_text: str = "",
) -> Dict[str, object]:
    """
    Build additive English-view metadata for existing grounded content.
    This never rewrites the source content in place and can be safely blocked.
    """
    original_text = str(text or "").strip()
    src = normalize_language_code(source_language, default="en", allow_auto=False)
    payload: Dict[str, object] = {
        "original_language": src,
        "translated_language": "en",
        "english_view_available": False,
        "translation_state": "blocked",
        "translation_warning": str(warning or "").strip(),
        "translation_blocked_reason": str(blocked_reason or "").strip(),
        "current_available_views": ["original"],
        "english_view_text": "",
    }
    if not original_text:
        logger.info("[EN_VIEW_TRANSLATION_BLOCKED] source_language=%s reason=%s", src, payload["translation_blocked_reason"] or "insufficient_grounded_text")
        payload["translation_blocked_reason"] = payload["translation_blocked_reason"] or "insufficient_grounded_text"
        return payload
    if src == "en":
        logger.info("[EN_VIEW_TRANSLATION_MODE] source_language=%s mode=same_as_original", src)
        payload["translation_state"] = "same_as_original"
        return payload
    if not allow_translation:
        logger.info("[EN_VIEW_TRANSLATION_BLOCKED] source_language=%s reason=%s", src, payload["translation_blocked_reason"] or "translation_blocked")
        payload["translation_blocked_reason"] = payload["translation_blocked_reason"] or "translation_blocked"
        return payload

    candidate = str(translated_text or "").strip()
    if not candidate:
        candidate = translate_text(
            original_text,
            source_language=src,
            target_language="en",
            preserve_format=preserve_format,
        ).strip()
    detected_lang, _, _, _ = detect_text_language(candidate, default="en")
    if not candidate:
        logger.info("[EN_VIEW_TRANSLATION_BLOCKED] source_language=%s reason=%s", src, payload["translation_blocked_reason"] or "translation_runtime_unavailable")
        payload["translation_blocked_reason"] = payload["translation_blocked_reason"] or "translation_runtime_unavailable"
        return payload
    if candidate == original_text and detected_lang != "en":
        logger.info("[EN_VIEW_TRANSLATION_BLOCKED] source_language=%s reason=%s", src, payload["translation_blocked_reason"] or "translation_runtime_unavailable")
        payload["translation_blocked_reason"] = payload["translation_blocked_reason"] or "translation_runtime_unavailable"
        return payload

    payload["english_view_available"] = True
    payload["translation_state"] = "translated" if candidate != original_text else "same_as_original"
    payload["translation_blocked_reason"] = ""
    payload["current_available_views"] = ["original", "english"]
    payload["english_view_text"] = candidate
    logger.info(
        "[EN_VIEW_TRANSLATION_RESULT] source_language=%s state=%s available=%s",
        src,
        payload["translation_state"],
        payload["english_view_available"],
    )
    return payload


def build_safe_english_view_structured_summary(
    payload: Dict[str, object],
    source_language: str,
    *,
    allow_translation: bool = True,
    blocked_reason: str = "",
    warning: str = "",
) -> Dict[str, object]:
    summary_payload = dict(payload or {})
    src = normalize_language_code(source_language, default="en", allow_auto=False)
    tldr = str(summary_payload.get("tldr", "") or "").strip()
    key_points = [str(item).strip() for item in (summary_payload.get("key_points") or []) if str(item).strip()]
    action_items = [str(item).strip() for item in (summary_payload.get("action_items") or []) if str(item).strip()]
    chapters = [dict(item) for item in (summary_payload.get("chapters") or []) if isinstance(item, dict)]
    joined = "\n".join([tldr, *key_points, *action_items, *[str(ch.get("title", "") or "").strip() for ch in chapters]]).strip()
    text_view = build_safe_english_view_text(
        joined,
        src,
        allow_translation=allow_translation,
        blocked_reason=blocked_reason,
        warning=warning or str(summary_payload.get("warning_message", "") or ""),
        preserve_format=True,
    )
    meta = {
        "summary_original_language": src,
        "summary_english_view_available": bool(text_view.get("english_view_available", False)),
        "summary_translation_state": str(text_view.get("translation_state", "") or ""),
        "summary_translation_warning": str(text_view.get("translation_warning", "") or ""),
        "summary_translation_blocked_reason": str(text_view.get("translation_blocked_reason", "") or ""),
        "summary_current_available_views": list(text_view.get("current_available_views") or ["original"]),
    }
    if not text_view.get("english_view_available"):
        return meta

    translated_payload = dict(summary_payload)
    translated_payload["tldr"] = translate_text(tldr, source_language=src, target_language="en", preserve_format=True).strip() if tldr else ""
    translated_payload["key_points"] = [
        translate_text(point, source_language=src, target_language="en", preserve_format=False).strip()
        for point in key_points
    ]
    translated_payload["action_items"] = [
        translate_text(item, source_language=src, target_language="en", preserve_format=False).strip()
        for item in action_items
    ]
    translated_payload["chapters"] = [
        {
            **chapter,
            "title": translate_text(str(chapter.get("title", "") or ""), source_language=src, target_language="en", preserve_format=False).strip(),
        }
        for chapter in chapters
    ]
    if translated_payload.get("warning_message"):
        translated_payload["warning_message"] = translate_text(
            str(translated_payload.get("warning_message") or ""),
            source_language=src,
            target_language="en",
            preserve_format=True,
        ).strip()
    meta["english_view_structured_summary"] = translated_payload
    return meta
