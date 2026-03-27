"""
Deepgram client for prerecorded multilingual transcription.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class DeepgramError(Exception):
    pass


def _parse_deepgram_response(payload: Dict) -> Dict:
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    channels = results.get("channels", []) if isinstance(results, dict) else []
    alternatives = []
    if channels and isinstance(channels[0], dict):
        alternatives = channels[0].get("alternatives", []) or []
    alt = alternatives[0] if alternatives and isinstance(alternatives[0], dict) else {}

    text = (alt.get("transcript") or "").strip()
    confidence = float(alt.get("confidence") or 0.0)
    detected_language = (
        (channels[0].get("detected_language") if channels and isinstance(channels[0], dict) else None)
        or payload.get("language")
        or "en"
    )

    word_timestamps: List[Dict] = []
    for w in (alt.get("words") or []):
        if not isinstance(w, dict):
            continue
        word_timestamps.append({
            "word": (w.get("punctuated_word") or w.get("word") or "").strip(),
            "start": float(w.get("start", 0.0) or 0.0),
            "end": float(w.get("end", 0.0) or 0.0),
            "probability": float(w.get("confidence", 0.0) or 0.0),
        })

    segments: List[Dict] = []
    utterances = results.get("utterances", []) if isinstance(results, dict) else []
    if isinstance(utterances, list) and utterances:
        for i, u in enumerate(utterances):
            if not isinstance(u, dict):
                continue
            seg_text = (u.get("transcript") or "").strip()
            if not seg_text:
                continue
            segments.append({
                "id": i,
                "start": float(u.get("start", 0.0) or 0.0),
                "end": float(u.get("end", 0.0) or 0.0),
                "text": seg_text,
            })
    else:
        paragraphs = (alt.get("paragraphs") or {}).get("paragraphs", []) if isinstance(alt.get("paragraphs"), dict) else []
        if isinstance(paragraphs, list) and paragraphs:
            for i, p in enumerate(paragraphs):
                if not isinstance(p, dict):
                    continue
                seg_text = (p.get("sentences", [{}])[0].get("text", "") if p.get("sentences") else "") or p.get("text", "")
                seg_text = str(seg_text).strip()
                if not seg_text:
                    continue
                segments.append({
                    "id": i,
                    "start": float(p.get("start", 0.0) or 0.0),
                    "end": float(p.get("end", 0.0) or 0.0),
                    "text": seg_text,
                })

    if not segments and text:
        end_t = float(word_timestamps[-1]["end"]) if word_timestamps else 0.0
        segments = [{"id": 0, "start": 0.0, "end": end_t, "text": text}]

    return {
        "text": text,
        "segments": segments,
        "word_timestamps": word_timestamps,
        "language": str(detected_language).lower(),
        "language_probability": confidence,
        "confidence": confidence,
        "metadata": {
            "asr_provider_used": "deepgram",
            "fallback_reason": "",
        },
    }


def transcribe_prerecorded_audio(
    audio_path: str,
    language: Optional[str] = None,
    detect_language: bool = True
) -> Dict:
    """
    Call Deepgram prerecorded transcription endpoint.
    """
    api_key = getattr(settings, "DEEPGRAM_API_KEY", "")
    if not api_key:
        raise DeepgramError("DEEPGRAM_API_KEY is not configured")

    model = getattr(settings, "DEEPGRAM_MODEL", "nova-2")
    base_url = getattr(settings, "DEEPGRAM_BASE_URL", "https://api.deepgram.com/v1/listen")
    timeout_sec = int(getattr(settings, "DEEPGRAM_TIMEOUT_SEC", "180"))

    params = {
        "model": model,
        "smart_format": "true",
        "punctuate": "true",
        "utterances": "true",
    }

    if language and language != "auto":
        params["language"] = language
        params["detect_language"] = "false"
    else:
        params["detect_language"] = "true" if detect_language else "false"

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }

    try:
        with open(audio_path, "rb") as fp:
            resp = requests.post(
                base_url,
                params=params,
                headers=headers,
                data=fp,
                timeout=timeout_sec,
            )
    except Exception as e:
        raise DeepgramError(f"Deepgram request failed: {e}") from e

    if resp.status_code >= 400:
        raise DeepgramError(f"Deepgram error {resp.status_code}: {resp.text[:500]}")

    try:
        payload = resp.json()
    except Exception as e:
        raise DeepgramError(f"Invalid Deepgram JSON response: {e}") from e

    parsed = _parse_deepgram_response(payload)
    if not parsed.get("text"):
        raise DeepgramError("Deepgram returned empty transcript")
    return parsed
