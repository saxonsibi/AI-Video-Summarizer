"""
Language utilities for multilingual ASR/summarization/chat.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional, Tuple, List


LANGUAGE_ALIASES = {
    "auto": "auto",
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "kannada": "kn",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "chinese-simplified": "zh",
    "portuguese": "pt",
    "italian": "it",
    "arabic": "ar",
    "russian": "ru",
}


SCRIPT_TO_LANG = {
    "latin": "en",
    "devanagari": "hi",
    "tamil": "ta",
    "telugu": "te",
    "malayalam": "ml",
    "kannada": "kn",
    "arabic": "ar",
    "cyrillic": "ru",
    "hangul": "ko",
    "cjk": "zh",
}


def normalize_language_code(
    language: Optional[str],
    default: str = "en",
    allow_auto: bool = True
) -> str:
    """Normalize language value to short code (e.g. en, hi, ml)."""
    if language is None:
        return default
    value = str(language).strip().lower()
    if not value:
        return default
    if value in LANGUAGE_ALIASES:
        mapped = LANGUAGE_ALIASES[value]
        if mapped == "auto" and not allow_auto:
            return default
        return mapped
    if value == "auto":
        return "auto" if allow_auto else default
    if re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", value):
        return value.split("-")[0]
    return default


def _script_distribution(text: str) -> Dict[str, int]:
    buckets = {
        "latin": 0,
        "cyrillic": 0,
        "devanagari": 0,
        "arabic": 0,
        "cjk": 0,
        "hangul": 0,
        "tamil": 0,
        "telugu": 0,
        "malayalam": 0,
        "kannada": 0,
        "bengali": 0,
        "other": 0,
    }
    for ch in text or "":
        if ch.isspace() or ch.isdigit():
            continue
        if unicodedata.category(ch).startswith("P"):
            continue
        cp = ord(ch)
        if 0x0041 <= cp <= 0x024F:
            buckets["latin"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            buckets["cyrillic"] += 1
        elif 0x0900 <= cp <= 0x097F:
            buckets["devanagari"] += 1
        elif 0x0600 <= cp <= 0x06FF:
            buckets["arabic"] += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            buckets["cjk"] += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            buckets["hangul"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            buckets["tamil"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            buckets["telugu"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            buckets["malayalam"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            buckets["kannada"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            buckets["bengali"] += 1
        else:
            buckets["other"] += 1
    return buckets


def detect_script_type(text: str) -> str:
    """Return dominant script type for text."""
    dist = _script_distribution(text or "")
    total = sum(dist.values())
    if total <= 0:
        return "unknown"
    script, count = max(dist.items(), key=lambda kv: kv[1])
    if count / total < 0.45:
        return "mixed"
    return script


def detect_text_language(text: str, default: str = "en") -> Tuple[str, float, str, str]:
    """
    Detect text language with graceful fallbacks.
    Returns (language_code, confidence, script_type, detector_name).
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return normalize_language_code(default, default="en", allow_auto=False), 0.0, "unknown", "empty"

    script_type = detect_script_type(cleaned)
    script_lang = SCRIPT_TO_LANG.get(script_type, default)

    # Optional detector: langid
    try:
        import langid  # type: ignore
        lang, score = langid.classify(cleaned[:5000])
        lang = normalize_language_code(lang, default=script_lang, allow_auto=False)
        conf = max(0.0, min(1.0, (float(score) + 200.0) / 400.0))  # rough score squashing
        return lang, conf, script_type, "langid"
    except Exception:
        pass

    # Fallback detector: heuristic script mapping.
    return normalize_language_code(script_lang, default=default, allow_auto=False), 0.35, script_type, "script"


def candidate_languages_for_script(script_type: str) -> List[str]:
    """Candidate languages used for ASR probing when auto-mode is unstable."""
    script = (script_type or "").lower()
    if script == "malayalam":
        return ["ml", "ta", "te", "kn", "hi", "en"]
    if script == "devanagari":
        return ["hi", "mr", "ne", "en"]
    if script == "tamil":
        return ["ta", "ml", "te", "en"]
    if script == "telugu":
        return ["te", "kn", "ta", "ml", "en"]
    if script == "kannada":
        return ["kn", "te", "ta", "ml", "en"]
    if script == "arabic":
        return ["ar", "ur", "fa", "en"]
    if script == "cyrillic":
        return ["ru", "uk", "bg", "en"]
    if script == "cjk":
        return ["zh", "ja", "ko", "en"]
    if script == "hangul":
        return ["ko", "en"]
    return ["en", "hi", "ml", "ta", "te", "kn"]
