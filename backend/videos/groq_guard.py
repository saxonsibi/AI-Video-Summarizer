"""
Shared Groq rate-limit guard to prevent repeated 429 hammering.
"""

from __future__ import annotations

import re
import threading
import time

_LOCK = threading.Lock()
_BLOCK_UNTIL_TS = 0.0


def is_groq_rate_limit_error(message: str) -> bool:
    msg = (message or "").lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "rate_limit_exceeded" in msg
        or "tokens per day" in msg
        or "tokens per minute" in msg
        or "too many requests" in msg
    )


def parse_retry_after_seconds(message: str, default_seconds: int = 180) -> int:
    """
    Parse Groq-style retry hints:
    - "Please try again in 3m20.448s"
    - "Please try again in 51s"
    """
    msg = (message or "").lower()

    m = re.search(r"please try again in\s*(\d+)m(\d+(?:\.\d+)?)s", msg)
    if m:
        mins = int(m.group(1))
        secs = float(m.group(2))
        return max(5, int(mins * 60 + secs))

    s = re.search(r"please try again in\s*(\d+(?:\.\d+)?)s", msg)
    if s:
        return max(5, int(float(s.group(1))))

    return max(5, int(default_seconds))


def mark_groq_rate_limited(message: str, default_seconds: int = 180) -> int:
    cooldown = parse_retry_after_seconds(message, default_seconds=default_seconds)
    until_ts = time.time() + cooldown
    with _LOCK:
        global _BLOCK_UNTIL_TS
        _BLOCK_UNTIL_TS = max(_BLOCK_UNTIL_TS, until_ts)
    return cooldown


def is_groq_temporarily_blocked() -> bool:
    with _LOCK:
        return time.time() < _BLOCK_UNTIL_TS


def groq_block_remaining_seconds() -> int:
    with _LOCK:
        rem = _BLOCK_UNTIL_TS - time.time()
    return max(0, int(rem))

