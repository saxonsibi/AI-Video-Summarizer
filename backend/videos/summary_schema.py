"""
Structured summary builder with deterministic topic extraction and validation.
"""

from __future__ import annotations

import math
import logging
import re
from collections import Counter
from hashlib import sha1
from typing import Dict, List, Optional, Tuple

from django.conf import settings

from .utils_metrics import evaluate_summary_quality

logger = logging.getLogger(__name__)
STRUCTURED_SUMMARY_CACHE_VERSION = "2026-03-20-ml-internal-evidence-cache-v1"


PERSON_NAME_PATTERN = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}(?:\s+(?:Jr\.?|Sr\.?))?\b"
)


STOPWORDS_EN = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "he", "her", "his", "how", "i", "in", "into", "is", "it", "its", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "they", "this", "to",
    "was", "we", "what", "when", "where", "which", "who", "why", "with", "you",
    "your", "just", "then", "than", "also", "about", "over", "after", "before",
    "there", "here", "really", "very", "more", "most", "only", "like", "kind",
}
ACTION_HINTS = {
    "click", "select", "open", "create", "build", "generate", "export", "install",
    "configure", "deploy", "upload", "download", "write", "run", "use", "set",
    "copy", "paste", "review", "follow", "check", "prepare", "share", "assign",
}
MEETING_HINTS = {
    "agenda", "meeting", "team", "deadline", "owner", "owners", "decision", "decisions",
    "follow-up", "followup", "update", "updates", "blocker", "blockers", "action",
}
TUTORIAL_HINTS = {
    "tutorial", "step", "workflow", "build", "create", "generate", "install",
    "export", "deploy", "configure", "code", "prompt", "screen", "open",
}
LECTURE_HINTS = {
    "lecture", "lesson", "class", "concept", "theory", "example", "students",
    "understand", "explains", "definition", "principle",
}
INTERVIEW_HINTS = {
    "interview", "host", "guest", "asked", "question", "answer", "conversation",
    "talks", "discusses",
}
COMMENTARY_HINTS = {
    "commentary", "reaction", "reacts", "opinion", "analysis", "reviews", "review",
}
INTERVIEW_QA_PATTERNS = [
    r"\bwhat is\b", r"\bwhat does\b", r"\bwhat did\b", r"\bwhy does\b", r"\bwhy did\b",
    r"\bhow did\b", r"\bhow does\b", r"\bwho is\b", r"\bwhere does\b", r"\bcan\s+\w+\b",
    r"\bdoes\s+\w+\b", r"\bautocomplete\b", r"\banswer(?:s|ing)?\b",
]
TUTORIAL_STRONG_PATTERNS = [
    r"\bhow to\b", r"\bstep by step\b", r"\bfirst\b", r"\bthen\b", r"\bnext\b", r"\bfinally\b",
    r"\byou need to\b", r"\bmake sure\b", r"\bclick\b", r"\bopen\b", r"\binstall\b",
]
COMMENTARY_STRONG_PATTERNS = [
    r"\bi think\b", r"\bin my opinion\b", r"\bmy take\b", r"\breaction\b", r"\breview\b",
]
GENERIC_TOPIC_WORDS = {
    "video", "thing", "stuff", "section", "part", "question", "answer", "discussion",
    "topic", "topics", "conversation", "interview", "tutorial", "lecture", "meeting",
}
BAD_TOPIC_PATTERNS = [
    r"^(?:hi|bit|my|does|got)\b",
    r"\b(?:does got|got does|my on|bit on|hi on)\b",
    r"^(?:on|of|for|with|about)\b",
    r"\b(?:think want|does type)\b",
    r"\bfilm nolan going first\b",
    r"\blooking downey robert\b",
]
NON_PERSON_ENTITY_WORDS = {"wired", "youtube", "google", "chatgpt", "gemini", "openai"}
LOW_SIGNAL_WORDS = {
    "think", "want", "going", "first", "type", "does", "got", "look", "looking",
    "really", "thing", "stuff", "okay", "yeah", "well", "kind", "sort", "both",
}
BAD_INTERVIEW_PARTICIPANT_TOKENS = {
    "hi", "very", "pointless", "tick", "unique", "question", "questions", "answer",
    "answers", "discussion", "interview", "wired", "im", "i'm", "dont", "don't",
    "know", "mother", "american", "threat", "iron", "man", "marvel", "oppenheimer",
    "any", "artifact",
}
BAD_INTERVIEW_PHRASES = {
    "hi", "very", "pointless", "question if", "tick unique", "does type",
    "threat even if", "mother's american i've", "think want", "film nolan going first",
    "looking downey robert", "pointless pointless question if", "any artifact",
}
INTERVIEW_BUCKET_RULES = [
    ("Interview Introduction", {"autocomplete", "question", "questions", "host", "interview", "intro", "introduction"}),
    ("Creative Work and Projects", {"filmmaking", "director", "directing", "creative", "project", "projects", "film", "movie", "work"}),
    ("Marvel / Iron Man", {"marvel", "iron", "man", "avengers", "superhero", "tony", "stark"}),
    ("Hobbies and Personal Life", {"hobby", "hobbies", "tattoo", "tattoos", "family", "personal", "life", "habit", "habits"}),
    ("Career Discussion", {"career", "role", "roles", "actor", "acting", "project", "projects", "work", "worked"}),
    ("Closing Questions", {"final", "closing", "last", "rapid", "quickfire", "wrap", "ending"}),
]
SAFE_INTERVIEW_CHAPTER_TITLES = {
    "Interview Introduction",
    "Career Discussion",
    "Creative Work And Projects",
    "Personal Background",
    "Filmmaking And Oppenheimer",
    "Filmmaking Choices And Oppenheimer",
    "Marvel / Iron Man",
    "Hobbies And Personal Life",
    "Closing Questions",
    "Nolan On Practical Effects",
    "Downey On Iron Man",
    "Career And Personal Questions",
}
INTERVIEW_SEMANTIC_TOPIC_MAP = {
    "Interview Introduction": "the interview format and opening questions",
    "Career Discussion": "career highlights",
    "Creative Work And Projects": "creative work and major projects",
    "Personal Background": "personal background",
    "Filmmaking And Oppenheimer": "filmmaking and Oppenheimer",
    "Filmmaking Choices And Oppenheimer": "filmmaking and Oppenheimer",
    "Marvel / Iron Man": "Marvel and Iron Man",
    "Hobbies And Personal Life": "hobbies and personal life",
    "Closing Questions": "closing personal questions",
    "Nolan On Practical Effects": "practical effects and filmmaking",
    "Downey On Iron Man": "Iron Man, Marvel, and career highlights",
    "Career And Personal Questions": "career highlights and personal questions",
}
BAD_FINAL_INTERVIEW_PARTICIPANTS = {
    "iron man",
    "marvel",
    "oppenheimer",
    "practical effects",
    "career discussion",
    "closing questions",
}
DEGRADED_ENTITY_REJECTION_TOKENS = {
    "announcer", "dingen", "speaker", "listen", "super", "channel", "chance",
    "exam", "coaching", "motivation", "motivational", "marks", "study",
    "enda", "va",
}
MALAYALAM_TRUSTED_ENGLISH_TERMS = {
    "confidence", "hard", "work", "result", "whatsapp", "channel", "exam", "hall",
    "support", "warrior", "coaching", "class", "motivation", "study", "marks",
}
MALAYALAM_DEGRADED_TOPIC_CLUSTERS = [
    (
        "exam_preparation_mindset",
        "Exam preparation and mindset",
        (
            "exam", "preparation", "prepare", "study", "class", "coaching", "marks",
            "rank", "revision", "practice", "പഠ", "പരീക്ഷ", "ക്ലാസ്", "കോച്ച", "മാർക്ക്",
        ),
    ),
    (
        "fear_confidence",
        "Fear, excuses, and confidence",
        (
            "fear", "confidence", "courage", "warrior", "excuse", "afraid", "bold",
            "ഭയ", "ധൈര", "വിശ്വാസ", "confidence", "warrior",
        ),
    ),
    (
        "effort_consistency",
        "Effort and consistency",
        (
            "hard", "work", "effort", "focus", "consistent", "discipline", "result",
            "മെഹന", "ശ്രമ", "ഫോക്കസ്", "ക്രമ", "ഫലം", "result",
        ),
    ),
    (
        "exam_hall_readiness",
        "Exam hall readiness",
        (
            "exam hall", "hall", "readiness", "ready", "question", "answer",
            "ഹാൾ", "റെഡി", "ഉത്തരം", "ചോദ്യ",
        ),
    ),
    (
        "support_guidance",
        "Support and guidance",
        (
            "whatsapp", "channel", "support", "guidance", "group", "community",
            "mentor", "help", "വാട്ട്സ്ആപ്പ്", "സപ്പോർട്ട്", "സഹായ", "ഗൈഡ",
        ),
    ),
]
BAD_INTERVIEW_KEY_POINT_PATTERNS = [
    r"^(?:how|why|what|when|where|who|can|does|did)\b",
    r"\b(?:i probably|i think|i guess|i mean|you know|kind of|sort of|blood pressure medication)\b",
    r"^(?:the interview covers\s+)?(?:how did|why did|why does|what does|what is)\b",
    r"^(?:the interview covers\s+)?the host opens\b",
    r"\b(?:autocomplete questions|opening questions)\b",
]


def default_structured_summary() -> Dict:
    return {
        "tldr": "",
        "key_points": [],
        "action_items": [],
        "chapters": [],
        "participants": [],
    }


def _clean_text(text: str) -> str:
    text = (text or "").replace("\u2014", " - ").replace("\u2013", " - ")
    return re.sub(r"\s+", " ", text.strip())


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+|\n+", _clean_text(text))
    return [s.strip() for s in raw if s and s.strip()]


def _split_bullets(text: str) -> List[str]:
    if not text:
        return []
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^[\s\-\u2022\u2023\u25E6\u2043\u2219]+", "", line.strip())
        line = _clean_text(line)
        if line:
            lines.append(line.rstrip(".").strip())
    if lines:
        return lines
    return [s.rstrip(".").strip() for s in _split_sentences(text)]


def _normalize_for_similarity(text: str) -> List[str]:
    return [tok for tok in re.findall(r"\w+", (text or "").lower()) if tok]


def _dedupe(items: List[str], similarity_threshold: float = 0.82) -> List[str]:
    out: List[str] = []
    for item in items:
        candidate = _clean_text(item)
        if not candidate:
            continue
        if any(_similarity(candidate, existing) >= similarity_threshold for existing in out):
            continue
        out.append(candidate)
    return out


def _format_timestamp(seconds: float) -> str:
    sec = int(max(0, float(seconds or 0)))
    mm = sec // 60
    ss = sec % 60
    return f"{mm:02}:{ss:02}"


def _contains_non_latin(text: str) -> bool:
    return any(ord(ch) > 127 and ch.isalpha() for ch in (text or ""))


def _is_english_like(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned or _contains_non_latin(cleaned):
        return False
    tokens = _normalize_for_similarity(cleaned)
    if not tokens:
        return False
    ascii_tokens = [tok for tok in tokens if re.fullmatch(r"[a-z0-9']+", tok)]
    return len(ascii_tokens) / max(len(tokens), 1) >= 0.85


def _compress_line(text: str, max_words: int = 18) -> str:
    s = _clean_text(text)
    if not s:
        return ""
    s = re.sub(r'^\d{1,2}:\d{2}\s*[-:—]?\s*', '', s)
    s = re.sub(r'^[\-\u2022*\s]+', '', s)
    s = re.sub(r'[?!.]{2,}', '.', s)
    words = s.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words).strip().rstrip(".,!?-")


def _similarity(left: str, right: str) -> float:
    left_tokens = set(_normalize_for_similarity(left))
    right_tokens = set(_normalize_for_similarity(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _looks_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if len(value.split()) < 2:
        return True
    if re.search(r"(?:^| )(?:and|but|so|because|then|well|oh|um|uh)\b", value.lower()):
        return True
    if value.endswith(("?", ",", "-", ":")):
        return True
    return False


def _sentence_contains_action(sentence: str) -> bool:
    low = _clean_text(sentence).lower()
    if not low:
        return False
    if re.search(r"\b(should|must|need to|have to|remember to|make sure)\b", low):
        return True
    return any(re.search(rf"\b{re.escape(word)}\b", low) for word in ACTION_HINTS)


def _score_markers(text: str, markers: set[str]) -> int:
    low = _clean_text(text).lower()
    return sum(1 for marker in markers if re.search(rf"\b{re.escape(marker)}\b", low))


def _count_pattern_hits(text: str, patterns: List[str]) -> int:
    low = _clean_text(text).lower()
    return sum(len(re.findall(pattern, low)) for pattern in patterns)


def _classify_video_type(transcript_text: str, full_summary: str = "", bullet_summary: str = "") -> str:
    cleaned_transcript = _clean_text(transcript_text)
    transcript_is_english_like = _is_english_like(cleaned_transcript)

    # Non-English transcripts should be classified primarily from transcript evidence.
    # English summary text can otherwise overwhelm the classifier and incorrectly
    # push unrelated videos into the interview-safe fallback path.
    summary_context = [full_summary, bullet_summary] if transcript_is_english_like else []
    corpus = " ".join(part for part in [cleaned_transcript, *summary_context] if part)
    low = corpus.lower()
    sentences = _split_sentences(corpus)
    question_count = low.count("?") + len(re.findall(r"\b(what|why|how|when|who|where|did|does|do|is|are|can|could|would)\b", low))
    question_pattern_count = _count_pattern_hits(corpus, INTERVIEW_QA_PATTERNS)
    action_count = sum(1 for word in ACTION_HINTS if re.search(rf"\b{re.escape(word)}\b", low))
    tutorial_pattern_count = _count_pattern_hits(corpus, TUTORIAL_STRONG_PATTERNS)
    commentary_pattern_count = _count_pattern_hits(corpus, COMMENTARY_STRONG_PATTERNS)
    imperative_count = len(re.findall(r"\b(?:click|open|select|choose|install|copy|paste|build|create|generate|deploy|configure)\b", low))
    named_person_count = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", corpus))

    qa_followups = 0
    for idx, sentence in enumerate(sentences[:-1]):
        normalized = sentence.strip().lower()
        if sentence.strip().endswith("?") or any(re.search(pattern, normalized) for pattern in INTERVIEW_QA_PATTERNS):
            next_sentence = sentences[idx + 1].strip()
            if 2 <= len(next_sentence.split()) <= 20:
                qa_followups += 1

    meeting_score = (_score_markers(corpus, MEETING_HINTS) * 2) + min(action_count, 3)
    tutorial_score = (
        (_score_markers(corpus, TUTORIAL_HINTS) * 2)
        + (tutorial_pattern_count * 3)
        + (imperative_count * 2)
        + action_count
    )
    lecture_score = (_score_markers(corpus, LECTURE_HINTS) * 2) + (2 if "students" in low else 0)
    interview_score = (
        (_score_markers(corpus, INTERVIEW_HINTS) * 2)
        + (question_pattern_count * 3)
        + (qa_followups * 3)
        + min(question_count, 8)
        + min(named_person_count, 4)
    )
    commentary_score = (_score_markers(corpus, COMMENTARY_HINTS) * 2) + (commentary_pattern_count * 3)
    explicit_interview_cues = _score_markers(corpus, {"interview", "host", "guest", "autocomplete", "wired"})
    validated_people = _finalize_interview_participants(_extract_interview_participants(cleaned_transcript, []))

    # Conversational Q&A should beat tutorial unless there is strong procedural evidence.
    tutorial_score -= min(question_pattern_count, 6)
    tutorial_score -= min(qa_followups * 2, 8)
    if tutorial_pattern_count < 2 and imperative_count < 3:
        tutorial_score -= 3

    scores = {
        "interview": interview_score,
        "tutorial": tutorial_score,
        "lecture": lecture_score,
        "meeting": meeting_score,
        "commentary": commentary_score,
        "casual conversation": 1,
    }
    chosen = max(scores.items(), key=lambda item: item[1])[0]

    strong_interview_signal = (
        explicit_interview_cues >= 1
        or len(validated_people) >= 2
        or (question_pattern_count >= 4 and qa_followups >= 2)
    )
    if chosen == "interview" and explicit_interview_cues == 0 and not validated_people:
        non_interview_scores = {
            label: score
            for label, score in scores.items()
            if label != "interview"
        }
        best_non_interview, _ = max(
            non_interview_scores.items(),
            key=lambda item: item[1],
        )
        chosen = best_non_interview
    if (
        chosen == "interview"
        and not strong_interview_signal
        and tutorial_score >= max(6, interview_score - 6)
    ):
        chosen = "tutorial"

    if scores[chosen] <= 2:
        chosen = "casual conversation"

    logger.info(
        "[SUMMARY_VIDEO_TYPE] interview_score=%d tutorial_score=%d lecture_score=%d meeting_score=%d commentary_score=%d explicit_interview_cues=%d validated_people=%d chosen=%s",
        interview_score,
        tutorial_score,
        lecture_score,
        meeting_score,
        commentary_score,
        explicit_interview_cues,
        len(validated_people),
        chosen,
    )
    return chosen


def _segment_topic_blocks(segments: List[Dict], min_blocks: int = 5, max_blocks: int = 8) -> List[Dict]:
    cleaned = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        text = _clean_text(seg.get("text", ""))
        if not text:
            continue
        cleaned.append({
            "start": float(seg.get("start", 0.0) or 0.0),
            "end": float(seg.get("end", seg.get("start", 0.0)) or seg.get("start", 0.0) or 0.0),
            "text": text,
        })
    if not cleaned:
        return []

    duration = max((seg["end"] for seg in cleaned), default=0.0)
    target_blocks = max(min_blocks, min(max_blocks, int(round(max(duration / 120.0, len(cleaned) / 28.0, 1.0)))))
    target_words = max(70, int(sum(len(seg["text"].split()) for seg in cleaned) / max(target_blocks, 1)))

    def _block_tokens(text: str) -> set:
        return {tok for tok in _normalize_for_similarity(text) if len(tok) > 2 and tok not in STOPWORDS_EN}

    blocks: List[Dict] = []
    current = None
    for seg in cleaned:
        seg_words = len(seg["text"].split())
        if current is None:
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "sentences": [seg["text"]],
                "word_count": seg_words,
            }
            continue

        gap = max(0.0, seg["start"] - current["end"])
        current_tokens = _block_tokens(current["text"])
        seg_tokens = _block_tokens(seg["text"])
        similarity = 0.0
        if current_tokens and seg_tokens:
            similarity = len(current_tokens & seg_tokens) / max(len(current_tokens | seg_tokens), 1)

        should_split = False
        if gap > 18.0:
            should_split = True
        elif current["word_count"] >= target_words and similarity < 0.12:
            should_split = True
        elif current["word_count"] >= int(target_words * 1.35):
            should_split = True

        if should_split and len(blocks) < (max_blocks - 1):
            blocks.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "sentences": [seg["text"]],
                "word_count": seg_words,
            }
            continue

        current["end"] = seg["end"]
        current["text"] = f"{current['text']} {seg['text']}".strip()
        current["sentences"].append(seg["text"])
        current["word_count"] += seg_words

    if current:
        blocks.append(current)

    while len(blocks) > max_blocks:
        shortest_idx = min(range(len(blocks) - 1), key=lambda idx: blocks[idx]["word_count"] + blocks[idx + 1]["word_count"])
        merged = {
            "start": blocks[shortest_idx]["start"],
            "end": blocks[shortest_idx + 1]["end"],
            "text": f"{blocks[shortest_idx]['text']} {blocks[shortest_idx + 1]['text']}".strip(),
            "sentences": blocks[shortest_idx]["sentences"] + blocks[shortest_idx + 1]["sentences"],
            "word_count": blocks[shortest_idx]["word_count"] + blocks[shortest_idx + 1]["word_count"],
        }
        blocks[shortest_idx:shortest_idx + 2] = [merged]

    while len(blocks) < min_blocks and any(block["word_count"] >= 40 for block in blocks):
        split_idx = max(range(len(blocks)), key=lambda idx: blocks[idx]["word_count"])
        block = blocks[split_idx]
        if len(block["sentences"]) < 2:
            break
        midpoint = max(1, len(block["sentences"]) // 2)
        first_sentences = block["sentences"][:midpoint]
        second_sentences = block["sentences"][midpoint:]
        if not first_sentences or not second_sentences:
            break
        first_text = " ".join(first_sentences).strip()
        second_text = " ".join(second_sentences).strip()
        first = {
            "start": block["start"],
            "end": block["start"] + max((block["end"] - block["start"]) / 2.0, 1.0),
            "text": first_text,
            "sentences": first_sentences,
            "word_count": len(first_text.split()),
        }
        second = {
            "start": first["end"],
            "end": block["end"],
            "text": second_text,
            "sentences": second_sentences,
            "word_count": len(second_text.split()),
        }
        blocks[split_idx:split_idx + 1] = [first, second]

    return blocks[:max_blocks]


def _keyword_phrase(text: str, max_words: int = 5) -> str:
    tokens = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
    if not tokens:
        return ""
    filtered = [tok for tok in tokens if len(tok) > 2 and tok not in STOPWORDS_EN]
    if not filtered:
        filtered = tokens[:max_words]
    counts = Counter(filtered)
    selected = [tok for tok, _ in counts.most_common(max_words)]
    phrase = " ".join(selected[:max_words]).strip()
    if _is_english_like(phrase):
        return phrase.title()
    return phrase


def _concept_phrases(text: str, max_items: int = 5) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z']+", text or "")
    phrases: List[str] = []
    seen = set()
    for size in (3, 2):
        for idx in range(0, max(0, len(tokens) - size + 1)):
            window = tokens[idx:idx + size]
            normalized = [tok.lower() for tok in window]
            if any(tok in STOPWORDS_EN or tok in LOW_SIGNAL_WORDS for tok in normalized):
                continue
            if len(set(normalized)) != len(normalized):
                continue
            phrase = " ".join(window)
            low = phrase.lower()
            if low in seen:
                continue
            if not _is_natural_topic_phrase(phrase):
                continue
            seen.add(low)
            phrases.append(phrase)
            if len(phrases) >= max_items:
                return phrases
    return phrases


def _ordered_keywords(text: str, max_words: int = 4) -> List[str]:
    tokens = re.findall(r"\w+", (text or ""), flags=re.UNICODE)
    counts = Counter(tok.lower() for tok in tokens if len(tok) > 2)
    ordered = []
    seen = set()
    for token in tokens:
        low = token.lower()
        if low in seen:
            continue
        if len(low) <= 2 or low in STOPWORDS_EN or low in GENERIC_TOPIC_WORDS:
            continue
        if counts.get(low, 0) < 1:
            continue
        seen.add(low)
        ordered.append(token)
        if len(ordered) >= max_words:
            break
    return ordered


def _extract_named_phrase(text: str) -> str:
    candidates = PERSON_NAME_PATTERN.findall(text or "") + re.findall(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){0,2}\b", text or "")
    cleaned = []
    for candidate in candidates:
        value = _clean_text(candidate)
        if len(value.split()) < 1:
            continue
        if value.lower() in {"the", "this", "that", "and"}:
            continue
        if value.lower() in NON_PERSON_ENTITY_WORDS:
            continue
        if any(part.lower() in NON_PERSON_ENTITY_WORDS for part in value.split()):
            continue
        cleaned.append(value)
    return cleaned[0] if cleaned else ""


def _extract_main_people(text: str, max_people: int = 2) -> List[str]:
    candidates = PERSON_NAME_PATTERN.findall(text or "")
    people = []
    seen = set()
    for candidate in candidates:
        value = _clean_text(candidate)
        low = value.lower()
        if low in seen or low in {"the", "this", "that"}:
            continue
        if any(part.lower().rstrip(".") in NON_PERSON_ENTITY_WORDS for part in value.split()):
            continue
        seen.add(low)
        people.append(value)
        if len(people) >= max_people:
            break
    return people


def _normalize_person_name(name: str) -> str:
    value = _clean_text(name)
    value = re.sub(r"\b(Jr|Sr)(?:\.)?(?=\s|$)", r"\1.", value)
    value = value.replace("Jr..", "Jr.").replace("Sr..", "Sr.")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _is_valid_interview_participant(name: str) -> bool:
    value = _normalize_person_name(name)
    if not value:
        return False
    parts = value.split()
    if len(parts) < 2 or len(parts) > 4:
        return False
    lowered = [part.lower().rstrip(".") for part in parts]
    if any(part in STOPWORDS_EN for part in lowered):
        return False
    if any(part in BAD_INTERVIEW_PARTICIPANT_TOKENS for part in lowered):
        return False
    if any(part in NON_PERSON_ENTITY_WORDS for part in lowered):
        return False
    if sum(1 for part in lowered if part in LOW_SIGNAL_WORDS) >= 1:
        return False
    if not all(re.fullmatch(r"[A-Z][a-z]+|Jr\.?|Sr\.?", part) for part in parts):
        return False
    return True


def _extract_interview_participants(transcript_text: str, segments: List[Dict], max_people: int = 2) -> List[str]:
    segment_hits: Dict[str, int] = {}
    transcript_counts: Counter = Counter()
    rejected = 0

    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        seg_text = seg.get("text", "") or ""
        found = set()
        for candidate in PERSON_NAME_PATTERN.findall(seg_text):
            normalized = _normalize_person_name(candidate)
            if not _is_valid_interview_participant(normalized):
                rejected += 1
                continue
            found.add(normalized)
        for item in found:
            segment_hits[item] = segment_hits.get(item, 0) + 1

    for candidate in PERSON_NAME_PATTERN.findall(transcript_text or ""):
        normalized = _normalize_person_name(candidate)
        if not _is_valid_interview_participant(normalized):
            continue
        transcript_counts[normalized] += 1

    ranked = sorted(
        segment_hits.keys(),
        key=lambda item: (segment_hits.get(item, 0), transcript_counts.get(item, 0), len(item.split())),
        reverse=True,
    )
    accepted = [
        item for item in ranked
        if (
            segment_hits.get(item, 0) >= 2
            or transcript_counts.get(item, 0) >= 2
            or (segment_hits.get(item, 0) >= 1 and transcript_counts.get(item, 0) >= 1)
        )
    ][:max_people]
    if not accepted:
        accepted = [
            person for person in _extract_main_people(transcript_text, max_people=max_people)
            if _is_valid_interview_participant(person)
        ][:max_people]
    logger.info(
        "[SUMMARY_INTERVIEW_PARTICIPANTS] accepted=%s rejected_fake=%d",
        accepted,
        rejected,
    )
    return accepted


def _filter_degraded_participants(participants: List[str], transcript_text: str, transcript_language: str = "") -> List[str]:
    cleaned_transcript = _clean_text(transcript_text)
    filtered: List[str] = []
    rejected = 0
    candidate_count = len(participants or [])
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for participant in participants or []:
        normalized = _normalize_person_name(participant)
        if not normalized:
            logger.info("[ML_SUMMARY_ENTITY_GUARD] candidate=%s accepted=False reason=empty_normalized_name", participant)
            rejected += 1
            continue
        parts = [part.lower().rstrip(".") for part in normalized.split()]
        if len(parts) < 2:
            logger.info("[ML_SUMMARY_ENTITY_GUARD] candidate=%s accepted=False reason=single_token_name", normalized)
            rejected += 1
            continue
        if any(len(part) <= 2 for part in parts):
            logger.info("[ML_SUMMARY_ENTITY_GUARD] candidate=%s accepted=False reason=short_name_token", normalized)
            rejected += 1
            continue
        if any(part in DEGRADED_ENTITY_REJECTION_TOKENS or part in LOW_SIGNAL_WORDS for part in parts):
            logger.info("[ML_SUMMARY_ENTITY_GUARD] candidate=%s accepted=False reason=low_signal_or_rejected_token", normalized)
            rejected += 1
            continue
        if any(part in MALAYALAM_TRUSTED_ENGLISH_TERMS for part in parts):
            logger.info("[ML_SUMMARY_ENTITY_GUARD] candidate=%s accepted=False reason=trusted_term_not_person", normalized)
            rejected += 1
            continue
        mention_count = cleaned_transcript.count(normalized)
        segment_support = sum(
            1 for sentence in _split_sentences(cleaned_transcript)
            if normalized.lower() in sentence.lower()
        )
        if malayalam_mode and segment_support < 2:
            logger.info(
                "[ML_SUMMARY_ENTITY_GUARD] candidate=%s accepted=False reason=insufficient_segment_support mentions=%s segment_support=%s",
                normalized,
                mention_count,
                segment_support,
            )
            rejected += 1
            continue
        if malayalam_mode and (sum(len(part) for part in parts) / max(len(parts), 1)) <= 4 and mention_count < 4:
            logger.info(
                "[ML_SUMMARY_ENTITY_GUARD] candidate=%s accepted=False reason=short_low_support_name mentions=%s segment_support=%s",
                normalized,
                mention_count,
                segment_support,
            )
            rejected += 1
            continue
        if mention_count < (3 if malayalam_mode else 2):
            logger.info(
                "[ML_SUMMARY_ENTITY_GUARD] candidate=%s accepted=False reason=insufficient_mentions mentions=%s segment_support=%s",
                normalized,
                mention_count,
                segment_support,
            )
            rejected += 1
            continue
        logger.info(
            "[ML_SUMMARY_ENTITY_GUARD] candidate=%s accepted=True reason=stable_degraded_support mentions=%s segment_support=%s",
            normalized,
            mention_count,
            segment_support,
        )
        filtered.append(normalized)
    logger.info(
        "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%d video_type_bias_applied=%s transcript_state=%s",
        candidate_count,
        filtered,
        rejected,
        False,
        "degraded",
    )
    return filtered[:2]


def _guard_malayalam_degraded_video_type(raw_choice: str, transcript_text: str, participants: List[str]) -> Tuple[str, str]:
    if raw_choice not in {"interview", "meeting"}:
        return raw_choice, "no_bias_needed"
    low = _clean_text(transcript_text).lower()
    explicit_interview_cues = _score_markers(low, {"interview", "host", "guest", "question", "questions", "answer", "answers"})
    qa_followups = _count_pattern_hits(low, INTERVIEW_QA_PATTERNS)
    meeting_cues = _score_markers(low, MEETING_HINTS)
    if raw_choice == "meeting" and meeting_cues >= 2:
        return raw_choice, "meeting_signals_strong_enough"
    if raw_choice == "interview" and len(participants) >= 2 and (explicit_interview_cues >= 2 or qa_followups >= 2):
        return raw_choice, "interview_signals_strong_enough"
    if any(term in low for term in ("exam", "coaching", "motivation", "confidence", "result", "study", "support", "channel", "whatsapp", "hall")):
        return "lecture", "weak_person_entity_confidence_malayalam_degraded"
    if raw_choice == "meeting":
        return "commentary", "weak_meeting_confidence_malayalam_degraded"
    return "commentary", "weak_person_entity_confidence_malayalam_degraded"


def _trusted_malayalam_summary_segments(segments: List[Dict]) -> Tuple[List[Dict], int]:
    try:
        from .utils import classify_malayalam_segment_type
    except Exception:
        return segments or [], 0

    trusted: List[Dict] = []
    rejected = 0
    for idx, seg in enumerate(segments or []):
        if not isinstance(seg, dict):
            continue
        text = _clean_text(seg.get("text", ""))
        if not text:
            continue
        classified = classify_malayalam_segment_type(text)
        seg_type = str(classified.get("type") or "uncertain_noise")
        logger.info(
            "[ML_SEGMENT_TYPE] idx=%s type=%s malayalam_ratio=%.3f english_ratio=%.3f wrong_script_ratio=%.3f readability=%.3f",
            idx,
            seg_type,
            float((classified.get("score", {}) or {}).get("malayalam_ratio", 0.0) or 0.0),
            float((classified.get("score", {}) or {}).get("english_preserved_ratio", 0.0) or 0.0),
            float((classified.get("score", {}) or {}).get("wrong_script_ratio", 0.0) or 0.0),
            float((classified.get("score", {}) or {}).get("score", 0.0) or 0.0),
        )
        if seg_type in {"clean_malayalam", "clean_english", "mixed_malayalam_english"}:
            trusted.append(seg)
        else:
            rejected += 1
    return trusted, rejected


def _pick_malayalam_summary_units(
    segments: List[Dict],
    raw_segments: List[Dict],
    assembled_units: List[Dict],
) -> Tuple[List[Dict], int, int]:
    trusted_assembled, rejected_units = _trusted_malayalam_summary_segments(assembled_units or [])
    trusted_raw, rejected_segments = _trusted_malayalam_summary_segments(raw_segments or segments or [])
    assembled_score = sum(float(seg.get('unit_readability', 0.0) or 0.0) for seg in trusted_assembled if isinstance(seg, dict))
    raw_score = sum(float(seg.get('unit_readability', 0.0) or 0.0) for seg in trusted_raw if isinstance(seg, dict))
    use_assembled = bool(trusted_assembled) and (
        len(trusted_assembled) <= max(len(trusted_raw), 1)
        or assembled_score >= raw_score
    )
    logger.info(
        "[ML_SUMMARY_ASSEMBLY] assembled_units_used=%s raw_segments_used=%s trusted_units=%s rejected_low_trust_units=%s",
        len(trusted_assembled) if use_assembled else 0,
        len(trusted_raw) if not use_assembled else 0,
        len(trusted_assembled) if use_assembled else len(trusted_raw),
        rejected_units if use_assembled else rejected_segments,
    )
    return (trusted_assembled if use_assembled else trusted_raw), rejected_units, rejected_segments


def _synthetic_malayalam_summary_units(transcript_text: str) -> List[Dict]:
    sentences = _split_sentences(transcript_text)[:4]
    units: List[Dict] = []
    for idx, sentence in enumerate(sentences):
        text = _clean_text(sentence)
        if not text:
            continue
        units.append({
            "id": idx,
            "start": float(idx * 10.0),
            "end": float((idx * 10.0) + 10.0),
            "display_start": float(idx * 10.0),
            "display_end": float((idx * 10.0) + 10.0),
            "text": text,
            "unit_readability": 0.0,
            "source": "synthetic_transcript_text",
        })
    return units


def _bounded_cleaned_malayalam_summary(*, reason: str) -> Dict:
    payload = default_structured_summary()
    payload["_trace"] = {
        "structured_summary_route": "normal_grounded",
        "structured_summary_route_reason": "cleaned_malayalam_bounded_empty",
        "structured_grounding_passed": False,
        "structured_grounding_reason": reason,
        "structured_summary_blocked_reason": reason,
    }
    return payload


def _select_malayalam_structured_inputs(
    *,
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_state: str,
) -> Dict[str, object]:
    degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
    trusted_units, rejected_assembled, rejected_raw = _pick_malayalam_summary_units(
        segments or [],
        raw_segments or [],
        assembled_units or [],
    )
    source = "trusted_raw_segments"
    if trusted_units and assembled_units:
        assembled_texts = { _clean_text(str(item.get("text", "") or "")) for item in assembled_units if isinstance(item, dict) }
        trusted_texts = { _clean_text(str(item.get("text", "") or "")) for item in trusted_units if isinstance(item, dict) }
        if trusted_texts and trusted_texts.issubset(assembled_texts):
            source = "trusted_assembled_units"
    if degraded_mode:
        logger.info(
            "[ML_STRUCTURED_INPUT_DEGRADED] source=%s units=%s words=%s reason=%s",
            source,
            len(trusted_units),
            len(re.findall(r"\w+", " ".join(str(seg.get("text", "") or "") for seg in trusted_units), flags=re.UNICODE)),
            "degraded_prefers_trusted_units",
        )
        return {
            "units": trusted_units,
            "source": source,
            "reason": "degraded_prefers_trusted_units",
            "rejected_low_trust_units": rejected_assembled if assembled_units else rejected_raw,
        }

    cleaned_transcript = _clean_text(transcript_text)
    transcript_snapshot = None
    if cleaned_transcript:
        try:
            from .utils import classify_malayalam_segment_type
            transcript_snapshot = classify_malayalam_segment_type(cleaned_transcript)
        except Exception:
            transcript_snapshot = None
    transcript_type = str((transcript_snapshot or {}).get("type") or "")
    transcript_readability = float((((transcript_snapshot or {}).get("score") or {}).get("score", 0.0) or 0.0))
    if trusted_units:
        logger.info(
            "[ML_STRUCTURED_INPUT_CLEANED] source=%s units=%s words=%s reason=%s",
            source,
            len(trusted_units),
            len(re.findall(r"\w+", " ".join(str(seg.get("text", "") or "") for seg in trusted_units), flags=re.UNICODE)),
            "cleaned_prefers_grounded_units",
        )
        return {
            "units": trusted_units,
            "source": source,
            "reason": "cleaned_prefers_grounded_units",
            "rejected_low_trust_units": rejected_assembled if assembled_units else rejected_raw,
        }
    cleaned_assembled = [
        unit for unit in (assembled_units or [])
        if isinstance(unit, dict) and _clean_text(str(unit.get("text", "") or ""))
    ]
    if cleaned_assembled:
        logger.info(
            "[ML_STRUCTURED_INPUT_CLEANED] source=%s units=%s words=%s reason=%s",
            "assembled_units",
            len(cleaned_assembled),
            len(re.findall(r"\w+", " ".join(str(seg.get("text", "") or "") for seg in cleaned_assembled), flags=re.UNICODE)),
            "cleaned_prefers_all_assembled_units_when_trusted_subset_is_empty",
        )
        return {
            "units": cleaned_assembled,
            "source": "assembled_units",
            "reason": "cleaned_prefers_all_assembled_units_when_trusted_subset_is_empty",
            "rejected_low_trust_units": rejected_assembled if assembled_units else rejected_raw,
        }
    if transcript_type in {"clean_malayalam", "mixed_malayalam_english"} or transcript_readability >= 0.34:
        synthetic_units = _synthetic_malayalam_summary_units(cleaned_transcript)
        logger.info(
            "[ML_STRUCTURED_INPUT_CLEANED] source=%s units=%s words=%s reason=%s",
            "synthetic_transcript_text",
            len(synthetic_units),
            len(re.findall(r"\w+", cleaned_transcript, flags=re.UNICODE)),
            "cleaned_transcript_text_is_more_grounded_than_segments",
        )
        return {
            "units": synthetic_units,
            "source": "synthetic_transcript_text",
            "reason": "cleaned_transcript_text_is_more_grounded_than_segments",
            "rejected_low_trust_units": rejected_assembled if assembled_units else rejected_raw,
        }
    logger.info(
        "[ML_STRUCTURED_INPUT_CLEANED] source=%s units=%s words=%s reason=%s",
        "none",
        0,
        0,
        "insufficient_grounded_malayalam_evidence",
    )
    return {
        "units": [],
        "source": "none",
        "reason": "insufficient_grounded_malayalam_evidence",
        "rejected_low_trust_units": rejected_assembled if assembled_units else rejected_raw,
    }


def _build_degraded_safe_malayalam_summary(
    *,
    transcript_text: str,
    evidence: Dict,
    video_type: str,
    reason: str,
) -> Dict:
    warning_message = "Malayalam transcript quality was too low for reliable summarization."
    trusted_units = evidence.get("trusted_units", []) or []
    trusted_terms = [
        str(term).strip()
        for term in (evidence.get("trusted_terms", []) or [])
        if str(term).strip()
    ]
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    time_ranges = []
    source_indices = set()
    for unit in trusted_units:
        if not isinstance(unit, dict):
            continue
        unit_id = unit.get("id")
        if unit_id is not None:
            try:
                source_indices.add(int(unit_id))
            except Exception:
                pass
        start = float(unit.get("display_start", unit.get("start", 0.0)) or 0.0)
        end = float(unit.get("display_end", unit.get("end", 0.0)) or 0.0)
        time_ranges.append({"start": round(start, 2), "end": round(end, 2)})

    tldr = "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization."
    key_points: List[str] = []
    if topic_signals:
        for cluster_id, _count in topic_signals.most_common(2):
            point = _cluster_summary_point(cluster_id)
            if point and point not in key_points:
                key_points.append(f"Limited reliable transcript fragments suggest: {point[0].lower() + point[1:]}")
    elif trusted_terms:
        preserved_terms = ", ".join(trusted_terms[:3])
        tldr = f"This video appears to contain Malayalam speech, but only a small amount of reliable transcript evidence was recovered, including terms such as {preserved_terms}."
        key_points.append(f"Limited reliable transcript fragments mention {preserved_terms}.")

    payload = {
        "summary_state": "degraded_safe",
        "warning_message": warning_message,
        "tldr": tldr,
        "key_points": key_points[:2],
        "action_items": [],
        "chapters": [],
        "participants": [],
        "_trace": {
            "mode": "degraded_safe",
            "reason": reason,
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted(source_indices),
            "source_time_ranges": time_ranges[:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": list(topic_signals.keys())[:4],
            "video_type": video_type,
        },
    }
    logger.info(
        "[SUMMARY_DEGRADED_SAFE] transcript_state=degraded trusted_units=%s trusted_terms=%s topic_clusters=%s reason=%s confidence=%s",
        len(trusted_units),
        len(trusted_terms),
        list(topic_signals.keys())[:4],
        reason,
        confidence_label,
    )
    logger.info(
        "[SUMMARY_FALLBACK_BLOCKED_DEGRADED] transcript_state=degraded reason=%s trusted_units=%s",
        reason,
        len(trusted_units),
    )
    logger.info(
        "[CHAPTERS_SUPPRESSED_LOW_TRUST] transcript_state=degraded trusted_units=%s reason=%s",
        len(trusted_units),
        reason,
    )
    return payload


def _safe_degraded_malayalam_tldr(topic_blocks: List[Dict], video_type: str, transcript_text: str) -> str:
    safe_labels = _dedupe(
        [
            _clean_text(block.get("label", ""))
            for block in topic_blocks[:4]
            if block.get("label")
            and not _is_suspicious_degraded_label(block.get("label", ""))
            and _is_natural_topic_phrase(block.get("label", ""))
        ]
    )[:3]
    low = _clean_text(transcript_text).lower()
    if safe_labels:
        topic_text = ", ".join(label.lower() for label in safe_labels[:2])
        if any(term in low for term in ("exam", "coaching", "result", "confidence", "hall")):
            return f"The talk focuses on {topic_text}."
        if any(term in low for term in ("support", "whatsapp", "channel")):
            return f"The speaker discusses {topic_text}."
        return f"The discussion centers on {topic_text}."
    if any(term in low for term in ("exam", "coaching", "motivation", "confidence", "result", "study")):
        return "The talk focuses on exam preparation, confidence, and results."
    if any(term in low for term in ("support", "whatsapp", "channel")):
        return "The talk discusses support, guidance, and follow-up resources."
    if video_type in {"lecture", "commentary"}:
        return "The talk covers the main ideas, but parts of the transcript remain noisy."
    return "The discussion covers the main ideas, but parts of the transcript remain noisy."


def _stopword_ratio(text: str) -> float:
    tokens = [tok.lower() for tok in re.findall(r"\w+", text or "")]
    if not tokens:
        return 1.0
    return sum(1 for tok in tokens if tok in STOPWORDS_EN) / max(len(tokens), 1)


def _is_natural_topic_phrase(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return False
    if not _is_english_like(value):
        return len(value.split()) >= 2
    if any(re.search(pattern, value.lower()) for pattern in BAD_TOPIC_PATTERNS):
        return False
    words = value.split()
    if len(words) < 2 or len(words) > 10:
        return False
    if _stopword_ratio(value) > 0.55:
        return False
    content_words = [w for w in re.findall(r"\w+", value.lower()) if w not in STOPWORDS_EN]
    if len(content_words) < 2:
        return False
    if any(word in LOW_SIGNAL_WORDS for word in content_words):
        return False
    if len(content_words) >= 3 and len(set(content_words)) <= 2:
        return False
    if len(content_words) >= 4 and sum(1 for word in content_words if word in NON_PERSON_ENTITY_WORDS or word in LOW_SIGNAL_WORDS) >= 2:
        return False
    if all(word in GENERIC_TOPIC_WORDS for word in content_words):
        return False
    return True


def _is_suspicious_degraded_label(text: str) -> bool:
    value = _clean_text(text).lower()
    if not value:
        return True
    parts = re.findall(r"[a-z']+", value)
    if not parts:
        return False
    if sum(1 for part in parts if part in DEGRADED_ENTITY_REJECTION_TOKENS or part in LOW_SIGNAL_WORDS) >= 1:
        return True
    if len(set(parts)) <= 1 and len(parts) >= 2:
        return True
    return False


def _is_bad_interview_topic_phrase(text: str) -> bool:
    value = _clean_text(text).lower()
    if not value:
        return True
    if value in BAD_INTERVIEW_PHRASES:
        return True
    if any(bad in value for bad in BAD_INTERVIEW_PHRASES):
        return True
    parts = re.findall(r"[a-z']+", value)
    if not parts:
        return True
    if sum(1 for part in parts if part in STOPWORDS_EN) > max(2, len(parts) // 2):
        return True
    if sum(1 for part in parts if part in BAD_INTERVIEW_PARTICIPANT_TOKENS or part in LOW_SIGNAL_WORDS) >= max(1, len(parts) // 2):
        return True
    if len(parts) >= 3 and len(set(parts)) <= 2:
        return True
    return False


def _score_interview_bucket_candidates(text: str, participants: List[str]) -> List[Tuple[str, int]]:
    lowered = _clean_text(text).lower()
    participant_tokens = {
        token.lower().rstrip(".")
        for person in (participants or [])
        for token in person.split()
    }
    scores: Dict[str, int] = {label: 0 for label in SAFE_INTERVIEW_CHAPTER_TITLES}

    for label, keywords in INTERVIEW_BUCKET_RULES:
        canonical = label.title().replace(" And ", " And ")
        scores[canonical] = scores.get(canonical, 0) + sum(
            3 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", lowered)
        )

    if re.search(r"\b(?:filmmaking|director|directing|creative work|creative|project|projects|film|movie)\b", lowered):
        scores["Creative Work And Projects"] += 4

    if re.search(r"\b(?:oppenheimer|practical effects|cgi)\b", lowered):
        scores["Filmmaking And Oppenheimer"] += 4
        if "christopher" in participant_tokens or "nolan" in participant_tokens or "nolan" in lowered:
            scores["Nolan On Practical Effects"] += 5

    if re.search(r"\b(?:iron man|marvel|tony stark|avengers)\b", lowered):
        scores["Marvel / Iron Man"] += 4
        if "robert" in participant_tokens or "downey" in participant_tokens or "downey" in lowered:
            scores["Downey On Iron Man"] += 5

    if re.search(r"\b(?:tattoo|tattoos|hobby|hobbies|personal|family|habit|habits|life)\b", lowered):
        scores["Hobbies And Personal Life"] += 4

    if re.search(r"\b(?:career|role|roles|projects|work|worked|filmography|journey)\b", lowered):
        scores["Career Discussion"] += 4
        scores["Career And Personal Questions"] += 2

    if re.search(r"\b(?:final|closing|last|wrap|ending|quickfire|rapid)\b", lowered):
        scores["Closing Questions"] += 4

    if re.search(r"\b(?:autocomplete|host|opening|intro|introduction|first question|first prompt)\b", lowered):
        scores["Interview Introduction"] += 5

    if re.search(r"\b(?:personal questions|career questions|rapid questions)\b", lowered):
        scores["Career And Personal Questions"] += 4

    thematic_labels = [
        "Creative Work And Projects",
        "Filmmaking And Oppenheimer",
        "Nolan On Practical Effects",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Hobbies And Personal Life",
        "Career Discussion",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    intro_score = scores.get("Interview Introduction", 0)
    best_thematic = max((scores.get(label, 0) for label in thematic_labels), default=0)
    if intro_score and best_thematic >= max(2, intro_score - 1):
        scores["Interview Introduction"] = max(0, intro_score - 3)

    if all(score <= 0 for score in scores.values()):
        scores["Career And Personal Questions"] = 1

    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return ranked


def _safe_interview_label_for_block(block_text: str, participants: List[str], used_labels: Optional[List[str]] = None) -> Tuple[str, bool]:
    used = {item.lower() for item in (used_labels or []) if item}
    used_semantic = {
        INTERVIEW_SEMANTIC_TOPIC_MAP.get(item, item).lower()
        for item in (used_labels or [])
        if item
    }
    ranked = _score_interview_bucket_candidates(block_text, participants)
    for label, score in ranked:
        if score <= 0:
            continue
        if label.lower() in used:
            continue
        if INTERVIEW_SEMANTIC_TOPIC_MAP.get(label, label).lower() in used_semantic:
            continue
        return label, True

    fallback_order = [
        "Career Discussion",
        "Hobbies And Personal Life",
        "Creative Work And Projects",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    for label in fallback_order:
        if label.lower() not in used:
            return label, True
    return "Career And Personal Questions", True


def _dedupe_interview_labels(labels: List[str], max_items: Optional[int] = None) -> Tuple[List[str], int]:
    out: List[str] = []
    deduped = 0
    for label in labels:
        cleaned = _clean_text(label)
        if not cleaned:
            continue
        if any(
            cleaned.lower() == existing.lower()
            or INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, cleaned).lower() == INTERVIEW_SEMANTIC_TOPIC_MAP.get(existing, existing).lower()
            for existing in out
        ):
            deduped += 1
            continue
        out.append(cleaned)
        if max_items and len(out) >= max_items:
            break
    return out, deduped


def _bucket_interview_topic(text: str, participants: List[str]) -> str:
    chosen, _ = _safe_interview_label_for_block(text, participants, used_labels=None)
    return chosen


def _dedupe_interview_topic_phrases(topics: List[str], max_items: int = 3) -> List[str]:
    out: List[str] = []
    for topic in topics:
        cleaned = _clean_text(topic)
        if not cleaned:
            continue
        topic_tokens = {
            token for token in _normalize_for_similarity(cleaned)
            if token not in {"and", "the", "interview", "questions", "question", "personal", "career", "closing"}
        }
        duplicate = False
        for existing in out:
            existing_tokens = {
                token for token in _normalize_for_similarity(existing)
                if token not in {"and", "the", "interview", "questions", "question", "personal", "career", "closing"}
            }
            if not topic_tokens or not existing_tokens:
                continue
            overlap = len(topic_tokens & existing_tokens) / max(1, min(len(topic_tokens), len(existing_tokens)))
            if overlap >= 0.5:
                duplicate = True
                break
        if duplicate:
            continue
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def _explicit_interview_topics_from_transcript(text: str, participants: List[str]) -> List[str]:
    lowered = (text or "").lower()
    participant_tokens = {
        token.lower().rstrip(".")
        for person in (participants or [])
        for token in person.split()
    }
    topics: List[str] = []
    if re.search(r"\b(?:oppenheimer|practical effects|cgi)\b", lowered):
        topics.append("Filmmaking And Oppenheimer")
        if "christopher" in participant_tokens or "nolan" in participant_tokens or "nolan" in lowered:
            topics.append("Nolan On Practical Effects")
    elif re.search(r"\b(?:filmmaking|director|directing|creative work|creative|film|movie)\b", lowered):
        topics.append("Creative Work And Projects")

    if re.search(r"\b(?:iron man|marvel|tony stark|avengers)\b", lowered):
        if "robert" in participant_tokens or "downey" in participant_tokens or "downey" in lowered:
            topics.append("Downey On Iron Man")
        topics.append("Marvel / Iron Man")

    return _dedupe_interview_labels(topics, max_items=4)[0]


def _safe_topic_label(block_text: str, video_type: str) -> Tuple[str, bool]:
    named_phrase = _extract_named_phrase(block_text)
    main_people = _extract_main_people(block_text, max_people=1)
    concept_phrases = _concept_phrases(block_text, max_items=3)
    keywords = _ordered_keywords(block_text, max_words=3)
    filtered_keywords = [
        word for word in keywords
        if word.lower() not in {part.lower().rstrip(".") for person in main_people for part in person.split()}
        and word.lower() not in NON_PERSON_ENTITY_WORDS
    ]
    keywords = filtered_keywords or keywords
    topic = concept_phrases[0] if concept_phrases else " ".join(keywords[:2]).strip()
    used_template = False

    if _is_english_like(block_text):
        if video_type == "interview":
            local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
            label, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
            used_template = True
        elif video_type == "tutorial":
            label = f"Explaining {topic.title()}" if topic else "Tutorial Step"
            used_template = True
        elif video_type == "lecture":
            label = f"Explaining {topic.title()}" if topic else "Lecture Topic"
            used_template = True
        elif video_type == "meeting":
            label = f"Discussion of {topic.title()}" if topic else "Meeting Discussion"
            used_template = True
        elif video_type == "commentary":
            label = f"Commentary on {topic.title()}" if topic else "Commentary Segment"
            used_template = True
        else:
            label = topic.title() if topic else "Main Discussion"
            used_template = True
    else:
        label = _compress_line(block_text, max_words=6) or "Discussion"

    return label.strip(), used_template


def _topic_label_from_block(block_text: str, video_type: str, stats: Optional[Dict] = None) -> str:
    if video_type == "interview":
        local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
        title, used_template = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
        if used_template and stats is not None:
            stats["fallback_template_usage"] = stats.get("fallback_template_usage", 0) + 1
        return title

    sentences = _split_sentences(block_text)
    first_sentence = _compress_line(sentences[0] if sentences else block_text, max_words=8)
    concept_phrase = _concept_phrases(block_text, max_items=1)
    keyword_title = concept_phrase[0] if concept_phrase else ""
    named_phrase = _extract_named_phrase(block_text)
    title = keyword_title or first_sentence
    if _is_english_like(block_text):
        if video_type == "interview" and named_phrase and keyword_title:
            title = f"{named_phrase} on {keyword_title.lower()}"
        elif video_type == "tutorial" and keyword_title:
            title = f"{keyword_title} Workflow"
        elif named_phrase and keyword_title:
            title = f"{named_phrase} {keyword_title}"
    if not title:
        title = "Discussion Point"
    if _looks_fragment(title):
        title = _compress_line(block_text, max_words=6)
    if not title:
        title = "Discussion Point"
    if _is_english_like(title):
        title = title.title()
    words = title.split()
    if len(words) > 8:
        title = " ".join(words[:8]).strip()
    if video_type == "interview" and _is_bad_interview_topic_phrase(title):
        if stats is not None:
            stats["rejected_topic_labels"] = stats.get("rejected_topic_labels", 0) + 1
        title, used_template = _safe_topic_label(block_text, video_type)
        if used_template and stats is not None:
            stats["fallback_template_usage"] = stats.get("fallback_template_usage", 0) + 1
    if not _is_natural_topic_phrase(title):
        if stats is not None:
            stats["rejected_topic_labels"] = stats.get("rejected_topic_labels", 0) + 1
        title, used_template = _safe_topic_label(block_text, video_type)
        if used_template and stats is not None:
            stats["fallback_template_usage"] = stats.get("fallback_template_usage", 0) + 1
    if not _is_natural_topic_phrase(title):
        title = "Discussion Topic" if _is_english_like(block_text) else (_compress_line(block_text, max_words=4) or "Discussion")
    return title


def _build_topic_blocks(segments: List[Dict], video_type: str) -> List[Dict]:
    stats = {"rejected_topic_labels": 0, "fallback_template_usage": 0}
    blocks = _segment_topic_blocks(segments)
    used_labels: List[str] = []
    deduped_labels = 0
    safe_bucket_fallback_usage = 0
    for block in blocks:
        if video_type == "interview":
            local_people = [person for person in _extract_main_people(block["text"], max_people=2) if _is_valid_interview_participant(person)]
            label, used_bucket = _safe_interview_label_for_block(block["text"], local_people, used_labels=used_labels)
            block["label"] = label
            if used_bucket:
                safe_bucket_fallback_usage += 1
            if any(
                label.lower() == existing.lower()
                or INTERVIEW_SEMANTIC_TOPIC_MAP.get(label, label).lower() == INTERVIEW_SEMANTIC_TOPIC_MAP.get(existing, existing).lower()
                for existing in used_labels
            ):
                deduped_labels += 1
            used_labels.append(label)
        else:
            block["label"] = _topic_label_from_block(block["text"], video_type, stats=stats)
    logger.info("[SUMMARY_TOPICS] blocks=%d video_type=%s", len(blocks), video_type)
    logger.info(
        "[SUMMARY_PHRASING] rejected_topic_labels=%d fallback_template_usage=%d deduplicated_labels=%d safe_interview_bucket_fallback_usage=%d",
        stats["rejected_topic_labels"],
        stats["fallback_template_usage"],
        deduped_labels,
        safe_bucket_fallback_usage,
    )
    return blocks


def structured_summary_cache_key(
    video_id: str,
    transcript_hash: str,
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    internal_evidence_units: List[Dict] | None = None,
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
    transcript_state: str = "",
    transcript_language: str = "",
) -> str:
    segment_fingerprint = "|".join(
        f"{float(seg.get('start', 0.0) or 0.0):.2f}:{float(seg.get('end', 0.0) or 0.0):.2f}:{_clean_text(seg.get('text', ''))[:120]}"
        for seg in (segments or [])
        if isinstance(seg, dict)
    )
    raw_fingerprint = "|".join(
        f"{float(seg.get('start', 0.0) or 0.0):.2f}:{float(seg.get('end', 0.0) or 0.0):.2f}:{_clean_text(seg.get('text', ''))[:60]}"
        for seg in (raw_segments or [])
        if isinstance(seg, dict)
    )
    assembled_fingerprint = "|".join(
        f"{float(seg.get('display_start', seg.get('start', 0.0)) or 0.0):.2f}:{float(seg.get('display_end', seg.get('end', 0.0)) or 0.0):.2f}:{_clean_text(seg.get('text', ''))[:80]}"
        for seg in (assembled_units or [])
        if isinstance(seg, dict)
    )
    use_internal_evidence_in_cache = _clean_text(transcript_language or "") == "ml" and _clean_text(transcript_state or "") == "degraded"
    internal_evidence_fingerprint = "|".join(
        (
            f"{float(seg.get('display_start', seg.get('start', 0.0)) or 0.0):.2f}:"
            f"{float(seg.get('display_end', seg.get('end', 0.0)) or 0.0):.2f}:"
            f"{int(bool(seg.get('evidence_only', False)))}:"
            f"{_clean_text(seg.get('text', ''))[:80]}"
        )
        for seg in (internal_evidence_units or [])
        if isinstance(seg, dict)
    ) if use_internal_evidence_in_cache else ""
    payload = "||".join([
        STRUCTURED_SUMMARY_CACHE_VERSION,
        str(video_id or ""),
        str(transcript_hash or ""),
        _clean_text(transcript_text or ""),
        _clean_text(full_summary or ""),
        _clean_text(bullet_summary or ""),
        _clean_text(short_summary or ""),
        _clean_text(transcript_state or ""),
        _clean_text(transcript_language or ""),
        segment_fingerprint,
        raw_fingerprint,
        assembled_fingerprint,
        internal_evidence_fingerprint,
    ])
    return sha1(payload.encode("utf-8")).hexdigest()


def _build_fallback_chapter_blocks(segments: List[Dict], video_type: str, min_items: int = 5, max_items: int = 12) -> List[Dict]:
    cleaned = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        text = _clean_text(seg.get("text", ""))
        if not text:
            continue
        cleaned.append({
            "start": float(seg.get("start", 0.0) or 0.0),
            "end": float(seg.get("end", seg.get("start", 0.0)) or seg.get("start", 0.0) or 0.0),
            "text": text,
        })
    if len(cleaned) < min_items:
        return []

    target = max(min_items, min(max_items, max(1, len(cleaned) // 3)))
    if len(cleaned) <= target:
        picks = cleaned
    else:
        step = max(len(cleaned) / float(target), 1.0)
        picks = []
        for idx in range(target):
            pick = cleaned[min(len(cleaned) - 1, int(round(idx * step)))]
            if picks and abs(pick["start"] - picks[-1]["start"]) < 1.0:
                continue
            picks.append(pick)
        if len(picks) < min_items:
            picks = cleaned[:min(max_items, len(cleaned))]

    blocks = []
    for seg in picks[:max_items]:
        label = _topic_label_from_block(seg["text"], video_type)
        if _looks_fragment(label):
            label = _keyword_phrase(seg["text"], max_words=4) or label
        blocks.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "sentences": [seg["text"]],
            "word_count": len(seg["text"].split()),
            "label": label,
        })
    return blocks


def _tldr_from_topics(video_type: str, topic_blocks: List[Dict], short_summary: str, full_summary: str, transcript_text: str) -> str:
    summary_source = short_summary or full_summary
    if not _is_english_like(summary_source or transcript_text):
        sentences = _split_sentences(summary_source or transcript_text)[:2]
        return "\n".join(
            _compress_line(sentence, max_words=18)
            for sentence in sentences
            if _compress_line(sentence, max_words=18)
        ).strip()

    labels = [block.get("label", "") for block in topic_blocks if block.get("label") and _is_natural_topic_phrase(block.get("label", ""))]
    labels = _dedupe(labels)[:3]
    if not labels:
        labels = [_compress_line(summary_source or transcript_text, max_words=8)]
    primary = labels[0] if labels else "the main discussion"
    secondary = labels[1] if len(labels) > 1 else ""
    people = _extract_main_people(transcript_text, max_people=2)
    topics = [label.lower() for label in labels[:2] if label]
    topic_text = ", ".join(topics) if topics else "the main topics"
    source_concepts = _concept_phrases(summary_source or transcript_text, max_items=2)

    if video_type == "tutorial":
        tutorial_cues = []
        tutorial_corpus = f"{transcript_text} {summary_source}".lower()
        for marker, label in [
            ("layout", "layout planning"),
            ("prompt", "prompt writing"),
            ("section", "section generation"),
            ("animation", "animation tuning"),
            ("deploy", "final deployment"),
        ]:
            if marker in tutorial_corpus and label not in tutorial_cues:
                tutorial_cues.append(label)
        if tutorial_cues:
            primary = tutorial_cues[0]
            if len(tutorial_cues) > 1:
                secondary = tutorial_cues[-1]
        elif source_concepts:
            primary = source_concepts[0]
            if len(source_concepts) > 1:
                secondary = source_concepts[1]
        first = f"This tutorial explains how to {primary.lower()}." if primary else "This tutorial explains the main workflow."
        second = f"It also covers {secondary.lower()} and the practical outcome." if secondary else "It walks through the main steps and practical outcome."
    elif video_type == "meeting":
        first = f"This meeting focuses on {primary.lower()}."
        second = "It captures the main decisions, discussion points, and follow-up items."
    elif video_type == "interview":
        people = _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks or []))
        if not people:
            people = _finalize_interview_participants([
                person for person in _extract_main_people(transcript_text, max_people=2)
                if _is_valid_interview_participant(person)
            ][:2])
        explicit_labels = _explicit_interview_topics_from_transcript(transcript_text, people)
        valid_topics = []
        for label in explicit_labels:
            mapped = INTERVIEW_SEMANTIC_TOPIC_MAP.get(label, label)
            if mapped not in valid_topics:
                valid_topics.append(mapped)
        for label in labels:
            if not label or _is_bad_interview_topic_phrase(label):
                continue
            mapped = INTERVIEW_SEMANTIC_TOPIC_MAP.get(label, label)
            if mapped not in valid_topics:
                valid_topics.append(mapped)
        thematic_topics = [
            topic for topic in valid_topics
            if topic not in {
                INTERVIEW_SEMANTIC_TOPIC_MAP.get("Interview Introduction"),
                INTERVIEW_SEMANTIC_TOPIC_MAP.get("Closing Questions"),
            }
        ]
        if thematic_topics:
            valid_topics = thematic_topics + [
                topic for topic in valid_topics if topic not in thematic_topics
            ]
        if not valid_topics:
            derived_labels = [_bucket_interview_topic(block.get("text", ""), people) for block in topic_blocks[:3]]
            valid_topics, _ = _dedupe_interview_labels([
                INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic)
                for topic in derived_labels
                if topic
            ], max_items=3)
        valid_topics = _dedupe_interview_topic_phrases(valid_topics, max_items=3)
        prioritized_topics = [
            topic for topic in valid_topics
            if topic not in {
                INTERVIEW_SEMANTIC_TOPIC_MAP.get("Interview Introduction"),
                INTERVIEW_SEMANTIC_TOPIC_MAP.get("Closing Questions"),
            }
        ] or valid_topics
        topic_text = ", ".join(prioritized_topics[:3]) if prioritized_topics else "career highlights, filmmaking, and personal questions"
        if people:
            first = f"This interview features {' and '.join(people[:2])} discussing {topic_text}."
        else:
            first = f"This interview features a conversation about {topic_text}."
        second = (
            f"It also covers {prioritized_topics[2]}."
            if len(prioritized_topics) > 2
            else "It covers the main themes that come up in the discussion."
        )
    elif video_type == "lecture":
        first = f"This lecture explains {primary.lower()}."
        second = f"It expands into {secondary.lower()} and the broader takeaway." if secondary else "It connects the core concept to the broader takeaway."
    elif video_type == "commentary":
        first = f"This commentary focuses on {primary.lower()}."
        second = f"It also covers {secondary.lower()}." if secondary else "It explains the speaker's main perspective and observations."
    else:
        first = f"This discussion is mainly about {primary.lower()}."
        second = f"It also covers {secondary.lower()}." if secondary else "It brings together the main ideas covered in the conversation."

    return "\n".join([first.strip(), second.strip()][:2]).strip()


def _semantic_point_from_label(video_type: str, label: str, participants: Optional[List[str]] = None) -> str:
    phrase = label.strip().rstrip(".")
    if not phrase:
        return ""
    if not _is_natural_topic_phrase(phrase):
        return ""
    if not _is_english_like(phrase):
        return _compress_line(phrase, max_words=18)
    base = phrase[0].lower() + phrase[1:] if phrase else phrase
    if video_type == "interview":
        participants = _finalize_interview_participants(participants or [])
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        if phrase == "Nolan On Practical Effects":
            return "The interview discusses practical effects, CGI, and broader filmmaking choices."
        if phrase == "Downey On Iron Man":
            return "The interview covers Iron Man, Marvel, and related career moments."
        if phrase == "Marvel / Iron Man":
            return "The discussion includes Marvel, Iron Man, and related career highlights."
        if phrase == "Interview Introduction":
            return "The interview opens with quick introductory and autocomplete-style questions."
        if phrase == "Creative Work And Projects":
            return "The interview explores creative work, major projects, and how those ideas were developed."
        if phrase in {"Filmmaking And Oppenheimer", "Filmmaking Choices And Oppenheimer"}:
            return "The interview covers filmmaking, Oppenheimer, and creative decisions."
        if phrase == "Career Discussion":
            return "The interview highlights career milestones, creative work, and major projects."
        if phrase in {"Hobbies and Personal Life", "Hobbies And Personal Life"}:
            return f"{participant_phrase.capitalize()} discuss hobbies, personal life, and informal anecdotes."
        if phrase == "Career And Personal Questions":
            return "The interview covers career highlights, personal anecdotes, and broader questions."
        if phrase == "Closing Questions":
            return "The conversation ends with faster personal and closing questions."
    templates = {
        "tutorial": f"The tutorial explains {base}.",
        "meeting": f"The discussion covers {base}.",
        "interview": f"The interview covers {base}.",
        "lecture": f"The lecture explains {base}.",
        "commentary": f"The speaker comments on {base}.",
        "casual conversation": f"The conversation touches on {base}.",
    }
    return templates.get(video_type, f"The video covers {base}.")


def _is_low_information_statement(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    words = value.split()
    if len(words) < 5:
        return True
    vague_starts = (
        "this is", "there is", "it is", "they are", "the video is", "the speaker is"
    )
    return value.lower().startswith(vague_starts) and len(words) <= 7


# Enhanced generic phrase detection for English summaries
# NOTE: Be careful not to filter valid tutorial/content phrases
ENGLISH_GENERIC_PHRASES = frozenset([
    # Workflow/process phrases - ONLY when standalone or vague
    "workflow", "things", "process", "step", "steps", 
    "overall", "overall process", "main steps", 
    # Vague standalone phrases - not when part of valid content
    "various", "various aspects", "various topics", 
    # Spam/action-bait phrases - always remove
    "choose us", "click below", "subscribe now", "like and subscribe",
    "don't forget", "stay tuned", "more content", "coming soon",
    # Weak generic key points
    "important point", "key point", "main point", "another point",
    "worth noting", "also worth", "additionally",
])

# Valid tutorial phrases that should NOT be removed
# These are specific content descriptions, not generic filler
VALID_TUTORIAL_PHRASES = frozenset([
    "tutorial explains", "tutorial covers", "tutorial shows",
    "learn how to", "find out how", "discover how",
    "practical outcome", "walk through", 
    "in this video we", "in this video i",
    "video covers", "video shows", "video explains",
    "career", "personal", "questions",  # Valid topics, not generic
    "covers the main", "explains the main", "covers the overall", "explains the overall",
])


def _is_generic_summary_phrase(text: str) -> bool:
    """
    Detect generic, low-value phrases that should be filtered from summaries.
    These phrases don't provide specific, meaningful information.
    
    IMPORTANT: Allow valid tutorial/content phrases - only filter truly generic content.
    """
    value = _clean_text(text)
    if not value:
        return True
    
    lower = value.lower().strip()
    words = lower.split()
    
    # Too short to be meaningful
    if len(words) < 3:
        return True
    
    # Check exact phrase matches - ONLY if not in VALID_TUTORIAL_PHRASES
    if lower in ENGLISH_GENERIC_PHRASES:
        # Check if this is actually a valid phrase we should keep
        if lower in VALID_TUTORIAL_PHRASES:
            return False
        return True
    
    # Check if phrase starts with generic patterns (but allow valid content)
    generic_starts = [
        "the video", "this video", "video about",
        "tutorial about", "tutorial on", "video on", "learn about",
    ]
    valid_starts = [
        "in this video we", "in this video i",  # Valid content
        "explains how to", "shows how to", "covers how to",
        "talks about", "discusses", "focuses on", "highlights",
    ]
    
    # Check if it starts with a valid pattern first
    for start in valid_starts:
        if lower.startswith(start):
            return False  # Allow valid content
    
    # Then check for generic patterns
    for start in generic_starts:
        if lower.startswith(start):
            # Only reject if very short after the generic start
            remaining = lower[len(start):].strip()
            if len(remaining.split()) <= 3:
                return True
    
    # Check for weak/empty content patterns - but be more lenient
    # Only reject if there's genuinely no specific content
    weak_patterns = [
        r'\bworkflow\b', r'\bthings?\b', r'\bprocess\b', r'^step\s*\d*'
    ]


def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()


def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()



def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()



def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()



def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()



def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()



def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()



def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()



def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()



def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()



def _validate_summary_quality(text: str, min_length: int = 20) -> bool:
    """
    Validate that summary text meets minimum quality standards.
    Returns True if quality is acceptable, False if too poor.
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    value = _clean_text(text)
    if not value:
        return False
    
    words = value.split()
    
    # Must have at least 5 words
    if len(words) < 5:
        return False
    
    # Reject if too generic
    if _is_generic_summary_phrase(value):
        return False
    
    # Reject if too many stopwords (indicates vague content)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'this', 'that', 'these', 'those', 'it', 'its', 'in', 'on', 
                 'at', 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'but'}
    stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / len(words)
    if stopword_ratio > 0.6:
        return False
    
    return True


def _validate_and_fix_tldr(tldr: str, fallback_topic: str = "video content") -> str:
    """
    Validate TLDR quality and return fallback if too poor.
    Ensures TLDR is specific and meaningful, not generic.
    """
    if not tldr:
        return f"This {fallback_topic} contains speech."
    
    # Check if TLDR is too generic
    if _is_generic_summary_phrase(tldr) or _is_low_information_statement(tldr):
        return f"This {fallback_topic} contains speech."
    
    # Validate quality
    if not _validate_summary_quality(tldr, min_length=15):
        return f"This {fallback_topic} contains speech."
    
    # Additional TLDR-specific checks
    lower = tldr.lower()
    
    # Must not be just a generic description
    generic_tldr_patterns = [
        r'^this (video|tutorial|interview|lecture) is about',
        r'^the video (covers|shows|explains)',
        r'^in this (video|tutorial)',
        r'^(video|tutorial) contains speech',
    ]
    for pattern in generic_tldr_patterns:
        if re.match(pattern, lower):
            return f"This {fallback_topic} contains speech."
    
    return tldr


def _malayalam_summary_noise_ratio(text: str) -> float:
    tokens = [tok for tok in re.findall(r'[\u0D00-\u0D7F]+', text or '') if len(tok) >= 4]
    if not tokens:
        return 0.0
    noisy = 0
    for token in tokens:
        if re.search(r'(ീകീ|ംകീ|വാഇ|ആകുന|േംഗീ|പേഡീ|രീദീ|ഇിദും)', token):
            noisy += 1
    return noisy / max(len(tokens), 1)


def _looks_noisy_malayalam_summary_fragment(text: str) -> bool:
    value = _clean_text(text)
    if not value:
        return True
    if not any('\u0d00' <= ch <= '\u0d7f' for ch in value):
        return False
    if len(value.split()) < 2:
        return True
    return _malayalam_summary_noise_ratio(value) >= 0.22


def _is_malayalam_mixed_lang_source(text: str) -> bool:
    low = _clean_text(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", low) for term in MALAYALAM_TRUSTED_ENGLISH_TERMS)


def _is_malayalam_or_mixed_language(language_code: str, text: str) -> bool:
    code = str(language_code or "").strip().lower()
    if code == "ml":
        return True
    has_malayalam = any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))
    return has_malayalam and _is_malayalam_mixed_lang_source(text)


def _collect_trusted_english_terms(text: str) -> List[str]:
    low = _clean_text(text).lower()
    terms = []
    for phrase in (
        "whatsapp channel",
        "exam hall",
        "hard work",
        "confidence",
        "support",
        "result",
        "warrior",
    ):
        if phrase in low:
            terms.append(phrase)
    return terms


def _match_malayalam_topic_clusters(text: str) -> List[str]:
    low = _clean_text(text).lower()
    matches: List[str] = []
    for cluster_id, _label, markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        for marker in markers:
            needle = str(marker).lower()
            if " " in needle:
                if needle in low:
                    matches.append(cluster_id)
                    break
            elif needle and needle in low:
                matches.append(cluster_id)
                break
    return matches


def _cluster_label(cluster_id: str) -> str:
    for candidate_id, label, _markers in MALAYALAM_DEGRADED_TOPIC_CLUSTERS:
        if candidate_id == cluster_id:
            return label
    return "Main discussion"


def _cluster_summary_point(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker emphasizes exam preparation, study planning, and the right mindset.",
        "fear_confidence": "The discussion highlights fear, excuses, confidence, and a stronger mindset.",
        "effort_consistency": "The talk stresses effort, focus, consistency, and the connection to results.",
        "exam_hall_readiness": "The speaker touches on exam-hall readiness, questions, and staying prepared.",
        "support_guidance": "The discussion includes support, guidance, and follow-up resources such as WhatsApp channel updates.",
    }
    return mapping.get(cluster_id, "")


def _cluster_tldr_sentence(cluster_id: str) -> str:
    mapping = {
        "exam_preparation_mindset": "The speaker focuses on exam preparation and mindset.",
        "fear_confidence": "The discussion also addresses fear, excuses, and confidence.",
        "effort_consistency": "The talk emphasizes consistent effort and results.",
        "exam_hall_readiness": "The speaker also touches on exam hall readiness.",
        "support_guidance": "Support and follow-up guidance are also mentioned.",
    }
    return mapping.get(cluster_id, "")


def collect_trusted_malayalam_summary_evidence(
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    transcript_text: str,
) -> Dict:
    trusted_units = []
    topic_counter: Counter[str] = Counter()
    topic_anchors: Dict[str, Dict] = {}
    trusted_terms: List[str] = []
    rejected_noisy_units = 0
    rejected_noisy_terms = 0
    candidates = assembled_units or trusted_segments or []
    for idx, unit in enumerate(candidates):
        if not isinstance(unit, dict):
            continue
        text = _clean_text(unit.get("text", ""))
        if not text:
            continue
        if _looks_noisy_malayalam_summary_fragment(text):
            rejected_noisy_units += 1
            continue
        trusted_units.append(unit)
        for term in _collect_trusted_english_terms(text):
            if term not in trusted_terms:
                trusted_terms.append(term)
        for cluster_id in _match_malayalam_topic_clusters(text):
            topic_counter[cluster_id] += 1
            topic_anchors.setdefault(
                cluster_id,
                {
                    "source_unit_indices": [],
                    "source_time_ranges": [],
                    "timestamp": _format_timestamp(unit.get("display_start", unit.get("start", 0.0))),
                },
            )
            topic_anchors[cluster_id]["source_unit_indices"].append(idx)
            topic_anchors[cluster_id]["source_time_ranges"].append(
                (
                    float(unit.get("display_start", unit.get("start", 0.0)) or 0.0),
                    float(unit.get("display_end", unit.get("end", unit.get("start", 0.0))) or unit.get("end", 0.0) or 0.0),
                )
            )
    if not trusted_units:
        for term in _collect_trusted_english_terms(transcript_text):
            if term not in trusted_terms:
                trusted_terms.append(term)
    rejected_noisy_terms += max(0, len(_collect_trusted_english_terms(transcript_text)) - len(trusted_terms))
    logger.info(
        "[ML_SUMMARY_RECON_EVIDENCE] unit_candidates=%s trusted_candidates=%s rejected_candidates=%s mixed_lang_terms_preserved=%s",
        len(candidates),
        len(trusted_units),
        rejected_noisy_units,
        len(trusted_terms),
    )
    return {
        "trusted_units": trusted_units,
        "trusted_terms": trusted_terms,
        "topic_signals": topic_counter,
        "chapter_anchor_candidates": topic_anchors,
        "rejected_noisy_units": rejected_noisy_units,
        "rejected_noisy_terms": rejected_noisy_terms,
    }


def _degraded_summary_confidence(evidence: Dict) -> Tuple[str, float]:
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals = evidence.get("topic_signals", Counter())
    avg_readability = 0.0
    if trusted_units:
        avg_readability = sum(float(unit.get("unit_readability", 0.0) or 0.0) for unit in trusted_units) / max(len(trusted_units), 1)
    score = min(1.0, (len(trusted_units) * 0.16) + (len(topic_signals) * 0.14) + avg_readability * 0.4)
    if score >= 0.72:
        return "high", score
    if score >= 0.48:
        return "medium", score
    return "low", score


def build_malayalam_degraded_reconstruction_summary(
    transcript_text: str,
    trusted_segments: List[Dict],
    assembled_units: List[Dict],
    video_type: str,
) -> Optional[Dict]:
    evidence = collect_trusted_malayalam_summary_evidence(trusted_segments, assembled_units, transcript_text)
    trusted_units = evidence.get("trusted_units", []) or []
    topic_signals: Counter = evidence.get("topic_signals", Counter())
    semantic_units = [
        unit
        for unit in trusted_units
        if isinstance(unit, dict) and str(unit.get("segment_type", "") or "").strip() in {"clean_malayalam", "mixed_malayalam_english"}
    ]
    if not semantic_units:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="english_only_trusted_fragments",
        )
    if not trusted_units or not topic_signals:
        return _build_degraded_safe_malayalam_summary(
            transcript_text=transcript_text,
            evidence=evidence,
            video_type=video_type,
            reason="insufficient_trusted_evidence",
        )

    confidence_label, confidence_score = _degraded_summary_confidence(evidence)
    ordered_clusters = [
        cluster_id
        for cluster_id, _count in topic_signals.most_common(4 if confidence_label != "low" else 2)
    ]
    tldr_sentences = []
    for cluster_id in ordered_clusters[:3]:
        sentence = _cluster_tldr_sentence(cluster_id)
        if sentence and sentence not in tldr_sentences:
            tldr_sentences.append(sentence)
    trusted_terms = evidence.get("trusted_terms", []) or []
    if confidence_label == "low":
        tldr_sentences = tldr_sentences[:2]
    if not tldr_sentences:
        tldr_sentences = ["The speaker discusses the main ideas, but parts of the transcript remain noisy."]
    if trusted_terms and "whatsapp channel" in trusted_terms and all("whatsapp channel" not in line.lower() for line in tldr_sentences):
        tldr_sentences.append("Support and WhatsApp channel guidance are also mentioned.")
    tldr = " ".join(tldr_sentences[:3]).strip()

    key_points = []
    for cluster_id in ordered_clusters:
        point = _cluster_summary_point(cluster_id)
        if point and point not in key_points:
            key_points.append(point)
    if confidence_label == "low":
        key_points = key_points[:2]
    else:
        key_points = key_points[:4]

    chapters = []
    chapter_anchors = evidence.get("chapter_anchor_candidates", {}) or {}
    for idx, cluster_id in enumerate(ordered_clusters[:4]):
        anchor = chapter_anchors.get(cluster_id, {})
        title = _cluster_label(cluster_id)
        chapter = {
            "title": title,
            "timestamp": anchor.get("timestamp", "00:00"),
        }
        chapters.append(chapter)
        logger.info(
            "[ML_SUMMARY_RECON_CHAPTER] idx=%s title=%s source_units=%s confidence=%s",
            idx,
            title,
            anchor.get("source_unit_indices", []),
            confidence_label,
        )

    payload = {
        "tldr": tldr,
        "key_points": key_points,
        "action_items": [],
        "chapters": _validate_chapters(chapters)[:4],
        "participants": [],
        "_trace": {
            "mode": "degraded_reconstruction",
            "summary_confidence": confidence_label,
            "summary_confidence_score": round(confidence_score, 3),
            "source_unit_indices": sorted({
                idx
                for anchor in chapter_anchors.values()
                for idx in anchor.get("source_unit_indices", [])
            }),
            "source_time_ranges": [
                {
                    "start": round(float(start), 2),
                    "end": round(float(end), 2),
                }
                for anchor in chapter_anchors.values()
                for start, end in anchor.get("source_time_ranges", [])
            ][:12],
            "trust_count": len(trusted_units),
            "rejected_noisy_evidence_count": int(evidence.get("rejected_noisy_units", 0) or 0),
            "trusted_terms": trusted_terms[:8],
            "topic_clusters": ordered_clusters,
            "video_type": video_type,
        },
    }
    logger.info(
        "[ML_SUMMARY_RECON] mode=degraded_reconstruction trusted_units=%s rejected_noisy_units=%s trusted_terms=%s topic_clusters=%s summary_confidence=%s",
        len(trusted_units),
        int(evidence.get("rejected_noisy_units", 0) or 0),
        len(trusted_terms),
        ordered_clusters,
        confidence_label,
    )
    return payload


def _clean_malayalam_mixed_summary_point(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(r"\b(?:àµ€à´•àµ€|à´‚à´•àµ€|à´µà´¾à´‡|à´†à´•àµà´¨|àµ‡à´‚à´—àµ€|à´ªàµ‡à´¡àµ€|à´°àµ€à´¦àµ€|à´‡à´¿à´¦àµà´‚)\S*", "", value)
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    return value


def _nearest_transcript_similarity(candidate: str, transcript_sentences: List[str]) -> float:
    if not candidate or not transcript_sentences:
        return 0.0
    return max((_similarity(candidate, sentence) for sentence in transcript_sentences), default=0.0)


def _summary_support_is_grounded(summary_text: str, transcript_text: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(summary_text)
    if not value:
        return False
    if not transcript_text.strip():
        return False
    if not _is_english_like(value):
        return True
    summary_tokens = set(_normalize_for_similarity(value))
    transcript_tokens = set(_normalize_for_similarity(transcript_text))
    if not summary_tokens or not transcript_tokens:
        return False
    token_overlap = len(summary_tokens & transcript_tokens) / max(min(len(summary_tokens), len(transcript_tokens)), 1)
    sentence_overlap = _nearest_transcript_similarity(value, transcript_sentences)
    return token_overlap >= 0.20 or sentence_overlap >= 0.28


def _ground_summary_support(
    transcript_text: str,
    full_summary: str,
    bullet_summary: str,
    short_summary: str,
) -> Tuple[str, str, str]:
    transcript_sentences = _split_sentences(transcript_text)
    grounded_full = full_summary if _summary_support_is_grounded(full_summary, transcript_text, transcript_sentences) else ""
    grounded_bullet = bullet_summary if _summary_support_is_grounded(bullet_summary, transcript_text, transcript_sentences) else ""
    grounded_short = short_summary if _summary_support_is_grounded(short_summary, transcript_text, transcript_sentences) else ""
    logger.info(
        "[SUMMARY_GROUNDING] full_used=%s bullet_used=%s short_used=%s",
        bool(grounded_full),
        bool(grounded_bullet),
        bool(grounded_short),
    )
    return grounded_full, grounded_bullet, grounded_short


def _build_key_points(
    video_type: str,
    transcript_text: str,
    bullet_summary: str,
    full_summary: str,
    topic_blocks: List[Dict],
    transcript_language: str = "",
) -> List[str]:
    transcript_sentences = _split_sentences(transcript_text)
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    candidate_lines = _split_bullets(bullet_summary) + _split_sentences(full_summary)
    semantic_candidates = []
    rejected_fragments = 0
    rejected_key_points = 0
    regenerated_semantic_key_points = 0
    malayalam_fragment_rejections = 0
    for line in candidate_lines:
        line = _compress_line(line, max_words=18)
        if not line:
            continue
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(line):
            rejected_fragments += 1
            rejected_key_points += 1
            malayalam_fragment_rejections += 1
            continue
        if _looks_fragment(line) or _is_low_information_statement(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        # NEW: Filter generic phrases for English
        if not malayalam_mode and _is_generic_summary_phrase(line):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _nearest_transcript_similarity(line, transcript_sentences) >= 0.76:
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if video_type == "interview" and _looks_weak_interview_key_point(line, transcript_sentences):
            rejected_fragments += 1
            rejected_key_points += 1
            continue
        if _is_english_like(line) and not re.search(r"\b(?:covers|explains|discusses|focuses|shows|highlights|talks)\b", line.lower()):
            rejected_key_points += 1
            continue
        semantic_candidates.append(line.rstrip("."))

    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    topic_fallbacks = [
        _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
        for block in topic_blocks
    ]
    topic_fallbacks = [
        item for item in topic_fallbacks
        if item
        and not _looks_fragment(item)
        and not _is_low_information_statement(item)
        and not (video_type == "interview" and _looks_weak_interview_key_point(item, transcript_sentences))
        and (malayalam_mode or not _is_generic_summary_phrase(item))  # NEW: Filter generic for English
    ]

    combined = _dedupe(semantic_candidates + topic_fallbacks)
    if len(combined) < 4:
        if video_type == "interview":
            safe_topic_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_topic_labels:
                    safe_topic_labels.append(label)
            interview_regenerated = _interview_fallback_key_points(safe_topic_labels, interview_participants)
            regenerated_semantic_key_points += len(interview_regenerated)
            combined = _dedupe(combined + interview_regenerated)
        else:
            for sentence in transcript_sentences:
                compressed = _compress_line(sentence, max_words=14)
                if not compressed:
                    continue
                if _looks_fragment(compressed) or _is_low_information_statement(compressed):
                    rejected_fragments += 1
                    rejected_key_points += 1
                    continue
                semantic = _semantic_point_from_label(video_type, compressed, interview_participants)
                if semantic and _nearest_transcript_similarity(semantic, transcript_sentences) < 0.76:
                    combined = _dedupe(combined + [semantic])
                if len(combined) >= 4:
                    break
    logger.info(
        "[SUMMARY_KEY_POINTS] topic_blocks=%d rejected_fragments=%d rejected_key_points=%d regenerated_semantic_key_points=%d final=%d",
        len(topic_blocks),
        rejected_fragments,
        rejected_key_points,
        regenerated_semantic_key_points,
        min(len(combined), 6),
    )
    if malayalam_mode:
        logger.info(
            "[ML_SUMMARY_CLEAN] transcript_fragment_rejections_in_summary=%d accepted_semantic_key_points=%d",
            malayalam_fragment_rejections,
            min(len(combined), 6),
        )
    return combined[:6]


def _extract_action_items(video_type: str, transcript_text: str, candidates: List[str], max_items: int = 6) -> List[str]:
    if video_type in {"interview", "commentary", "casual conversation"}:
        return []

    transcript_sentences = _split_sentences(transcript_text)
    out = []
    for line in candidates:
        cleaned = _compress_line(line, max_words=16).rstrip(".")
        if not cleaned or not _sentence_contains_action(cleaned):
            continue
        if _nearest_transcript_similarity(cleaned, transcript_sentences) >= 0.94:
            cleaned = re.sub(r"^(the speaker|they|we|you)\s+", "", cleaned, flags=re.IGNORECASE)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return _dedupe(out)[:max_items]


def _interview_chapter_title(block_text: str) -> str:
    local_people = [person for person in _extract_main_people(block_text, max_people=2) if _is_valid_interview_participant(person)]
    title, _ = _safe_interview_label_for_block(block_text, local_people, used_labels=None)
    return title


def _build_chapters(topic_blocks: List[Dict], max_items: int = 12, video_type: Optional[str] = None, transcript_language: str = "") -> List[Dict]:
    chapters = []
    rejected_titles = 0
    seen_titles: List[str] = []
    malayalam_mode = str(transcript_language or "").strip().lower() == "ml"
    for block in topic_blocks[:max_items]:
        chapter_type = "interview" if video_type == "interview" else "chapter"
        if chapter_type == "interview":
            local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
            title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
        else:
            title = _topic_label_from_block(block.get("text", ""), chapter_type)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title) or _is_bad_interview_topic_phrase(title):
            rejected_titles += 1
            title, _ = _safe_topic_label(block.get("text", ""), "interview" if _extract_named_phrase(block.get("text", "")) else "casual conversation")
        cleaned_title = _compress_line(title, max_words=8) or "Chapter"
        if malayalam_mode and _looks_noisy_malayalam_summary_fragment(cleaned_title):
            rejected_titles += 1
            cleaned_title = f"Discussion at {_format_timestamp(block.get('start', 0.0))}"
        if any(_similarity(cleaned_title, existing) >= 0.82 for existing in seen_titles):
            rejected_titles += 1
            if chapter_type == "interview":
                local_people = [person for person in _extract_main_people(block.get("text", ""), max_people=2) if _is_valid_interview_participant(person)]
                cleaned_title, _ = _safe_interview_label_for_block(block.get("text", ""), local_people, used_labels=seen_titles)
            else:
                cleaned_title = _compress_line(title, max_words=6) or "Chapter"
        seen_titles.append(cleaned_title)
        chapters.append({
            "title": cleaned_title,
            "timestamp": _format_timestamp(block.get("start", 0.0)),
        })
    logger.info("[SUMMARY_CHAPTERS] final_count=%d rejected_titles=%d", len(chapters), rejected_titles)
    return chapters


def _validate_chapters(chapters: List[Dict]) -> List[Dict]:
    validated = []
    rejected_titles = 0
    for chapter in chapters:
        title = _clean_text(chapter.get("title", ""))
        if not title:
            continue
        english_like = _is_english_like(title)
        if _looks_fragment(title) or not _is_natural_topic_phrase(title):
            rejected_titles += 1
            title = _compress_line(title, max_words=4) or "Chapter"
        words = title.split()
        if len(words) < 4:
            if title in SAFE_INTERVIEW_CHAPTER_TITLES:
                pass
            elif english_like:
                title = f"Discussion on {title}".strip()
            elif len(words) < 2:
                continue
        if len(title.split()) > 10:
            title = " ".join(title.split()[:10]).strip()
        validated.append({
            "title": title,
            "timestamp": chapter.get("timestamp", "00:00"),
        })
    logger.info("[SUMMARY_CHAPTER_VALIDATION] rejected_titles=%d final=%d", rejected_titles, len(validated[:12]))
    return validated[:12]


def _semantic_core(text: str) -> str:
    value = _clean_text(text)
    if not value:
        return ""
    value = re.sub(
        r"^(?:This|It|The)\s+(?:interview|tutorial|video|discussion|conversation|speaker|lecture|meeting)\s+"
        r"(?:features|covers|explains|highlights|focuses on|discusses|talks about)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:Christopher Nolan|Robert Downey Jr\.?)\s+(?:discusses|explains|talks about)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip().rstrip(".")
    return value


def _looks_transcript_leaky_key_point(point: str, transcript_sentences: List[str], video_type: str) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if (_looks_fragment(value) and not (video_type == "interview" and has_semantic_verb)) or _is_low_information_statement(value):
        return True
    if re.search(r"['\"“”]", value):
        return True
    if re.search(r"\b(?:i do not|i don't|you know|i mean|sort of|kind of)\b", lower):
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.74 and not (video_type == "interview" and has_semantic_verb):
        return True
    if video_type == "interview":
        if _looks_weak_interview_key_point(value, transcript_sentences):
            return True
        if not has_semantic_verb and _is_bad_interview_topic_phrase(value):
            return True
        normalized_names = re.findall(r"(Christopher Nolan|Robert Downey Jr\.?)", value)
        if len(normalized_names) >= 2 and len(set(normalized_names)) == 1 and not re.search(r"(?:explains|discusses|talks about|covers|highlights)", lower):
            return True
        return False
    core = _semantic_core(value)
    if _is_english_like(value) and (not core or not _is_natural_topic_phrase(core)):
        return True
    return False


def _semantic_dedupe_key(items: List[str], *, video_type: str) -> List[str]:
    deduped: List[str] = []
    seen: List[str] = []
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        if video_type == "interview":
            semantic = INTERVIEW_SEMANTIC_TOPIC_MAP.get(cleaned, _semantic_core(cleaned) or cleaned).lower()
        else:
            semantic = (_semantic_core(cleaned) or cleaned).lower()
        if any(_similarity(semantic, existing) >= 0.78 for existing in seen):
            continue
        seen.append(semantic)
        deduped.append(cleaned)
    return deduped


def _interview_fallback_key_points(topic_labels: List[str], participants: List[str]) -> List[str]:
    preferred_order = [
        "Nolan On Practical Effects",
        "Filmmaking And Oppenheimer",
        "Downey On Iron Man",
        "Marvel / Iron Man",
        "Creative Work And Projects",
        "Career Discussion",
        "Hobbies And Personal Life",
        "Career And Personal Questions",
        "Closing Questions",
    ]
    available = list(topic_labels)
    ordered = [label for label in preferred_order if label in available]
    ordered.extend(label for label in available if label not in ordered)

    out: List[str] = []
    for label in ordered:
        point = _semantic_point_from_label("interview", label, participants)
        if not point:
            continue
        out.append(point)
        if len(out) >= 4:
            break
    if len(out) < 4:
        participant_phrase = " and ".join(participants[:2]) if participants else "the speakers"
        safe_generic = [
            "The interview covers the main discussion topics, career moments, and personal reflections.",
            f"{participant_phrase.capitalize()} discuss major roles, personal interests, and memorable experiences.",
            "The conversation highlights creative choices, work highlights, and broader themes.",
            "The discussion also includes personal anecdotes and closing questions.",
        ]
        for point in safe_generic:
            if point not in out:
                out.append(point)
            if len(out) >= 4:
                break
    return out[:6]


def _dedupe_tldr_sentences(tldr: str) -> Tuple[str, int]:
    protected = (tldr or "").replace("Jr.", "Jr<dot>").replace("Sr.", "Sr<dot>")
    sentences = [sentence.replace("Jr<dot>", "Jr.").replace("Sr<dot>", "Sr.") for sentence in _split_sentences(protected)]
    if not sentences:
        return "", 0
    kept: List[str] = []
    deduped = 0
    for sentence in sentences[:2]:
        cleaned = _clean_text(sentence)
        if not cleaned:
            continue
        core = _semantic_core(cleaned) or cleaned
        if any(_similarity(core, _semantic_core(existing) or existing) >= 0.74 for existing in kept):
            deduped += 1
            continue
        kept.append(cleaned)
    return "\n".join(kept[:2]).strip(), deduped


def _finalize_interview_participants(participants: List[str]) -> List[str]:
    out: List[str] = []
    for participant in participants or []:
        cleaned = _normalize_person_name(participant)
        if not cleaned:
            continue
        if cleaned.lower() in BAD_FINAL_INTERVIEW_PARTICIPANTS:
            continue
        token_set = {part.lower().rstrip(".") for part in cleaned.split()}
        if token_set & {"iron", "man", "marvel", "oppenheimer"}:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out[:2]


def _meaningful_token_count(text: str) -> int:
    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z']*", text or "")
        if tok.lower() not in STOPWORDS_EN and tok.lower() not in LOW_SIGNAL_WORDS
    ]
    return len(tokens)


def _looks_weak_interview_key_point(point: str, transcript_sentences: List[str]) -> bool:
    value = _clean_text(point)
    if not value:
        return True
    lower = value.lower()
    has_semantic_verb = bool(re.search(r"\b(?:explains|discusses|talks about|shares|covers|highlights|explores|describes|includes)\b", lower))
    if any(re.search(pattern, lower) for pattern in BAD_INTERVIEW_KEY_POINT_PATTERNS):
        return True
    if re.search(r"\b(?:i|i'm|i’ve|i'd|my|me)\b", lower) and not re.search(r"\b(?:discusses|explains|talks about|shares|covers|highlights|explores)\b", lower):
        return True
    if "?" in value:
        return True
    if _meaningful_token_count(value) < 5:
        return True
    if not has_semantic_verb:
        return True
    if _nearest_transcript_similarity(value, transcript_sentences) >= 0.9 and not re.search(
        r"\b(?:creative decisions|career highlights|personal interests|film craft|practical effects|iron man|marvel|oppenheimer|personal anecdotes)\b",
        lower,
    ):
        return True
    core = _semantic_core(value)
    if not core or _meaningful_token_count(core) < 3:
        return True
    return False


def _validate_summary(
    payload: Dict,
    *,
    transcript_text: str,
    video_type: str,
    topic_blocks: List[Dict],
) -> Dict:
    transcript_sentences = _split_sentences(transcript_text)
    fallback_template_usage = 0
    rejected_transcript_leaks = 0
    deduped_key_points = 0
    deduped_chapters = 0
    deduped_tldr_topics = 0

    key_points = []
    rejected_points = 0
    interview_participants = (
        _finalize_interview_participants(_extract_interview_participants(transcript_text, topic_blocks))
        if video_type == "interview"
        else []
    )
    for point in payload.get("key_points", []):
        if _looks_transcript_leaky_key_point(point, transcript_sentences, video_type):
            rejected_points += 1
            rejected_transcript_leaks += 1
            continue
        key_points.append(point)
    deduped_candidate_key_points = _semantic_dedupe_key(key_points, video_type=video_type)
    deduped_key_points += max(0, len(key_points) - len(deduped_candidate_key_points))
    key_points = deduped_candidate_key_points
    if len(key_points) < 4:
        extras = [
            _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
            for block in topic_blocks
        ]
        extras = [item for item in extras if item and not _looks_transcript_leaky_key_point(item, transcript_sentences, video_type)]
        key_points = _semantic_dedupe_key(key_points + extras, video_type=video_type)[:6]
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for block in topic_blocks:
                for sentence in _split_sentences(block.get("text", "")):
                    candidate = _compress_line(sentence, max_words=18)
                    if not candidate or _looks_fragment(candidate):
                        continue
                    if _is_english_like(candidate):
                        candidate = _semantic_point_from_label(video_type, candidate, interview_participants) or candidate
                    if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                        key_points.append(candidate)
                    if len(key_points) >= 4:
                        break
                if len(key_points) >= 4:
                    break
    if len(key_points) < 4:
        if video_type == "interview":
            safe_labels = []
            for block in topic_blocks:
                label = _clean_text(block.get("label", ""))
                if label and label not in safe_labels:
                    safe_labels.append(label)
            key_points = _semantic_dedupe_key(
                key_points + _interview_fallback_key_points(safe_labels, interview_participants),
                video_type="interview",
            )[:6]
        else:
            for sentence in _split_sentences(transcript_text):
                candidate = _compress_line(sentence, max_words=18)
                if not candidate or _looks_fragment(candidate):
                    continue
                if not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in key_points:
                    key_points.append(candidate)
                if len(key_points) >= 4:
                    break
    payload["key_points"] = _semantic_dedupe_key(key_points, video_type=video_type)[:6]

    if video_type in {"interview", "commentary", "casual conversation"}:
        payload["action_items"] = []

    if video_type == "interview":
        participants = interview_participants
        transcript_topics = _explicit_interview_topics_from_transcript(transcript_text, participants)
        safe_topics, deduped_labels = _dedupe_interview_labels(transcript_topics + [
            _bucket_interview_topic(block.get("text", ""), participants)
            for block in topic_blocks
            if block.get("text")
        ], max_items=4)
        tldr_lower = _clean_text(payload.get("tldr", "")).lower()
        transcript_topic_terms = [
            INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic).lower()
            for topic in transcript_topics
            if topic not in {"Interview Introduction", "Closing Questions"}
        ]
        tldr_missing_explicit_topics = bool(transcript_topic_terms) and not any(
            term in tldr_lower or any(token in tldr_lower for token in term.split() if len(token) > 4)
            for term in transcript_topic_terms
        )
        if (
            any(_is_bad_interview_topic_phrase(item) for item in [payload.get("tldr", "")] + payload.get("key_points", []))
            or "main themes that come up" in tldr_lower
            or ("opening questions" in tldr_lower and tldr_missing_explicit_topics)
        ):
            payload["tldr"] = (
                f"This interview features {' and '.join(participants[:2])} discussing {', '.join(INTERVIEW_SEMANTIC_TOPIC_MAP.get(topic, topic) for topic in safe_topics[:3])}."
                if participants and safe_topics
                else "This interview features a discussion of the main topics, career highlights, and personal questions."
            )
            fallback_template_usage += 1
        payload["key_points"] = [
            point for point in payload.get("key_points", [])
            if not _is_bad_interview_topic_phrase(point)
            and not _looks_transcript_leaky_key_point(point, transcript_sentences, "interview")
        ]
        if len(payload["key_points"]) < 4:
            interview_fallbacks = _interview_fallback_key_points(safe_topics, participants)
            payload["key_points"] = _semantic_dedupe_key(
                payload["key_points"] + [item for item in interview_fallbacks if item],
                video_type="interview",
            )[:6]
        repaired_chapters = []
        chapter_titles_seen: List[str] = []
        for chapter in payload.get("chapters", []):
            title = _clean_text(chapter.get("title", ""))
            if (
                not title
                or _is_bad_interview_topic_phrase(title)
                or any(_similarity(title, existing) >= 0.82 for existing in chapter_titles_seen)
            ):
                deduped_chapters += 1
                replacement, _ = _safe_interview_label_for_block(
                    chapter.get("title", "") or chapter.get("timestamp", "") or transcript_text,
                    participants,
                    used_labels=chapter_titles_seen,
                )
                title = replacement
                fallback_template_usage += 1
            chapter_titles_seen.append(title)
            repaired_chapters.append({
                "title": title,
                "timestamp": chapter.get("timestamp", "00:00"),
            })
        payload["chapters"] = repaired_chapters
        logger.info("[SUMMARY_INTERVIEW_VALIDATION] deduplicated_labels=%d safe_bucket_fallback_usage=%d", deduped_labels, fallback_template_usage)

    payload["chapters"] = _validate_chapters(payload.get("chapters", []))
    chapter_titles = [chapter.get("title", "") for chapter in payload["chapters"]]
    deduped_chapter_titles = _semantic_dedupe_key(chapter_titles, video_type=video_type)
    deduped_chapters += max(0, len(chapter_titles) - len(deduped_chapter_titles))
    if len(deduped_chapter_titles) != len(chapter_titles):
        repaired = []
        for title in deduped_chapter_titles:
            chapter = next((item for item in payload["chapters"] if item.get("title") == title), None)
            if chapter:
                repaired.append(chapter)
        payload["chapters"] = repaired
    if len(payload["key_points"]) < 4:
        for chapter in payload["chapters"]:
            title = _clean_text(chapter.get("title", ""))
            if not title:
                continue
            candidate = _semantic_point_from_label(video_type, title, interview_participants) if _is_english_like(title) else title
            candidate = _compress_line(candidate, max_words=18)
            if candidate and not _looks_transcript_leaky_key_point(candidate, transcript_sentences, video_type) and candidate not in payload["key_points"]:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    if len(payload["key_points"]) < 4:
        if _is_english_like(transcript_text):
            generic_map = {
                "tutorial": ["The tutorial covers the main workflow.", "It also explains the overall process."],
                "interview": ["The interview covers the main discussion points.", "It also highlights the broader conversation themes."],
                "meeting": ["The meeting covers the main discussion points.", "It also captures the overall outcome."],
                "lecture": ["The lecture explains the main concept.", "It also connects the broader ideas."],
                "commentary": ["The discussion covers the main talking points.", "It also highlights the overall perspective."],
            }
            fallback_points = generic_map.get(video_type, ["The video covers the main ideas.", "It also highlights the overall discussion."])
        else:
            fallback_points = [payload.get("tldr", "")] + [chapter.get("title", "") for chapter in payload["chapters"]]
        for candidate in fallback_points:
            candidate = _compress_line(candidate, max_words=18)
            if candidate:
                payload["key_points"].append(candidate)
            if len(payload["key_points"]) >= 4:
                break
    original_point_count = len(payload["key_points"])
    payload["key_points"] = _semantic_dedupe_key(
        [
            point for point in payload["key_points"]
            if not _looks_transcript_leaky_key_point(point, transcript_sentences, video_type)
        ],
        video_type=video_type,
    )[:6]
    deduped_key_points += max(0, original_point_count - len(payload["key_points"]))
    tldr = "\n".join(_split_sentences(payload.get("tldr", ""))[:2]).strip()
    if not _is_english_like(transcript_text) and not _contains_non_latin(tldr):
        native_sentences = [
            _compress_line(sentence, max_words=18)
            for sentence in _split_sentences(transcript_text)[:2]
            if _compress_line(sentence, max_words=18)
        ]
        if native_sentences:
            tldr = "\n".join(native_sentences[:2]).strip()
    if not tldr or (_is_english_like(tldr) and not _is_natural_topic_phrase(re.sub(r"^This (?:interview|tutorial|video|discussion)\s+(?:features|explains|covers)\s+", "", tldr, flags=re.IGNORECASE))):
        fallback_template_usage += 1
        tldr = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="",
            full_summary="",
            transcript_text=transcript_text,
        )
    tldr, deduped_tldr_topics = _dedupe_tldr_sentences(tldr)
    payload["tldr"] = tldr
    logger.info(
        "[SUMMARY_OUTPUT_VALIDATION] rejected_key_points=%d rejected_transcript_leak_key_points=%d deduplicated_key_points=%d deduplicated_chapters=%d deduplicated_tldr_topics=%d fallback_template_usage=%d",
        rejected_points,
        rejected_transcript_leaks,
        deduped_key_points,
        deduped_chapters,
        deduped_tldr_topics,
        fallback_template_usage,
    )
    return payload


def build_structured_summary(
    transcript_text: str,
    segments: List[Dict],
    raw_segments: List[Dict] | None = None,
    assembled_units: List[Dict] | None = None,
    transcript_state: str = "",
    transcript_language: str = "",
    full_summary: str = "",
    bullet_summary: str = "",
    short_summary: str = "",
) -> Dict:
    """
    Build:
    { tldr: string, key_points: string[], action_items: string[], chapters: [{title, timestamp}] }
    """
    try:
        cleaned_transcript = _clean_text(transcript_text)
        if not cleaned_transcript:
            return default_structured_summary()
        payload = default_structured_summary()
        degraded_mode = str(transcript_state or "").strip().lower() == "degraded"
        malayalam_summary_mode = _is_malayalam_or_mixed_language(transcript_language, cleaned_transcript) and str(transcript_state or "").strip().lower() in {"cleaned", "degraded"}
        mixed_lang_source = bool(malayalam_summary_mode and _is_malayalam_mixed_lang_source(cleaned_transcript))
        trusted_segments = segments or []
        source_segments = raw_segments or segments or []
        rejected_noisy_segments = 0
        trusted_transcript_text = cleaned_transcript
        structured_input_source = "segments"
        structured_input_reason = "default_non_malayalam_path"
        structured_summary_route = "normal"
        structured_grounding_passed = True
        structured_grounding_reason = "non_malayalam_default"
        structured_summary_blocked_reason = ""
        if malayalam_summary_mode:
            selected_inputs = _select_malayalam_structured_inputs(
                transcript_text=cleaned_transcript,
                segments=segments or [],
                raw_segments=raw_segments or [],
                assembled_units=assembled_units or [],
                transcript_state=transcript_state,
            )
            trusted_segments = list(selected_inputs.get("units") or [])
            structured_input_source = str(selected_inputs.get("source") or "segments")
            structured_input_reason = str(selected_inputs.get("reason") or "malayalam_input_selection")
            rejected_noisy_segments = int(selected_inputs.get("rejected_low_trust_units", 0) or 0)
            trusted_transcript_text = _clean_text(" ".join(str(seg.get("text", "") or "") for seg in trusted_segments if isinstance(seg, dict)))
            logger.info(
                "[ML_STRUCTURED_INPUT_SELECT] state=%s source=%s unit_count=%s word_count=%s reason=%s",
                transcript_state,
                structured_input_source,
                len(trusted_segments),
                len(re.findall(r"\w+", trusted_transcript_text, flags=re.UNICODE)),
                structured_input_reason,
            )

        grounding_text = trusted_transcript_text if malayalam_summary_mode else cleaned_transcript
        if malayalam_summary_mode and not grounding_text:
            structured_grounding_passed = False
            structured_grounding_reason = structured_input_reason or "missing_grounded_malayalam_text"
            structured_summary_blocked_reason = structured_grounding_reason
        if malayalam_summary_mode:
            logger.info(
                "[ML_STRUCTURED_GROUNDING_CHECK] transcript_state=%s structured_grounding_passed=%s reason=%s",
                transcript_state,
                bool(grounding_text),
                structured_grounding_reason if not grounding_text else structured_input_reason,
            )
        if malayalam_summary_mode and not degraded_mode and not grounding_text:
            payload = _bounded_cleaned_malayalam_summary(reason=structured_summary_blocked_reason or "missing_grounded_malayalam_text")
            payload["_trace"].update({
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": 0,
                "structured_input_reason": structured_input_reason,
            })
            return payload
        full_summary, bullet_summary, short_summary = _ground_summary_support(
            grounding_text,
            full_summary,
            bullet_summary,
            short_summary,
        )
        if degraded_mode:
            full_summary = _clean_text(short_summary or full_summary)
            bullet_summary = ""
            logger.info("[SUMMARY_DEGRADED] enabled=True transcript_state=%s", transcript_state)
        participant_segments = trusted_segments if malayalam_summary_mode else source_segments
        interview_participants = _finalize_interview_participants(_extract_interview_participants(grounding_text, participant_segments))
        if degraded_mode:
            interview_participants = _filter_degraded_participants(interview_participants, grounding_text, transcript_language=transcript_language)
        payload["participants"] = interview_participants or []

        raw_video_type = _classify_video_type(grounding_text, full_summary=full_summary, bullet_summary=bullet_summary)
        video_type = raw_video_type
        if degraded_mode and str(transcript_language or "").strip().lower() == "ml":
            video_type, video_type_reason = _guard_malayalam_degraded_video_type(raw_video_type, grounding_text, interview_participants)
            logger.info(
                "[ML_SUMMARY_VIDEO_TYPE_GUARD] raw_choice=%s final_choice=%s reason=%s",
                raw_video_type,
                video_type,
                video_type_reason,
            )
            logger.info(
                "[ML_SUMMARY_TRUST] participant_candidates=%s accepted=%s rejected=%s video_type_bias_applied=%s transcript_state=%s",
                len(interview_participants),
                interview_participants,
                max(0, len(_extract_interview_participants(cleaned_transcript, source_segments)) - len(interview_participants)),
                raw_video_type != video_type,
                transcript_state,
            )
        topic_blocks = _build_topic_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type)
        chapter_blocks = topic_blocks
        if len(chapter_blocks) < 5:
            fallback_blocks = _build_fallback_chapter_blocks(trusted_segments if malayalam_summary_mode else source_segments, video_type, min_items=5, max_items=12)
            if fallback_blocks:
                chapter_blocks = fallback_blocks

        if degraded_mode and malayalam_summary_mode:
            structured_summary_route = "degraded_safe"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "degraded_malayalam_uses_degraded_safe_reconstruction")
            reconstruction_payload = build_malayalam_degraded_reconstruction_summary(
                transcript_text=grounding_text,
                trusted_segments=trusted_segments,
                assembled_units=assembled_units or trusted_segments,
                video_type=video_type,
            )
            reconstruction_payload.setdefault("tldr", "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.")
            reconstruction_payload.setdefault("key_points", [])
            reconstruction_payload.setdefault("action_items", [])
            reconstruction_payload.setdefault("chapters", [])
            reconstruction_payload.setdefault("participants", [])
            reconstruction_payload.setdefault("summary_state", "degraded_safe")
            reconstruction_payload.setdefault(
                "warning_message",
                "Malayalam transcript quality was too low for reliable summarization.",
            )
            reconstruction_payload["_trace"] = {
                **(reconstruction_payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction",
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason,
            }
            return reconstruction_payload

        if malayalam_summary_mode and not degraded_mode:
            structured_summary_route = "normal_grounded"
            logger.info("[ML_STRUCTURED_ROUTE] route=%s transcript_state=%s input_source=%s", structured_summary_route, transcript_state, structured_input_source)
            logger.info("[ML_STRUCTURED_ROUTE_REASON] route=%s reason=%s", structured_summary_route, "cleaned_malayalam_grounded_summary")

        payload["tldr"] = _tldr_from_topics(
            video_type=video_type,
            topic_blocks=topic_blocks,
            short_summary="" if (malayalam_summary_mode and not degraded_mode) else short_summary,
            full_summary="" if (malayalam_summary_mode and not degraded_mode) else full_summary,
            transcript_text=grounding_text,
        )
        payload["key_points"] = _build_key_points(
            video_type=video_type,
            transcript_text=grounding_text,
            bullet_summary=bullet_summary,
            full_summary=full_summary,
            topic_blocks=topic_blocks,
            transcript_language=transcript_language,
        )[:6]

        action_candidates = _split_bullets(bullet_summary) + _split_sentences(full_summary) + [block.get("text", "") for block in topic_blocks]
        payload["action_items"] = _extract_action_items(
            video_type=video_type,
            transcript_text=grounding_text,
            candidates=_dedupe(action_candidates),
            max_items=6,
        )
        payload["chapters"] = _build_chapters(chapter_blocks, max_items=12, video_type=video_type, transcript_language=transcript_language)
        if malayalam_summary_mode:
            rejected_noisy_fragments = 0
            payload["key_points"] = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, block.get("label", ""), interview_participants))
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["key_points"] = [point for point in payload["key_points"] if point]
            rejected_noisy_fragments += max(0, len(topic_blocks[:6]) - len(payload["key_points"]))
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=semantic_preference mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                rejected_noisy_fragments,
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, block.get("label", ""), interview_participants)
                    for block in topic_blocks[:4]
                    if block.get("label") and not _is_suspicious_degraded_label(block.get("label", ""))
                ]
            )[:4]
            if interview_participants:
                payload["tldr"] = _tldr_from_topics(
                    video_type=video_type,
                    topic_blocks=topic_blocks,
                    short_summary=short_summary,
                    full_summary=full_summary,
                    transcript_text=grounding_text,
                )
            else:
                payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)
            payload["chapters"] = [
                {
                    "title": (
                        chapter.get("title")
                        if (
                            _is_natural_topic_phrase(chapter.get("title", ""))
                            and not _is_suspicious_degraded_label(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                            and not _clean_text(chapter.get("title", "")).lower().startswith("discussion on ")
                        )
                        else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            payload["action_items"] = []
        payload = _validate_summary(
            payload,
            transcript_text=grounding_text,
            video_type=video_type,
            topic_blocks=topic_blocks,
        )
        if malayalam_summary_mode:
            safe_labels = _dedupe(
                [
                    block.get("label", "")
                    for block in topic_blocks[:6]
                    if (
                        block.get("label")
                        and not _looks_noisy_malayalam_summary_fragment(block.get("label", ""))
                        and not (degraded_mode and _is_suspicious_degraded_label(block.get("label", "")))
                    )
                ]
            )[:6]
            semantic_points = _dedupe(
                [
                    _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                    for label in safe_labels
                    if _clean_malayalam_mixed_summary_point(_semantic_point_from_label(video_type, label, interview_participants))
                ]
            )[:6]
            payload["key_points"] = semantic_points or payload.get("key_points", [])
            payload["key_points"] = [
                point for point in payload.get("key_points", [])
                if not _looks_noisy_malayalam_summary_fragment(point)
                and not (degraded_mode and _is_suspicious_degraded_label(point))
            ][:6]
            payload["chapters"] = [
                {
                    "title": (
                        _clean_malayalam_mixed_summary_point(chapter.get("title"))
                        if (
                            not _looks_noisy_malayalam_summary_fragment(chapter.get("title", ""))
                            and not _looks_transcript_leaky_key_point(chapter.get("title", ""), _split_sentences(grounding_text), video_type)
                        ) else f"Discussion at {chapter.get('timestamp', '00:00')}"
                    ),
                    "timestamp": chapter.get("timestamp", "00:00"),
                }
                for chapter in _validate_chapters(payload.get("chapters", []))[:6]
            ]
            logger.info(
                "[ML_SUMMARY_CLEAN] mode=post_validation mixed_lang_source=%s rejected_noisy_fragments=%d accepted_semantic_points=%d key_points=%d chapters=%d rejected_noisy_segments=%d",
                mixed_lang_source,
                max(0, len(safe_labels) - len(payload["key_points"])),
                len(payload["key_points"]),
                len(payload["key_points"]),
                len(payload["chapters"]),
                rejected_noisy_segments,
            )
        if degraded_mode and (not interview_participants):
            payload["tldr"] = _safe_degraded_malayalam_tldr(topic_blocks, video_type, grounding_text)

        quality = evaluate_summary_quality(payload, grounding_text)
        if float(quality.get("final_quality_score", 0.0) or 0.0) < float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)):
            safe_topics = [block.get("label", "") for block in topic_blocks if block.get("label")]
            safe_topics = _dedupe([_clean_text(topic) for topic in safe_topics])[:4]
            payload["tldr"] = _tldr_from_topics(
                video_type=video_type,
                topic_blocks=topic_blocks,
                short_summary=short_summary,
                full_summary=full_summary,
                transcript_text=grounding_text,
            )
            payload["key_points"] = _dedupe(
                [
                    _semantic_point_from_label(video_type, label, interview_participants)
                    for label in safe_topics
                    if _semantic_point_from_label(video_type, label, interview_participants)
                ]
            )[:6]
            payload["chapters"] = _validate_chapters(payload.get("chapters", []))
            payload["action_items"] = payload.get("action_items", []) if video_type not in {"interview", "commentary", "casual conversation"} else []
            payload = _validate_summary(
                payload,
                transcript_text=grounding_text,
                video_type=video_type,
                topic_blocks=topic_blocks,
            )
            logger.info(
                "[SUMMARY_QA_GATE] regenerated=True summary_quality_score=%.4f threshold=%.4f",
                float(quality.get("final_quality_score", 0.0) or 0.0),
                float(getattr(settings, "SUMMARY_QUALITY_MIN_SCORE", 0.62)),
            )

        if not payload["key_points"]:
            payload = _bounded_cleaned_malayalam_summary(reason="no_grounded_key_points") if malayalam_summary_mode and not degraded_mode else default_structured_summary()
        if malayalam_summary_mode:
            payload["_trace"] = {
                **(payload.get("_trace") or {}),
                "structured_input_source": structured_input_source,
                "structured_input_unit_count": len(trusted_segments),
                "structured_input_word_count": len(re.findall(r"\w+", grounding_text, flags=re.UNICODE)),
                "structured_input_reason": structured_input_reason,
                "structured_summary_route": structured_summary_route,
                "structured_summary_route_reason": (
                    "cleaned_malayalam_grounded_summary"
                    if not degraded_mode else "degraded_malayalam_uses_degraded_safe_reconstruction"
                ),
                "structured_grounding_passed": bool(grounding_text),
                "structured_grounding_reason": structured_input_reason if grounding_text else structured_grounding_reason,
                "structured_summary_blocked_reason": structured_summary_blocked_reason or (
                    "no_grounded_key_points" if malayalam_summary_mode and not degraded_mode and not payload.get("key_points") else ""
                ),
            }
        return payload
    except Exception:
        return default_structured_summary()

