"""
Compatibility views for the Chrome extension API contract.

Contract:
- POST /api/extension/summarize {url, style}
- GET  /api/extension/status?job_id=...
- GET  /api/extension/result?job_id=...
- POST /api/extension/chat {job_id, message}
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import status, views
from rest_framework.parsers import JSONParser
from rest_framework.response import Response

from videos.models import Summary, Transcript, Video
from videos.views import _launch_youtube_processing
from videos.utils import normalize_language_code
from videos.language import detect_text_language
from videos.translation import translate_text

logger = logging.getLogger(__name__)


def _map_video_status(video_status: str) -> str:
    value = (video_status or "").lower()
    if value in {"pending"}:
        return "queued"
    if value in {"processing", "transcribing", "summarizing"}:
        return "processing"
    if value in {"completed"}:
        return "ready"
    if value in {"failed"}:
        return "error"
    return "processing"


def _is_valid_youtube_url(url: str) -> bool:
    patterns = [
        r"(?:https?://)?(?:www\.|m\.)?youtube\.com/watch\?v=[\w-]+",
        r"(?:https?://)?(?:www\.)?youtu\.be/[\w-]+",
        r"(?:https?://)?(?:www\.|m\.)?youtube\.com/shorts/[\w-]+",
        r"(?:https?://)?(?:www\.|m\.)?youtube\.com/embed/[\w-]+",
        r"(?:https?://)?(?:www\.|m\.)?youtube\.com/v/[\w-]+",
    ]
    return any(re.match(pattern, url) for pattern in patterns)


def _extract_transcript_segments(transcript: Transcript | None) -> List[Dict[str, Any]]:
    if not transcript:
        return []

    payload = transcript.json_data
    if isinstance(payload, dict):
        payload = payload.get("segments", [])

    segments: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = float(item.get("start", item.get("start_time", idx * 5)) or 0)
            end = float(item.get("end", item.get("end_time", start + 5)) or (start + 5))
            segments.append({"start": start, "end": end, "text": text})

    if segments:
        return segments

    # Fallback from plain transcript text
    text = transcript.full_text or ""
    if not text.strip():
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    for idx, sent in enumerate(sentences):
        start = float(idx * 5)
        segments.append({"start": start, "end": start + 5.0, "text": sent})
    return segments


def _summary_by_type(video: Video, summary_type: str) -> str:
    obj = (
        Summary.objects.filter(video=video, summary_type=summary_type)
        .order_by("-created_at")
        .first()
    )
    return obj.content.strip() if obj and obj.content else ""


def _parse_bullets(bullet_text: str) -> List[str]:
    if not bullet_text:
        return []
    lines = [line.strip() for line in bullet_text.splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        line = re.sub(r"^[-*•\d\.\)\s]+", "", line).strip()
        if line:
            cleaned.append(line)
    return cleaned[:10]


def _timeline_from_segments(segments: List[Dict[str, Any]]) -> List[str]:
    if not segments:
        return []
    picks = [segments[0]]
    if len(segments) >= 3:
        picks.append(segments[len(segments) // 2])
        picks.append(segments[-1])
    timeline = []
    for seg in picks[:5]:
        ts = int(seg["start"])
        timeline.append(f"{ts}s: {seg['text'][:140].strip()}")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for item in timeline:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _entities_from_text(text: str) -> List[str]:
    # Light heuristic: collect capitalized multi-letter tokens, dedupe.
    tokens = re.findall(r"\b[A-Z][a-zA-Z0-9][a-zA-Z0-9'-]*\b", text or "")
    deny = {"The", "And", "But", "This", "That", "With", "From", "Into", "When", "What", "How"}
    entities = []
    seen = set()
    for tok in tokens:
        if tok in deny:
            continue
        if tok not in seen:
            seen.add(tok)
            entities.append(tok)
        if len(entities) >= 12:
            break
    return entities


def _parse_source_timestamp(source_ts: str) -> float:
    """
    Parse formats like:
    - "12.3s - 22.9s"
    - "15s"
    """
    if not source_ts:
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s*s", source_ts)
    if not match:
        return 0.0
    return float(match.group(1))


class ExtensionSummarizeView(views.APIView):
    parser_classes = [JSONParser]

    def post(self, request):
        url = str(request.data.get("url", "")).strip()
        style = str(request.data.get("style", "detailed")).strip() or "detailed"
        transcription_language = normalize_language_code(
            request.data.get("transcription_language"),
            default='auto',
            allow_auto=True
        )
        output_language = normalize_language_code(
            request.data.get("output_language"),
            default='auto',
            allow_auto=True
        )
        summary_language_mode = (
            str(request.data.get("summary_language_mode", "same_as_transcript")).strip().lower()
            or "same_as_transcript"
        )

        if not url:
            return Response({"error": "url is required"}, status=status.HTTP_400_BAD_REQUEST)
        if not _is_valid_youtube_url(url):
            return Response({"error": "Invalid YouTube URL format"}, status=status.HTTP_400_BAD_REQUEST)

        video = Video.objects.create(
            title="YouTube Video",
            description=f"Extension style={style}",
            youtube_url=url,
            status="processing",
        )
        _launch_youtube_processing(
            str(video.id),
            transcription_language=transcription_language,
            output_language=output_language,
            summary_language_mode=summary_language_mode
        )

        return Response(
            {
                "job_id": str(video.id),
                "status": _map_video_status(video.status),
                "style": style,
                "transcription_language": transcription_language,
                "output_language": output_language,
                "summary_language_mode": summary_language_mode,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class ExtensionStatusView(views.APIView):
    def get(self, request):
        job_id = str(request.query_params.get("job_id", "")).strip()
        if not job_id:
            return Response({"error": "job_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        video = get_object_or_404(Video, id=job_id)
        mapped = _map_video_status(video.status)
        payload = {
            "job_id": str(video.id),
            "status": mapped,
            "progress": int(video.processing_progress or 0),
            "error": video.error_message or "",
        }
        return Response(payload)


class ExtensionResultView(views.APIView):
    def get(self, request):
        job_id = str(request.query_params.get("job_id", "")).strip()
        if not job_id:
            return Response({"error": "job_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        video = get_object_or_404(Video, id=job_id)
        mapped_status = _map_video_status(video.status)
        if mapped_status != "ready":
            return Response(
                {"error": f"job is not ready (status={mapped_status})"},
                status=status.HTTP_409_CONFLICT,
            )

        transcript = Transcript.objects.filter(video=video).order_by("-created_at").first()
        segments = _extract_transcript_segments(transcript)

        full_summary = _summary_by_type(video, "full")
        bullet_summary = _summary_by_type(video, "bullet")
        short_summary = _summary_by_type(video, "short")

        bullet_items = _parse_bullets(bullet_summary)
        timeline = _timeline_from_segments(segments)
        action_items = bullet_items[:5]
        entities = _entities_from_text(full_summary or transcript.full_text if transcript else "")

        return Response(
            {
                "job_id": str(video.id),
                "title": video.title,
                "channel": "",
                "thumbnail": "",
                "summary": {
                    "full_summary": full_summary,
                    "key_takeaways": bullet_items,
                    "timeline": timeline,
                    "action_items": action_items,
                    "entities": entities,
                    "short": short_summary,
                },
                "transcript_segments": segments,
            }
        )


class ExtensionChatView(views.APIView):
    parser_classes = [JSONParser]

    def post(self, request):
        job_id = str(request.data.get("job_id", "")).strip()
        message = str(request.data.get("message", "")).strip()
        output_language = normalize_language_code(
            request.data.get("output_language") or request.data.get("response_language"),
            default='auto',
            allow_auto=True
        )
        if not job_id or not message:
            return Response(
                {"error": "job_id and message are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        video = get_object_or_404(Video, id=job_id)
        transcript = Transcript.objects.filter(video=video).order_by("-created_at").first()
        if not transcript:
            return Response(
                {"error": "Transcript not available for this video yet"},
                status=status.HTTP_409_CONFLICT,
            )

        from chatbot.rag_engine import ChatbotEngine

        chatbot = ChatbotEngine(str(video.id))
        if not chatbot.initialize():
            payload = transcript.json_data
            if isinstance(payload, dict):
                canonical_segments = payload.get("canonical_segments", [])
                if isinstance(canonical_segments, list) and canonical_segments:
                    chatbot.build_from_transcript(canonical_segments)
                else:
                    chatbot.build_from_transcript(_extract_transcript_segments(transcript))
            else:
                chatbot.build_from_transcript(_extract_transcript_segments(transcript))

        transcript_language = normalize_language_code(
            transcript.transcript_language or transcript.language,
            default='en',
            allow_auto=False
        )
        retrieval_language = normalize_language_code(
            transcript.canonical_language or 'en',
            default='en',
            allow_auto=False
        )
        if output_language == 'auto':
            output_language = transcript_language

        q_language, _, _, _ = detect_text_language(message, default='en')
        retrieval_query = message
        if q_language != retrieval_language:
            retrieval_query = translate_text(
                message,
                source_language=q_language,
                target_language=retrieval_language,
                preserve_format=False
            )

        result = chatbot.ask(
            retrieval_query,
            use_llm=bool(getattr(settings, "CHATBOT_USE_GROQ_LLM", False)),
            strict_mode=False,
            response_language=retrieval_language
        )

        answer_text = result.get("answer", "")
        if retrieval_language != output_language:
            answer_text = translate_text(
                answer_text,
                source_language=retrieval_language,
                target_language=output_language,
                preserve_format=True
            )

        citations = []
        for src in result.get("sources", [])[:6]:
            ts_raw = str(src.get("timestamp", ""))
            quote = str(src.get("text", "")).strip()
            if retrieval_language != output_language and quote:
                quote = translate_text(
                    quote,
                    source_language=retrieval_language,
                    target_language=output_language,
                    preserve_format=False
                )
            citations.append(
                {
                    "timestamp": _parse_source_timestamp(ts_raw),
                    "quote": quote,
                }
            )

        return Response(
            {
                "answer": answer_text,
                "answer_language": output_language,
                "citations": citations,
            }
        )
