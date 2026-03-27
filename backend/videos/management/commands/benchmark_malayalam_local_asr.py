"""
Benchmark Malayalam local ASR profiles on a single local sample file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.test.utils import override_settings

from videos.tasks import _compute_transcript_state
from videos.utils import (
    _get_audio_duration_seconds,
    _transcribe_with_faster_whisper,
    clean_transcript,
    extract_audio,
)


VIDEO_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
}


def _prepare_audio_input(sample_path: str) -> tuple[str, bool]:
    path = Path(sample_path)
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return extract_audio(str(path)), True
    return str(path), False


def _selected_model_path(metadata: dict) -> str:
    if not isinstance(metadata, dict):
        return ""
    return str(
        metadata.get("resolved_local_model_name")
        or metadata.get("resolved_model_name")
        or metadata.get("actual_local_model_name")
        or metadata.get("configured_model_name")
        or ""
    ).strip()


def _benchmark_profile(
    *,
    audio_path: str,
    profile_name: str,
    fast_primary_model: str,
    allow_large_fallback: bool,
) -> dict:
    profile_settings = {
        "ASR_MALAYALAM_PRIMARY_MODEL": "",
        "WHISPER_MODEL_MALAYALAM_PRIMARY": "",
        "WHISPER_MODEL_MALAYALAM_SECONDARY": "",
        "ASR_MALAYALAM_FAST_PRIMARY_MODEL": fast_primary_model,
        "WHISPER_MODEL_MALAYALAM_FALLBACK": "large-v3",
        "ASR_MALAYALAM_ENABLE_LARGE_FALLBACK": allow_large_fallback,
    }
    audio_duration = max(0.0, float(_get_audio_duration_seconds(audio_path) or 0.0))
    started = time.perf_counter()
    with override_settings(**profile_settings):
        payload = _transcribe_with_faster_whisper(
            audio_path=audio_path,
            source_type="file",
            transcription_language="ml",
        )
    elapsed = round(time.perf_counter() - started, 3)

    cleaned = clean_transcript(payload.get("segments", []) or [])
    cleaned_text = str(cleaned.get("full_text") or payload.get("text") or "").strip()
    cleaned_segments = cleaned.get("segments") or payload.get("segments") or []
    quality = _compute_transcript_state(
        cleaned_text=cleaned_text,
        cleaned_segments=cleaned_segments,
        transcript_payload={**payload, "text": cleaned_text},
        audio_duration_seconds=audio_duration,
        transcript_language="ml",
    )
    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    selected_model_path = _selected_model_path(metadata)

    return {
        "profile": profile_name,
        "requested_fast_primary_model": fast_primary_model or "",
        "allow_large_v3_fallback": bool(allow_large_fallback),
        "total_transcription_time_seconds": elapsed,
        "real_time_factor": round((elapsed / audio_duration), 4) if audio_duration > 0 else 0.0,
        "audio_duration_seconds": round(audio_duration, 3),
        "transcript_state": str(quality.get("state", "") or ""),
        "fallback_triggered": bool(metadata.get("fallback_triggered", False)),
        "fallback_reason": str(metadata.get("fallback_reason", "") or ""),
        "selected_model_path": selected_model_path,
        "selected_model_name": str(metadata.get("actual_local_model_name") or metadata.get("configured_model_name") or ""),
    }


class Command(BaseCommand):
    help = "Benchmark Malayalam local ASR on one sample file for small-first vs large-v3 profiles."

    def add_arguments(self, parser):
        parser.add_argument(
            "sample_path",
            type=str,
            help="Path to a local Malayalam audio or video file.",
        )
        parser.add_argument(
            "--small-model",
            type=str,
            default=str(getattr(settings, "ASR_MALAYALAM_FAST_PRIMARY_MODEL", "small") or "small"),
            help="Malayalam fast primary model to benchmark against large-v3 (default: current ASR_MALAYALAM_FAST_PRIMARY_MODEL or small).",
        )
        parser.add_argument(
            "--disable-small-fallback",
            action="store_true",
            help="Disable large-v3 fallback during the small-first profile benchmark.",
        )

    def handle(self, *args, **options):
        sample_path = str(options.get("sample_path") or "").strip()
        if not sample_path:
            raise CommandError("sample_path is required.")
        if not Path(sample_path).exists():
            raise CommandError(f"Sample file not found: {sample_path}")

        small_model = str(options.get("small_model") or "small").strip() or "small"
        allow_small_fallback = not bool(options.get("disable_small_fallback", False))

        audio_path = sample_path
        cleanup_audio = False
        try:
            audio_path, cleanup_audio = _prepare_audio_input(sample_path)
            small_first = _benchmark_profile(
                audio_path=audio_path,
                profile_name="small_first_pass",
                fast_primary_model=small_model,
                allow_large_fallback=allow_small_fallback,
            )
            large_v3 = _benchmark_profile(
                audio_path=audio_path,
                profile_name="large_v3",
                fast_primary_model="",
                allow_large_fallback=False,
            )
        finally:
            if cleanup_audio:
                try:
                    Path(audio_path).unlink(missing_ok=True)
                except Exception:
                    pass

        report = {
            "sample_path": sample_path,
            "benchmarks": [small_first, large_v3],
        }
        self.stdout.write(json.dumps(report, indent=2))
