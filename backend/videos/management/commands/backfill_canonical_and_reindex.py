"""
Backfill canonical transcript fields and rebuild chatbot RAG indexes.

Usage examples:
  python manage.py backfill_canonical_and_reindex
  python manage.py backfill_canonical_and_reindex --dry-run
  python manage.py backfill_canonical_and_reindex --video-id <uuid>
  python manage.py backfill_canonical_and_reindex --all-transcripts
  python manage.py backfill_canonical_and_reindex --skip-reindex
"""

from __future__ import annotations

from typing import List, Dict, Iterable, Optional

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from videos.models import Transcript, Video
from videos.canonical import build_canonical_text
from videos.language import normalize_language_code, detect_script_type
from chatbot.models import VideoIndex


def _extract_original_segments(json_data) -> List[Dict]:
    if isinstance(json_data, dict):
        segs = json_data.get("segments", [])
        return segs if isinstance(segs, list) else []
    if isinstance(json_data, list):
        return json_data
    return []


def _segmentize_text(text: str) -> List[Dict]:
    import re
    if not text:
        return []
    chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    out = []
    for i, c in enumerate(chunks):
        out.append({
            "id": i,
            "start": float(i * 5),
            "end": float((i + 1) * 5),
            "text": c,
            "original_text": c,
        })
    return out


class Command(BaseCommand):
    help = "Backfill canonical transcript data and rebuild RAG indexes for existing videos."

    def add_arguments(self, parser):
        parser.add_argument("--video-id", type=str, default="", help="Process only one video UUID.")
        parser.add_argument(
            "--all-transcripts",
            action="store_true",
            help="Process every transcript row (default: latest transcript per video).",
        )
        parser.add_argument(
            "--skip-reindex",
            action="store_true",
            help="Backfill transcript fields only, skip RAG index rebuild.",
        )
        parser.add_argument(
            "--canonical-language",
            type=str,
            default="",
            help="Override canonical language for this run (default from settings EMBED_CANONICAL_LANGUAGE).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Process at most N transcripts (0 means all selected).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be changed without writing DB/index files.",
        )

    def _iter_target_transcripts(self, video_id: str, all_transcripts: bool) -> Iterable[Transcript]:
        if video_id:
            qs = Transcript.objects.filter(video_id=video_id).order_by("-created_at")
            if all_transcripts:
                return qs.iterator()
            first = qs.first()
            return [first] if first else []

        if all_transcripts:
            return Transcript.objects.all().order_by("video_id", "-created_at").iterator()

        # Latest transcript per video.
        video_ids = Video.objects.values_list("id", flat=True)
        targets = []
        for vid in video_ids:
            tr = Transcript.objects.filter(video_id=vid).order_by("-created_at").first()
            if tr:
                targets.append(tr)
        return targets

    def handle(self, *args, **options):
        dry_run = bool(options.get("dry_run", False))
        all_transcripts = bool(options.get("all_transcripts", False))
        skip_reindex = bool(options.get("skip_reindex", False))
        video_id = (options.get("video_id") or "").strip()
        limit = int(options.get("limit") or 0)

        canonical_language = normalize_language_code(
            options.get("canonical_language") or getattr(settings, "EMBED_CANONICAL_LANGUAGE", "en"),
            default="en",
            allow_auto=False,
        )

        targets = list(self._iter_target_transcripts(video_id, all_transcripts))
        if limit > 0:
            targets = targets[:limit]

        if not targets:
            self.stdout.write(self.style.WARNING("No transcripts found for requested scope."))
            return

        self.stdout.write(
            f"Starting backfill for {len(targets)} transcript(s). "
            f"canonical_language={canonical_language}, dry_run={dry_run}, skip_reindex={skip_reindex}"
        )

        updated_count = 0
        reindex_ok = 0
        reindex_fail = 0
        errors = 0

        for tr in targets:
            try:
                source_language = normalize_language_code(
                    tr.transcript_language or tr.language or "en",
                    default="en",
                    allow_auto=False,
                )
                original_text = (tr.transcript_original_text or tr.full_text or "").strip()
                original_segments = _extract_original_segments(tr.json_data)
                if not original_segments:
                    original_segments = _segmentize_text(original_text)

                canonical = build_canonical_text(
                    transcript_text=original_text,
                    transcript_segments=original_segments,
                    transcript_language=source_language,
                    canonical_language=canonical_language,
                )

                script_type = detect_script_type(original_text)
                metadata_source = tr.json_data if isinstance(tr.json_data, dict) else {}
                asr_engine = "faster_whisper"
                if isinstance(metadata_source, dict):
                    asr_engine = (
                        metadata_source.get("metadata", {}).get("asr_provider_used")
                        if isinstance(metadata_source.get("metadata"), dict)
                        else None
                    ) or tr.asr_engine or "faster_whisper"
                quality_score = float(tr.transcript_quality_score or 0.0)

                if dry_run:
                    self.stdout.write(
                        f"[DRY] video={tr.video_id} transcript={tr.id} "
                        f"src={source_language} canon={canonical.get('canonical_language')} "
                        f"segs={len(canonical.get('canonical_segments', []))}"
                    )
                    updated_count += 1
                else:
                    with transaction.atomic():
                        tr.language = source_language
                        tr.transcript_language = source_language
                        tr.canonical_language = canonical.get("canonical_language", canonical_language)
                        tr.script_type = script_type
                        tr.asr_engine = asr_engine
                        tr.asr_engine_used = asr_engine
                        if not tr.detection_confidence:
                            tr.detection_confidence = 0.0
                        tr.transcript_quality_score = quality_score
                        tr.full_text = original_text or tr.full_text
                        tr.transcript_original_text = original_text
                        tr.transcript_canonical_text = canonical.get("canonical_text", original_text)
                        tr.transcript_canonical_en_text = canonical.get("canonical_text", original_text)
                        tr.json_data = {
                            "segments": original_segments,
                            "canonical_segments": canonical.get("canonical_segments", []),
                        }
                        tr.save(
                            update_fields=[
                                "language",
                                "transcript_language",
                                "canonical_language",
                                "script_type",
                                "asr_engine",
                                "asr_engine_used",
                                "detection_confidence",
                                "transcript_quality_score",
                                "full_text",
                                "transcript_original_text",
                                "transcript_canonical_text",
                                "transcript_canonical_en_text",
                                "json_data",
                            ]
                        )
                    updated_count += 1

                if skip_reindex:
                    continue

                canonical_segments = canonical.get("canonical_segments", [])
                if not canonical_segments:
                    canonical_segments = _segmentize_text(canonical.get("canonical_text", ""))

                if not canonical_segments:
                    self.stdout.write(
                        self.style.WARNING(f"[SKIP-INDEX] video={tr.video_id} no canonical segments available")
                    )
                    reindex_fail += 1
                    continue

                if dry_run:
                    self.stdout.write(
                        f"[DRY-INDEX] video={tr.video_id} segments={len(canonical_segments)}"
                    )
                    reindex_ok += 1
                else:
                    from chatbot.rag_engine import ChatbotEngine
                    engine = ChatbotEngine(str(tr.video_id))
                    ok = engine.build_from_transcript(canonical_segments)
                    if ok:
                        VideoIndex.objects.update_or_create(
                            video_id=tr.video_id,
                            defaults={
                                "is_indexed": True,
                                "index_created_at": timezone.now(),
                                "num_documents": len(engine.rag_engine.documents),
                                "embedding_model": getattr(settings, "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
                            },
                        )
                        reindex_ok += 1
                    else:
                        reindex_fail += 1
                        self.stdout.write(self.style.WARNING(f"[INDEX-FAIL] video={tr.video_id}"))

            except Exception as e:
                errors += 1
                self.stdout.write(self.style.ERROR(f"[ERROR] transcript={getattr(tr, 'id', '?')} detail={e}"))

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Backfill completed."))
        self.stdout.write(f"  updated_transcripts: {updated_count}")
        self.stdout.write(f"  reindex_ok:          {reindex_ok}")
        self.stdout.write(f"  reindex_fail:        {reindex_fail}")
        self.stdout.write(f"  errors:              {errors}")
