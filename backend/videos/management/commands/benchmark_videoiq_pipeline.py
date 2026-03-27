"""
Run a small production-style benchmark over already processed videos.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from videos.benchmarking import benchmark_video, write_benchmark_outputs
from videos.models import Video


class Command(BaseCommand):
    help = "Benchmark transcript, summary, and chatbot quality on processed videos."

    def add_arguments(self, parser):
        parser.add_argument(
            "--video-id",
            action="append",
            dest="video_ids",
            default=[],
            help="Benchmark a specific video UUID. Pass multiple times for multiple videos.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=3,
            help="Number of latest completed videos to benchmark when --video-id is not provided.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="backend/benchmark_reports",
            help="Directory where JSON/CSV/checklist files will be written.",
        )
        parser.add_argument(
            "--report-name",
            type=str,
            default="",
            help="Optional report filename prefix. Defaults to a timestamped name.",
        )
        parser.add_argument(
            "--use-llm",
            action="store_true",
            help="Use the normal chatbot LLM answer path instead of the deterministic retrieval formatter.",
        )
        parser.add_argument(
            "--response-language",
            type=str,
            default="auto",
            help="Response language for chatbot probes (default: transcript language).",
        )
        parser.add_argument(
            "--include-moment-probes",
            action="store_true",
            help="Also benchmark timestamp-grounded moment explanation quality.",
        )

    def _target_videos(self, video_ids, limit):
        if video_ids:
            videos = list(Video.objects.filter(id__in=video_ids).order_by("-created_at"))
            if len(videos) != len(set(video_ids)):
                found = {str(video.id) for video in videos}
                missing = [video_id for video_id in video_ids if video_id not in found]
                raise CommandError(f"Video(s) not found: {', '.join(missing)}")
            return videos

        return list(
            Video.objects.filter(status__in=["completed", "indexing_chat", "summarizing_final", "transcript_ready"])
            .order_by("-created_at")[: max(1, int(limit or 1))]
        )

    def handle(self, *args, **options):
        videos = self._target_videos(options.get("video_ids") or [], options.get("limit") or 3)
        if not videos:
            raise CommandError("No eligible videos found for benchmarking.")

        output_dir = Path(options.get("output_dir") or "backend/benchmark_reports")
        report_name = (options.get("report_name") or "").strip() or f"videoiq_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        use_llm = bool(options.get("use_llm", False))
        response_language = (options.get("response_language") or "auto").strip() or "auto"
        include_moment_probes = bool(options.get("include_moment_probes", False))

        self.stdout.write(
            f"Benchmarking {len(videos)} video(s) "
            f"(use_llm={use_llm}, response_language={response_language}, include_moment_probes={include_moment_probes})"
        )

        results = []
        for video in videos:
            self.stdout.write(f"- {video.id} | {video.title}")
            results.append(
                benchmark_video(
                    video,
                    use_llm=use_llm,
                    response_language=response_language,
                    include_moment_probes=include_moment_probes,
                )
            )

        paths = write_benchmark_outputs(results, output_dir, report_name)
        self.stdout.write(self.style.SUCCESS("Benchmark completed."))
        self.stdout.write(f"JSON: {paths['json']}")
        self.stdout.write(f"CSV: {paths['csv']}")
        self.stdout.write(f"Checklist: {paths['checklist']}")
