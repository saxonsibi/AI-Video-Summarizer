"""
Standalone Malayalam WER/CER evaluation harness.

This script is intentionally outside the production pipeline. It benchmarks
Malayalam ASR output across a small ablation matrix and writes CSV/JSON results
 under backend/videos/eval_results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "videoiq.settings")

import django  # noqa: E402

django.setup()

from django.conf import settings  # noqa: E402

from videos.asr_router import transcribe_video_router  # noqa: E402
from videos.tasks import _prepare_audio_for_pipeline  # noqa: E402


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "malayalam_eval"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
RESULTS_DIR = Path(__file__).resolve().parent / "eval_results"

VARIANTS = [
    {"name": "dynaudnorm=off filters=off", "dynaudnorm": False, "speech_band_filter": False},
    {"name": "dynaudnorm=on filters=off", "dynaudnorm": True, "speech_band_filter": False},
    {"name": "dynaudnorm=off filters=on", "dynaudnorm": False, "speech_band_filter": True},
    {"name": "dynaudnorm=on filters=on", "dynaudnorm": True, "speech_band_filter": True},
]


@dataclass
class FixtureCase:
    case_id: str
    label: str
    audio_file: str
    reference_file: str
    tags: List[str]
    notes: str = ""


@contextmanager
def temporary_settings(**overrides):
    original = {}
    for key, value in overrides.items():
        original[key] = getattr(settings, key, None)
        setattr(settings, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


def load_fixture_cases(manifest_path: Path) -> List[FixtureCase]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fixture manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = payload.get("fixtures", []) if isinstance(payload, dict) else []
    return [FixtureCase(**item) for item in cases]


def normalize_eval_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def compute_wer_cer(hypothesis: str, reference: str) -> dict:
    """Return {"wer": float, "cer": float}."""
    try:
        import jiwer
    except Exception as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "jiwer is required for Malayalam WER/CER evaluation. Install it in the backend environment."
        ) from exc

    hypothesis = normalize_eval_text(hypothesis)
    reference = normalize_eval_text(reference)
    return {
        "wer": float(jiwer.wer(reference, hypothesis)),
        "cer": float(jiwer.cer(reference, hypothesis)),
    }


def run_variant(audio_path, config) -> str:
    """Run preprocessing + ASR with given config, return transcript text."""
    audio_path = str(audio_path)
    with temporary_settings():
        prepared_audio_path, prep_meta = _prepare_audio_for_pipeline(
            audio_path,
            apply_dynaudnorm=bool(config.get("dynaudnorm", False)),
            apply_speech_band_filter=bool(config.get("speech_band_filter", False)),
        )
        payload = transcribe_video_router(
            audio_path=prepared_audio_path,
            source_type="eval_fixture",
            requested_language="ml",
            chunks=prep_meta.get("chunks"),
        )
    return normalize_eval_text(payload.get("text", ""))


def _fixture_paths(case: FixtureCase) -> tuple[Path, Path]:
    return FIXTURE_DIR / case.audio_file, FIXTURE_DIR / case.reference_file


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def run_matrix(cases: List[FixtureCase], manifest_path: Path) -> Dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    skipped = []

    for variant in VARIANTS:
        per_clip = []
        for case in cases:
            audio_path, ref_path = _fixture_paths(case)
            if not audio_path.exists() or not ref_path.exists():
                skipped.append({
                    "variant": variant["name"],
                    "case_id": case.case_id,
                    "audio_exists": audio_path.exists(),
                    "reference_exists": ref_path.exists(),
                })
                continue
            reference = ref_path.read_text(encoding="utf-8")
            hypothesis = run_variant(audio_path, variant)
            metrics = compute_wer_cer(hypothesis, reference)
            per_clip.append({
                "case_id": case.case_id,
                "label": case.label,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
            })

        row = {
            "variant": variant["name"],
            "dynaudnorm": bool(variant["dynaudnorm"]),
            "speech_band_filter": bool(variant["speech_band_filter"]),
            "clips_evaluated": len(per_clip),
            "mean_wer": _mean(item["wer"] for item in per_clip),
            "mean_cer": _mean(item["cer"] for item in per_clip),
            "per_clip": per_clip,
        }
        rows.append(row)

    csv_path = RESULTS_DIR / f"malayalam_wer_matrix_{timestamp}.csv"
    json_path = RESULTS_DIR / f"malayalam_wer_matrix_{timestamp}.json"

    case_ids = [case.case_id for case in cases]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["variant", *[f"{case_id}_wer" for case_id in case_ids], "mean_wer", "mean_cer"]
        writer.writerow(header)
        for row in rows:
            clip_map = {item["case_id"]: item for item in row["per_clip"]}
            writer.writerow([
                row["variant"],
                *[
                    f"{clip_map[case_id]['wer']:.4f}" if case_id in clip_map else ""
                    for case_id in case_ids
                ],
                f"{row['mean_wer']:.4f}",
                f"{row['mean_cer']:.4f}",
            ])

    report = {
        "generated_at": timestamp,
        "manifest_path": str(manifest_path),
        "results_csv": str(csv_path),
        "results_json": str(json_path),
        "warning": "",
        "fixtures": [asdict(case) for case in cases],
        "variants": rows,
        "skipped": skipped,
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_report(report: Dict[str, object]) -> None:
    print("Variant                          | Mean WER | Mean CER | Clips")
    print("---------------------------------|----------|----------|------")
    for row in report.get("variants", []):
        print(
            f"{str(row.get('variant', ''))[:32]:32} | "
            f"{float(row.get('mean_wer', 0.0)):.4f}   | "
            f"{float(row.get('mean_cer', 0.0)):.4f}   | "
            f"{int(row.get('clips_evaluated', 0))}"
        )
    skipped = list(report.get("skipped", []) or [])
    if skipped:
        print("")
        print("Skipped fixtures:")
        for item in skipped:
            print(
                f"- {item['case_id']} ({item['variant']}): "
                f"audio_exists={item['audio_exists']} reference_exists={item['reference_exists']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Malayalam WER/CER evaluation harness")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    cases = load_fixture_cases(manifest_path)
    report = run_matrix(cases, manifest_path)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
