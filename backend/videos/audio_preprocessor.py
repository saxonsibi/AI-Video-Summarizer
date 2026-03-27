from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


LUFS_TARGET = -16.0
LUFS_TOLERANCE = 1.0
MAX_CHUNK_DURATION_S = 30
MIN_SILENCE_DURATION_MS = 500
SILENCE_THRESHOLD_DBFS = -40

_MIN_CHUNK_DURATION_S = 3.0
_FFMPEG_NULL_SINK = "NUL"


@dataclass(frozen=True)
class NormalizationResult:
    input_lufs: float
    output_lufs: float
    was_reencoded: bool
    output_path: str


@dataclass(frozen=True)
class ChunkMetadata:
    chunk_id: int
    start_s: float
    end_s: float
    path: str
    duration_s: float


@dataclass(frozen=True)
class ConditioningResult:
    input_path: str
    output_path: str
    dynaudnorm_applied: bool
    speech_band_filter_applied: bool
    was_noop: bool


def _run_command(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    )


def _probe_duration_seconds(path: str) -> float:
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
    )
    try:
        return max(0.0, float((result.stdout or "").strip() or 0.0))
    except ValueError:
        return 0.0


def _extract_last_json_block(stderr_text: str) -> dict:
    text = str(stderr_text or "")
    stack = 0
    end = None
    for index in range(len(text) - 1, -1, -1):
        char = text[index]
        if char == "}":
            if end is None:
                end = index
            stack += 1
            continue
        if char == "{":
            if end is None:
                continue
            stack -= 1
            if stack == 0:
                candidate = text[index : end + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    end = None
                    stack = 0
                    continue
    raise RuntimeError("Failed to parse ffmpeg loudnorm JSON output")


def _measure_integrated_lufs(path: str) -> float:
    result = _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            path,
            "-af",
            f"loudnorm=I={LUFS_TARGET}:LRA=11:TP=-1.5:print_format=json",
            "-f",
            "null",
            _FFMPEG_NULL_SINK,
        ]
    )
    payload = _extract_last_json_block(result.stderr)
    try:
        return float(payload.get("input_i"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ffmpeg loudnorm did not return input_i") from exc


def _measure_silence_boundaries(path: str) -> List[float]:
    result = _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            path,
            "-af",
            f"silencedetect=n={SILENCE_THRESHOLD_DBFS}dB:d={MIN_SILENCE_DURATION_MS / 1000.0}",
            "-f",
            "null",
            _FFMPEG_NULL_SINK,
        ]
    )
    silence_ends = []
    for match in re.finditer(r"silence_end:\s*(?P<end>[0-9]+(?:\.[0-9]+)?)", result.stderr or ""):
        try:
            silence_ends.append(float(match.group("end")))
        except ValueError:
            continue
    return sorted(set(round(v, 3) for v in silence_ends if v > 0))


def normalize_to_lufs(input_path: str, output_path: str) -> NormalizationResult:
    input_lufs = _measure_integrated_lufs(input_path)
    if math.isfinite(input_lufs) and abs(input_lufs - LUFS_TARGET) <= LUFS_TOLERANCE:
        if Path(input_path).resolve() != Path(output_path).resolve():
            shutil.copyfile(input_path, output_path)
        return NormalizationResult(
            input_lufs=input_lufs,
            output_lufs=input_lufs,
            was_reencoded=False,
            output_path=output_path,
        )

    _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            input_path,
            "-af",
            f"loudnorm=I={LUFS_TARGET}:LRA=11:TP=-1.5",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )
    output_lufs = _measure_integrated_lufs(output_path)
    return NormalizationResult(
        input_lufs=input_lufs,
        output_lufs=output_lufs,
        was_reencoded=True,
        output_path=output_path,
    )


def condition_audio_for_asr(
    input_path: str,
    output_path: str,
    apply_dynaudnorm: bool = False,
    apply_speech_band_filter: bool = False,
) -> ConditioningResult:
    """
    Apply optional perceptual conditioning after LUFS normalization.
    Output remains PCM 16kHz mono WAV.
    """
    filters = []
    if apply_dynaudnorm:
        filters.append("dynaudnorm=f=120:g=15")
    if apply_speech_band_filter:
        filters.append("highpass=f=120")
        filters.append("lowpass=f=3800")

    if not filters:
        return ConditioningResult(
            input_path=input_path,
            output_path=input_path,
            dynaudnorm_applied=False,
            speech_band_filter_applied=False,
            was_noop=True,
        )

    _run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            input_path,
            "-af",
            ",".join(filters),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )
    return ConditioningResult(
        input_path=input_path,
        output_path=output_path,
        dynaudnorm_applied=bool(apply_dynaudnorm),
        speech_band_filter_applied=bool(apply_speech_band_filter),
        was_noop=False,
    )


def chunk_on_silence_boundaries(
    audio_path: str,
    output_dir: str,
    *,
    max_chunk_duration_s: float = MAX_CHUNK_DURATION_S,
) -> List[ChunkMetadata]:
    duration_s = _probe_duration_seconds(audio_path)
    if duration_s <= 0:
        return []
    resolved_max_chunk_duration_s = max(float(max_chunk_duration_s or MAX_CHUNK_DURATION_S), _MIN_CHUNK_DURATION_S)

    silence_ends = _measure_silence_boundaries(audio_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[ChunkMetadata] = []
    current_start = 0.0
    chunk_id = 0

    while current_start < duration_s - 0.01:
        hard_end = min(current_start + resolved_max_chunk_duration_s, duration_s)
        candidate_end = None
        for boundary in silence_ends:
            if boundary <= current_start + _MIN_CHUNK_DURATION_S:
                continue
            if boundary <= hard_end + 0.001:
                candidate_end = boundary
            else:
                break
        current_end = candidate_end if candidate_end is not None else hard_end
        current_end = min(max(current_end, current_start + min(_MIN_CHUNK_DURATION_S, duration_s - current_start)), duration_s)
        if duration_s - current_end < 0.35:
            current_end = duration_s

        chunk_path = out_dir / f"chunk_{chunk_id:04d}.wav"
        _run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                audio_path,
                "-ss",
                f"{current_start:.3f}",
                "-t",
                f"{max(current_end - current_start, 0.001):.3f}",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(chunk_path),
            ]
        )
        chunks.append(
            ChunkMetadata(
                chunk_id=chunk_id,
                start_s=round(current_start, 3),
                end_s=round(current_end, 3),
                path=str(chunk_path),
                duration_s=round(current_end - current_start, 3),
            )
        )
        chunk_id += 1
        current_start = current_end

    return chunks


__all__ = [
    "ConditioningResult",
    "condition_audio_for_asr",
    "normalize_to_lufs",
    "chunk_on_silence_boundaries",
]
