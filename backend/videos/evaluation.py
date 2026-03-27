"""
Evaluation and calibration helpers for multilingual transcript-product behavior.
"""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional

from django.conf import settings

from .processing_metadata import build_processing_metadata
from .translation import normalize_language_code

logger = logging.getLogger(__name__)

EVALUATION_VERSION = "ml_eval_v1"
BENCHMARK_SUITE_VERSION = "ml_benchmark_v1"
THRESHOLD_PROFILE_VERSION = "ml_threshold_profile_v1"
FIRST_CANDIDATE_EXPERIMENT_VERSION = "ml_candidate_experiment_v1"
EXPERIMENT_REPORT_VERSION = "ml_experiment_report_v1"
SECOND_CANDIDATE_EXPERIMENT_VERSION = "ml_second_candidate_experiment_v1"
SECOND_EXPERIMENT_REPORT_VERSION = "ml_second_experiment_report_v1"
REAL_AUDIO_STRATEGY_COMPARISON_VERSION = "ml_real_audio_strategy_compare_v1"
MALAYALAM_ASR_REVIEW_STRATEGIES = (
    "current_default",
    "fast_first",
    "quality_first",
    "hybrid_retry",
)


DEFAULT_MALAYALAM_THRESHOLDS: Dict[str, float] = {
    "cleaned_min_trusted_visible_words": 10.0,
    "cleaned_min_trusted_display_units": 1.0,
    "cleaned_min_readability": 0.34,
    "cleaned_min_lexical_trust": 0.40,
    "cleaned_max_wrong_script_burden": 0.18,
    "degraded_low_evidence_max_trusted_visible_words": 0.0,
    "hopeless_wrong_script_min_burden": 0.28,
    "hopeless_wrong_script_max_lexical_trust": 0.18,
    "english_contamination_min_burden": 0.52,
    "english_contamination_max_lexical_trust": 0.28,
    "degraded_useful_min_trusted_visible_words": 8.0,
    "degraded_useful_min_trusted_display_units": 1.0,
    "degraded_useful_min_readability": 0.18,
    "degraded_useful_min_lexical_trust": 0.16,
    "borderline_min_readability": 0.28,
    "borderline_max_readability": 0.38,
    "borderline_min_lexical_trust": 0.18,
    "borderline_max_lexical_trust": 0.42,
    "borderline_min_wrong_script": 0.18,
    "borderline_max_wrong_script": 0.30,
    "borderline_min_contamination": 0.35,
    "borderline_max_contamination": 0.58,
    "low_evidence_max_trusted_display_units": 0.0,
    "rescue_recoverability_min_score": 0.25,
}


@dataclass
class BenchmarkCase:
    case_id: str
    label: str
    expected_language: str = ""
    expected_quality_bucket: str = ""
    expected_downstream_decision: str = ""
    expected_english_view_decision: str = ""
    benchmark_group: str = ""
    benchmark_tags: list[str] = field(default_factory=list)
    expected_flags: list[str] = field(default_factory=list)
    human_review_required: bool = False
    notes: str = ""
    transcript_snapshot: Dict[str, Any] = field(default_factory=dict)
    metadata_snapshot: Dict[str, Any] = field(default_factory=dict)
    summary_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealAudioReviewCase:
    case_id: str
    label: str
    media_path: str = ""
    input_reference: str = ""
    expected_language: str = ""
    expected_quality_bucket: str = ""
    expected_downstream_decision: str = ""
    expected_english_view_decision: str = ""
    benchmark_group: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    human_reference_transcript: str = ""
    human_reference_summary: str = ""
    pipeline_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealAudioReviewHistoryEntry:
    history_version: str
    review_run_id: str
    created_at: str
    suite_name: str
    case_id: str
    label: str = ""
    benchmark_group: str = ""
    input_reference: str = ""
    transcript_state: str = ""
    rescue_ran: bool = False
    downstream_decision: str = ""
    summary_state: str = ""
    english_view_decision: str = ""
    dominant_safety_reason: str = ""
    dominant_failure_reason: str = ""
    fixture_alignment_status: str = ""
    mismatch_summary: list[str] = field(default_factory=list)
    should_add_or_update_fixture: bool = False
    recommended_fixture_case_id: str = ""
    promotion_priority: str = ""


@dataclass
class MalayalamThresholdProfile:
    profile_name: str = "runtime_default"
    profile_version: str = THRESHOLD_PROFILE_VERSION
    base_profile: str = "runtime_default"
    overridden_thresholds: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


def _coerce_threshold_profile(profile: Optional[MalayalamThresholdProfile | Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(profile, MalayalamThresholdProfile):
        data = asdict(profile)
    elif isinstance(profile, dict):
        data = dict(profile)
    else:
        data = {}
    overridden = data.get("overridden_thresholds", {})
    merged_thresholds = dict(DEFAULT_MALAYALAM_THRESHOLDS)
    if isinstance(overridden, dict):
        for key, value in overridden.items():
            try:
                merged_thresholds[str(key)] = float(value)
            except Exception:
                continue
    normalized = {
        "profile_name": str(data.get("profile_name", "runtime_default") or "runtime_default"),
        "profile_version": str(data.get("profile_version", THRESHOLD_PROFILE_VERSION) or THRESHOLD_PROFILE_VERSION),
        "base_profile": str(data.get("base_profile", "runtime_default") or "runtime_default"),
        "overridden_thresholds": {
            key: merged_thresholds[key]
            for key in merged_thresholds
            if key in (overridden or {})
        },
        "notes": str(data.get("notes", "") or ""),
        "thresholds": merged_thresholds,
    }
    logger.info(
        "[ML_THRESH_PROFILE] profile=%s base=%s overrides=%s",
        normalized["profile_name"],
        normalized["base_profile"],
        ",".join(sorted(normalized["overridden_thresholds"].keys())) or "none",
    )
    return normalized


def _coerce_real_audio_review_case(case: RealAudioReviewCase | Dict[str, Any]) -> RealAudioReviewCase:
    if isinstance(case, RealAudioReviewCase):
        return case
    data = dict(case or {})
    return RealAudioReviewCase(
        case_id=str(data.get("case_id", "") or ""),
        label=str(data.get("label", "") or ""),
        media_path=str(data.get("media_path", "") or ""),
        input_reference=str(data.get("input_reference", "") or ""),
        expected_language=str(data.get("expected_language", "") or ""),
        expected_quality_bucket=str(data.get("expected_quality_bucket", "") or ""),
        expected_downstream_decision=str(data.get("expected_downstream_decision", "") or ""),
        expected_english_view_decision=str(data.get("expected_english_view_decision", "") or ""),
        benchmark_group=str(data.get("benchmark_group", "") or ""),
        tags=list(data.get("tags", []) or []),
        notes=str(data.get("notes", "") or ""),
        human_reference_transcript=str(data.get("human_reference_transcript", "") or ""),
        human_reference_summary=str(data.get("human_reference_summary", "") or ""),
        pipeline_kwargs=dict(data.get("pipeline_kwargs", {}) or {}),
    )


@contextmanager
def _temporary_setting(name: str, value: Any):
    sentinel = object()
    previous = getattr(settings, name, sentinel)
    setattr(settings, name, value)
    try:
        yield
    finally:
        if previous is sentinel:
            try:
                delattr(settings, name)
            except Exception:
                pass
        else:
            setattr(settings, name, previous)


def build_first_candidate_threshold_profile() -> Dict[str, Any]:
    profile = _coerce_threshold_profile(
        MalayalamThresholdProfile(
            profile_name="candidate_borderline_recoverable_malayalam_v1",
            base_profile="runtime_default",
            overridden_thresholds={
                "degraded_useful_min_trusted_visible_words": 7.0,
                "degraded_useful_min_readability": 0.20,
                "rescue_recoverability_min_score": 0.23,
            },
            notes="Slightly relax borderline recoverable Malayalam thresholds without touching wrong-script, contamination, or low-evidence protections.",
        )
    )
    profile["candidate_profile_notes"] = profile["notes"]
    profile["intended_effect"] = "Promote only borderline degraded-but-useful Malayalam cases with some visible trusted evidence."
    profile["protected_constraints"] = [
        "wrong_script_rejection_unchanged",
        "english_contamination_protection_unchanged",
        "low_evidence_downstream_suppression_unchanged",
        "protected_english_view_blocking_unchanged",
    ]
    logger.info(
        "[ML_CANDIDATE_PROFILE] profile=%s intended_effect=%s protected_constraints=%s",
        profile["profile_name"],
        profile["intended_effect"],
        ",".join(profile["protected_constraints"]),
    )
    return profile


def _benchmark_case(
    *,
    case_id: str,
    label: str,
    benchmark_group: str,
    transcript_state: str,
    language: str,
    transcript_snapshot: Optional[Dict[str, Any]] = None,
    metadata_snapshot: Optional[Dict[str, Any]] = None,
    summary_snapshot: Optional[Dict[str, Any]] = None,
    expected_quality_bucket: str = "",
    expected_downstream_decision: str = "",
    expected_english_view_decision: str = "",
    expected_language: str = "",
    expected_flags: Optional[list[str]] = None,
    benchmark_tags: Optional[list[str]] = None,
    human_review_required: bool = False,
    notes: str = "",
) -> BenchmarkCase:
    transcript_payload = {
        "language": language,
        "transcript_state": transcript_state,
        **(transcript_snapshot or {}),
    }
    metadata_payload = {
        "detected_language": language,
        "transcript_state": transcript_state,
        **(metadata_snapshot or {}),
    }
    return BenchmarkCase(
        case_id=case_id,
        label=label,
        expected_language=expected_language or language,
        expected_quality_bucket=expected_quality_bucket,
        expected_downstream_decision=expected_downstream_decision,
        expected_english_view_decision=expected_english_view_decision,
        benchmark_group=benchmark_group,
        benchmark_tags=list(benchmark_tags or []),
        expected_flags=list(expected_flags or []),
        human_review_required=human_review_required,
        notes=notes,
        transcript_snapshot=transcript_payload,
        metadata_snapshot=metadata_payload,
        summary_snapshot=dict(summary_snapshot or {}),
    )


def _build_cleaned_malayalam_case(
    *,
    case_id: str,
    label: str,
    transcript_text: str,
    trusted_visible_word_count: int,
    trusted_display_unit_count: int,
    lexical_trust_score: float,
    overall_readability: float,
    wrong_script_burden: float,
    contamination_burden: float,
    english_view_available: bool = True,
    summary_route: str = "normal_grounded",
    benchmark_tags: Optional[list[str]] = None,
) -> BenchmarkCase:
    return _benchmark_case(
        case_id=case_id,
        label=label,
        benchmark_group="cleaned_malayalam",
        transcript_state="cleaned",
        language="ml",
        transcript_snapshot={
            "display_readable_transcript": transcript_text,
            "trusted_visible_word_count": trusted_visible_word_count,
            "trusted_display_unit_count": trusted_display_unit_count,
            "quality_metrics": {
                "lexical_trust_score": lexical_trust_score,
                "overall_readability": overall_readability,
                "wrong_script_burden": wrong_script_burden,
                "contamination_burden": contamination_burden,
            },
        },
        metadata_snapshot={
            "downstream_suppressed": False,
            "trusted_visible_word_count": trusted_visible_word_count,
            "trusted_display_unit_count": trusted_display_unit_count,
            "english_view_available": english_view_available,
            "translation_state": "translated" if english_view_available else "same_as_original",
        },
        summary_snapshot={
            "_trace": {
                "structured_summary_route": summary_route,
                "structured_summary_route_reason": "cleaned_malayalam_grounded_summary",
            }
        },
        expected_quality_bucket="clearly_cleaned",
        expected_downstream_decision="allowed",
        expected_english_view_decision="available" if english_view_available else "same_as_original",
        benchmark_tags=benchmark_tags,
    )


def _build_degraded_low_evidence_case(
    *,
    case_id: str,
    label: str,
    wrong_script_burden: float,
    contamination_burden: float,
    lexical_trust_score: float,
    overall_readability: float,
    translation_blocked_reason: str = "degraded_safe_translation_blocked",
    benchmark_group: str = "degraded_low_evidence_malayalam",
    benchmark_tags: Optional[list[str]] = None,
) -> BenchmarkCase:
    return _benchmark_case(
        case_id=case_id,
        label=label,
        benchmark_group=benchmark_group,
        transcript_state="degraded",
        language="ml",
        transcript_snapshot={
            "display_readable_transcript": "",
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "low_evidence_malayalam": True,
            "quality_metrics": {
                "lexical_trust_score": lexical_trust_score,
                "overall_readability": overall_readability,
                "wrong_script_burden": wrong_script_burden,
                "contamination_burden": contamination_burden,
            },
        },
        metadata_snapshot={
            "downstream_suppressed": True,
            "downstream_suppression_reason": "no_trusted_display_units",
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "low_evidence_malayalam": True,
            "english_view_available": False,
            "translation_blocked_reason": translation_blocked_reason,
        },
        summary_snapshot={
            "summary_state": "degraded_safe",
            "_trace": {"structured_summary_route": "degraded_safe"},
        },
        expected_quality_bucket="degraded_low_evidence",
        expected_downstream_decision="suppressed",
        expected_english_view_decision="blocked",
        expected_flags=["safe_downstream_suppression_expected"],
        benchmark_tags=benchmark_tags,
    )


def _build_wrong_script_case(
    *,
    case_id: str,
    label: str,
    wrong_script_burden: float,
    contamination_burden: float,
    lexical_trust_score: float = 0.05,
    overall_readability: float = 0.06,
    benchmark_tags: Optional[list[str]] = None,
) -> BenchmarkCase:
    return _benchmark_case(
        case_id=case_id,
        label=label,
        benchmark_group="wrong_script_hopeless",
        transcript_state="failed",
        language="ml",
        transcript_snapshot={
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "quality_metrics": {
                "lexical_trust_score": lexical_trust_score,
                "overall_readability": overall_readability,
                "wrong_script_burden": wrong_script_burden,
                "contamination_burden": contamination_burden,
            },
        },
        metadata_snapshot={
            "downstream_suppressed": True,
            "english_view_available": False,
        },
        expected_quality_bucket="hopeless_wrong_script",
        expected_downstream_decision="suppressed",
        expected_english_view_decision="blocked",
        expected_flags=["verify_wrong_script_rejection", "safe_downstream_suppression_expected"],
        benchmark_tags=benchmark_tags,
    )


def _build_mixed_educational_case(
    *,
    case_id: str,
    label: str,
    transcript_state: str = "degraded",
    trusted_visible_word_count: int = 8,
    trusted_display_unit_count: int = 1,
    lexical_trust_score: float = 0.22,
    overall_readability: float = 0.23,
    summary_route: str = "degraded_safe",
    benchmark_tags: Optional[list[str]] = None,
) -> BenchmarkCase:
    return _benchmark_case(
        case_id=case_id,
        label=label,
        benchmark_group="mixed_malayalam_english_educational",
        transcript_state=transcript_state,
        language="ml",
        transcript_snapshot={
            "display_readable_transcript": "exam hall result confidence class teacher support",
            "trusted_visible_word_count": trusted_visible_word_count,
            "trusted_display_unit_count": trusted_display_unit_count,
            "quality_metrics": {
                "lexical_trust_score": lexical_trust_score,
                "overall_readability": overall_readability,
                "wrong_script_burden": 0.10,
                "contamination_burden": 0.18,
            },
        },
        metadata_snapshot={
            "downstream_suppressed": False,
            "trusted_visible_word_count": trusted_visible_word_count,
            "trusted_display_unit_count": trusted_display_unit_count,
            "english_view_available": False,
        },
        summary_snapshot={
            "_trace": {
                "structured_summary_route": summary_route,
                "structured_summary_route_reason": "mixed_malayalam_english_grounded_evidence",
            }
        },
        expected_quality_bucket="degraded_but_useful" if transcript_state == "degraded" else "clearly_cleaned",
        expected_downstream_decision="allowed",
        expected_english_view_decision="blocked",
        expected_flags=["degraded_safe_output_expected"] if transcript_state == "degraded" else [],
        benchmark_tags=benchmark_tags,
    )


def _build_english_reference_case(*, case_id: str, label: str) -> BenchmarkCase:
    return _benchmark_case(
        case_id=case_id,
        label=label,
        benchmark_group="english_reference",
        transcript_state="cleaned",
        language="en",
        transcript_snapshot={
            "display_readable_transcript": "English transcript remains stable.",
            "quality_metrics": {"overall_readability": 0.92},
        },
        metadata_snapshot={
            "downstream_suppressed": False,
            "english_view_available": False,
            "translation_state": "same_as_original",
        },
        expected_quality_bucket="english_stable",
        expected_downstream_decision="allowed",
        expected_english_view_decision="same_as_original",
    )


def _build_non_malayalam_reference_case(*, case_id: str, label: str, language: str = "hi") -> BenchmarkCase:
    return _benchmark_case(
        case_id=case_id,
        label=label,
        benchmark_group="non_malayalam_reference",
        transcript_state="cleaned",
        language=language,
        transcript_snapshot={
            "display_readable_transcript": "Reference non-Malayalam transcript.",
            "quality_metrics": {"overall_readability": 0.78},
        },
        metadata_snapshot={
            "downstream_suppressed": False,
            "english_view_available": True,
            "translation_state": "translated",
        },
        expected_quality_bucket="non_malayalam_reference",
        expected_downstream_decision="allowed",
        expected_english_view_decision="available",
    )


def build_default_multilingual_benchmark_cases() -> list[BenchmarkCase]:
    return [
        _build_cleaned_malayalam_case(
            case_id="ml_cleaned_grounded_exam_case_01",
            label="Cleaned Malayalam Grounded Exam",
            transcript_text="confidence result exam hall support guidance",
            trusted_visible_word_count=12,
            trusted_display_unit_count=2,
            lexical_trust_score=0.58,
            overall_readability=0.55,
            wrong_script_burden=0.06,
            contamination_burden=0.12,
        ),
        _build_degraded_low_evidence_case(
            case_id="ml_degraded_low_evidence_case_01",
            label="Low Evidence Malayalam",
            wrong_script_burden=0.24,
            contamination_burden=0.67,
            lexical_trust_score=0.08,
            overall_readability=0.10,
        ),
    ]


def build_expanded_malayalam_benchmark_cases() -> list[BenchmarkCase]:
    return [
        _build_cleaned_malayalam_case(
            case_id="ml_cleaned_grounded_exam_case_01",
            label="Cleaned Malayalam Grounded Exam",
            transcript_text="confidence result exam hall support guidance",
            trusted_visible_word_count=12,
            trusted_display_unit_count=2,
            lexical_trust_score=0.58,
            overall_readability=0.55,
            wrong_script_burden=0.06,
            contamination_burden=0.12,
            benchmark_tags=["grounded", "exam"],
        ),
        _build_cleaned_malayalam_case(
            case_id="ml_cleaned_sparse_grounded_case_01",
            label="Cleaned Malayalam Sparse Grounded",
            transcript_text="result guidance support marks",
            trusted_visible_word_count=10,
            trusted_display_unit_count=1,
            lexical_trust_score=0.44,
            overall_readability=0.39,
            wrong_script_burden=0.08,
            contamination_burden=0.10,
            benchmark_tags=["sparse", "grounded"],
        ),
        _build_mixed_educational_case(
            case_id="ml_degraded_useful_case_01",
            label="Degraded but Useful Malayalam",
            transcript_state="degraded",
            trusted_visible_word_count=8,
            trusted_display_unit_count=1,
            lexical_trust_score=0.19,
            overall_readability=0.21,
            benchmark_tags=["degraded_useful"],
        ),
        _build_degraded_low_evidence_case(
            case_id="ml_degraded_low_evidence_case_01",
            label="Low Evidence Malayalam",
            wrong_script_burden=0.24,
            contamination_burden=0.67,
            lexical_trust_score=0.08,
            overall_readability=0.10,
            benchmark_tags=["protected_low_evidence"],
        ),
        _build_wrong_script_case(
            case_id="ml_wrong_script_hopeless_case_01",
            label="Wrong Script Hopeless Malayalam",
            wrong_script_burden=0.42,
            contamination_burden=0.73,
            benchmark_tags=["protected_wrong_script"],
        ),
        _benchmark_case(
            case_id="ml_english_contaminated_case_01",
            label="English Contaminated Malayalam",
            benchmark_group="english_contaminated",
            transcript_state="degraded",
            language="ml",
            transcript_snapshot={
                "trusted_visible_word_count": 2,
                "trusted_display_unit_count": 0,
                "quality_metrics": {
                    "lexical_trust_score": 0.18,
                    "overall_readability": 0.19,
                    "wrong_script_burden": 0.14,
                    "contamination_burden": 0.61,
                },
            },
            metadata_snapshot={
                "downstream_suppressed": True,
                "english_view_available": False,
            },
            expected_quality_bucket="english_contaminated",
            expected_downstream_decision="suppressed",
            expected_english_view_decision="blocked",
            expected_flags=["review_english_contamination_guard"],
            benchmark_tags=["protected_english_contamination"],
        ),
        _build_mixed_educational_case(
            case_id="ml_mixed_edu_case_01",
            label="Mixed Malayalam English Educational",
            transcript_state="degraded",
            trusted_visible_word_count=9,
            trusted_display_unit_count=1,
            lexical_trust_score=0.24,
            overall_readability=0.24,
            benchmark_tags=["mixed_educational"],
        ),
        _build_degraded_low_evidence_case(
            case_id="ml_degraded_safe_sparse_summary_case_01",
            label="Degraded Safe Sparse Summary",
            wrong_script_burden=0.20,
            contamination_burden=0.56,
            lexical_trust_score=0.12,
            overall_readability=0.14,
            benchmark_group="degraded_safe_sparse",
            benchmark_tags=["degraded_safe", "sparse_summary"],
        ),
        _build_english_reference_case(
            case_id="en_reference_case_01",
            label="English Stable Reference",
        ),
        _build_non_malayalam_reference_case(
            case_id="hi_reference_case_01",
            label="Hindi Reference Case",
            language="hi",
        ),
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return int(default)


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _avg_metric(units: Any, key: str) -> float:
    if not isinstance(units, list):
        return 0.0
    values = []
    for unit in units:
        if isinstance(unit, dict) and unit.get(key) is not None:
            values.append(_safe_float(unit.get(key), 0.0))
    if not values:
        return 0.0
    return round(sum(values) / max(len(values), 1), 4)


def _extract_units(transcript_json: Dict[str, Any]) -> list:
    for key in ("display_transcript_units", "assembled_transcript_units", "segments", "raw_transcript_segments"):
        value = transcript_json.get(key, [])
        if isinstance(value, list) and value:
            return value
    return []


def _derive_wrong_script_burden(transcript_json: Dict[str, Any], quality_metrics: Dict[str, Any]) -> float:
    direct = _safe_float(
        quality_metrics.get("wrong_script_burden", quality_metrics.get("wrong_script_ratio", 0.0)),
        0.0,
    )
    if direct > 0.0:
        return round(direct, 4)
    return _avg_metric(_extract_units(transcript_json), "wrong_script_ratio")


def _derive_contamination_burden(transcript_json: Dict[str, Any], quality_metrics: Dict[str, Any]) -> float:
    direct = _safe_float(quality_metrics.get("contamination_burden", quality_metrics.get("contamination_score", 0.0)), 0.0)
    if direct > 0.0:
        return round(direct, 4)
    return _avg_metric(_extract_units(transcript_json), "contamination_score")


def _derive_readability(transcript_json: Dict[str, Any], quality_metrics: Dict[str, Any]) -> float:
    direct = _safe_float(quality_metrics.get("overall_readability", quality_metrics.get("score", 0.0)), 0.0)
    if direct > 0.0:
        return round(direct, 4)
    return _avg_metric(_extract_units(transcript_json), "unit_readability")


def _derive_lexical_trust(transcript_json: Dict[str, Any], quality_metrics: Dict[str, Any]) -> float:
    direct = _safe_float(quality_metrics.get("lexical_trust_score", 0.0), 0.0)
    if direct > 0.0:
        return round(direct, 4)
    return _avg_metric(_extract_units(transcript_json), "malayalam_trust")


def _summarize_rescue_effect(language: str, observability: Dict[str, Any], evaluation_status: str, trusted_visible_word_count: int) -> str:
    if language != "ml":
        return "not_applicable"
    if bool(observability.get("retry_executed", False)):
        if evaluation_status == "cleaned" and trusted_visible_word_count >= 8:
            return "helped"
        return "no_change"
    if bool(observability.get("retry_considered", False)) or _first_non_empty(observability.get("retry_skipped_reason"), observability.get("retry_decision_reason")):
        return "skipped"
    return "not_applicable"


def classify_malayalam_calibration_bucket(
    snapshot: Dict[str, Any],
    threshold_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = _coerce_threshold_profile(threshold_profile)
    thresholds = profile["thresholds"]
    language = normalize_language_code(snapshot.get("detected_language") or snapshot.get("language"), default="en", allow_auto=False)
    state = str(snapshot.get("transcript_state", "") or "").strip().lower()
    lexical = _safe_float(snapshot.get("lexical_trust_score", 0.0), 0.0)
    readability = _safe_float(snapshot.get("overall_readability", 0.0), 0.0)
    wrong_script = _safe_float(snapshot.get("wrong_script_burden", 0.0), 0.0)
    contamination = _safe_float(snapshot.get("contamination_burden", 0.0), 0.0)
    trusted_visible = _safe_int(snapshot.get("trusted_visible_word_count", 0), 0)
    trusted_units = _safe_int(snapshot.get("trusted_display_unit_count", 0), 0)
    downstream_suppressed = bool(snapshot.get("downstream_suppressed", False))
    low_evidence = bool(snapshot.get("low_evidence_malayalam", False))
    english_decision = str(snapshot.get("english_view_decision", "") or "").strip().lower()
    structured_route = str(snapshot.get("structured_summary_route", "") or "").strip().lower()
    recommendation_flags = []
    decisive_signals = []
    borderline_signals = []

    if language == "en":
        bucket = "english_stable"
        decisive_signals.append("already_english")
        recommendation_flags.append("no_action_needed")
    elif language != "ml":
        bucket = "non_malayalam_reference"
        decisive_signals.append("non_malayalam_language")
        recommendation_flags.append("no_action_needed")
    elif (
        state == "cleaned"
        and trusted_visible >= int(thresholds["cleaned_min_trusted_visible_words"])
        and trusted_units >= int(thresholds["cleaned_min_trusted_display_units"])
        and readability >= thresholds["cleaned_min_readability"]
        and lexical >= thresholds["cleaned_min_lexical_trust"]
        and wrong_script <= thresholds["cleaned_max_wrong_script_burden"]
        and structured_route.startswith("normal")
    ):
        bucket = "clearly_cleaned"
        decisive_signals.extend(["cleaned_state", "grounded_summary_route", "trusted_visible_content"])
        recommendation_flags.append("no_action_needed")
    elif (
        state == "degraded"
        and (low_evidence or downstream_suppressed)
        and trusted_visible <= int(thresholds["degraded_low_evidence_max_trusted_visible_words"])
        and trusted_units <= int(thresholds["low_evidence_max_trusted_display_units"])
    ):
        bucket = "degraded_low_evidence"
        decisive_signals.extend(["degraded_state", "low_evidence_gate", "no_trusted_visible_content"])
        recommendation_flags.append("safe_downstream_suppression_expected")
    elif (
        state in {"degraded", "failed"}
        and wrong_script >= thresholds["hopeless_wrong_script_min_burden"]
        and lexical < thresholds["hopeless_wrong_script_max_lexical_trust"]
        and trusted_visible <= int(thresholds["degraded_low_evidence_max_trusted_visible_words"])
    ):
        bucket = "hopeless_wrong_script"
        decisive_signals.extend(["wrong_script_burden_high", "lexical_trust_very_low", "no_trusted_visible_content"])
        recommendation_flags.extend(["verify_wrong_script_rejection", "safe_downstream_suppression_expected"])
    elif contamination >= thresholds["english_contamination_min_burden"] and lexical < thresholds["english_contamination_max_lexical_trust"]:
        bucket = "english_contaminated"
        decisive_signals.extend(["english_contamination_high", "low_lexical_trust"])
        recommendation_flags.append("review_english_contamination_guard")
    elif (
        state == "degraded"
        and trusted_visible >= int(thresholds["degraded_useful_min_trusted_visible_words"])
        and trusted_units >= int(thresholds["degraded_useful_min_trusted_display_units"])
        and readability >= thresholds["degraded_useful_min_readability"]
        and lexical >= thresholds["degraded_useful_min_lexical_trust"]
    ):
        bucket = "degraded_but_useful"
        decisive_signals.extend(["degraded_state", "trusted_visible_content_present"])
        recommendation_flags.append("degraded_safe_output_expected")
    else:
        bucket = "borderline_review"
        recommendation_flags.append("manual_threshold_review")
        if thresholds["borderline_min_readability"] <= readability <= thresholds["borderline_max_readability"]:
            borderline_signals.append("readability_borderline")
        if thresholds["borderline_min_lexical_trust"] <= lexical <= thresholds["borderline_max_lexical_trust"]:
            borderline_signals.append("lexical_trust_borderline")
        if thresholds["borderline_min_wrong_script"] <= wrong_script <= thresholds["borderline_max_wrong_script"]:
            borderline_signals.append("wrong_script_borderline")
        if thresholds["borderline_min_contamination"] <= contamination <= thresholds["borderline_max_contamination"]:
            borderline_signals.append("contamination_borderline")
        if english_decision in {"blocked", "available"}:
            borderline_signals.append(f"english_view_{english_decision}")

    logger.info(
        "[ML_CALIBRATION_BUCKET] clip=%s bucket=%s state=%s language=%s",
        snapshot.get("clip_identifier", ""),
        bucket,
        state,
        language,
    )
    logger.info(
        "[ML_CALIBRATION_SIGNALS] clip=%s decisive=%s borderline=%s",
        snapshot.get("clip_identifier", ""),
        ",".join(decisive_signals) or "none",
        ",".join(borderline_signals) or "none",
    )
    return {
        "calibration_bucket": bucket,
        "decisive_signals": decisive_signals,
        "borderline_signals": borderline_signals,
        "suggested_review_reason": borderline_signals[0] if borderline_signals else "",
        "recommendation_flags": recommendation_flags,
        "threshold_profile_name": profile["profile_name"],
        "threshold_profile_version": profile["profile_version"],
    }


def _build_decision_trace(snapshot: Dict[str, Any]) -> Dict[str, str]:
    rescue_reason = _first_non_empty(snapshot.get("rescue_selection_reason"), snapshot.get("retry_decision_reason"), snapshot.get("retry_skipped_reason"), snapshot.get("rescue_effect"))
    transcript_reason = _first_non_empty(snapshot.get("transcript_state_reason"), snapshot.get("dominant_safety_reason"), snapshot.get("transcript_warning_message"))
    summary_reason = _first_non_empty(snapshot.get("structured_summary_route_reason"), snapshot.get("summary_state"), snapshot.get("summary_translation_blocked_reason"))
    downstream_reason = _first_non_empty(snapshot.get("downstream_suppression_reason"), snapshot.get("downstream_decision"))
    english_reason = _first_non_empty(snapshot.get("translation_blocked_reason"), snapshot.get("summary_translation_blocked_reason"), snapshot.get("english_view_decision"))
    decision_trace = {
        "rescue": rescue_reason,
        "transcript": transcript_reason,
        "summary": summary_reason,
        "downstream": downstream_reason,
        "english_view": english_reason,
    }
    short = " | ".join(
        f"{key}:{value}" for key, value in decision_trace.items() if str(value or "").strip()
    )
    return {
        "decision_trace": decision_trace,
        "decision_trace_short": short,
        "dominant_failure_reason": transcript_reason if snapshot.get("transcript_state") in {"degraded", "failed"} else "",
        "dominant_safety_reason": downstream_reason or english_reason or summary_reason,
    }


def build_multilingual_evaluation_result(
    *,
    clip_identifier: str,
    transcript=None,
    transcript_json: Optional[Dict[str, Any]] = None,
    processing_metadata: Optional[Dict[str, Any]] = None,
    structured_summary: Optional[Dict[str, Any]] = None,
    evaluation_notes: str = "",
    threshold_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    transcript_data = transcript_json or (transcript.json_data if transcript and isinstance(transcript.json_data, dict) else {}) or {}
    if processing_metadata is None and transcript is not None:
        processing_metadata = build_processing_metadata(getattr(transcript, "video", None), transcript)
    processing_metadata = processing_metadata or {}
    structured_summary = structured_summary or (
        transcript_data.get("structured_summary_cache", {}).get("payload", {})
        if isinstance(transcript_data.get("structured_summary_cache", {}), dict) else {}
    ) or {}
    quality_metrics = transcript_data.get("quality_metrics", {}) if isinstance(transcript_data.get("quality_metrics", {}), dict) else {}
    processing_metrics = transcript_data.get("processing_metrics", {}) if isinstance(transcript_data.get("processing_metrics", {}), dict) else {}
    malayalam_obs = processing_metadata.get("malayalam_observability", {}) if isinstance(processing_metadata.get("malayalam_observability", {}), dict) else {}
    structured_trace = structured_summary.get("_trace", {}) if isinstance(structured_summary.get("_trace", {}), dict) else {}
    readable_text = _first_non_empty(
        transcript_data.get("display_readable_transcript"),
        transcript_data.get("readable_transcript"),
    )

    snapshot = {
        "evaluation_version": EVALUATION_VERSION,
        "clip_identifier": clip_identifier,
        "detected_language": _first_non_empty(processing_metadata.get("detected_language"), transcript_data.get("detected_language"), processing_metadata.get("language"), transcript_data.get("language"), getattr(transcript, "transcript_language", "")),
        "detected_language_confidence": _safe_float(processing_metadata.get("detected_language_confidence", transcript_data.get("detected_language_confidence", 0.0)), 0.0),
        "transcript_state": _first_non_empty(processing_metadata.get("transcript_state"), transcript_data.get("transcript_state")),
        "evaluation_status": _first_non_empty(processing_metadata.get("transcript_state"), transcript_data.get("transcript_state")),
        "rescue_ran": bool(malayalam_obs.get("retry_executed", False)),
        "downstream_suppressed": bool(processing_metadata.get("downstream_suppressed", False)),
        "downstream_suppression_reason": _first_non_empty(processing_metadata.get("downstream_suppression_reason"), processing_metrics.get("downstream_suppression_reason")),
        "trusted_visible_word_count": _safe_int(processing_metadata.get("trusted_visible_word_count", transcript_data.get("trusted_visible_word_count", 0)), 0),
        "trusted_display_unit_count": _safe_int(processing_metadata.get("trusted_display_unit_count", transcript_data.get("trusted_display_unit_count", 0)), 0),
        "low_evidence_malayalam": bool(processing_metadata.get("low_evidence_malayalam", transcript_data.get("low_evidence_malayalam", False))),
        "lexical_trust_score": _derive_lexical_trust(transcript_data, quality_metrics),
        "overall_readability": _derive_readability(transcript_data, quality_metrics),
        "wrong_script_burden": _derive_wrong_script_burden(transcript_data, quality_metrics),
        "contamination_burden": _derive_contamination_burden(transcript_data, quality_metrics),
        "summary_state": _first_non_empty(structured_summary.get("summary_state"), structured_trace.get("structured_summary_route")),
        "structured_summary_route": _first_non_empty(structured_trace.get("structured_summary_route"), structured_summary.get("summary_state")),
        "structured_summary_route_reason": _first_non_empty(structured_trace.get("structured_summary_route_reason"), structured_trace.get("structured_summary_blocked_reason")),
        "structured_input_source": _first_non_empty(structured_trace.get("structured_input_source"), structured_trace.get("structured_grounding_reason")),
        "transcript_display_available": bool(readable_text),
        "english_view_available": bool(processing_metadata.get("english_view_available", transcript_data.get("english_view_available", False))),
        "english_view_decision": "same_as_original" if _first_non_empty(processing_metadata.get("translation_state"), transcript_data.get("translation_state")) == "same_as_original" else ("available" if bool(processing_metadata.get("english_view_available", transcript_data.get("english_view_available", False))) else "blocked"),
        "translation_blocked_reason": _first_non_empty(processing_metadata.get("translation_blocked_reason"), transcript_data.get("translation_blocked_reason")),
        "summary_translation_blocked_reason": _first_non_empty(structured_summary.get("summary_translation_blocked_reason")),
        "retry_decision_reason": _first_non_empty(malayalam_obs.get("retry_decision_reason")),
        "retry_skipped_reason": _first_non_empty(malayalam_obs.get("retry_skipped_reason")),
        "transcript_state_reason": _first_non_empty(quality_metrics.get("malayalam_post_asr_accept_reason"), transcript_data.get("transcript_warning_message")),
        "evaluation_notes": str(evaluation_notes or "").strip(),
    }
    snapshot["rescue_effect"] = _summarize_rescue_effect(
        snapshot["detected_language"],
        malayalam_obs,
        snapshot["evaluation_status"],
        snapshot["trusted_visible_word_count"],
    )
    snapshot["downstream_decision"] = "suppressed" if snapshot["downstream_suppressed"] else "allowed"
    profile = _coerce_threshold_profile(threshold_profile)
    snapshot["threshold_profile_name"] = profile["profile_name"]
    snapshot["threshold_profile_version"] = profile["profile_version"]
    bucket = classify_malayalam_calibration_bucket(snapshot, threshold_profile=profile)
    trace = _build_decision_trace({**snapshot, **bucket})
    evaluation = {
        **snapshot,
        **bucket,
        **trace,
    }
    if not evaluation["evaluation_notes"]:
        evaluation["evaluation_notes"] = evaluation.get("suggested_review_reason", "") or ""
    logger.info(
        "[ML_EVAL_RESULT] clip=%s status=%s bucket=%s rescue=%s downstream=%s english_view=%s summary_state=%s",
        evaluation["clip_identifier"],
        evaluation["evaluation_status"],
        evaluation["calibration_bucket"],
        evaluation["rescue_effect"],
        evaluation["downstream_decision"],
        evaluation["english_view_decision"],
        evaluation["summary_state"],
    )
    logger.info(
        "[ML_EVAL_RECOMMENDATION] clip=%s flags=%s dominant_failure_reason=%s dominant_safety_reason=%s",
        evaluation["clip_identifier"],
        ",".join(evaluation.get("recommendation_flags", [])) or "none",
        evaluation.get("dominant_failure_reason", ""),
        evaluation.get("dominant_safety_reason", ""),
    )
    return evaluation


def evaluate_benchmark_case(
    case: BenchmarkCase | Dict[str, Any],
    *,
    threshold_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    case_data = asdict(case) if isinstance(case, BenchmarkCase) else dict(case or {})
    profile = _coerce_threshold_profile(threshold_profile)
    evaluation = build_multilingual_evaluation_result(
        clip_identifier=str(case_data.get("case_id", "") or case_data.get("label", "") or "benchmark-case"),
        transcript_json=case_data.get("transcript_snapshot", {}) if isinstance(case_data.get("transcript_snapshot", {}), dict) else {},
        processing_metadata=case_data.get("metadata_snapshot", {}) if isinstance(case_data.get("metadata_snapshot", {}), dict) else {},
        structured_summary=case_data.get("summary_snapshot", {}) if isinstance(case_data.get("summary_snapshot", {}), dict) else {},
        evaluation_notes=str(case_data.get("notes", "") or ""),
        threshold_profile=profile,
    )
    mismatches = []
    expected_language = normalize_language_code(case_data.get("expected_language"), default="", allow_auto=False) if str(case_data.get("expected_language", "") or "").strip() else ""
    if expected_language and normalize_language_code(evaluation.get("detected_language"), default="", allow_auto=False) != expected_language:
        mismatches.append(
            {
                "field": "detected_language",
                "expected": expected_language,
                "actual": normalize_language_code(evaluation.get("detected_language"), default="", allow_auto=False),
            }
        )
    for field_name, expected_key in (
        ("calibration_bucket", "expected_quality_bucket"),
        ("downstream_decision", "expected_downstream_decision"),
        ("english_view_decision", "expected_english_view_decision"),
    ):
        expected_value = str(case_data.get(expected_key, "") or "").strip()
        actual_value = str(evaluation.get(field_name, "") or "").strip()
        if expected_value and actual_value != expected_value:
            mismatches.append(
                {
                    "field": field_name,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    expected_flags = [str(flag).strip() for flag in (case_data.get("expected_flags") or []) if str(flag).strip()]
    for flag in expected_flags:
        if flag not in list(evaluation.get("recommendation_flags", []) or []):
            mismatches.append(
                {
                    "field": "recommendation_flags",
                    "expected": flag,
                    "actual": list(evaluation.get("recommendation_flags", []) or []),
                }
            )
    result = {
        "case_id": str(case_data.get("case_id", "") or ""),
        "label": str(case_data.get("label", "") or ""),
        "benchmark_group": str(case_data.get("benchmark_group", "") or ""),
        "benchmark_tags": list(case_data.get("benchmark_tags") or []),
        "human_review_required": bool(case_data.get("human_review_required", False)),
        "evaluation": evaluation,
        "threshold_profile_name": profile["profile_name"],
        "passed_expectations": not mismatches,
        "mismatches": mismatches,
        "decision_trace_excerpt": str(evaluation.get("decision_trace_short", "") or ""),
    }
    logger.info(
        "[ML_BENCH_CASE] case_id=%s label=%s expected_bucket=%s expected_downstream=%s expected_english_view=%s",
        result["case_id"],
        result["label"],
        str(case_data.get("expected_quality_bucket", "") or ""),
        str(case_data.get("expected_downstream_decision", "") or ""),
        str(case_data.get("expected_english_view_decision", "") or ""),
    )
    logger.info(
        "[ML_BENCH_RESULT] case_id=%s passed=%s bucket=%s downstream=%s english_view=%s mismatches=%s",
        result["case_id"],
        result["passed_expectations"],
        evaluation.get("calibration_bucket", ""),
        evaluation.get("downstream_decision", ""),
        evaluation.get("english_view_decision", ""),
        len(mismatches),
    )
    return result


def summarize_benchmark_suite_results(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    bucket_counts: Dict[str, int] = {}
    downstream_counts: Dict[str, int] = {}
    english_view_counts: Dict[str, int] = {}
    suppression_reasons: Dict[str, int] = {}
    translation_block_reasons: Dict[str, int] = {}
    failure_categories: Dict[str, int] = {}
    borderline_case_ids = []
    helpful_rescue_case_ids = []
    persistent_failure_case_ids = []
    review_required_cases = []
    mismatched_cases = []

    for row in results or []:
        evaluation = row.get("evaluation", {}) if isinstance(row.get("evaluation", {}), dict) else {}
        bucket = str(evaluation.get("calibration_bucket", "") or "")
        downstream = str(evaluation.get("downstream_decision", "") or "")
        english_view = str(evaluation.get("english_view_decision", "") or "")
        suppression_reason = str(evaluation.get("downstream_suppression_reason", "") or "")
        translation_reason = str(evaluation.get("translation_blocked_reason", "") or evaluation.get("summary_translation_blocked_reason", "") or "")
        dominant_failure = str(evaluation.get("dominant_failure_reason", "") or "")
        case_id = str(row.get("case_id", "") or "")

        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        downstream_counts[downstream] = downstream_counts.get(downstream, 0) + 1
        english_view_counts[english_view] = english_view_counts.get(english_view, 0) + 1
        if suppression_reason:
            suppression_reasons[suppression_reason] = suppression_reasons.get(suppression_reason, 0) + 1
        if translation_reason:
            translation_block_reasons[translation_reason] = translation_block_reasons.get(translation_reason, 0) + 1
        if dominant_failure:
            failure_categories[dominant_failure] = failure_categories.get(dominant_failure, 0) + 1
        if bucket == "borderline_review":
            borderline_case_ids.append(case_id)
        if str(evaluation.get("rescue_effect", "") or "") == "helped":
            helpful_rescue_case_ids.append(case_id)
        if bucket in {"hopeless_wrong_script", "degraded_low_evidence"}:
            persistent_failure_case_ids.append(case_id)
        if bool(row.get("human_review_required", False)) or bucket == "borderline_review":
            review_required_cases.append(case_id)
        if not bool(row.get("passed_expectations", False)):
            mismatched_cases.append(
                {
                    "case_id": case_id,
                    "label": row.get("label", ""),
                    "mismatches": row.get("mismatches", []),
                    "decision_trace_excerpt": row.get("decision_trace_excerpt", ""),
                }
            )

    return {
        "bucket_counts": bucket_counts,
        "downstream_decision_counts": downstream_counts,
        "english_view_decision_counts": english_view_counts,
        "dominant_failure_categories": failure_categories,
        "dominant_suppression_reasons": suppression_reasons,
        "dominant_translation_block_reasons": translation_block_reasons,
        "borderline_case_ids": borderline_case_ids,
        "helpful_rescue_case_ids": helpful_rescue_case_ids,
        "persistent_failure_case_ids": persistent_failure_case_ids,
        "review_required_cases": review_required_cases,
        "mismatched_cases": mismatched_cases,
    }


def run_multilingual_benchmark_suite(
    cases: list[BenchmarkCase | Dict[str, Any]],
    *,
    threshold_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = _coerce_threshold_profile(threshold_profile)
    results = [evaluate_benchmark_case(case, threshold_profile=profile) for case in (cases or [])]
    passed = sum(1 for row in results if bool(row.get("passed_expectations", False)))
    failed = len(results) - passed
    aggregate = summarize_benchmark_suite_results(results)
    report = {
        "suite_version": BENCHMARK_SUITE_VERSION,
        "threshold_profile_name": profile["profile_name"],
        "total_cases": len(results),
        "passed_expectations": passed,
        "failed_expectations": failed,
        **aggregate,
    }
    logger.info(
        "[ML_BENCH_SUMMARY] suite_version=%s total_cases=%s passed=%s failed=%s bucket_counts=%s",
        report["suite_version"],
        report["total_cases"],
        report["passed_expectations"],
        report["failed_expectations"],
        report["bucket_counts"],
    )
    return {
        "suite_version": BENCHMARK_SUITE_VERSION,
        "threshold_profile_name": profile["profile_name"],
        "results": results,
        "summary": report,
    }


def _default_real_audio_pipeline_runner(case: RealAudioReviewCase) -> Dict[str, Any]:
    pipeline_kwargs = dict(case.pipeline_kwargs or {})
    video_id = pipeline_kwargs.pop("video_id", None)
    if video_id is None:
        return {
            "status": "blocked",
            "message": "No default real-audio pipeline input was provided.",
            "blocked_reason": "missing_video_id_for_default_runner",
        }
    from .tasks import process_video_transcription_sync

    result = process_video_transcription_sync(
        video_id,
        transcription_language=str(pipeline_kwargs.pop("transcription_language", "auto") or "auto"),
        output_language=str(pipeline_kwargs.pop("output_language", "auto") or "auto"),
        summary_language_mode=str(pipeline_kwargs.pop("summary_language_mode", "same_as_transcript") or "same_as_transcript"),
    )
    return result if isinstance(result, dict) else {"status": "unknown", "raw_result": result}


def _extract_review_snapshots_from_pipeline_result(
    case: RealAudioReviewCase,
    pipeline_result: Dict[str, Any],
) -> Dict[str, Any]:
    transcript_snapshot = dict(pipeline_result.get("transcript_snapshot", {}) or {})
    metadata_snapshot = dict(pipeline_result.get("metadata_snapshot", {}) or {})
    summary_snapshot = dict(pipeline_result.get("summary_snapshot", {}) or {})

    if transcript_snapshot or metadata_snapshot or summary_snapshot:
        return {
            "transcript_snapshot": transcript_snapshot,
            "metadata_snapshot": metadata_snapshot,
            "summary_snapshot": summary_snapshot,
        }

    video = pipeline_result.get("video")
    transcript = pipeline_result.get("transcript")
    if video is not None and transcript is not None:
        processing_metadata = build_processing_metadata(video, transcript)
        transcript_json = getattr(transcript, "json_data", {}) or {}
        transcript_snapshot = {
            "language": getattr(transcript, "language", "") or processing_metadata.get("detected_language", ""),
            "transcript_state": getattr(transcript, "transcript_state", "") or processing_metadata.get("transcript_state", ""),
            "display_readable_transcript": getattr(transcript, "display_readable_transcript", "") or getattr(transcript, "readable_transcript", ""),
            "trusted_visible_word_count": processing_metadata.get("trusted_visible_word_count", 0),
            "trusted_display_unit_count": processing_metadata.get("trusted_display_unit_count", 0),
            "quality_metrics": dict(transcript_json.get("quality_metrics", {}) or {}),
        }
        summary_snapshot = dict(
            transcript_json.get("structured_summary_cache", {}).get("payload", {})
            or transcript_json.get("structured_summary", {})
            or {}
        )
        return {
            "transcript_snapshot": transcript_snapshot,
            "metadata_snapshot": processing_metadata,
            "summary_snapshot": summary_snapshot,
        }

    return {
        "transcript_snapshot": {
            "language": str(pipeline_result.get("language", case.expected_language) or ""),
            "transcript_state": str(pipeline_result.get("transcript_state", "") or ""),
            "display_readable_transcript": str(pipeline_result.get("display_readable_transcript", "") or ""),
            "trusted_visible_word_count": int(pipeline_result.get("trusted_visible_word_count", 0) or 0),
            "trusted_display_unit_count": int(pipeline_result.get("trusted_display_unit_count", 0) or 0),
            "quality_metrics": dict(pipeline_result.get("quality_metrics", {}) or {}),
        },
        "metadata_snapshot": dict(pipeline_result.get("processing_metadata", {}) or {}),
        "summary_snapshot": dict(pipeline_result.get("structured_summary", {}) or {}),
    }


def compare_real_audio_review_to_fixture(
    review_result: Dict[str, Any],
    *,
    benchmark_case: Optional[BenchmarkCase | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    evaluation = dict(review_result.get("evaluation_result", {}) or {})
    mismatches = list(review_result.get("expectation_mismatches", review_result.get("mismatches", [])) or [])
    if benchmark_case is None:
        fixture_alignment_status = "divergent" if mismatches else "aligned"
        fixture_alignment_notes = (
            "Real clip output diverged from case expectations."
            if mismatches
            else "Real clip output matches the current review expectations."
        )
        return _json_safe(
            {
                "fixture_alignment_status": fixture_alignment_status,
                "fixture_alignment_notes": fixture_alignment_notes,
                "should_add_or_update_fixture": bool(mismatches),
                "recommended_fixture_case_id": str(review_result.get("case_id", "") or ""),
                "divergence_reason": ",".join(sorted({str(item.get("field", "") or "") for item in mismatches if isinstance(item, dict)})),
            }
        )

    fixture_case = asdict(benchmark_case) if isinstance(benchmark_case, BenchmarkCase) else dict(benchmark_case or {})
    divergence_reason = []
    if str(evaluation.get("calibration_bucket", "") or "") != str(fixture_case.get("expected_quality_bucket", "") or "") and str(fixture_case.get("expected_quality_bucket", "") or ""):
        divergence_reason.append("quality_bucket_mismatch")
    if str(evaluation.get("downstream_decision", "") or "") != str(fixture_case.get("expected_downstream_decision", "") or "") and str(fixture_case.get("expected_downstream_decision", "") or ""):
        divergence_reason.append("downstream_decision_mismatch")
    if str(evaluation.get("english_view_decision", "") or "") != str(fixture_case.get("expected_english_view_decision", "") or "") and str(fixture_case.get("expected_english_view_decision", "") or ""):
        divergence_reason.append("english_view_decision_mismatch")
    fixture_alignment_status = "aligned" if not divergence_reason else "divergent"
    return _json_safe(
        {
            "fixture_alignment_status": fixture_alignment_status,
            "fixture_alignment_notes": (
                "Real clip behavior matches the benchmark fixture."
                if not divergence_reason
                else "Real clip behavior diverges from the current benchmark fixture."
            ),
            "should_add_or_update_fixture": bool(divergence_reason),
            "recommended_fixture_case_id": str(fixture_case.get("case_id", review_result.get("case_id", "")) or ""),
            "divergence_reason": ",".join(divergence_reason),
        }
    )


def run_real_audio_review_case(
    case: RealAudioReviewCase | Dict[str, Any],
    *,
    pipeline_runner: Optional[Callable[[RealAudioReviewCase], Dict[str, Any]]] = None,
    threshold_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
    benchmark_case: Optional[BenchmarkCase | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    real_case = _coerce_real_audio_review_case(case)
    runner = pipeline_runner or _default_real_audio_pipeline_runner
    profile = _coerce_threshold_profile(threshold_profile)
    pipeline_result = runner(real_case) or {}
    run_status = str(pipeline_result.get("status", "") or "completed")
    snapshots = _extract_review_snapshots_from_pipeline_result(real_case, pipeline_result)

    bench_case = BenchmarkCase(
        case_id=real_case.case_id,
        label=real_case.label,
        expected_language=real_case.expected_language,
        expected_quality_bucket=real_case.expected_quality_bucket,
        expected_downstream_decision=real_case.expected_downstream_decision,
        expected_english_view_decision=real_case.expected_english_view_decision,
        benchmark_group=real_case.benchmark_group,
        benchmark_tags=list(real_case.tags or []),
        notes=real_case.notes,
        transcript_snapshot=snapshots["transcript_snapshot"],
        metadata_snapshot=snapshots["metadata_snapshot"],
        summary_snapshot=snapshots["summary_snapshot"],
    )
    benchmark_eval = evaluate_benchmark_case(bench_case, threshold_profile=profile)
    evaluation_result = dict(benchmark_eval.get("evaluation", {}) or {})
    metadata_snapshot = dict(snapshots.get("metadata_snapshot", {}) or {})
    pipeline_result_summary = {
        "status": run_status,
        "input_reference": real_case.input_reference or real_case.media_path,
        "transcript_state": str(evaluation_result.get("transcript_state", "") or ""),
        "rescue_ran": bool(evaluation_result.get("rescue_ran", False)),
        "rescue_effect": str(evaluation_result.get("rescue_effect", "") or ""),
        "downstream_decision": str(evaluation_result.get("downstream_decision", "") or ""),
        "summary_state": str(evaluation_result.get("summary_state", "") or ""),
        "english_view_decision": str(evaluation_result.get("english_view_decision", "") or ""),
    }
    strategy_metadata = {
        "malayalam_asr_strategy": str(
            metadata_snapshot.get("malayalam_asr_strategy", getattr(settings, "ASR_MALAYALAM_STRATEGY", "current_default")) or ""
        ),
        "primary_model_used": str(metadata_snapshot.get("primary_model_used", "") or ""),
        "fallback_model_used": str(metadata_snapshot.get("fallback_model_used", "") or ""),
        "retry_model_used": str(metadata_snapshot.get("retry_model_used", "") or ""),
        "second_pass_asr_attempted": bool(metadata_snapshot.get("second_pass_asr_attempted", False)),
        "second_pass_asr_reason": str(metadata_snapshot.get("second_pass_asr_reason", "") or ""),
        "second_pass_asr_improved": bool(metadata_snapshot.get("second_pass_asr_improved", False)),
        "second_pass_asr_blocked_reason": str(metadata_snapshot.get("second_pass_asr_blocked_reason", "") or ""),
    }
    fixture_alignment = compare_real_audio_review_to_fixture(benchmark_eval, benchmark_case=benchmark_case)
    result = {
        "case_id": real_case.case_id,
        "label": real_case.label,
        "benchmark_group": real_case.benchmark_group,
        "run_status": run_status,
        "pipeline_result_summary": pipeline_result_summary,
        "evaluation_result": evaluation_result,
        "expectation_mismatches": list(benchmark_eval.get("mismatches", []) or []),
        "review_notes": real_case.notes,
        "generated_artifact_paths": list(pipeline_result.get("generated_artifact_paths", []) or []),
        "strategy_metadata": strategy_metadata,
        **fixture_alignment,
    }
    return _json_safe(result)


def summarize_real_audio_review_run(review_result: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = dict(review_result.get("evaluation_result", {}) or {})
    mismatches = list(review_result.get("expectation_mismatches", []) or [])
    return _json_safe(
        {
            "case_id": str(review_result.get("case_id", "") or ""),
            "transcript_state": str(evaluation.get("transcript_state", "") or ""),
            "rescue_ran": bool(evaluation.get("rescue_ran", False)),
            "rescue_effect_summary": str(evaluation.get("rescue_effect", "") or ""),
            "downstream_decision": str(evaluation.get("downstream_decision", "") or ""),
            "summary_state": str(evaluation.get("summary_state", "") or ""),
            "display_transcript_available": bool(evaluation.get("transcript_display_available", False)),
            "english_view_decision": str(evaluation.get("english_view_decision", "") or ""),
            "dominant_safety_reason": str(evaluation.get("dominant_safety_reason", "") or ""),
            "dominant_failure_reason": str(evaluation.get("dominant_failure_reason", "") or ""),
            "mismatch_summary": [str(item.get("field", "") or "") for item in mismatches if isinstance(item, dict)],
        }
    )


def run_real_audio_review_suite(
    cases: list[RealAudioReviewCase | Dict[str, Any]],
    *,
    pipeline_runner: Optional[Callable[[RealAudioReviewCase], Dict[str, Any]]] = None,
    threshold_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    results = [
        run_real_audio_review_case(case, pipeline_runner=pipeline_runner, threshold_profile=threshold_profile)
        for case in (cases or [])
    ]
    return _json_safe(
        {
            "total_cases": len(results),
            "completed_cases": sum(1 for row in results if str(row.get("run_status", "") or "") == "completed"),
            "results": results,
        }
    )


def _invoke_strategy_pipeline_runner(
    runner: Callable[..., Dict[str, Any]],
    case: RealAudioReviewCase,
    strategy: str,
) -> Dict[str, Any]:
    try:
        signature = inspect.signature(runner)
        supports_strategy_kwarg = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD or name == "strategy"
            for name, parameter in signature.parameters.items()
        )
    except Exception:
        supports_strategy_kwarg = False
    if supports_strategy_kwarg:
        return runner(case, strategy=strategy)
    return runner(case)


def _run_real_audio_case_under_strategy(
    case: RealAudioReviewCase,
    *,
    strategy: str,
    pipeline_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    threshold_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runner = pipeline_runner or _default_real_audio_pipeline_runner

    def wrapped_runner(real_case: RealAudioReviewCase) -> Dict[str, Any]:
        with _temporary_setting("ASR_MALAYALAM_STRATEGY", strategy):
            return _invoke_strategy_pipeline_runner(runner, real_case, strategy)

    return run_real_audio_review_case(
        case,
        pipeline_runner=wrapped_runner,
        threshold_profile=threshold_profile,
    )


def _extract_strategy_case_metrics(review_result: Dict[str, Any], *, strategy: str) -> Dict[str, Any]:
    evaluation = dict(review_result.get("evaluation_result", {}) or {})
    strategy_metadata = dict(review_result.get("strategy_metadata", {}) or {})
    return _json_safe(
        {
            "case_id": str(review_result.get("case_id", "") or ""),
            "label": str(review_result.get("label", "") or ""),
            "benchmark_group": str(review_result.get("benchmark_group", "") or ""),
            "strategy": strategy,
            "run_status": str(review_result.get("run_status", "") or ""),
            "transcript_state": str(evaluation.get("transcript_state", "") or ""),
            "rescue_ran": bool(evaluation.get("rescue_ran", False)),
            "rescue_effect": str(evaluation.get("rescue_effect", "") or ""),
            "downstream_decision": str(evaluation.get("downstream_decision", "") or ""),
            "summary_state": str(evaluation.get("summary_state", "") or ""),
            "english_view_decision": str(evaluation.get("english_view_decision", "") or ""),
            "dominant_safety_reason": str(evaluation.get("dominant_safety_reason", "") or ""),
            "dominant_failure_reason": str(evaluation.get("dominant_failure_reason", "") or ""),
            "trusted_visible_word_count": int(evaluation.get("trusted_visible_word_count", 0) or 0),
            "trusted_display_unit_count": int(evaluation.get("trusted_display_unit_count", 0) or 0),
            "readability": float(evaluation.get("overall_readability", 0.0) or 0.0),
            "lexical_trust": float(evaluation.get("lexical_trust_score", 0.0) or 0.0),
            "fixture_alignment_status": str(review_result.get("fixture_alignment_status", "") or ""),
            "expectation_mismatches": list(review_result.get("expectation_mismatches", []) or []),
            "second_pass_asr_attempted": bool(strategy_metadata.get("second_pass_asr_attempted", False)),
            "second_pass_asr_improved": bool(strategy_metadata.get("second_pass_asr_improved", False)),
            "second_pass_asr_reason": str(strategy_metadata.get("second_pass_asr_reason", "") or ""),
            "second_pass_asr_blocked_reason": str(strategy_metadata.get("second_pass_asr_blocked_reason", "") or ""),
            "primary_model_used": str(strategy_metadata.get("primary_model_used", "") or ""),
            "fallback_model_used": str(strategy_metadata.get("fallback_model_used", "") or ""),
            "retry_model_used": str(strategy_metadata.get("retry_model_used", "") or ""),
        }
    )


def run_malayalam_asr_strategy_review_case(
    case: RealAudioReviewCase | Dict[str, Any],
    *,
    strategies: Optional[list[str]] = None,
    pipeline_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    threshold_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    real_case = _coerce_real_audio_review_case(case)
    selected_strategies = [
        strategy
        for strategy in (strategies or list(MALAYALAM_ASR_REVIEW_STRATEGIES))
        if strategy in MALAYALAM_ASR_REVIEW_STRATEGIES
    ] or ["current_default"]
    per_strategy_results: Dict[str, Any] = {}
    for strategy in dict.fromkeys(selected_strategies):
        review_result = _run_real_audio_case_under_strategy(
            real_case,
            strategy=strategy,
            pipeline_runner=pipeline_runner,
            threshold_profile=threshold_profile,
        )
        per_strategy_results[strategy] = {
            "review_result": review_result,
            "summary": _extract_strategy_case_metrics(review_result, strategy=strategy),
        }
    return _json_safe(
        {
            "case_id": real_case.case_id,
            "label": real_case.label,
            "benchmark_group": real_case.benchmark_group,
            "selected_strategies": list(per_strategy_results.keys()),
            "strategy_results": per_strategy_results,
        }
    )


def _transcript_state_rank(value: str) -> int:
    return {"failed": 0, "degraded": 1, "cleaned": 2}.get(str(value or ""), 0)


def _decision_rank(value: str) -> int:
    return {"suppressed": 0, "blocked": 0, "allowed": 1, "available": 1, "same_as_original": 1}.get(str(value or ""), 0)


def _classify_strategy_case_change(
    case: RealAudioReviewCase,
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    if str(candidate.get("run_status", "") or "") != "completed":
        return {"change_classification": "review_needed", "reason": "candidate_run_incomplete", "safety_regression": False}
    if str(baseline.get("run_status", "") or "") != "completed":
        return {"change_classification": "review_needed", "reason": "baseline_run_incomplete", "safety_regression": False}

    protected_case = (
        str(case.expected_downstream_decision or "") == "suppressed"
        or str(case.expected_english_view_decision or "") == "blocked"
        or "protected" in set(case.tags or [])
        or case.benchmark_group in {"wrong_script_hopeless", "degraded_low_evidence_malayalam", "english_contaminated_malayalam"}
    )
    safety_regression = False
    reasons: list[str] = []

    if protected_case and str(baseline.get("downstream_decision", "") or "") == "suppressed" and str(candidate.get("downstream_decision", "") or "") == "allowed":
        safety_regression = True
        reasons.append("protected_downstream_regression")
    if protected_case and str(case.expected_english_view_decision or "") == "blocked" and str(candidate.get("english_view_decision", "") or "") == "available":
        safety_regression = True
        reasons.append("protected_english_view_regression")
    if str(baseline.get("fixture_alignment_status", "") or "") == "aligned" and str(candidate.get("fixture_alignment_status", "") or "") == "divergent":
        reasons.append("fixture_alignment_loss")
        if protected_case:
            safety_regression = True

    baseline_score = (
        _transcript_state_rank(baseline.get("transcript_state", ""))
        + _decision_rank(baseline.get("downstream_decision", ""))
        + _decision_rank(baseline.get("english_view_decision", ""))
    )
    candidate_score = (
        _transcript_state_rank(candidate.get("transcript_state", ""))
        + _decision_rank(candidate.get("downstream_decision", ""))
        + _decision_rank(candidate.get("english_view_decision", ""))
    )

    if safety_regression:
        return {"change_classification": "regression", "reason": ",".join(reasons) or "safety_regression", "safety_regression": True}

    if candidate_score > baseline_score:
        reasons.append("higher_decision_score")
    if int(candidate.get("trusted_visible_word_count", 0) or 0) > int(baseline.get("trusted_visible_word_count", 0) or 0):
        reasons.append("more_trusted_visible_words")
    if float(candidate.get("readability", 0.0) or 0.0) > float(baseline.get("readability", 0.0) or 0.0) + 0.03:
        reasons.append("better_readability")
    if bool(candidate.get("second_pass_asr_improved", False)):
        reasons.append("second_pass_helped")
    if str(baseline.get("fixture_alignment_status", "") or "") == "divergent" and str(candidate.get("fixture_alignment_status", "") or "") == "aligned":
        reasons.append("fixture_alignment_gain")

    if reasons:
        return {"change_classification": "improvement", "reason": ",".join(reasons), "safety_regression": False}

    regression_reasons: list[str] = []
    if candidate_score < baseline_score:
        regression_reasons.append("lower_decision_score")
    if str(candidate.get("dominant_failure_reason", "") or "") and str(candidate.get("dominant_failure_reason", "") or "") != str(baseline.get("dominant_failure_reason", "") or ""):
        regression_reasons.append("failure_reason_shift")
    if regression_reasons:
        return {"change_classification": "regression", "reason": ",".join(regression_reasons), "safety_regression": False}

    return {"change_classification": "unchanged", "reason": "no_material_change", "safety_regression": False}


def run_malayalam_asr_strategy_review_suite(
    cases: list[RealAudioReviewCase | Dict[str, Any]],
    *,
    strategies: Optional[list[str]] = None,
    baseline_strategy: str = "current_default",
    pipeline_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    threshold_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected_strategies = [
        strategy
        for strategy in (strategies or list(MALAYALAM_ASR_REVIEW_STRATEGIES))
        if strategy in MALAYALAM_ASR_REVIEW_STRATEGIES
    ] or ["current_default"]
    if baseline_strategy not in selected_strategies:
        selected_strategies = [baseline_strategy] + selected_strategies
    selected_strategies = list(dict.fromkeys(selected_strategies))

    case_results = []
    improved_case_ids_by_strategy = {strategy: [] for strategy in selected_strategies if strategy != baseline_strategy}
    regressed_case_ids_by_strategy = {strategy: [] for strategy in selected_strategies if strategy != baseline_strategy}
    unchanged_case_ids_by_strategy = {strategy: [] for strategy in selected_strategies if strategy != baseline_strategy}
    safety_regression_case_ids_by_strategy = {strategy: [] for strategy in selected_strategies if strategy != baseline_strategy}
    second_pass_helped_case_ids = {strategy: [] for strategy in selected_strategies}
    second_pass_not_helpful_case_ids = {strategy: [] for strategy in selected_strategies}
    strategy_case_counts = {strategy: 0 for strategy in selected_strategies}
    strategy_fixture_alignment_gain = {strategy: [] for strategy in selected_strategies if strategy != baseline_strategy}
    strategy_fixture_alignment_loss = {strategy: [] for strategy in selected_strategies if strategy != baseline_strategy}
    fixture_alignment_case_changes = {strategy: [] for strategy in selected_strategies if strategy != baseline_strategy}

    for raw_case in cases or []:
        real_case = _coerce_real_audio_review_case(raw_case)
        case_run = run_malayalam_asr_strategy_review_case(
            real_case,
            strategies=selected_strategies,
            pipeline_runner=pipeline_runner,
            threshold_profile=threshold_profile,
        )
        summaries = {
            strategy: dict(payload.get("summary", {}) or {})
            for strategy, payload in dict(case_run.get("strategy_results", {}) or {}).items()
        }
        baseline_summary = dict(summaries.get(baseline_strategy, {}) or {})
        per_strategy_change = {}
        for strategy, summary in summaries.items():
            strategy_case_counts[strategy] = strategy_case_counts.get(strategy, 0) + 1
            if bool(summary.get("second_pass_asr_improved", False)):
                second_pass_helped_case_ids[strategy].append(real_case.case_id)
            elif bool(summary.get("second_pass_asr_attempted", False)):
                second_pass_not_helpful_case_ids[strategy].append(real_case.case_id)
            if strategy == baseline_strategy:
                continue
            classified = _classify_strategy_case_change(real_case, baseline_summary, summary)
            per_strategy_change[strategy] = classified
            change_classification = classified["change_classification"]
            if change_classification == "improvement":
                improved_case_ids_by_strategy[strategy].append(real_case.case_id)
            elif change_classification == "regression":
                regressed_case_ids_by_strategy[strategy].append(real_case.case_id)
            elif change_classification == "unchanged":
                unchanged_case_ids_by_strategy[strategy].append(real_case.case_id)
            else:
                unchanged_case_ids_by_strategy[strategy].append(real_case.case_id)
            if bool(classified.get("safety_regression", False)):
                safety_regression_case_ids_by_strategy[strategy].append(real_case.case_id)
            baseline_alignment = str(baseline_summary.get("fixture_alignment_status", "") or "")
            candidate_alignment = str(summary.get("fixture_alignment_status", "") or "")
            if baseline_alignment == "divergent" and candidate_alignment == "aligned":
                strategy_fixture_alignment_gain[strategy].append(real_case.case_id)
                fixture_alignment_case_changes[strategy].append(real_case.case_id)
            elif baseline_alignment == "aligned" and candidate_alignment == "divergent":
                strategy_fixture_alignment_loss[strategy].append(real_case.case_id)
                fixture_alignment_case_changes[strategy].append(real_case.case_id)
        case_results.append(
            {
                "case_id": real_case.case_id,
                "label": real_case.label,
                "benchmark_group": real_case.benchmark_group,
                "strategy_summaries": summaries,
                "per_strategy_change": per_strategy_change,
            }
        )

    return _json_safe(
        {
            "comparison_version": REAL_AUDIO_STRATEGY_COMPARISON_VERSION,
            "baseline_strategy": baseline_strategy,
            "candidate_strategies": [strategy for strategy in selected_strategies if strategy != baseline_strategy],
            "total_cases": len(case_results),
            "strategy_case_counts": strategy_case_counts,
            "case_results": case_results,
            "improved_case_ids_by_strategy": improved_case_ids_by_strategy,
            "regressed_case_ids_by_strategy": regressed_case_ids_by_strategy,
            "unchanged_case_ids_by_strategy": unchanged_case_ids_by_strategy,
            "safety_regression_case_ids_by_strategy": safety_regression_case_ids_by_strategy,
            "second_pass_helped_case_ids": second_pass_helped_case_ids,
            "second_pass_not_helpful_case_ids": second_pass_not_helpful_case_ids,
            "strategy_fixture_alignment_gain": strategy_fixture_alignment_gain,
            "strategy_fixture_alignment_loss": strategy_fixture_alignment_loss,
            "fixture_alignment_case_changes": fixture_alignment_case_changes,
        }
    )


def summarize_malayalam_asr_strategy_comparison(comparison_result: Dict[str, Any]) -> Dict[str, Any]:
    baseline_strategy = str(comparison_result.get("baseline_strategy", "current_default") or "current_default")
    candidate_strategies = list(comparison_result.get("candidate_strategies", []) or [])
    improved_case_ids_by_strategy = {
        str(key): list(value or [])
        for key, value in dict(comparison_result.get("improved_case_ids_by_strategy", {}) or {}).items()
    }
    regressed_case_ids_by_strategy = {
        str(key): list(value or [])
        for key, value in dict(comparison_result.get("regressed_case_ids_by_strategy", {}) or {}).items()
    }
    unchanged_case_ids_by_strategy = {
        str(key): list(value or [])
        for key, value in dict(comparison_result.get("unchanged_case_ids_by_strategy", {}) or {}).items()
    }
    safety_regression_case_ids_by_strategy = {
        str(key): list(value or [])
        for key, value in dict(comparison_result.get("safety_regression_case_ids_by_strategy", {}) or {}).items()
    }
    second_pass_helped_case_ids = {
        str(key): list(value or [])
        for key, value in dict(comparison_result.get("second_pass_helped_case_ids", {}) or {}).items()
    }
    second_pass_not_helpful_case_ids = {
        str(key): list(value or [])
        for key, value in dict(comparison_result.get("second_pass_not_helpful_case_ids", {}) or {}).items()
    }

    dominant_improvement_reasons_by_strategy: Dict[str, list[str]] = {}
    dominant_regression_reasons_by_strategy: Dict[str, list[str]] = {}
    recommended_default_strategy = baseline_strategy
    recommended_default_strategy_reason = "No candidate strategy clearly outperformed the current default without added risk."
    best_improvement_count = 0

    for strategy in candidate_strategies:
        change_entries = [
            dict(item.get("per_strategy_change", {}).get(strategy, {}) or {})
            for item in list(comparison_result.get("case_results", []) or [])
            if strategy in dict(item.get("per_strategy_change", {}) or {})
        ]
        improvement_reasons: Dict[str, int] = {}
        regression_reasons: Dict[str, int] = {}
        for change in change_entries:
            reason = str(change.get("reason", "") or "")
            if not reason:
                continue
            if str(change.get("change_classification", "") or "") == "improvement":
                for part in [segment for segment in reason.split(",") if segment]:
                    improvement_reasons[part] = improvement_reasons.get(part, 0) + 1
            elif str(change.get("change_classification", "") or "") == "regression":
                for part in [segment for segment in reason.split(",") if segment]:
                    regression_reasons[part] = regression_reasons.get(part, 0) + 1
        dominant_improvement_reasons_by_strategy[strategy] = [
            key for key, _ in sorted(improvement_reasons.items(), key=lambda item: (-item[1], item[0]))
        ][:3]
        dominant_regression_reasons_by_strategy[strategy] = [
            key for key, _ in sorted(regression_reasons.items(), key=lambda item: (-item[1], item[0]))
        ][:3]

        improvement_count = len(improved_case_ids_by_strategy.get(strategy, []))
        regression_count = len(regressed_case_ids_by_strategy.get(strategy, []))
        safety_count = len(safety_regression_case_ids_by_strategy.get(strategy, []))
        if safety_count:
            continue
        if improvement_count > best_improvement_count and improvement_count > regression_count:
            best_improvement_count = improvement_count
            recommended_default_strategy = strategy
            recommended_default_strategy_reason = (
                f"{strategy} improved more curated Malayalam review cases than the baseline without observed safety regressions."
            )

    if recommended_default_strategy != baseline_strategy and any(
        safety_regression_case_ids_by_strategy.get(strategy) for strategy in candidate_strategies
    ):
        recommended_default_strategy = baseline_strategy
        recommended_default_strategy_reason = "At least one candidate strategy caused safety regressions, so the current default remains the safer choice."

    recommended_followup_cases_for_audio_review = sorted(
        {
            case_id
            for values in regressed_case_ids_by_strategy.values()
            for case_id in values
        }
        | {
            case_id
            for values in safety_regression_case_ids_by_strategy.values()
            for case_id in values
        }
    )
    recommended_followup_cases_for_asr_investigation = sorted(
        {
            case_id
            for values in second_pass_not_helpful_case_ids.values()
            for case_id in values
        }
        | {
            case_id
            for values in comparison_result.get("fixture_alignment_case_changes", {}).values()
            for case_id in values
        }
    )

    return _json_safe(
        {
            "baseline_strategy": baseline_strategy,
            "candidate_strategies": candidate_strategies,
            "total_cases": int(comparison_result.get("total_cases", 0) or 0),
            "strategy_case_counts": dict(comparison_result.get("strategy_case_counts", {}) or {}),
            "improved_case_ids_by_strategy": improved_case_ids_by_strategy,
            "regressed_case_ids_by_strategy": regressed_case_ids_by_strategy,
            "unchanged_case_ids_by_strategy": unchanged_case_ids_by_strategy,
            "safety_regression_case_ids_by_strategy": safety_regression_case_ids_by_strategy,
            "second_pass_helped_case_ids": second_pass_helped_case_ids,
            "second_pass_not_helpful_case_ids": second_pass_not_helpful_case_ids,
            "dominant_improvement_reasons_by_strategy": dominant_improvement_reasons_by_strategy,
            "dominant_regression_reasons_by_strategy": dominant_regression_reasons_by_strategy,
            "strategy_fixture_alignment_gain": dict(comparison_result.get("strategy_fixture_alignment_gain", {}) or {}),
            "strategy_fixture_alignment_loss": dict(comparison_result.get("strategy_fixture_alignment_loss", {}) or {}),
            "fixture_alignment_case_changes": dict(comparison_result.get("fixture_alignment_case_changes", {}) or {}),
            "recommended_default_strategy": recommended_default_strategy,
            "recommended_default_strategy_reason": recommended_default_strategy_reason,
            "recommended_followup_cases_for_audio_review": recommended_followup_cases_for_audio_review,
            "recommended_followup_cases_for_asr_investigation": recommended_followup_cases_for_asr_investigation,
        }
    )


def conclude_malayalam_asr_strategy_default(
    strategy_summary_or_comparison: Dict[str, Any],
) -> Dict[str, Any]:
    summary = (
        summarize_malayalam_asr_strategy_comparison(strategy_summary_or_comparison)
        if "recommended_default_strategy" not in dict(strategy_summary_or_comparison or {})
        else _json_safe(strategy_summary_or_comparison)
    )
    baseline_strategy = str(summary.get("baseline_strategy", "current_default") or "current_default")
    candidate_strategies = list(summary.get("candidate_strategies", []) or [])
    improved_case_ids_by_strategy = {
        str(key): list(value or [])
        for key, value in dict(summary.get("improved_case_ids_by_strategy", {}) or {}).items()
    }
    regressed_case_ids_by_strategy = {
        str(key): list(value or [])
        for key, value in dict(summary.get("regressed_case_ids_by_strategy", {}) or {}).items()
    }
    safety_regression_case_ids_by_strategy = {
        str(key): list(value or [])
        for key, value in dict(summary.get("safety_regression_case_ids_by_strategy", {}) or {}).items()
    }
    second_pass_helped_case_ids = {
        str(key): list(value or [])
        for key, value in dict(summary.get("second_pass_helped_case_ids", {}) or {}).items()
    }
    second_pass_not_helpful_case_ids = {
        str(key): list(value or [])
        for key, value in dict(summary.get("second_pass_not_helpful_case_ids", {}) or {}).items()
    }
    fixture_alignment_gain = {
        str(key): list(value or [])
        for key, value in dict(summary.get("strategy_fixture_alignment_gain", {}) or {}).items()
    }
    fixture_alignment_loss = {
        str(key): list(value or [])
        for key, value in dict(summary.get("strategy_fixture_alignment_loss", {}) or {}).items()
    }

    strategy_rank_rows = []
    safe_candidate_strategies = []
    rejected_candidate_strategies = []
    review_candidate_strategies = []
    strategy_rank_notes: Dict[str, str] = {}

    for strategy in candidate_strategies:
        improved = len(improved_case_ids_by_strategy.get(strategy, []))
        regressed = len(regressed_case_ids_by_strategy.get(strategy, []))
        safety = len(safety_regression_case_ids_by_strategy.get(strategy, []))
        second_pass_helped = len(second_pass_helped_case_ids.get(strategy, []))
        second_pass_not_helpful = len(second_pass_not_helpful_case_ids.get(strategy, []))
        alignment_gain = len(fixture_alignment_gain.get(strategy, []))
        alignment_loss = len(fixture_alignment_loss.get(strategy, []))
        score = (improved * 4) + alignment_gain + second_pass_helped - (regressed * 3) - (alignment_loss * 2) - (safety * 10) - second_pass_not_helpful

        if safety > 0:
            note = "Rejected because it caused protected or safety-sensitive regressions."
            rejected_candidate_strategies.append(strategy)
        elif improved > 0 and improved > regressed:
            note = "Safe enough for manual adoption review because it improved curated cases without observed safety regressions."
            safe_candidate_strategies.append(strategy)
            review_candidate_strategies.append(strategy)
        else:
            note = "No clear safe advantage over the current default."
        strategy_rank_notes[strategy] = note
        strategy_rank_rows.append(
            {
                "strategy": strategy,
                "score": score,
                "improved_cases": improved,
                "regressed_cases": regressed,
                "safety_regressions": safety,
                "fixture_alignment_gains": alignment_gain,
                "fixture_alignment_losses": alignment_loss,
                "second_pass_helped_cases": second_pass_helped,
                "second_pass_not_helpful_cases": second_pass_not_helpful,
                "note": note,
            }
        )

    ranked_strategies = [baseline_strategy] + [
        row["strategy"]
        for row in sorted(
            strategy_rank_rows,
            key=lambda item: (
                item["safety_regressions"],
                -item["improved_cases"],
                item["regressed_cases"],
                -item["fixture_alignment_gains"],
                item["fixture_alignment_losses"],
                item["strategy"],
            ),
        )
    ]

    winning_strategy = baseline_strategy
    winning_strategy_status = "baseline_preferred"
    final_strategy_recommendation = "keep_current_default"
    final_strategy_recommendation_reason = "No candidate strategy showed strong enough safe gains to justify changing the production default."
    adoption_review_justified = False
    keep_current_default = True
    reject_candidate_strategies = bool(rejected_candidate_strategies)

    if len(review_candidate_strategies) == 1:
        winning_strategy = review_candidate_strategies[0]
        winning_strategy_status = "safe_candidate_for_manual_review"
        final_strategy_recommendation = "manual_adoption_review_only"
        final_strategy_recommendation_reason = (
            f"{winning_strategy} showed safe gains on curated real-audio cases without observed protected regressions, but still needs manual adoption review."
        )
        adoption_review_justified = True
        keep_current_default = False
    elif rejected_candidate_strategies and not review_candidate_strategies:
        final_strategy_recommendation = "reject_all_candidates_for_now"
        final_strategy_recommendation_reason = "Candidate strategies introduced safety-sensitive regressions or failed to produce clear safe gains."
        winning_strategy_status = "all_candidates_rejected"

    strongest_positive_signal = ""
    if review_candidate_strategies:
        best = review_candidate_strategies[0]
        strongest_positive_signal = f"{best}_safe_improvements={len(improved_case_ids_by_strategy.get(best, []))}"
    strongest_negative_signal = ""
    if rejected_candidate_strategies:
        worst = rejected_candidate_strategies[0]
        strongest_negative_signal = f"{worst}_safety_regressions={len(safety_regression_case_ids_by_strategy.get(worst, []))}"
    elif all(not improved_case_ids_by_strategy.get(strategy) for strategy in candidate_strategies):
        strongest_negative_signal = "no_candidate_clear_safe_gain"

    recommended_audio_review_case_ids = sorted(
        {
            case_id
            for strategy in candidate_strategies
            for case_id in regressed_case_ids_by_strategy.get(strategy, [])
        }
        | {
            case_id
            for strategy in candidate_strategies
            for case_id in safety_regression_case_ids_by_strategy.get(strategy, [])
        }
    )
    recommended_asr_investigation_case_ids = sorted(
        {
            case_id
            for strategy in candidate_strategies
            for case_id in second_pass_not_helpful_case_ids.get(strategy, [])
        }
        | set(summary.get("recommended_followup_cases_for_asr_investigation", []) or [])
    )

    if final_strategy_recommendation == "manual_adoption_review_only":
        recommended_next_action = "run_manual_adoption_review_on_single_safe_candidate"
        recommended_strategy_followup = f"review_only_{winning_strategy}"
        recommended_non_threshold_work = [
            "listen_to_improved_real_audio_cases_manually",
            "verify_no_hidden_safety_regressions_on_additional Malayalam clips",
            "keep current production default unchanged until review completes",
        ]
        decision_confidence = "medium"
    elif final_strategy_recommendation == "reject_all_candidates_for_now":
        recommended_next_action = "keep_current_default_and_shift_effort_to_asr_source_quality"
        recommended_strategy_followup = "reject_candidates_keep_baseline"
        recommended_non_threshold_work = [
            "expand real Malayalam audio coverage",
            "investigate source ASR/model quality on hard clips",
            "review Malayalam preprocessing on divergent cases",
        ]
        decision_confidence = "high" if rejected_candidate_strategies else "medium"
    else:
        recommended_next_action = "keep_current_default_and_continue_real_audio_review"
        recommended_strategy_followup = "retain_baseline_as_reference"
        recommended_non_threshold_work = [
            "collect more hard Malayalam clips",
            "investigate second-pass cases that did not help",
            "promote divergent real-audio cases into fixture coverage",
        ]
        decision_confidence = "medium" if candidate_strategies else "low"

    rationale_bullets = [
        f"Baseline strategy remained {baseline_strategy}.",
        f"Safe candidate strategies: {', '.join(review_candidate_strategies) or 'none'}.",
        f"Rejected candidate strategies: {', '.join(rejected_candidate_strategies) or 'none'}.",
        f"Audio review follow-up cases: {len(recommended_audio_review_case_ids)}.",
        f"ASR investigation follow-up cases: {len(recommended_asr_investigation_case_ids)}.",
    ]
    rationale_summary = final_strategy_recommendation_reason

    conclusion = {
        "final_strategy_recommendation": final_strategy_recommendation,
        "final_strategy_recommendation_reason": final_strategy_recommendation_reason,
        "recommended_production_default": baseline_strategy if keep_current_default else winning_strategy,
        "adoption_review_justified": adoption_review_justified,
        "keep_current_default": keep_current_default,
        "reject_candidate_strategies": reject_candidate_strategies,
        "winning_strategy": winning_strategy,
        "winning_strategy_status": winning_strategy_status,
        "ranked_strategies": ranked_strategies,
        "strategy_rank_notes": strategy_rank_notes,
        "safe_candidate_strategies": safe_candidate_strategies,
        "rejected_candidate_strategies": rejected_candidate_strategies,
        "review_candidate_strategies": review_candidate_strategies,
        "recommended_next_action": recommended_next_action,
        "recommended_audio_review_case_ids": recommended_audio_review_case_ids,
        "recommended_asr_investigation_case_ids": recommended_asr_investigation_case_ids,
        "recommended_strategy_followup": recommended_strategy_followup,
        "recommended_non_threshold_work": recommended_non_threshold_work,
        "rationale_summary": rationale_summary,
        "rationale_bullets": rationale_bullets,
        "strongest_positive_signal": strongest_positive_signal,
        "strongest_negative_signal": strongest_negative_signal,
        "decision_confidence": decision_confidence,
        "supporting_metrics_summary": {
            "improved_case_ids_by_strategy": improved_case_ids_by_strategy,
            "regressed_case_ids_by_strategy": regressed_case_ids_by_strategy,
            "safety_regression_case_ids_by_strategy": safety_regression_case_ids_by_strategy,
            "second_pass_helped_case_ids": second_pass_helped_case_ids,
            "second_pass_not_helpful_case_ids": second_pass_not_helpful_case_ids,
            "fixture_alignment_gain": fixture_alignment_gain,
            "fixture_alignment_loss": fixture_alignment_loss,
        },
    }
    return _json_safe(conclusion)


def export_malayalam_asr_strategy_decision(
    strategy_summary_or_comparison: Dict[str, Any],
) -> Dict[str, Any]:
    conclusion = conclude_malayalam_asr_strategy_default(strategy_summary_or_comparison)
    return _json_safe(
        {
            "export_version": "ml_asr_strategy_decision_v1",
            "recommendation": {
                "final_strategy_recommendation": conclusion["final_strategy_recommendation"],
                "final_strategy_recommendation_reason": conclusion["final_strategy_recommendation_reason"],
                "recommended_production_default": conclusion["recommended_production_default"],
                "winning_strategy": conclusion["winning_strategy"],
                "winning_strategy_status": conclusion["winning_strategy_status"],
                "adoption_review_justified": conclusion["adoption_review_justified"],
                "keep_current_default": conclusion["keep_current_default"],
                "reject_candidate_strategies": conclusion["reject_candidate_strategies"],
            },
            "rationale": {
                "rationale_summary": conclusion["rationale_summary"],
                "rationale_bullets": conclusion["rationale_bullets"],
                "strongest_positive_signal": conclusion["strongest_positive_signal"],
                "strongest_negative_signal": conclusion["strongest_negative_signal"],
                "decision_confidence": conclusion["decision_confidence"],
            },
            "ranking": {
                "ranked_strategies": conclusion["ranked_strategies"],
                "strategy_rank_notes": conclusion["strategy_rank_notes"],
                "safe_candidate_strategies": conclusion["safe_candidate_strategies"],
                "rejected_candidate_strategies": conclusion["rejected_candidate_strategies"],
                "review_candidate_strategies": conclusion["review_candidate_strategies"],
            },
            "follow_up_guidance": {
                "recommended_next_action": conclusion["recommended_next_action"],
                "recommended_audio_review_case_ids": conclusion["recommended_audio_review_case_ids"],
                "recommended_asr_investigation_case_ids": conclusion["recommended_asr_investigation_case_ids"],
                "recommended_strategy_followup": conclusion["recommended_strategy_followup"],
                "recommended_non_threshold_work": conclusion["recommended_non_threshold_work"],
            },
            "supporting_metrics_summary": conclusion["supporting_metrics_summary"],
        }
    )


def run_default_malayalam_asr_strategy_decision_flow(
    *,
    pipeline_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    strategies: Optional[list[str]] = None,
    baseline_strategy: str = "current_default",
    threshold_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    suite_result = run_malayalam_asr_strategy_review_suite(
        build_malayalam_real_audio_review_cases(),
        strategies=strategies or list(MALAYALAM_ASR_REVIEW_STRATEGIES),
        baseline_strategy=baseline_strategy,
        pipeline_runner=pipeline_runner,
        threshold_profile=threshold_profile,
    )
    comparison_summary = summarize_malayalam_asr_strategy_comparison(suite_result)
    final_decision = conclude_malayalam_asr_strategy_default(comparison_summary)
    exported_decision = export_malayalam_asr_strategy_decision(final_decision)
    return _json_safe(
        {
            "suite_result": suite_result,
            "comparison_summary": comparison_summary,
            "final_decision": final_decision,
            "exported_decision": exported_decision,
        }
    )


def _real_audio_case(
    *,
    case_id: str,
    label: str,
    benchmark_group: str,
    media_path: str = "",
    input_reference: str = "",
    expected_language: str = "",
    expected_quality_bucket: str = "",
    expected_downstream_decision: str = "",
    expected_english_view_decision: str = "",
    tags: Optional[list[str]] = None,
    notes: str = "",
    human_reference_transcript: str = "",
    human_reference_summary: str = "",
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
) -> RealAudioReviewCase:
    return RealAudioReviewCase(
        case_id=case_id,
        label=label,
        media_path=media_path,
        input_reference=input_reference,
        expected_language=expected_language,
        expected_quality_bucket=expected_quality_bucket,
        expected_downstream_decision=expected_downstream_decision,
        expected_english_view_decision=expected_english_view_decision,
        benchmark_group=benchmark_group,
        tags=list(tags or []),
        notes=notes,
        human_reference_transcript=human_reference_transcript,
        human_reference_summary=human_reference_summary,
        pipeline_kwargs=dict(pipeline_kwargs or {}),
    )


def build_malayalam_real_audio_review_cases() -> list[RealAudioReviewCase]:
    return [
        _real_audio_case(
            case_id="real_audio_ml_cleaned_grounded_01",
            label="Malayalam Cleaned Grounded Sample",
            benchmark_group="cleaned_malayalam",
            media_path="review_samples/ml_cleaned_grounded_01.wav",
            expected_language="ml",
            expected_quality_bucket="clearly_cleaned",
            expected_downstream_decision="allowed",
            expected_english_view_decision="available",
            tags=["real_audio", "grounded", "cleaned"],
            notes="Reference cleaned Malayalam educational clip.",
        ),
        _real_audio_case(
            case_id="real_audio_ml_degraded_useful_01",
            label="Malayalam Degraded Useful Sample",
            benchmark_group="degraded_useful_malayalam",
            media_path="review_samples/ml_degraded_useful_01.wav",
            expected_language="ml",
            expected_quality_bucket="degraded_but_useful",
            expected_downstream_decision="allowed",
            expected_english_view_decision="blocked",
            tags=["real_audio", "degraded_useful"],
        ),
        _real_audio_case(
            case_id="real_audio_ml_degraded_low_evidence_01",
            label="Malayalam Low Evidence Sample",
            benchmark_group="degraded_low_evidence_malayalam",
            media_path="review_samples/ml_degraded_low_evidence_01.wav",
            expected_language="ml",
            expected_quality_bucket="degraded_low_evidence",
            expected_downstream_decision="suppressed",
            expected_english_view_decision="blocked",
            tags=["real_audio", "suppressed", "protected"],
        ),
        _real_audio_case(
            case_id="real_audio_ml_wrong_script_01",
            label="Malayalam Wrong Script Sample",
            benchmark_group="wrong_script_hopeless",
            media_path="review_samples/ml_wrong_script_01.wav",
            expected_language="ml",
            expected_quality_bucket="hopeless_wrong_script",
            expected_downstream_decision="suppressed",
            expected_english_view_decision="blocked",
            tags=["real_audio", "wrong_script", "protected"],
        ),
        _real_audio_case(
            case_id="real_audio_ml_mixed_edu_01",
            label="Malayalam Mixed Educational Sample",
            benchmark_group="mixed_malayalam_english_educational",
            media_path="review_samples/ml_mixed_edu_01.wav",
            expected_language="ml",
            expected_quality_bucket="degraded_but_useful",
            expected_downstream_decision="allowed",
            expected_english_view_decision="blocked",
            tags=["real_audio", "mixed_edu"],
        ),
    ]


def build_default_real_audio_review_cases() -> list[RealAudioReviewCase]:
    return build_malayalam_real_audio_review_cases() + [
        _real_audio_case(
            case_id="real_audio_en_reference_01",
            label="English Reference Sample",
            benchmark_group="english_reference",
            media_path="review_samples/en_reference_01.wav",
            expected_language="en",
            expected_quality_bucket="english_stable",
            expected_downstream_decision="allowed",
            expected_english_view_decision="same_as_original",
            tags=["real_audio", "english_reference"],
        )
    ]


def export_real_audio_review_suite_report(suite_result: Dict[str, Any]) -> Dict[str, Any]:
    results = list(suite_result.get("results", []) or [])
    case_reports = [summarize_real_audio_review_run(row) | {
        "label": str(row.get("label", "") or ""),
        "benchmark_group": str(row.get("benchmark_group", "") or ""),
        "input_reference": str(row.get("pipeline_result_summary", {}).get("input_reference", "") or ""),
        "run_status": str(row.get("run_status", "") or ""),
        "fixture_alignment_status": str(row.get("fixture_alignment_status", "") or ""),
        "mismatch_summary": list((summarize_real_audio_review_run(row)).get("mismatch_summary", []) or []),
    } for row in results]

    summary_by_transcript_state: Dict[str, int] = {}
    summary_by_downstream_decision: Dict[str, int] = {}
    summary_by_english_view_decision: Dict[str, int] = {}
    summary_by_benchmark_group: Dict[str, int] = {}
    changed_fixture_recommendations = []
    successful_runs = 0
    blocked_runs = 0
    failed_runs = 0
    cases_with_expectation_mismatches = []
    cases_aligned_with_fixture = []
    cases_diverging_from_fixture = []

    for row, case_report in zip(results, case_reports):
        run_status = str(row.get("run_status", "") or "")
        if run_status == "completed":
            successful_runs += 1
        elif run_status == "blocked":
            blocked_runs += 1
        else:
            failed_runs += 1

        transcript_state = str(case_report.get("transcript_state", "") or "")
        downstream_decision = str(case_report.get("downstream_decision", "") or "")
        english_view_decision = str(case_report.get("english_view_decision", "") or "")
        benchmark_group = str(row.get("benchmark_group", "") or "")
        summary_by_transcript_state[transcript_state] = summary_by_transcript_state.get(transcript_state, 0) + 1
        summary_by_downstream_decision[downstream_decision] = summary_by_downstream_decision.get(downstream_decision, 0) + 1
        summary_by_english_view_decision[english_view_decision] = summary_by_english_view_decision.get(english_view_decision, 0) + 1
        summary_by_benchmark_group[benchmark_group] = summary_by_benchmark_group.get(benchmark_group, 0) + 1

        if row.get("expectation_mismatches"):
            cases_with_expectation_mismatches.append(str(row.get("case_id", "") or ""))
        if str(row.get("fixture_alignment_status", "") or "") == "aligned":
            cases_aligned_with_fixture.append(str(row.get("case_id", "") or ""))
        if str(row.get("fixture_alignment_status", "") or "") == "divergent":
            cases_diverging_from_fixture.append(str(row.get("case_id", "") or ""))
        if bool(row.get("should_add_or_update_fixture", False)):
            changed_fixture_recommendations.append(
                {
                    "case_id": str(row.get("case_id", "") or ""),
                    "recommended_fixture_case_id": str(row.get("recommended_fixture_case_id", "") or ""),
                    "recommended_fixture_group": benchmark_group,
                    "divergence_reason": str(row.get("divergence_reason", "") or ""),
                    "promotion_priority": "high" if row.get("expectation_mismatches") else "medium",
                }
            )

    return _json_safe(
        {
            "report_version": "ml_real_audio_review_report_v1",
            "total_cases": len(results),
            "successful_runs": successful_runs,
            "blocked_runs": blocked_runs,
            "failed_runs": failed_runs,
            "cases_with_expectation_mismatches": cases_with_expectation_mismatches,
            "cases_aligned_with_fixture": cases_aligned_with_fixture,
            "cases_diverging_from_fixture": cases_diverging_from_fixture,
            "summary_by_transcript_state": summary_by_transcript_state,
            "summary_by_downstream_decision": summary_by_downstream_decision,
            "summary_by_english_view_decision": summary_by_english_view_decision,
            "summary_by_benchmark_group": summary_by_benchmark_group,
            "changed_fixture_recommendations": changed_fixture_recommendations,
            "case_reports": case_reports,
        }
    )


def review_real_audio_suite_report(report: Dict[str, Any]) -> Dict[str, Any]:
    changed_fixture_recommendations = list(report.get("changed_fixture_recommendations", []) or [])
    case_reports = list(report.get("case_reports", []) or [])
    divergent_cases = list(report.get("cases_diverging_from_fixture", []) or [])
    blocked_runs = int(report.get("blocked_runs", 0) or 0)

    recommended_audio_review_case_ids = [
        str(item.get("case_id", "") or "")
        for item in changed_fixture_recommendations
        if str(item.get("promotion_priority", "") or "") == "high"
    ] or list(divergent_cases)
    recommended_fixture_updates = [
        str(item.get("recommended_fixture_case_id", "") or item.get("case_id", "") or "")
        for item in changed_fixture_recommendations
    ]
    recommended_asr_investigation_case_ids = [
        str(item.get("case_id", "") or "")
        for item in case_reports
        if str(item.get("transcript_state", "") or "") in {"degraded", "failed"}
        and str(item.get("dominant_failure_reason", "") or "") != ""
    ]
    recommended_threshold_revisit_case_ids = [
        str(item.get("case_id", "") or "")
        for item in case_reports
        if str(item.get("transcript_state", "") or "") == "degraded"
        and str(item.get("downstream_decision", "") or "") == "allowed"
        and str(item.get("english_view_decision", "") or "") == "blocked"
    ]

    if divergent_cases:
        recommended_next_action = "inspect_divergent_real_audio_cases_and_update_fixture_pack"
        recommended_non_threshold_work = [
            "listen_to_divergent_audio_cases_manually",
            "compare_real_output_with_offline_fixture_assumptions",
            "investigate_source_asr_quality_before_threshold_changes",
        ]
    elif blocked_runs:
        recommended_next_action = "unblock_review_inputs_and_rerun_real_audio_suite"
        recommended_non_threshold_work = ["supply_video_ids_or_pipeline_inputs_for_blocked_cases"]
    else:
        recommended_next_action = "use_current_real_audio_suite_as_reference_and expand clips gradually"
        recommended_non_threshold_work = ["continue collecting representative Malayalam clips"]

    strongest_positive_signal = "real_audio_cases_match_existing_fixture_expectations" if not divergent_cases else ""
    strongest_negative_signal = "real_audio_fixture_divergence_detected" if divergent_cases else ("blocked_real_audio_runs_present" if blocked_runs else "")

    return _json_safe(
        {
            "recommended_next_action": recommended_next_action,
            "recommended_fixture_updates": recommended_fixture_updates,
            "recommended_audio_review_case_ids": recommended_audio_review_case_ids,
            "recommended_asr_investigation_case_ids": recommended_asr_investigation_case_ids,
            "recommended_threshold_revisit_case_ids": recommended_threshold_revisit_case_ids,
            "recommended_non_threshold_work": recommended_non_threshold_work,
            "strongest_positive_signal": strongest_positive_signal,
            "strongest_negative_signal": strongest_negative_signal,
        }
    )


def build_real_audio_review_history_entries(
    report: Dict[str, Any],
    *,
    review_run_id: str,
    created_at: str,
    suite_name: str = "real_audio_review_suite",
) -> list[Dict[str, Any]]:
    recommendations_by_case = {
        str(item.get("case_id", "") or ""): dict(item)
        for item in list(report.get("changed_fixture_recommendations", []) or [])
        if str(item.get("case_id", "") or "")
    }
    entries: list[Dict[str, Any]] = []
    for case_report in list(report.get("case_reports", []) or []):
        case_id = str(case_report.get("case_id", "") or "")
        recommendation = recommendations_by_case.get(case_id, {})
        entry = RealAudioReviewHistoryEntry(
            history_version="ml_real_audio_review_history_v1",
            review_run_id=str(review_run_id or ""),
            created_at=str(created_at or ""),
            suite_name=str(suite_name or "real_audio_review_suite"),
            case_id=case_id,
            label=str(case_report.get("label", "") or ""),
            benchmark_group=str(case_report.get("benchmark_group", "") or ""),
            input_reference=str(case_report.get("input_reference", "") or ""),
            transcript_state=str(case_report.get("transcript_state", "") or ""),
            rescue_ran=bool(case_report.get("rescue_ran", False)),
            downstream_decision=str(case_report.get("downstream_decision", "") or ""),
            summary_state=str(case_report.get("summary_state", "") or ""),
            english_view_decision=str(case_report.get("english_view_decision", "") or ""),
            dominant_safety_reason=str(case_report.get("dominant_safety_reason", "") or ""),
            dominant_failure_reason=str(case_report.get("dominant_failure_reason", "") or ""),
            fixture_alignment_status=str(case_report.get("fixture_alignment_status", "") or ""),
            mismatch_summary=[str(item) for item in list(case_report.get("mismatch_summary", []) or []) if str(item or "")],
            should_add_or_update_fixture=bool(recommendation),
            recommended_fixture_case_id=str(
                recommendation.get("recommended_fixture_case_id", "") or case_id
            ),
            promotion_priority=str(recommendation.get("promotion_priority", "") or ""),
        )
        entries.append(asdict(entry))
    return _json_safe(sorted(entries, key=lambda item: str(item.get("case_id", "") or "")))


def export_real_audio_review_history_snapshot(
    report: Dict[str, Any],
    *,
    review_run_id: str,
    created_at: str,
    suite_name: str = "real_audio_review_suite",
) -> Dict[str, Any]:
    entries = build_real_audio_review_history_entries(
        report,
        review_run_id=review_run_id,
        created_at=created_at,
        suite_name=suite_name,
    )
    return _json_safe(
        {
            "history_version": "ml_real_audio_review_history_v1",
            "review_run_id": str(review_run_id or ""),
            "created_at": str(created_at or ""),
            "suite_name": str(suite_name or "real_audio_review_suite"),
            "total_entries": len(entries),
            "entries": entries,
        }
    )


def compare_real_audio_review_history(
    previous_snapshot: Dict[str, Any],
    current_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    previous_entries = {
        str(item.get("case_id", "") or ""): dict(item)
        for item in list(previous_snapshot.get("entries", []) or [])
        if str(item.get("case_id", "") or "")
    }
    current_entries = {
        str(item.get("case_id", "") or ""): dict(item)
        for item in list(current_snapshot.get("entries", []) or [])
        if str(item.get("case_id", "") or "")
    }

    improved_case_ids: list[str] = []
    regressed_case_ids: list[str] = []
    unchanged_case_ids: list[str] = []
    review_needed_case_ids: list[str] = []
    new_case_ids = sorted(case_id for case_id in current_entries if case_id not in previous_entries)
    removed_case_ids = sorted(case_id for case_id in previous_entries if case_id not in current_entries)
    change_summaries: list[Dict[str, Any]] = []

    for case_id in sorted(set(previous_entries) & set(current_entries)):
        previous = previous_entries[case_id]
        current = current_entries[case_id]
        reasons: list[str] = []
        classification = "unchanged"

        if previous.get("transcript_state") != current.get("transcript_state"):
            reasons.append("transcript_state_changed")
        if previous.get("downstream_decision") != current.get("downstream_decision"):
            reasons.append("downstream_decision_changed")
        if previous.get("english_view_decision") != current.get("english_view_decision"):
            reasons.append("english_view_decision_changed")
        if previous.get("fixture_alignment_status") != current.get("fixture_alignment_status"):
            reasons.append("fixture_alignment_changed")
        if bool(previous.get("mismatch_summary")) and not bool(current.get("mismatch_summary")):
            reasons.append("mismatch_resolved")
        if not bool(previous.get("mismatch_summary")) and bool(current.get("mismatch_summary")):
            reasons.append("new_mismatch_appeared")
        if previous.get("dominant_failure_reason") != current.get("dominant_failure_reason"):
            reasons.append("dominant_failure_reason_changed")
        if previous.get("dominant_safety_reason") != current.get("dominant_safety_reason"):
            reasons.append("dominant_safety_reason_changed")

        previous_safety_risk = (
            previous.get("transcript_state") == "failed"
            or previous.get("downstream_decision") == "suppressed"
            or previous.get("fixture_alignment_status") == "divergent"
            or bool(previous.get("mismatch_summary"))
        )
        current_safety_risk = (
            current.get("transcript_state") == "failed"
            or current.get("downstream_decision") == "suppressed"
            or current.get("fixture_alignment_status") == "divergent"
            or bool(current.get("mismatch_summary"))
        )

        if "new_mismatch_appeared" in reasons or current.get("fixture_alignment_status") == "divergent":
            classification = "review_needed"
            review_needed_case_ids.append(case_id)
        elif current_safety_risk and not previous_safety_risk:
            classification = "regression"
            regressed_case_ids.append(case_id)
        elif previous_safety_risk and not current_safety_risk:
            classification = "improvement"
            improved_case_ids.append(case_id)
        elif reasons:
            old_state = str(previous.get("transcript_state", "") or "")
            new_state = str(current.get("transcript_state", "") or "")
            if old_state in {"degraded", "failed"} and new_state == "cleaned":
                classification = "improvement"
                improved_case_ids.append(case_id)
            elif old_state == "cleaned" and new_state in {"degraded", "failed"}:
                classification = "regression"
                regressed_case_ids.append(case_id)
            else:
                classification = "review_needed"
                review_needed_case_ids.append(case_id)
        else:
            unchanged_case_ids.append(case_id)

        change_summaries.append(
            {
                "case_id": case_id,
                "change_classification": classification,
                "change_reasons": reasons,
                "old_transcript_state": str(previous.get("transcript_state", "") or ""),
                "new_transcript_state": str(current.get("transcript_state", "") or ""),
                "old_downstream_decision": str(previous.get("downstream_decision", "") or ""),
                "new_downstream_decision": str(current.get("downstream_decision", "") or ""),
                "old_english_view_decision": str(previous.get("english_view_decision", "") or ""),
                "new_english_view_decision": str(current.get("english_view_decision", "") or ""),
                "old_fixture_alignment_status": str(previous.get("fixture_alignment_status", "") or ""),
                "new_fixture_alignment_status": str(current.get("fixture_alignment_status", "") or ""),
            }
        )

    for case_id in new_case_ids:
        change_summaries.append(
            {
                "case_id": case_id,
                "change_classification": "new_case",
                "change_reasons": ["new_case_added"],
            }
        )
    for case_id in removed_case_ids:
        change_summaries.append(
            {
                "case_id": case_id,
                "change_classification": "removed_case",
                "change_reasons": ["case_removed_from_suite"],
            }
        )

    return _json_safe(
        {
            "previous_review_run_id": str(previous_snapshot.get("review_run_id", "") or ""),
            "current_review_run_id": str(current_snapshot.get("review_run_id", "") or ""),
            "improved_case_ids": sorted(set(improved_case_ids)),
            "regressed_case_ids": sorted(set(regressed_case_ids)),
            "unchanged_case_ids": sorted(set(unchanged_case_ids)),
            "review_needed_case_ids": sorted(set(review_needed_case_ids)),
            "new_case_ids": new_case_ids,
            "removed_case_ids": removed_case_ids,
            "change_summaries": sorted(change_summaries, key=lambda item: str(item.get("case_id", "") or "")),
        }
    )


def summarize_real_audio_review_history_comparison(comparison: Dict[str, Any]) -> Dict[str, Any]:
    improved_case_ids = list(comparison.get("improved_case_ids", []) or [])
    regressed_case_ids = list(comparison.get("regressed_case_ids", []) or [])
    review_needed_case_ids = list(comparison.get("review_needed_case_ids", []) or [])
    new_case_ids = list(comparison.get("new_case_ids", []) or [])
    removed_case_ids = list(comparison.get("removed_case_ids", []) or [])
    change_summaries = list(comparison.get("change_summaries", []) or [])

    recommended_case_ids_for_fixture_update = [
        str(item.get("case_id", "") or "")
        for item in change_summaries
        if "fixture_alignment_changed" in list(item.get("change_reasons", []) or [])
        or "new_mismatch_appeared" in list(item.get("change_reasons", []) or [])
    ]
    recommended_case_ids_for_asr_investigation = [
        str(item.get("case_id", "") or "")
        for item in change_summaries
        if "dominant_failure_reason_changed" in list(item.get("change_reasons", []) or [])
        or str(item.get("new_transcript_state", "") or "") in {"degraded", "failed"}
    ]
    recommended_case_ids_for_manual_audio_review = sorted(
        set(regressed_case_ids) | set(review_needed_case_ids) | set(recommended_case_ids_for_fixture_update)
    )

    if regressed_case_ids:
        recommended_next_action = "investigate_regressions_before_accepting_real_audio_changes"
    elif review_needed_case_ids or new_case_ids or removed_case_ids:
        recommended_next_action = "manually_review_divergent_or_changed_real_audio_cases"
    elif improved_case_ids:
        recommended_next_action = "keep_positive_run_as_reference_and_continue_monitoring"
    else:
        recommended_next_action = "store_snapshot_as_stable_reference"

    strongest_improvement_signal = "mismatch_resolved_or_state_improved" if improved_case_ids else ""
    strongest_regression_signal = "safety_or_fixture_regression_detected" if regressed_case_ids else (
        "fixture_divergence_requires_review" if review_needed_case_ids else ""
    )

    return _json_safe(
        {
            "total_cases_compared": len(improved_case_ids) + len(regressed_case_ids) + len(comparison.get("unchanged_case_ids", []) or []) + len(review_needed_case_ids),
            "total_improved_cases": len(improved_case_ids),
            "total_regressed_cases": len(regressed_case_ids),
            "total_review_needed_cases": len(review_needed_case_ids),
            "strongest_improvement_signal": strongest_improvement_signal,
            "strongest_regression_signal": strongest_regression_signal,
            "recommended_next_action": recommended_next_action,
            "recommended_case_ids_for_manual_audio_review": recommended_case_ids_for_manual_audio_review,
            "recommended_case_ids_for_fixture_update": sorted(set(recommended_case_ids_for_fixture_update)),
            "recommended_case_ids_for_asr_investigation": sorted(set(recommended_case_ids_for_asr_investigation)),
        }
    )


def _compare_single_case_result(base_result: Dict[str, Any], candidate_result: Dict[str, Any]) -> Dict[str, Any]:
    base_eval = base_result.get("evaluation", {}) if isinstance(base_result.get("evaluation", {}), dict) else {}
    candidate_eval = candidate_result.get("evaluation", {}) if isinstance(candidate_result.get("evaluation", {}), dict) else {}
    changes = {}
    for field in (
        "calibration_bucket",
        "downstream_decision",
        "english_view_decision",
        "dominant_safety_reason",
        "recommendation_flags",
    ):
        base_value = base_eval.get(field)
        candidate_value = candidate_eval.get(field)
        if base_value != candidate_value:
            changes[field] = {"base": base_value, "candidate": candidate_value}

    helpful = False
    risky = False
    review_required = False
    new_bucket = str(candidate_eval.get("calibration_bucket", "") or "")
    old_bucket = str(base_eval.get("calibration_bucket", "") or "")
    old_downstream = str(base_eval.get("downstream_decision", "") or "")
    new_downstream = str(candidate_eval.get("downstream_decision", "") or "")
    old_english = str(base_eval.get("english_view_decision", "") or "")
    new_english = str(candidate_eval.get("english_view_decision", "") or "")

    protected_regressions = []
    if old_bucket in {"degraded_low_evidence", "hopeless_wrong_script", "english_contaminated"} and new_downstream == "allowed":
        risky = True
        protected_regressions.append("protected_case_became_downstream_allowed")
    if old_bucket == "degraded_low_evidence" and new_bucket != "degraded_low_evidence":
        risky = True
        protected_regressions.append("low_evidence_bucket_softened")
    if old_bucket == "hopeless_wrong_script" and new_bucket != "hopeless_wrong_script":
        risky = True
        protected_regressions.append("wrong_script_bucket_softened")
    if old_bucket == "english_contaminated" and new_bucket == "degraded_but_useful":
        risky = True
        protected_regressions.append("english_contamination_became_useful")
    if old_english == "blocked" and new_english == "available" and old_bucket in {"degraded_low_evidence", "hopeless_wrong_script", "english_contaminated"}:
        risky = True
        protected_regressions.append("protected_case_english_view_unblocked")
    if old_downstream == "suppressed" and new_downstream == "allowed" and old_bucket == "degraded_but_useful":
        helpful = True
    if old_bucket == "borderline_review" and new_bucket in {"degraded_but_useful", "clearly_cleaned"}:
        helpful = True
    if new_bucket == "borderline_review":
        review_required = True
    if old_bucket == new_bucket and not changes:
        classification = "neutral"
    elif risky:
        classification = "risky"
    elif helpful:
        classification = "helpful"
    else:
        classification = "review_required" if review_required else "neutral"

    return {
        "case_id": str(base_result.get("case_id", "") or candidate_result.get("case_id", "") or ""),
        "label": str(base_result.get("label", "") or candidate_result.get("label", "") or ""),
        "changed": bool(changes),
        "classification": classification,
        "changes": changes,
        "protected_regressions": protected_regressions,
        "decision_trace_excerpt": str(candidate_result.get("decision_trace_excerpt", "") or ""),
    }


def compare_threshold_profiles_on_benchmark(
    cases: list[BenchmarkCase | Dict[str, Any]],
    *,
    base_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
    candidate_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = _coerce_threshold_profile(base_profile)
    candidate = _coerce_threshold_profile(candidate_profile)
    logger.info(
        "[ML_THRESH_PROFILE_COMPARE] base=%s candidate=%s",
        base["profile_name"],
        candidate["profile_name"],
    )
    base_suite = run_multilingual_benchmark_suite(cases, threshold_profile=base)
    candidate_suite = run_multilingual_benchmark_suite(cases, threshold_profile=candidate)
    changed_cases = []
    improved_case_ids = []
    regressed_case_ids = []
    unchanged_case_ids = []
    review_required_case_ids = []
    changed_bucket_cases = []
    changed_downstream_cases = []
    changed_english_view_cases = []
    protected_case_regressions = []

    for base_result, candidate_result in zip(base_suite.get("results", []), candidate_suite.get("results", [])):
        comparison = _compare_single_case_result(base_result, candidate_result)
        if not comparison["changed"]:
            unchanged_case_ids.append(comparison["case_id"])
        else:
            changed_cases.append(comparison)
            if "calibration_bucket" in comparison["changes"]:
                changed_bucket_cases.append(comparison["case_id"])
            if "downstream_decision" in comparison["changes"]:
                changed_downstream_cases.append(comparison["case_id"])
            if "english_view_decision" in comparison["changes"]:
                changed_english_view_cases.append(comparison["case_id"])
            if comparison["classification"] == "helpful":
                improved_case_ids.append(comparison["case_id"])
            elif comparison["classification"] == "risky":
                regressed_case_ids.append(comparison["case_id"])
            else:
                review_required_case_ids.append(comparison["case_id"])
        protected_case_regressions.extend(
            {"case_id": comparison["case_id"], "reason": reason}
            for reason in comparison.get("protected_regressions", [])
        )
        logger.info(
            "[ML_THRESH_DRYRUN_CASE] case_id=%s changed=%s classification=%s changed_fields=%s",
            comparison["case_id"],
            comparison["changed"],
            comparison["classification"],
            ",".join(sorted(comparison["changes"].keys())) or "none",
        )

    risky_threshold_changes = []
    if protected_case_regressions:
        risky_threshold_changes.extend(sorted({item["reason"] for item in protected_case_regressions}))
    safety_regression_flags = []
    if any(item["reason"] in {"protected_case_became_downstream_allowed", "low_evidence_bucket_softened"} for item in protected_case_regressions):
        safety_regression_flags.append("low_evidence_suppression_regressed")
    if any(item["reason"] == "wrong_script_bucket_softened" for item in protected_case_regressions):
        safety_regression_flags.append("wrong_script_rejection_regressed")
    if any(item["reason"] == "english_contamination_became_useful" for item in protected_case_regressions):
        safety_regression_flags.append("english_contamination_regressed")
    if any(item["reason"] == "protected_case_english_view_unblocked" for item in protected_case_regressions):
        safety_regression_flags.append("english_view_safety_regressed")

    recommended_reject_reason = ",".join(safety_regression_flags) if safety_regression_flags else ""
    logger.info(
        "[ML_THRESH_SAFETY_CHECK] candidate=%s protected_regressions=%s flags=%s",
        candidate["profile_name"],
        len(protected_case_regressions),
        ",".join(safety_regression_flags) or "none",
    )
    if safety_regression_flags:
        logger.warning(
            "[ML_THRESH_SAFETY_WARNING] candidate=%s reject_reason=%s",
            candidate["profile_name"],
            recommended_reject_reason,
        )

    summary = {
        "base_profile_name": base["profile_name"],
        "candidate_profile_name": candidate["profile_name"],
        "total_changed_cases": len(changed_cases),
        "total_improved_cases": len(improved_case_ids),
        "total_regressed_cases": len(regressed_case_ids),
        "protected_case_regressions": protected_case_regressions,
        "safety_regression_flags": safety_regression_flags,
        "risky_threshold_changes": risky_threshold_changes,
        "recommended_reject_reason": recommended_reject_reason,
        "best_candidate_wins": improved_case_ids,
        "borderline_case_movements": [
            item["case_id"]
            for item in changed_cases
            if "calibration_bucket" in item["changes"]
            and (
                item["changes"]["calibration_bucket"]["base"] == "borderline_review"
                or item["changes"]["calibration_bucket"]["candidate"] == "borderline_review"
            )
        ],
        "suppression_to_useful_case_ids": [
            item["case_id"]
            for item in changed_cases
            if item["changes"].get("downstream_decision", {}).get("base") == "suppressed"
            and item["changes"].get("downstream_decision", {}).get("candidate") == "allowed"
        ],
        "useful_to_suppressed_case_ids": [
            item["case_id"]
            for item in changed_cases
            if item["changes"].get("downstream_decision", {}).get("base") == "allowed"
            and item["changes"].get("downstream_decision", {}).get("candidate") == "suppressed"
        ],
        "review_required_case_ids": review_required_case_ids,
        "improved_case_ids": improved_case_ids,
        "regressed_case_ids": regressed_case_ids,
        "unchanged_case_ids": unchanged_case_ids,
        "changed_bucket_cases": changed_bucket_cases,
        "changed_downstream_cases": changed_downstream_cases,
        "changed_english_view_cases": changed_english_view_cases,
    }
    logger.info(
        "[ML_THRESH_DRYRUN_SUMMARY] base=%s candidate=%s changed=%s improved=%s regressed=%s",
        base["profile_name"],
        candidate["profile_name"],
        summary["total_changed_cases"],
        summary["total_improved_cases"],
        summary["total_regressed_cases"],
    )
    return {
        "base_profile": base,
        "candidate_profile": candidate,
        "base_suite": base_suite,
        "candidate_suite": candidate_suite,
        "changed_cases": changed_cases,
        "summary": summary,
    }


def summarize_first_candidate_experiment(comparison: Dict[str, Any]) -> Dict[str, Any]:
    summary = comparison.get("summary", {}) if isinstance(comparison.get("summary", {}), dict) else {}
    changed_cases = comparison.get("changed_cases", []) if isinstance(comparison.get("changed_cases", []), list) else []
    base_results = {
        str(row.get("case_id", "") or ""): row.get("evaluation", {})
        for row in (comparison.get("base_suite", {}) or {}).get("results", [])
        if isinstance(row, dict)
    }
    dominant_improvement_reasons: Dict[str, int] = {}
    dominant_regression_reasons: Dict[str, int] = {}
    changed_case_summaries = []
    cleaned_only_promotions = []

    for item in changed_cases:
        classification = str(item.get("classification", "") or "")
        changes = item.get("changes", {}) if isinstance(item.get("changes", {}), dict) else {}
        change_keys = sorted(changes.keys())
        if classification == "helpful":
            for key in change_keys:
                dominant_improvement_reasons[key] = dominant_improvement_reasons.get(key, 0) + 1
        elif classification == "risky":
            for key in change_keys:
                dominant_regression_reasons[key] = dominant_regression_reasons.get(key, 0) + 1
        changed_case_summaries.append(
            {
                "case_id": item.get("case_id", ""),
                "label": item.get("label", ""),
                "classification": classification,
                "changed_fields": change_keys,
                "decision_trace_excerpt": item.get("decision_trace_excerpt", ""),
            }
        )
        bucket_change = changes.get("calibration_bucket", {}) if isinstance(changes.get("calibration_bucket", {}), dict) else {}
        if (
            classification == "helpful"
            and bucket_change.get("base") == "borderline_review"
            and bucket_change.get("candidate") == "clearly_cleaned"
            and str((base_results.get(str(item.get("case_id", "") or ""), {}) or {}).get("evaluation_status", "") or "") == "cleaned"
        ):
            cleaned_only_promotions.append(str(item.get("case_id", "") or ""))

    safety_regressions = list(summary.get("safety_regression_flags", []) or [])
    protected_case_regressions = list(summary.get("protected_case_regressions", []) or [])
    total_improved = int(summary.get("total_improved_cases", 0) or 0)
    total_regressed = int(summary.get("total_regressed_cases", 0) or 0)
    total_changed = int(summary.get("total_changed_cases", 0) or 0)

    has_borderline_movements = bool(summary.get("borderline_case_movements"))

    if safety_regressions:
        experiment_decision = "reject_candidate"
        experiment_decision_reason = ",".join(safety_regressions)
    elif total_improved > 0 and total_regressed == 0 and (
        not has_borderline_movements
        or (
            cleaned_only_promotions
            and len(cleaned_only_promotions) == total_improved
        )
    ):
        experiment_decision = "cautiously_promising"
        experiment_decision_reason = (
            "cleaned_grounded_borderline_cases_promoted_without_safety_regressions"
            if cleaned_only_promotions and len(cleaned_only_promotions) == total_improved
            else "improvements_without_protected_regressions"
        )
    elif total_changed == 0 or total_regressed > 0 or has_borderline_movements:
        experiment_decision = "review_candidate"
        experiment_decision_reason = "borderline_or_ambiguous_movements_require_manual_review"
    else:
        experiment_decision = "review_candidate"
        experiment_decision_reason = "insufficient_signal_for_safe_promotion"

    return {
        "experiment_decision": experiment_decision,
        "experiment_decision_reason": experiment_decision_reason,
        "total_changed_cases": total_changed,
        "total_improved_cases": total_improved,
        "total_regressed_cases": total_regressed,
        "protected_case_regressions": protected_case_regressions,
        "suppression_to_useful_case_ids": list(summary.get("suppression_to_useful_case_ids", []) or []),
        "useful_to_suppressed_case_ids": list(summary.get("useful_to_suppressed_case_ids", []) or []),
        "changed_english_view_cases": list(summary.get("changed_english_view_cases", []) or []),
        "changed_bucket_cases": list(summary.get("changed_bucket_cases", []) or []),
        "borderline_case_movements": list(summary.get("borderline_case_movements", []) or []),
        "dominant_improvement_reasons": dominant_improvement_reasons,
        "dominant_regression_reasons": dominant_regression_reasons,
        "changed_case_summaries": changed_case_summaries,
        "cleaned_only_promotions": cleaned_only_promotions,
    }


def run_first_candidate_threshold_experiment(
    cases: list[BenchmarkCase | Dict[str, Any]],
    *,
    candidate_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
    base_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base = _coerce_threshold_profile(base_profile)
    candidate = _coerce_threshold_profile(candidate_profile) if candidate_profile is not None else build_first_candidate_threshold_profile()
    logger.info(
        "[ML_CANDIDATE_EXPERIMENT] base=%s candidate=%s total_cases=%s",
        base["profile_name"],
        candidate["profile_name"],
        len(cases or []),
    )
    comparison = compare_threshold_profiles_on_benchmark(
        cases,
        base_profile=base,
        candidate_profile=candidate,
    )
    experiment_summary = summarize_first_candidate_experiment(comparison)
    logger.info(
        "[ML_CANDIDATE_DECISION] candidate=%s decision=%s reason=%s changed=%s improved=%s regressed=%s",
        candidate["profile_name"],
        experiment_summary["experiment_decision"],
        experiment_summary["experiment_decision_reason"],
        experiment_summary["total_changed_cases"],
        experiment_summary["total_improved_cases"],
        experiment_summary["total_regressed_cases"],
    )
    return {
        "experiment_version": FIRST_CANDIDATE_EXPERIMENT_VERSION,
        "base_profile": base,
        "candidate_profile": candidate,
        "comparison": comparison,
        "report": experiment_summary,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _build_changed_case_report_entries(experiment_result: Dict[str, Any]) -> list[Dict[str, Any]]:
    comparison = experiment_result.get("comparison", {}) if isinstance(experiment_result.get("comparison", {}), dict) else {}
    base_results = {
        str(row.get("case_id", "") or ""): row
        for row in (comparison.get("base_suite", {}) or {}).get("results", [])
        if isinstance(row, dict)
    }
    candidate_results = {
        str(row.get("case_id", "") or ""): row
        for row in (comparison.get("candidate_suite", {}) or {}).get("results", [])
        if isinstance(row, dict)
    }
    entries = []
    for changed in comparison.get("changed_cases", []) or []:
        case_id = str(changed.get("case_id", "") or "")
        base_eval = ((base_results.get(case_id, {}) or {}).get("evaluation", {}) or {})
        candidate_eval = ((candidate_results.get(case_id, {}) or {}).get("evaluation", {}) or {})
        protected_reasons = set(changed.get("protected_regressions", []) or [])
        classification = str(changed.get("classification", "") or "neutral")
        if protected_reasons:
            change_classification = "protected_regression"
        elif classification == "helpful":
            change_classification = "helpful_change"
        elif classification == "review_required":
            change_classification = "ambiguous_change"
        else:
            change_classification = "neutral_change"
        entries.append(
            {
                "case_id": case_id,
                "label": str(changed.get("label", "") or ""),
                "benchmark_group": str((base_results.get(case_id, {}) or {}).get("benchmark_group", "") or ""),
                "old_bucket": str(base_eval.get("calibration_bucket", "") or ""),
                "new_bucket": str(candidate_eval.get("calibration_bucket", "") or ""),
                "old_downstream_decision": str(base_eval.get("downstream_decision", "") or ""),
                "new_downstream_decision": str(candidate_eval.get("downstream_decision", "") or ""),
                "old_english_view_decision": str(base_eval.get("english_view_decision", "") or ""),
                "new_english_view_decision": str(candidate_eval.get("english_view_decision", "") or ""),
                "change_classification": change_classification,
                "decision_trace_excerpt": str(changed.get("decision_trace_excerpt", "") or ""),
                "requires_manual_review": bool(classification == "review_required" or change_classification == "protected_regression"),
            }
        )
    return entries


def export_first_candidate_experiment_report(experiment_result: Dict[str, Any]) -> Dict[str, Any]:
    comparison = experiment_result.get("comparison", {}) if isinstance(experiment_result.get("comparison", {}), dict) else {}
    report = experiment_result.get("report", {}) if isinstance(experiment_result.get("report", {}), dict) else {}
    base_profile = experiment_result.get("base_profile", {}) if isinstance(experiment_result.get("base_profile", {}), dict) else {}
    candidate_profile = experiment_result.get("candidate_profile", {}) if isinstance(experiment_result.get("candidate_profile", {}), dict) else {}
    changed_case_summaries = _build_changed_case_report_entries(experiment_result)

    protected_regressions = list(report.get("protected_case_regressions", []) or [])
    safety_regression_flags = list((comparison.get("summary", {}) or {}).get("safety_regression_flags", []) or [])
    total_cases = int((((comparison.get("base_suite", {}) or {}).get("summary", {}) or {}).get("total_cases", 0) or 0))
    total_changed_cases = int(report.get("total_changed_cases", 0) or 0)
    total_improved_cases = int(report.get("total_improved_cases", 0) or 0)
    total_regressed_cases = int(report.get("total_regressed_cases", 0) or 0)

    executive_summary = {
        "decision": str(report.get("experiment_decision", "") or ""),
        "reason": str(report.get("experiment_decision_reason", "") or ""),
        "candidate_profile_name": str(candidate_profile.get("profile_name", "") or ""),
        "base_profile_name": str(base_profile.get("profile_name", "") or ""),
        "total_cases": total_cases,
        "total_changed_cases": total_changed_cases,
        "total_improved_cases": total_improved_cases,
        "total_regressed_cases": total_regressed_cases,
    }
    protected_regression_summary = {
        "has_protected_regressions": bool(protected_regressions or safety_regression_flags),
        "safety_regression_flags": safety_regression_flags,
        "protected_case_regressions": protected_regressions,
    }
    improvement_summary = {
        "suppression_to_useful_case_ids": list(report.get("suppression_to_useful_case_ids", []) or []),
        "dominant_improvement_reasons": dict(report.get("dominant_improvement_reasons", {}) or {}),
        "changed_bucket_cases": list(report.get("changed_bucket_cases", []) or []),
    }
    regression_summary = {
        "useful_to_suppressed_case_ids": list(report.get("useful_to_suppressed_case_ids", []) or []),
        "changed_english_view_cases": list(report.get("changed_english_view_cases", []) or []),
        "dominant_regression_reasons": dict(report.get("dominant_regression_reasons", {}) or {}),
    }
    manual_review_recommendations = {
        "borderline_case_movements": list(report.get("borderline_case_movements", []) or []),
        "review_case_ids": [
            entry["case_id"]
            for entry in changed_case_summaries
            if bool(entry.get("requires_manual_review", False))
        ],
    }

    exported = {
        "report_version": EXPERIMENT_REPORT_VERSION,
        "experiment_version": str(experiment_result.get("experiment_version", "") or ""),
        "base_profile": base_profile,
        "candidate_profile": candidate_profile,
        "experiment_decision": str(report.get("experiment_decision", "") or ""),
        "experiment_decision_reason": str(report.get("experiment_decision_reason", "") or ""),
        "total_cases": total_cases,
        "total_changed_cases": total_changed_cases,
        "total_improved_cases": total_improved_cases,
        "total_regressed_cases": total_regressed_cases,
        "protected_case_regressions": protected_regressions,
        "safety_regression_flags": safety_regression_flags,
        "suppression_to_useful_case_ids": list(report.get("suppression_to_useful_case_ids", []) or []),
        "useful_to_suppressed_case_ids": list(report.get("useful_to_suppressed_case_ids", []) or []),
        "changed_bucket_cases": list(report.get("changed_bucket_cases", []) or []),
        "changed_english_view_cases": list(report.get("changed_english_view_cases", []) or []),
        "borderline_case_movements": list(report.get("borderline_case_movements", []) or []),
        "dominant_improvement_reasons": dict(report.get("dominant_improvement_reasons", {}) or {}),
        "dominant_regression_reasons": dict(report.get("dominant_regression_reasons", {}) or {}),
        "cleaned_only_promotions": list(report.get("cleaned_only_promotions", []) or []),
        "changed_case_summaries": changed_case_summaries,
        "executive_summary": executive_summary,
        "protected_regression_summary": protected_regression_summary,
        "improvement_summary": improvement_summary,
        "regression_summary": regression_summary,
        "manual_review_recommendations": manual_review_recommendations,
    }
    return _json_safe(exported)


def _increment_reason_counts(target: Dict[str, int], values: Any) -> None:
    if isinstance(values, dict):
        iterable = values.keys()
    elif isinstance(values, list):
        iterable = values
    else:
        iterable = []
    for value in iterable:
        key = str(value or "").strip()
        if not key:
            continue
        target[key] = target.get(key, 0) + 1


def _prioritize_changed_cases(exported_report: Dict[str, Any]) -> Dict[str, list[str]]:
    priorities = {
        "safety_critical": [],
        "borderline_useful": [],
        "promising_cleaned_promotions": [],
        "translation_sensitive": [],
        "likely_fixture_gap": [],
    }
    cleaned_promotions = set(exported_report.get("cleaned_only_promotions", []) or [])
    changed_english = set(exported_report.get("changed_english_view_cases", []) or [])
    useful_to_suppressed = set(exported_report.get("useful_to_suppressed_case_ids", []) or [])
    suppression_to_useful = set(exported_report.get("suppression_to_useful_case_ids", []) or [])
    borderline = set(exported_report.get("borderline_case_movements", []) or [])

    for entry in exported_report.get("changed_case_summaries", []) or []:
        if not isinstance(entry, dict):
            continue
        case_id = str(entry.get("case_id", "") or "")
        if not case_id:
            continue
        classification = str(entry.get("change_classification", "") or "")
        if classification == "protected_regression":
            priorities["safety_critical"].append(case_id)
            continue
        if case_id in changed_english:
            priorities["translation_sensitive"].append(case_id)
        if case_id in useful_to_suppressed:
            priorities["safety_critical"].append(case_id)
        elif case_id in cleaned_promotions:
            priorities["promising_cleaned_promotions"].append(case_id)
        elif case_id in suppression_to_useful or case_id in borderline:
            priorities["borderline_useful"].append(case_id)
        elif classification == "ambiguous_change":
            priorities["likely_fixture_gap"].append(case_id)

    return priorities


def _build_first_candidate_decision_rationale(exported_report: Dict[str, Any]) -> Dict[str, Any]:
    protected_regressions = list(exported_report.get("protected_case_regressions", []) or [])
    safety_flags = list(exported_report.get("safety_regression_flags", []) or [])
    total_improved = int(exported_report.get("total_improved_cases", 0) or 0)
    total_regressed = int(exported_report.get("total_regressed_cases", 0) or 0)
    total_changed = int(exported_report.get("total_changed_cases", 0) or 0)
    cleaned_only_promotions = list(exported_report.get("cleaned_only_promotions", []) or [])
    borderline_movements = list(exported_report.get("borderline_case_movements", []) or [])
    improvement_reasons = dict(exported_report.get("dominant_improvement_reasons", {}) or {})
    regression_reasons = dict(exported_report.get("dominant_regression_reasons", {}) or {})

    strongest_positive_signal = ""
    if cleaned_only_promotions:
        strongest_positive_signal = "cleaned_grounded_borderline_promotions_without_protected_regressions"
    elif total_improved > 0:
        strongest_positive_signal = "candidate_improves_some_cases_without_direct_protected_failures"

    strongest_negative_signal = ""
    if safety_flags:
        strongest_negative_signal = ",".join(safety_flags)
    elif total_regressed > 0:
        strongest_negative_signal = "candidate_regresses_non_protected_cases"
    elif borderline_movements:
        strongest_negative_signal = "candidate_changes_are_mostly_borderline_and_need_manual_review"

    rationale_bullets = []
    if safety_flags:
        rationale_bullets.append("Protected safety regressions were detected.")
    if total_improved:
        rationale_bullets.append(f"{total_improved} case(s) improved under the candidate profile.")
    if total_regressed:
        rationale_bullets.append(f"{total_regressed} case(s) regressed under the candidate profile.")
    if borderline_movements:
        rationale_bullets.append(f"{len(borderline_movements)} case(s) moved through borderline buckets and need manual review.")
    if cleaned_only_promotions:
        rationale_bullets.append("Helpful changes are concentrated in cleaned grounded borderline promotions.")
    if not rationale_bullets:
        rationale_bullets.append("No meaningful case movement was observed.")

    if safety_flags:
        decision_confidence = "high"
        rationale_summary = "Protected safety regressions outweigh any candidate gains."
    elif total_changed == 0:
        decision_confidence = "high"
        rationale_summary = "The candidate is effectively neutral against the current benchmark pack."
    elif total_improved > 0 and total_regressed == 0 and cleaned_only_promotions:
        decision_confidence = "medium"
        rationale_summary = "The candidate shows narrow cleaned-grounded upside without protected regressions."
    else:
        decision_confidence = "medium" if total_changed > 0 else "low"
        rationale_summary = "The candidate produces mixed or borderline movement and needs manual inspection."

    return {
        "rationale_summary": rationale_summary,
        "rationale_bullets": rationale_bullets,
        "strongest_positive_signal": strongest_positive_signal,
        "strongest_negative_signal": strongest_negative_signal,
        "decision_confidence": decision_confidence,
        "dominant_improvement_reasons": improvement_reasons,
        "dominant_regression_reasons": regression_reasons,
        "protected_regression_count": len(protected_regressions),
    }


def review_first_candidate_experiment(report_or_experiment: Dict[str, Any]) -> Dict[str, Any]:
    if "report_version" in (report_or_experiment or {}):
        exported_report = dict(report_or_experiment)
    else:
        exported_report = export_first_candidate_experiment_report(report_or_experiment)

    protected_regression_present = bool(
        exported_report.get("protected_case_regressions") or exported_report.get("safety_regression_flags")
    )
    total_changed = int(exported_report.get("total_changed_cases", 0) or 0)
    total_improved = int(exported_report.get("total_improved_cases", 0) or 0)
    total_regressed = int(exported_report.get("total_regressed_cases", 0) or 0)
    borderline_movements = list(exported_report.get("borderline_case_movements", []) or [])
    cleaned_only_promotions = list(exported_report.get("cleaned_only_promotions", []) or [])
    changed_english = list(exported_report.get("changed_english_view_cases", []) or [])
    useful_to_suppressed = list(exported_report.get("useful_to_suppressed_case_ids", []) or [])
    priorities = _prioritize_changed_cases(exported_report)
    rationale = _build_first_candidate_decision_rationale(exported_report)

    if protected_regression_present:
        final_recommendation = "reject_and_archive"
        final_recommendation_reason = "protected_safety_regressions_present"
        candidate_status = "rejected"
    elif total_changed == 0:
        final_recommendation = "keep_as_reference_only"
        final_recommendation_reason = "no_material_benchmark_change_detected"
        candidate_status = "neutral_reference"
    elif total_improved > 0 and total_regressed == 0 and cleaned_only_promotions:
        final_recommendation = "consider_narrow_followup_candidate"
        final_recommendation_reason = "narrow_cleaned_grounded_promotions_without_protected_regressions"
        candidate_status = "narrow_promising"
    else:
        final_recommendation = "manual_review_needed"
        final_recommendation_reason = "borderline_or_mixed_case_movements_require_review"
        candidate_status = "manual_review"

    manual_review_required = final_recommendation in {"manual_review_needed", "reject_and_archive"}
    should_try_followup_candidate = final_recommendation == "consider_narrow_followup_candidate"
    should_reject_current_candidate = final_recommendation == "reject_and_archive"
    should_keep_for_reference_only = final_recommendation == "keep_as_reference_only"

    if should_reject_current_candidate:
        recommended_next_action = "archive_current_candidate_and_do_not_relax_protected_thresholds"
        recommended_followup_scope = "none_until_protected_regressions_are_understood"
        recommended_protected_constraints = [
            "do_not_touch_wrong_script_thresholds",
            "do_not_touch_english_contamination_thresholds",
            "do_not_touch_low_evidence_suppression_thresholds",
            "keep_protected_english_view_blocking",
        ]
    elif should_try_followup_candidate:
        recommended_next_action = "try_one_narrow_followup_candidate_after_targeted_audio_review"
        recommended_followup_scope = "cleaned_degraded_borderline_and_rescue_recoverability_only"
        recommended_protected_constraints = [
            "keep_wrong_script_thresholds_fixed",
            "keep_english_contamination_thresholds_fixed",
            "keep_low_evidence_suppression_fixed",
            "keep_english_view_safety_fixed",
        ]
    elif should_keep_for_reference_only:
        recommended_next_action = "archive_report_for_reference_and_make_no_threshold_change"
        recommended_followup_scope = "none"
        recommended_protected_constraints = ["runtime_thresholds_unchanged"]
    else:
        recommended_next_action = "manually_review_changed_cases_before_considering_any_followup"
        recommended_followup_scope = "only_borderline_changed_cases"
        recommended_protected_constraints = [
            "do_not_broaden_threshold_relaxation",
            "avoid_wrong_script_or_contamination_changes",
            "keep_low_evidence_suppression_intact",
        ]

    priority_case_ids_for_manual_review = list(
        dict.fromkeys(
            priorities["safety_critical"]
            + priorities["borderline_useful"]
            + priorities["translation_sensitive"]
            + priorities["likely_fixture_gap"]
        )
    )
    priority_case_ids_for_audio_review = list(
        dict.fromkeys(
            priorities["safety_critical"]
            + priorities["borderline_useful"]
            + priorities["promising_cleaned_promotions"]
        )
    )
    priority_case_ids_for_threshold_focus = list(
        dict.fromkeys(
            priorities["promising_cleaned_promotions"]
            + borderline_movements
            + useful_to_suppressed
            + changed_english
        )
    )

    review = {
        "report_version": str(exported_report.get("report_version", "") or ""),
        "experiment_version": str(exported_report.get("experiment_version", "") or ""),
        "final_recommendation": final_recommendation,
        "final_recommendation_reason": final_recommendation_reason,
        "candidate_status": candidate_status,
        "protected_regression_present": protected_regression_present,
        "manual_review_required": manual_review_required,
        "should_try_followup_candidate": should_try_followup_candidate,
        "should_reject_current_candidate": should_reject_current_candidate,
        "should_keep_for_reference_only": should_keep_for_reference_only,
        "recommended_next_action": recommended_next_action,
        "recommended_followup_scope": recommended_followup_scope,
        "recommended_protected_constraints": recommended_protected_constraints,
        "priority_case_ids_for_manual_review": priority_case_ids_for_manual_review,
        "priority_case_ids_for_audio_review": priority_case_ids_for_audio_review,
        "priority_case_ids_for_threshold_focus": priority_case_ids_for_threshold_focus,
        "priority_groups": priorities,
        **rationale,
    }
    logger.info(
        "[ML_CANDIDATE_REVIEW] recommendation=%s protected=%s changed=%s improved=%s regressed=%s",
        review["final_recommendation"],
        review["protected_regression_present"],
        total_changed,
        total_improved,
        total_regressed,
    )
    return _json_safe(review)


def should_build_second_candidate(review_or_report: Dict[str, Any]) -> Dict[str, Any]:
    if "final_recommendation" in (review_or_report or {}):
        review = dict(review_or_report)
    else:
        review = review_first_candidate_experiment(review_or_report)

    source_recommendation = str(review.get("final_recommendation", "") or "")
    protected_regression_present = bool(review.get("protected_regression_present", False))
    required_followup_scope = str(review.get("recommended_followup_scope", "") or "")

    if source_recommendation == "reject_and_archive":
        allow_second_candidate = False
        gate_reason = "first_candidate_rejected_due_to_protected_regressions"
    elif source_recommendation == "keep_as_reference_only":
        allow_second_candidate = False
        gate_reason = "first_candidate_archived_as_reference_only"
    elif source_recommendation == "manual_review_needed":
        allow_second_candidate = False
        gate_reason = "first_candidate_requires_manual_review_before_any_followup"
    elif source_recommendation == "consider_narrow_followup_candidate":
        allow_second_candidate = True
        gate_reason = "narrow_followup_candidate_allowed_by_first_review"
    else:
        allow_second_candidate = False
        gate_reason = "first_candidate_review_not_actionable_for_followup"

    return _json_safe(
        {
            "allow_second_candidate": allow_second_candidate,
            "gate_reason": gate_reason,
            "source_recommendation": source_recommendation,
            "protected_regression_present": protected_regression_present,
            "required_followup_scope": required_followup_scope,
        }
    )


def build_second_candidate_threshold_profile(review_or_report: Dict[str, Any]) -> Dict[str, Any]:
    review = dict(review_or_report) if "final_recommendation" in (review_or_report or {}) else review_first_candidate_experiment(review_or_report)
    gate = should_build_second_candidate(review)
    if not gate.get("allow_second_candidate", False):
        return _json_safe(
            {
                "second_candidate_profile_name": "",
                "derived_from_first_candidate": False,
                "followup_scope": str(gate.get("required_followup_scope", "") or ""),
                "protected_constraints": list(review.get("recommended_protected_constraints", []) or []),
                "intended_effect": "",
                "blocked": True,
                "blocked_reason": str(gate.get("gate_reason", "") or ""),
                "overridden_thresholds": {},
            }
        )

    priority_threshold_focus = list(review.get("priority_case_ids_for_threshold_focus", []) or [])
    promising_promotions = set(((review.get("priority_groups", {}) or {}).get("promising_cleaned_promotions", []) or []))
    borderline_useful = set(((review.get("priority_groups", {}) or {}).get("borderline_useful", []) or []))

    if promising_promotions:
        overridden_thresholds = {"cleaned_min_trusted_visible_words": 9.0}
        intended_effect = "Probe only cleaned-grounded borderline promotions without touching degraded safety boundaries."
        focus_area = "cleaned_grounded_boundary_only"
    elif borderline_useful:
        overridden_thresholds = {"rescue_recoverability_min_score": 0.24}
        intended_effect = "Probe only borderline rescue recoverability selection without softening protected safety thresholds."
        focus_area = "rescue_recoverability_only"
    else:
        overridden_thresholds = {"rescue_recoverability_min_score": 0.24}
        intended_effect = "Probe one narrow recoverability threshold only."
        focus_area = "minimal_recoverability_probe"

    profile = _coerce_threshold_profile(
        MalayalamThresholdProfile(
            profile_name="candidate_narrow_followup_malayalam_v2",
            base_profile="runtime_default",
            overridden_thresholds=overridden_thresholds,
            notes="Second candidate derived from first-candidate review; narrower than first candidate and preserves all protected constraints.",
        )
    )
    profile.update(
        {
            "second_candidate_profile_name": profile["profile_name"],
            "derived_from_first_candidate": True,
            "followup_scope": focus_area,
            "protected_constraints": list(review.get("recommended_protected_constraints", []) or []),
            "intended_effect": intended_effect,
            "priority_case_ids_for_threshold_focus": priority_threshold_focus,
        }
    )
    logger.info(
        "[ML_SECOND_CANDIDATE_PROFILE] profile=%s scope=%s overrides=%s",
        profile["profile_name"],
        focus_area,
        ",".join(sorted(profile.get("overridden_thresholds", {}).keys())) or "none",
    )
    return _json_safe(profile)


def export_second_candidate_experiment_report(experiment_result: Dict[str, Any]) -> Dict[str, Any]:
    if bool(experiment_result.get("second_candidate_blocked", False)):
        return _json_safe(
            {
                "report_version": SECOND_EXPERIMENT_REPORT_VERSION,
                "experiment_version": SECOND_CANDIDATE_EXPERIMENT_VERSION,
                "second_candidate_attempted": bool(experiment_result.get("second_candidate_attempted", False)),
                "second_candidate_built": bool(experiment_result.get("second_candidate_built", False)),
                "second_candidate_blocked": True,
                "second_candidate_block_reason": str(experiment_result.get("second_candidate_block_reason", "") or ""),
                "gate": dict(experiment_result.get("gate", {}) or {}),
                "base_profile": dict(experiment_result.get("base_profile", {}) or {}),
                "candidate_profile": dict(experiment_result.get("candidate_profile", {}) or {}),
                "experiment_decision": "blocked_no_second_candidate",
                "experiment_decision_reason": str(experiment_result.get("second_candidate_block_reason", "") or ""),
                "changed_case_summaries": [],
            }
        )

    exported = export_first_candidate_experiment_report(experiment_result)
    exported["report_version"] = SECOND_EXPERIMENT_REPORT_VERSION
    exported["experiment_version"] = SECOND_CANDIDATE_EXPERIMENT_VERSION
    exported["second_candidate_attempted"] = bool(experiment_result.get("second_candidate_attempted", False))
    exported["second_candidate_built"] = bool(experiment_result.get("second_candidate_built", False))
    exported["second_candidate_blocked"] = False
    exported["second_candidate_block_reason"] = ""
    exported["gate"] = dict(experiment_result.get("gate", {}) or {})
    return _json_safe(exported)


def review_second_candidate_experiment(report_or_experiment: Dict[str, Any]) -> Dict[str, Any]:
    if "report_version" in (report_or_experiment or {}):
        exported = dict(report_or_experiment)
    else:
        exported = export_second_candidate_experiment_report(report_or_experiment)

    if bool(exported.get("second_candidate_blocked", False)):
        block_reason = str(exported.get("second_candidate_block_reason", "") or "")
        if "rejected" in block_reason or "protected" in block_reason:
            final_recommendation = "reject_and_archive"
        elif "reference" in block_reason:
            final_recommendation = "keep_as_reference_only"
        else:
            final_recommendation = "manual_review_needed"
        return _json_safe(
            {
                "final_recommendation": final_recommendation,
                "final_recommendation_reason": block_reason,
                "candidate_status": "blocked",
                "manual_review_required": final_recommendation == "manual_review_needed",
                "should_keep_for_reference_only": final_recommendation == "keep_as_reference_only",
                "should_reject_current_candidate": final_recommendation == "reject_and_archive",
                "should_try_followup_candidate": False,
            }
        )

    review = review_first_candidate_experiment(exported)
    if review.get("final_recommendation") == "consider_narrow_followup_candidate":
        review["final_recommendation"] = "consider_threshold_adoption_review"
        review["final_recommendation_reason"] = "second_candidate_shows_narrow_promising_movement_without_protected_regressions"
        review["should_try_followup_candidate"] = False
    return _json_safe(review)


def run_second_candidate_threshold_experiment(
    cases: list[BenchmarkCase | Dict[str, Any]],
    *,
    first_candidate_review: Optional[Dict[str, Any]] = None,
    first_candidate_result: Optional[Dict[str, Any]] = None,
    base_profile: Optional[MalayalamThresholdProfile | Dict[str, Any]] = None,
) -> Dict[str, Any]:
    review = (
        dict(first_candidate_review)
        if first_candidate_review is not None and "final_recommendation" in first_candidate_review
        else review_first_candidate_experiment(first_candidate_review or first_candidate_result or {})
    )
    gate = should_build_second_candidate(review)
    base = _coerce_threshold_profile(base_profile)

    if not gate.get("allow_second_candidate", False):
        return _json_safe(
            {
                "experiment_version": SECOND_CANDIDATE_EXPERIMENT_VERSION,
                "second_candidate_attempted": True,
                "second_candidate_built": False,
                "second_candidate_blocked": True,
                "second_candidate_block_reason": str(gate.get("gate_reason", "") or ""),
                "gate": gate,
                "base_profile": base,
                "candidate_profile": {},
                "comparison": {},
                "report": {},
            }
        )

    candidate = build_second_candidate_threshold_profile(review)
    comparison = compare_threshold_profiles_on_benchmark(
        cases,
        base_profile=base,
        candidate_profile=candidate,
    )
    experiment_summary = summarize_first_candidate_experiment(comparison)
    return _json_safe(
        {
            "experiment_version": SECOND_CANDIDATE_EXPERIMENT_VERSION,
            "second_candidate_attempted": True,
            "second_candidate_built": True,
            "second_candidate_blocked": False,
            "second_candidate_block_reason": "",
            "gate": gate,
            "base_profile": base,
            "candidate_profile": candidate,
            "comparison": comparison,
            "report": experiment_summary,
        }
    )


def _summarize_candidate_for_cycle(review: Dict[str, Any], experiment_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    experiment_result = experiment_result or {}
    protected = bool(review.get("protected_regression_present", False) or review.get("should_reject_current_candidate", False))
    changed = 0
    improved = 0
    regressed = 0
    if "report" in experiment_result and isinstance(experiment_result.get("report"), dict):
        changed = int(experiment_result["report"].get("total_changed_cases", 0) or 0)
        improved = int(experiment_result["report"].get("total_improved_cases", 0) or 0)
        regressed = int(experiment_result["report"].get("total_regressed_cases", 0) or 0)
    else:
        changed = int(review.get("changed_cases", review.get("total_changed_cases", 0)) or 0)
        improved = int(review.get("improved_cases", review.get("total_improved_cases", 0)) or 0)
        regressed = int(review.get("regressed_cases", review.get("total_regressed_cases", 0)) or 0)
    return {
        "final_recommendation": str(review.get("final_recommendation", "") or ""),
        "candidate_status": str(review.get("candidate_status", "") or ""),
        "protected_regression_present": protected,
        "manual_review_required": bool(review.get("manual_review_required", False)),
        "changed_cases": changed,
        "improved_cases": improved,
        "regressed_cases": regressed,
        "priority_case_ids_for_audio_review": list(review.get("priority_case_ids_for_audio_review", []) or []),
        "priority_case_ids_for_threshold_focus": list(review.get("priority_case_ids_for_threshold_focus", []) or []),
        "strongest_positive_signal": str(review.get("strongest_positive_signal", "") or ""),
        "strongest_negative_signal": str(review.get("strongest_negative_signal", "") or ""),
    }


def conclude_malayalam_calibration_cycle(
    *,
    first_candidate_result: Optional[Dict[str, Any]] = None,
    first_candidate_review: Optional[Dict[str, Any]] = None,
    second_candidate_result: Optional[Dict[str, Any]] = None,
    second_candidate_review: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    first_review = (
        dict(first_candidate_review)
        if first_candidate_review is not None and "final_recommendation" in first_candidate_review
        else review_first_candidate_experiment(first_candidate_review or first_candidate_result or {})
    )
    second_result = dict(second_candidate_result or {})
    if second_candidate_review is not None and "final_recommendation" in second_candidate_review:
        second_review = dict(second_candidate_review)
    elif second_result:
        second_review = review_second_candidate_experiment(second_result)
    else:
        second_review = {}

    first_summary = _summarize_candidate_for_cycle(first_review, first_candidate_result)
    second_summary = _summarize_candidate_for_cycle(second_review, second_result) if second_review else {}
    second_blocked = bool(second_result.get("second_candidate_blocked", False))
    second_noop = second_blocked or not bool(second_result.get("second_candidate_built", False))

    first_reco = str(first_review.get("final_recommendation", "") or "")
    second_reco = str(second_review.get("final_recommendation", "") or "") if second_review else ""
    protected_regression_present = bool(
        first_summary.get("protected_regression_present", False)
        or second_summary.get("protected_regression_present", False)
    )

    better_candidate = ""
    better_candidate_reason = ""
    if second_summary and second_reco == "consider_threshold_adoption_review":
        better_candidate = "second_candidate"
        better_candidate_reason = "second_candidate_reached_adoption_review_without_protected_regressions"
    elif first_reco == "consider_narrow_followup_candidate":
        better_candidate = "first_candidate"
        better_candidate_reason = "first_candidate_showed_narrow_promising_cleaned_grounded_signal"
    elif second_summary and second_summary.get("improved_cases", 0) > first_summary.get("improved_cases", 0):
        better_candidate = "second_candidate"
        better_candidate_reason = "second_candidate_improved_more_cases"
    elif first_summary.get("improved_cases", 0) > 0:
        better_candidate = "first_candidate"
        better_candidate_reason = "first_candidate_provided_the_only_observed_improvement"

    candidate_comparison_notes = []
    if protected_regression_present:
        candidate_comparison_notes.append("Protected safety regressions remain the dominant cycle risk.")
    if second_noop:
        candidate_comparison_notes.append("The second candidate was blocked or produced no meaningful follow-up signal.")
    if first_reco == "keep_as_reference_only":
        candidate_comparison_notes.append("The first candidate is useful as a reference baseline only.")
    if better_candidate:
        candidate_comparison_notes.append(f"{better_candidate} is the stronger candidate for this cycle.")
    if not candidate_comparison_notes:
        candidate_comparison_notes.append("Neither candidate produced decisive adoption-grade evidence.")

    if protected_regression_present and second_noop:
        calibration_cycle_status = "closed_no_safe_candidate"
        final_cycle_recommendation = "archive_and_stop"
        final_cycle_recommendation_reason = "protected_regressions_and_no_safe_followup_candidate"
        winning_candidate = ""
        winning_candidate_status = "none"
        adoption_review_justified = False
        no_threshold_adoption_recommended = True
        archive_candidates_for_reference = True
        future_calibration_worth_retrying = False
    elif first_reco == "keep_as_reference_only" and (not second_review or second_reco in {"", "manual_review_needed", "keep_as_reference_only"}):
        calibration_cycle_status = "closed_reference_only"
        final_cycle_recommendation = "keep_reference_and_stop"
        final_cycle_recommendation_reason = "candidate_signal_is_neutral_or_reference_only"
        winning_candidate = "first_candidate"
        winning_candidate_status = "reference_only"
        adoption_review_justified = False
        no_threshold_adoption_recommended = True
        archive_candidates_for_reference = True
        future_calibration_worth_retrying = True
    elif second_reco == "consider_threshold_adoption_review":
        calibration_cycle_status = "closed_with_manual_adoption_review"
        final_cycle_recommendation = "manual_adoption_review_only"
        final_cycle_recommendation_reason = "second_candidate_showed_narrow_promising_signal_without_protected_regressions"
        winning_candidate = "second_candidate"
        winning_candidate_status = "manual_adoption_review_only"
        adoption_review_justified = True
        no_threshold_adoption_recommended = False
        archive_candidates_for_reference = False
        future_calibration_worth_retrying = True
    elif first_reco == "consider_narrow_followup_candidate" and second_noop:
        calibration_cycle_status = "closed_without_adoption"
        final_cycle_recommendation = "revisit_after_fixture_expansion"
        final_cycle_recommendation_reason = "first_candidate_showed_narrow_signal_but_second_candidate_did_not_justify_adoption"
        winning_candidate = "first_candidate"
        winning_candidate_status = "promising_but_not_adoptable"
        adoption_review_justified = False
        no_threshold_adoption_recommended = True
        archive_candidates_for_reference = True
        future_calibration_worth_retrying = True
    else:
        calibration_cycle_status = "closed_without_adoption"
        final_cycle_recommendation = "no_threshold_adoption"
        final_cycle_recommendation_reason = "cycle_did_not_produce_strong_safe_candidate_evidence"
        winning_candidate = better_candidate
        winning_candidate_status = "reference_only" if better_candidate else "none"
        adoption_review_justified = False
        no_threshold_adoption_recommended = True
        archive_candidates_for_reference = True
        future_calibration_worth_retrying = True

    safety_critical_cases = list(
        dict.fromkeys(
            list(((first_review.get("priority_groups", {}) or {}).get("safety_critical", []) or []))
            + list(((second_review.get("priority_groups", {}) or {}).get("safety_critical", []) or []))
        )
    )
    promising_cases = list(
        dict.fromkeys(
            list(((first_review.get("priority_groups", {}) or {}).get("promising_cleaned_promotions", []) or []))
            + list(((second_review.get("priority_groups", {}) or {}).get("promising_cleaned_promotions", []) or []))
        )
    )
    ambiguous_cases = list(
        dict.fromkeys(
            list(((first_review.get("priority_groups", {}) or {}).get("borderline_useful", []) or []))
            + list(((second_review.get("priority_groups", {}) or {}).get("borderline_useful", []) or []))
            + list(((first_review.get("priority_groups", {}) or {}).get("translation_sensitive", []) or []))
            + list(((second_review.get("priority_groups", {}) or {}).get("translation_sensitive", []) or []))
        )
    )
    likely_fixture_gap_cases = list(
        dict.fromkeys(
            list(((first_review.get("priority_groups", {}) or {}).get("likely_fixture_gap", []) or []))
            + list(((second_review.get("priority_groups", {}) or {}).get("likely_fixture_gap", []) or []))
        )
    )
    audio_review_priority_cases = list(
        dict.fromkeys(
            list(first_review.get("priority_case_ids_for_audio_review", []) or [])
            + list(second_review.get("priority_case_ids_for_audio_review", []) or [])
        )
    )

    if adoption_review_justified:
        recommended_next_action = "run_manual_adoption_review_with_targeted_audio_checks"
        recommended_future_work_type = "manual_adoption_review"
        recommended_fixture_expansion_focus = []
        recommended_threshold_focus_if_revisited = list(second_review.get("priority_case_ids_for_threshold_focus", []) or [])
        recommended_non_threshold_work = ["validate_candidate_against_real_audio_before_any_adoption"]
    elif protected_regression_present:
        recommended_next_action = "stop_threshold_tuning_and_shift_to_fixture_expansion_and_source_asr_review"
        recommended_future_work_type = "fixture_expansion_and_source_asr_review"
        recommended_fixture_expansion_focus = ["wrong_script_hopeless", "english_contaminated", "low_evidence_malayalam"]
        recommended_threshold_focus_if_revisited = []
        recommended_non_threshold_work = [
            "listen_to_safety_critical_audio_cases",
            "improve_source_asr_quality_before_more_threshold_changes",
        ]
    else:
        recommended_next_action = "revisit_after_fixture_expansion_and_real_audio_review"
        recommended_future_work_type = "fixture_expansion_then_audio_review"
        recommended_fixture_expansion_focus = ["borderline_cleaned_vs_degraded", "recoverable_malayalam_rescue_cases", "mixed_educational_cases"]
        recommended_threshold_focus_if_revisited = list(
            dict.fromkeys(
                list(first_review.get("priority_case_ids_for_threshold_focus", []) or [])
                + list(second_review.get("priority_case_ids_for_threshold_focus", []) or [])
            )
        )
        recommended_non_threshold_work = [
            "review_real_audio_for_borderline_cases",
            "prioritize_source_asr_improvements_over_broader_threshold_relaxation",
        ]

    strongest_cycle_positive_signal = str(
        second_summary.get("strongest_positive_signal", "")
        or first_summary.get("strongest_positive_signal", "")
        or ""
    )
    strongest_cycle_negative_signal = str(
        first_summary.get("strongest_negative_signal", "")
        or second_summary.get("strongest_negative_signal", "")
        or ""
    )

    cycle_rationale_bullets = []
    if protected_regression_present:
        cycle_rationale_bullets.append("Protected safety concerns still dominate this calibration cycle.")
    if better_candidate:
        cycle_rationale_bullets.append(f"{better_candidate} was the strongest candidate examined.")
    if second_noop:
        cycle_rationale_bullets.append("The second candidate did not provide a stronger safe signal than the first.")
    if promising_cases:
        cycle_rationale_bullets.append("Some promising cleaned-grounded promotions were observed, but they remain narrow.")
    if not cycle_rationale_bullets:
        cycle_rationale_bullets.append("No candidate produced enough trustworthy evidence for threshold adoption.")

    if adoption_review_justified:
        calibration_confidence = "medium"
        cycle_rationale_summary = "A narrow manual adoption review is justified, but only after targeted audio verification."
    elif protected_regression_present:
        calibration_confidence = "high"
        cycle_rationale_summary = "This calibration cycle should end without threshold adoption because protected safety risk remains too high."
    elif winning_candidate:
        calibration_confidence = "medium"
        cycle_rationale_summary = "The cycle found some useful signal, but not enough for threshold adoption."
    else:
        calibration_confidence = "high"
        cycle_rationale_summary = "The cycle produced no strong evidence for safe threshold adoption."

    conclusion = {
        "calibration_cycle_status": calibration_cycle_status,
        "final_cycle_recommendation": final_cycle_recommendation,
        "final_cycle_recommendation_reason": final_cycle_recommendation_reason,
        "winning_candidate": winning_candidate,
        "winning_candidate_status": winning_candidate_status,
        "protected_regression_present": protected_regression_present,
        "adoption_review_justified": adoption_review_justified,
        "no_threshold_adoption_recommended": no_threshold_adoption_recommended,
        "archive_candidates_for_reference": archive_candidates_for_reference,
        "future_calibration_worth_retrying": future_calibration_worth_retrying,
        "first_candidate_summary": first_summary,
        "second_candidate_summary": second_summary,
        "better_candidate": better_candidate,
        "better_candidate_reason": better_candidate_reason,
        "candidate_comparison_notes": candidate_comparison_notes,
        "recommended_next_action": recommended_next_action,
        "recommended_future_work_type": recommended_future_work_type,
        "recommended_fixture_expansion_focus": recommended_fixture_expansion_focus,
        "recommended_audio_review_case_ids": audio_review_priority_cases,
        "recommended_threshold_focus_if_revisited": recommended_threshold_focus_if_revisited,
        "recommended_non_threshold_work": recommended_non_threshold_work,
        "cycle_rationale_summary": cycle_rationale_summary,
        "cycle_rationale_bullets": cycle_rationale_bullets,
        "strongest_cycle_positive_signal": strongest_cycle_positive_signal,
        "strongest_cycle_negative_signal": strongest_cycle_negative_signal,
        "calibration_confidence": calibration_confidence,
        "safety_critical_cases": safety_critical_cases,
        "promising_cases": promising_cases,
        "ambiguous_cases": ambiguous_cases,
        "likely_fixture_gap_cases": likely_fixture_gap_cases,
        "audio_review_priority_cases": audio_review_priority_cases,
    }
    logger.info(
        "[ML_CALIBRATION_CYCLE] recommendation=%s winning_candidate=%s protected=%s adoption_review=%s",
        conclusion["final_cycle_recommendation"],
        conclusion["winning_candidate"] or "none",
        conclusion["protected_regression_present"],
        conclusion["adoption_review_justified"],
    )
    return _json_safe(conclusion)
