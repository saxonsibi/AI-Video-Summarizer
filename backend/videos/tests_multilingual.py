import json
import re
import shutil
from io import StringIO
from unittest.mock import patch
from pathlib import Path

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIRequestFactory

from videos.audio_preprocessor import ChunkMetadata
from videos.asr_router import transcribe_video_router, _choose_primary_engine, _assert_malayalam_provider_is_not_groq, _should_attempt_malayalam_second_pass, _apply_bounded_malayalam_post_asr_correction, _apply_bounded_malayalam_faithfulness_recovery, build_malayalam_specialist_candidate, build_malayalam_linguistic_correction_candidate, _get_malayalam_model as router_get_malayalam_model, _get_malayalam_retry_model, _malayalam_second_pass_model, _transcribe_with_malayalam_local, _build_terminal_malayalam_failure_payload, _is_valid_chunk_payload, _analyze_malayalam_asr_payload
from chatbot.rag_engine import ChatbotEngine
from videos.canonical import build_canonical_text
from videos.evaluation import (
    BenchmarkCase,
    MalayalamThresholdProfile,
    RealAudioReviewCase,
    build_default_real_audio_review_cases,
    build_malayalam_real_audio_review_cases,
    build_multilingual_evaluation_result,
    build_default_multilingual_benchmark_cases,
    build_expanded_malayalam_benchmark_cases,
    build_first_candidate_threshold_profile,
    build_second_candidate_threshold_profile,
    classify_malayalam_calibration_bucket,
    compare_real_audio_review_to_fixture,
    compare_real_audio_review_history,
    conclude_malayalam_asr_strategy_default,
    conclude_malayalam_calibration_cycle,
    compare_threshold_profiles_on_benchmark,
    evaluate_benchmark_case,
    export_malayalam_asr_strategy_decision,
    export_real_audio_review_history_snapshot,
    export_real_audio_review_suite_report,
    export_first_candidate_experiment_report,
    export_second_candidate_experiment_report,
    build_real_audio_review_history_entries,
    review_first_candidate_experiment,
    review_real_audio_suite_report,
    review_second_candidate_experiment,
    run_real_audio_review_case,
    run_real_audio_review_suite,
    run_malayalam_asr_strategy_review_case,
    run_malayalam_asr_strategy_review_suite,
    run_multilingual_benchmark_suite,
    run_first_candidate_threshold_experiment,
    run_default_malayalam_asr_strategy_decision_flow,
    run_second_candidate_threshold_experiment,
    summarize_malayalam_asr_strategy_comparison,
    summarize_real_audio_review_history_comparison,
    summarize_real_audio_review_run,
    should_build_second_candidate,
)
from videos.models import HighlightSegment, Summary, Transcript, Video
from videos.serializers import TranscriptSerializer, VideoDetailSerializer, _extract_structured_summary_inputs, _safe_transcript_english_view, _augment_structured_summary_with_english_view, get_or_build_structured_summary, _fidelity_failed_malayalam_summary_payload
from videos.summary_schema import build_structured_summary, default_structured_summary, structured_summary_cache_key, _classify_video_type, SAFE_INTERVIEW_CHAPTER_TITLES
from videos.tasks import _build_transcript_json_payload, _build_malayalam_observability, _compute_transcript_state, _persist_minimal_low_trust_malayalam_checkpoint, _run_audio_pipeline, _rebuild_highlights, _should_suppress_low_trust_malayalam_outputs, _upsert_all_summaries, _prepare_audio_for_pipeline, process_youtube_video_sync, process_video_transcription_sync, process_video_transcription, process_youtube_video
from videos.translation import translate_text
from videos.utils import summarize_text, build_summary_prompt, _get_local_whisper_model, _get_local_whisper_model_with_meta, _WHISPER_MODEL_CACHE, _SUMMARY_PIPELINE_CACHE, _load_hf_summary_pipeline, _transcribe_with_faster_whisper, _transcribe_with_faster_whisper_model, clean_transcript, _ensure_malayalam_ctranslate2_model, _stabilize_summary_faithfulness, _should_accept_malayalam_mixed_script_override, repair_malayalam_degraded_transcript, _garble_debug_snapshot, detect_bad_malayalam_segments, choose_best_malayalam_segment_candidate, classify_malayalam_segment_type, rescue_malayalam_segment_with_local_large_v3, _build_malayalam_rescue_windows, assemble_malayalam_transcript_units, build_malayalam_display_transcript_units, should_skip_malayalam_segment_rescue, should_attempt_malayalam_local_segment_rescue, build_malayalam_groq_prompt, build_malayalam_local_prompt, evaluate_malayalam_linguistic_correction
from videos.utils_metrics import evaluate_transcript_quality
from videos.views import VideoViewSet


class CanonicalPipelineTests(SimpleTestCase):
    @patch("videos.canonical.translate_segments")
    @override_settings(CANONICAL_TRANSLATE_SEGMENTS=True, TRANSLATION_PROVIDER="local")
    def test_malayalam_to_canonical_english(self, mock_translate_segments):
        mock_translate_segments.return_value = [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "Work hard daily.", "original_text": "à´¦à´¿à´µà´¸à´µàµà´‚ à´•à´ à´¿à´¨à´®à´¾à´¯à´¿ à´ªà´ à´¿à´•àµà´•àµ‚."},
            {"id": 1, "start": 5.0, "end": 10.0, "text": "Stay focused for exam.", "original_text": "à´ªà´°àµ€à´•àµà´·à´¯àµà´•àµà´•àµ à´¶àµà´°à´¦àµà´§ à´•àµ‡à´¨àµà´¦àµà´°àµ€à´•à´°à´¿à´•àµà´•àµ‚."},
        ]
        result = build_canonical_text(
            transcript_text="à´¦à´¿à´µà´¸à´µàµà´‚ à´•à´ à´¿à´¨à´®à´¾à´¯à´¿ à´ªà´ à´¿à´•àµà´•àµ‚. à´ªà´°àµ€à´•àµà´·à´¯àµà´•àµà´•àµ à´¶àµà´°à´¦àµà´§ à´•àµ‡à´¨àµà´¦àµà´°àµ€à´•à´°à´¿à´•àµà´•àµ‚.",
            transcript_segments=[
                {"id": 0, "start": 0.0, "end": 5.0, "text": "à´¦à´¿à´µà´¸à´µàµà´‚ à´•à´ à´¿à´¨à´®à´¾à´¯à´¿ à´ªà´ à´¿à´•àµà´•àµ‚."},
                {"id": 1, "start": 5.0, "end": 10.0, "text": "à´ªà´°àµ€à´•àµà´·à´¯àµà´•àµà´•àµ à´¶àµà´°à´¦àµà´§ à´•àµ‡à´¨àµà´¦àµà´°àµ€à´•à´°à´¿à´•àµà´•àµ‚."},
            ],
            transcript_language="ml",
            canonical_language="en",
        )
        self.assertEqual(result["canonical_language"], "en")
        self.assertTrue(result["translation_used"])
        self.assertIn("Work hard daily.", result["canonical_text"])
        self.assertEqual(len(result["canonical_segments"]), 2)


class GapAwareDownstreamTests(TestCase):
    def test_summary_prompt_includes_gap_notice_when_gaps_present(self):
        prompt = build_summary_prompt("Base summary prompt.", has_fidelity_gaps=True)
        self.assertIn("Do not speculate", prompt)
        self.assertIn("could not be recovered faithfully", prompt)

    def test_summary_prompt_unmodified_when_no_gaps(self):
        prompt = build_summary_prompt("Base summary prompt.", has_fidelity_gaps=False)
        self.assertEqual(prompt, "Base summary prompt.")

    def test_chat_context_includes_gap_system_note_when_gaps_present(self):
        engine = ChatbotEngine(video_id="demo-video")
        with patch.object(engine, "_current_transcript_has_fidelity_gaps", return_value=True):
            context = engine._build_context_for_segments([
                {"start": 0.0, "end": 3.0, "text": "ഇത് ഒരു ഭാഗമാണ്"}
            ])
        self.assertIn("fidelity gaps", context)
        self.assertIn("ആ ഭാഗം ട്രാൻസ്ക്രിപ്റ്റിൽ ലഭ്യമല്ല", context)

    def test_chat_context_unmodified_when_no_gaps(self):
        engine = ChatbotEngine(video_id="demo-video")
        with patch.object(engine, "_current_transcript_has_fidelity_gaps", return_value=False):
            context = engine._build_context_for_segments([
                {"start": 0.0, "end": 3.0, "text": "ഇത് ഒരു ഭാഗമാണ്"}
            ])
        self.assertNotIn("fidelity gaps", context)
        self.assertNotIn("ആ ഭാഗം ട്രാൻസ്ക്രിപ്റ്റിൽ ലഭ്യമല്ല", context)
        self.assertIn("ഇത് ഒരു ഭാഗമാണ്", context)

    def test_serializer_returns_partial_status_when_gaps_present(self):
        transcript = Transcript(
            language="ml",
            transcript_language="ml",
            json_data={
                "language": "ml",
                "readable_transcript": "ഇത് ഒരു ഭാഗമാണ്",
                "has_fidelity_gaps": True,
            },
        )
        data = TranscriptSerializer(transcript).data
        self.assertEqual(data["transcript_quality"]["status"], "partial")
        self.assertTrue(data["transcript_quality"]["warning"])
        self.assertTrue(data["transcript_quality"]["warning_ml"])

    def test_serializer_returns_complete_status_when_no_gaps(self):
        transcript = Transcript(
            language="ml",
            transcript_language="ml",
            json_data={"language": "ml", "readable_transcript": "ഇത് പൂര്‍ണ്ണമാണ്"},
        )
        data = TranscriptSerializer(transcript).data
        self.assertEqual(data["transcript_quality"]["status"], "complete")
        self.assertNotIn("warning", data["transcript_quality"])

    def test_full_fidelity_failure_behavior_unchanged(self):
        transcript = Transcript(
            language="ml",
            transcript_language="ml",
            full_text="english leakage",
            transcript_original_text="english leakage",
            transcript_canonical_text="english leakage",
            transcript_canonical_en_text="english leakage",
            json_data={
                "language": "ml",
                "transcript_state": "source_language_fidelity_failed",
                "source_language_fidelity_failed": True,
                "readable_transcript": "english leakage",
            },
        )
        data = TranscriptSerializer(transcript).data
        self.assertEqual(data["transcript_state"], "source_language_fidelity_failed")
        self.assertEqual(data["readable_transcript"], "")
        self.assertEqual(data["transcript_quality"]["status"], "blocked")

    def test_transcript_text_contains_no_gap_placeholders(self):
        transcript = Transcript(
            language="ml",
            transcript_language="ml",
            full_text="ഇത് ഒരു ഭാഗമാണ്",
            json_data={
                "language": "ml",
                "readable_transcript": "ഇത് ഒരു ഭാഗമാണ്",
                "display_readable_transcript": "ഇത് ഒരു ഭാഗമാണ്",
                "evidence_readable_transcript": "ഇത് ഒരു ഭാഗമാണ്",
                "has_fidelity_gaps": True,
            },
        )
        data = TranscriptSerializer(transcript).data
        self.assertNotIn("[gap]", data["readable_transcript"].lower())
        self.assertNotIn("[gap]", json.dumps(data["json_data"], ensure_ascii=False).lower())

    @patch("videos.tasks.summarize_text")
    def test_runtime_summary_path_passes_has_fidelity_gaps_through(self, mock_summarize):
        mock_summarize.return_value = {
            "title": "Summary",
            "content": "സംഗ്രഹം",
            "summary": "സംഗ്രഹം",
            "summary_en": "Summary",
            "summary_out": "സംഗ്രഹം",
            "key_topics": [],
            "summary_language": "ml",
            "summary_source_language": "ml",
            "translation_used": False,
            "model_used": "groq",
            "generation_time": 0.1,
            "summary_model_requested": "demo",
            "summary_model_used": "demo",
            "summary_model_fallback_used": False,
            "summary_generation_mode": "groq",
            "summary_runtime_error": "",
        }
        video = Video.objects.create(title="Gap demo", status="completed")
        transcript = Transcript.objects.create(
            video=video,
            language="ml",
            transcript_language="ml",
            transcript_original_text="ഇത് ഒരു ട്രാൻസ്ക്രിപ്റ്റ് ആണ്",
            full_text="ഇത് ഒരു ട്രാൻസ്ക്രിപ്റ്റ് ആണ്",
            transcript_canonical_text="This is a transcript",
            canonical_language="en",
            json_data={"language": "ml", "has_fidelity_gaps": True},
        )

        _upsert_all_summaries(video, transcript, summary_types=["short"], output_language="ml", source_language="ml")

        self.assertTrue(mock_summarize.called)
        self.assertTrue(mock_summarize.call_args.kwargs["has_fidelity_gaps"])


class TranscriptCleanupV2Tests(SimpleTestCase):
    def test_cleanup_removes_overlay_noise_and_normalizes_entities(self):
        cleaned = clean_transcript([
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Ask AI about this moment lovable dev builds a landingpage"},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "click the bell icon and improve the hero-section CTA"},
        ])
        text = cleaned.get("full_text", "")
        self.assertNotIn("Ask AI about this moment", text)
        self.assertNotIn("click the bell icon", text)
        self.assertIn("Lovable", text)
        self.assertIn("landing page", text)
        self.assertIn("hero section", text)
        self.assertIn("CTA", text)

    def test_transcript_quality_metrics_include_new_fields(self):
        metrics = evaluate_transcript_quality(
            "Ask AI about this moment. Ask AI about this moment. Lovable builds a landing page.",
            [{"text": "Ask AI about this moment"}, {"text": "Lovable builds a landing page"}],
        )
        self.assertIn("repeated_noise_phrase_count", metrics)
        self.assertIn("entity_normalization_count", metrics)
        self.assertIn("very_short_segment_ratio", metrics)
        self.assertIn("final_quality_score", metrics)

    def _old_test_malayalam_repair_transliterates_wrong_script_and_preserves_english(self):
        segment_text = "ਇന്നു ਇന്നു WhatsApp channel pakshe"
        repaired = repair_malayalam_degraded_transcript(
            segment_text,
            [{"start": 0.0, "end": 4.0, "text": segment_text}],
        )
        repaired_text = repaired["text"]
        self.assertIn("ഇ", repaired_text)
        self.assertIn("WhatsApp channel", repaired_text)
        self.assertIn("പക്ഷേ", repaired_text)
        self.assertTrue(repaired["metadata"]["segments_changed"] >= 1)


class MalayalamRepairTests(SimpleTestCase):
    def test_malayalam_repair_preserves_english_and_repairs_tokens(self):
        segment_text = "\u0a07\u0a28\u0a4d\u0a28\u0a41 \u0a07\u0a28\u0a4d\u0a28\u0a41 WhatsApp channel pakshe"
        repaired = repair_malayalam_degraded_transcript(
            segment_text,
            [{"start": 0.0, "end": 4.0, "text": segment_text}],
        )
        repaired_text = repaired["text"]
        self.assertRegex(repaired_text, r"[\u0d00-\u0d7f]")
        self.assertIn("WhatsApp channel", repaired_text)
        self.assertIn("\u0d2a\u0d15\u0d4d\u0d37\u0d47", repaired_text)
        self.assertGreaterEqual(repaired["metadata"]["segments_changed"], 1)

    def test_malayalam_repair_applies_phrase_level_and_token_level_normalization(self):
        segment_text = "പരീക്ഷക്ക വേണ്ടി നമ്മൾ എല്ലാവരും pakshe pakshe markk kittan പഠിക്കണ്ട"
        repaired = repair_malayalam_degraded_transcript(
            segment_text,
            [{"start": 0.0, "end": 6.0, "text": segment_text}],
        )
        meta = repaired["metadata"]
        repaired_text = repaired["text"]
        self.assertGreater(meta["lexical_replacements"], 0)
        self.assertGreaterEqual(meta["phrase_replacements"], 1)
        self.assertGreaterEqual(meta["token_replacements"], 1)
        self.assertIn("പക്ഷേ", repaired_text)
        self.assertIn("പഠിക്കണം", repaired_text)

    def test_malayalam_repair_matches_noisy_phrase_families_from_real_logs(self):
        segment_text = "ഇിദുംദു ഓരോ ഉത്തരോ നേംഗല കലീആകുന മാരകാ വാഇംകീനേംഗീല"
        repaired = repair_malayalam_degraded_transcript(
            segment_text,
            [{"start": 0.0, "end": 6.0, "text": segment_text}],
        )
        meta = repaired["metadata"]
        repaired_text = repaired["text"]
        self.assertGreaterEqual(meta["phrase_replacements"], 1)
        self.assertIn("ഇത് ഓരോ", repaired_text)
        self.assertIn("ഉത്തരം", repaired_text)
        self.assertIn("നിങ്ങൾ", repaired_text)

    def test_malayalam_phrase_bank_replaces_high_confidence_educational_terms_conservatively(self):
        from videos import utils as videos_utils

        segment_text = "കോസ്റ്റിൻ പേപ്പർ അൻസർ ഷീറ്റ് എക്സാം ഹാൾ മോഡൽ എക്സാം ക്ലാസ് ടീച്ചർ"
        repaired_text, meta = videos_utils._apply_malayalam_educational_phrase_bank(segment_text)
        self.assertIn("question paper", repaired_text)
        self.assertIn("answer sheet", repaired_text)
        self.assertIn("exam hall", repaired_text)
        self.assertIn("model exam", repaired_text)
        self.assertIn("class teacher", repaired_text)
        self.assertGreaterEqual(int(meta.get("applied", 0) or 0), 3)

    def test_malayalam_garble_snapshot_discounts_small_latin_and_other_residue_after_repair(self):
        text = (
            "ഇത് ഒരു പരീക്ഷയ്ക്ക് വേണ്ടി തയ്യാറാക്കിയ ക്ലാസാണ് motivational coaching WhatsApp "
            "ഇത് ഒരു പരീക്ഷയ്ക്ക് വേണ്ടി തയ്യാറാക്കിയ ക്ലാസാണ് ©©© "
            "ഇത് ഒരു പരീക്ഷയ്ക്ക് വേണ്ടി തയ്യാറാക്കിയ ക്ലാസാണ്"
        )
        generic = _garble_debug_snapshot(text)
        malayalam = _garble_debug_snapshot(text, language_hint="ml")
        self.assertIn("latin", generic["raw_active_scripts"])
        self.assertLessEqual(malayalam["garbled_score"], generic["garbled_score"])


    def test_malayalam_garble_snapshot_does_not_force_full_replacement_penalty_for_stray_fffd(self):
        text = (
            "ഇത് ഒരു പരീക്ഷയ്ക്ക് വേണ്ടി തയ്യാറാക്കിയ ക്ലാസാണ്. ഇത് ഒരു motivating session ആണ്. "
            "നിങ്ങൾ സ്ഥിരമായി പഠിച്ചാൽ നല്ല മാർക്ക് നേടാം. \ufffd \ufffd"
        )
        snapshot = _garble_debug_snapshot(text, language_hint="ml")
        self.assertGreater(snapshot["replacement_chars"], 0)
        self.assertLess(snapshot["replacement_penalty"], 1.0)
        self.assertNotEqual(snapshot["dominant_script"], "unknown")


    def test_malayalam_repair_tracks_segment_level_adoption_and_malformed_reduction(self):
        segment_text = "രീദീല നേംഗല കലീആകുന പേഡീകീദന"
        repaired = repair_malayalam_degraded_transcript(
            segment_text,
            [{"start": 0.0, "end": 4.0, "text": segment_text}],
        )
        meta = repaired["metadata"]
        self.assertGreaterEqual(meta["repaired_segments"], 1)
        self.assertGreaterEqual(meta["family_replacements"], 1)
        self.assertGreaterEqual(meta["malformed_token_reduction"], 0.0)

    def test_malayalam_repair_preserves_natural_english_insertions(self):
        segment_text = "നിങ്ങൾ confidence നിലനിർത്തണം WhatsApp channel വഴി support കിട്ടും exam hall ലും focus വേണം"
        repaired = repair_malayalam_degraded_transcript(
            segment_text,
            [{"id": 0, "start": 2.0, "end": 8.0, "text": segment_text}],
        )
        self.assertIn("confidence", repaired["text"])
        self.assertIn("WhatsApp channel", repaired["text"])
        self.assertIn("support", repaired["text"])
        meta = repaired["metadata"]
        self.assertGreater(meta["token_class_totals"]["trusted_english"], 0)
        self.assertEqual(meta["timestamp_qc"]["overlap_errors"], 0)

    def test_malayalam_repair_tracks_low_trust_bucket_without_crashing(self):
        segment_text = "à´‡à´¦àµà´¦àµà´‚à´¦àµ confidence support"
        repaired = repair_malayalam_degraded_transcript(
            segment_text,
            [{"id": 0, "start": 0.0, "end": 4.0, "text": segment_text}],
        )
        self.assertIn("low_trust_malayalam", repaired["metadata"]["token_class_totals"])

    def test_malayalam_repair_rejects_less_readable_pseudo_malayalam(self):
        from videos import utils as videos_utils

        original = "നിങ്ങൾ WhatsApp channel follow ചെയ്യണം result നല്ലതായിരിക്കും"
        worse = "നിങ്ങൾ വാട്സ്ആപ്പ് ചാനൽ ഫോളോ ചെയ്യണം result നല്ലതായിരിക്കും"
        accepted, reason = videos_utils._should_use_repaired_malayalam_segment(original, worse)
        self.assertFalse(accepted)
        self.assertEqual(reason, "trusted_english_preservation_regressed")

    def test_malayalam_repair_not_adopted_when_only_script_ratio_improves(self):
        from videos import utils as videos_utils

        original = "ਪੋਰതികരിന്ദു ਕੋਣത് confidence"
        repaired = "പോരതികരിന്ദു കോണത് confidence"
        accepted, reason = videos_utils._should_use_repaired_malayalam_segment(original, repaired)
        self.assertFalse(accepted)
        self.assertEqual(reason, "script_ratio_only_improvement")

    def test_malayalam_repair_preserves_segment_timestamps(self):
        segment_text = "നിങ്ങൾ confidence നിലനിർത്തണം WhatsApp channel വഴി support കിട്ടും"
        repaired = repair_malayalam_degraded_transcript(
            segment_text,
            [
                {"id": 0, "start": 1.25, "end": 4.5, "text": segment_text},
                {"id": 1, "start": 4.5, "end": 7.0, "text": "exam hall focus വേണം"},
            ],
        )
        self.assertEqual(repaired["segments"][0]["start"], 1.25)
        self.assertEqual(repaired["segments"][0]["end"], 4.5)
        self.assertEqual(repaired["segments"][1]["start"], 4.5)
        self.assertEqual(repaired["segments"][1]["end"], 7.0)

    def test_bad_malayalam_segment_detection_flags_only_noisy_segment(self):
        segments = [
            {"id": 0, "start": 0.0, "end": 4.0, "text": "നിങ്ങൾ confidence നന്നായി നിലനിർത്തണം exam hall ലും focus വേണം"},
            {"id": 1, "start": 4.0, "end": 8.0, "text": "ഇിദുംദു വാഇംകീനേംഗീല പേഡീകീദന നേംഗല കലീആകുന"},
        ]
        bad = detect_bad_malayalam_segments(segments)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["idx"], 1)

    def test_stable_english_only_segment_is_not_retried_as_bad_malayalam(self):
        segments = [
            {"id": 0, "start": 0.0, "end": 3.0, "text": "WhatsApp channel support and exam hall strategy"},
        ]
        self.assertEqual(detect_bad_malayalam_segments(segments), [])

    def test_suspicious_english_substitution_segment_is_flagged_for_bounded_rescue(self):
        segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "നിങ്ങൾ പരീക്ഷയ്ക്ക് തയ്യാറാകണം"},
            {"id": 1, "start": 2.0, "end": 5.0, "text": "When you exit the exam hall you will feel satisfied after checking the result"},
            {"id": 2, "start": 5.0, "end": 7.0, "text": "റിസൾട്ട് website ൽ വരും"},
        ]
        bad = detect_bad_malayalam_segments(segments)
        self.assertIn(1, [item["idx"] for item in bad])

    def test_suspicious_span_rescue_does_not_trigger_on_hopeless_junk(self):
        segments = [
            {"id": 0, "start": 0.0, "end": 3.0, "text": "zxqv qqq rrr"},
        ]
        self.assertEqual(detect_bad_malayalam_segments(segments), [])

    def test_rescue_ranking_prefers_recoverable_malayalam_over_hopeless_garbage(self):
        from videos import utils as videos_utils

        recoverable = videos_utils.score_malayalam_rescue_recoverability(
            {"id": 1, "start": 2.8, "end": 5.6, "text": "ഇദ്ദുംദു നിങ്ങൾ result confidence"},
            prev_text="നിങ്ങൾ exam hall ൽ confidence നിലനിറുത്തണം",
            next_text="നല്ല മാർക്ക് നേടാൻ പഠിക്കണം",
        )
        hopeless = videos_utils.score_malayalam_rescue_recoverability(
            {"id": 2, "start": 5.6, "end": 8.4, "text": "Fall of all of your please mute dublin"},
            prev_text="നിങ്ങൾ exam hall ൽ confidence നിലനിറുത്തണം",
            next_text="നല്ല മാർക്ക് നേടാൻ പഠിക്കണം",
        )
        self.assertGreater(float(recoverable.get("score", 0.0) or 0.0), float(hopeless.get("score", 0.0) or 0.0))

    def test_segment_classifier_marks_clean_english_and_corrupted_malayalam_like(self):
        english = classify_malayalam_segment_type("Think about it. People who are not prepared need support.")
        corrupted = classify_malayalam_segment_type("ഇദ്ദുംദു വാഇംകീനേംഗീൽ പേഡീകീദന നേംഗല")
        self.assertEqual(english["type"], "clean_english")
        self.assertEqual(corrupted["type"], "corrupted_malayalam_like")

    def test_hallucinated_english_fragment_in_malayalam_context_is_not_preserved_as_trusted_content(self):
        contaminated = classify_malayalam_segment_type(
            "Fall of all of your please mute dublin",
            prev_text="നിങ്ങൾ പരീക്ഷയ്ക്ക് തയ്യാറാകണം",
            next_text="നല്ല മാർക്ക് നേടാൻ പഠിക്കണം",
        )
        self.assertNotEqual(contaminated["type"], "clean_english")
        self.assertGreaterEqual(float(contaminated.get("contamination_score", 0.0) or 0.0), 0.58)
        self.assertFalse(bool(contaminated.get("english_preserve_allowed", True)))

    def test_genuine_mixed_malayalam_english_sentence_preserves_english_terms(self):
        classified = classify_malayalam_segment_type(
            "നിങ്ങൾ exam hall ൽ confidence നിലനിർത്തണം",
            prev_text="ഇത് പരീക്ഷയ്ക്ക് മുമ്പുള്ള ക്ലാസാണ്",
            next_text="അതുകൊണ്ട് focus വേണം",
        )
        self.assertEqual(classified["type"], "mixed_malayalam_english")
        self.assertTrue(bool(classified.get("english_preserve_allowed", False)))
        self.assertLess(float(classified.get("contamination_score", 1.0) or 1.0), 0.52)

    def test_rescue_adoption_accepts_materially_improved_malayalam_candidate(self):
        from videos import utils as videos_utils

        accepted, reason = videos_utils._should_use_repaired_malayalam_segment(
            "ഇദ്ദുംദു നേംഗല കളിയാകുന result",
            "നിങ്ങൾ confidence ഉണ്ടെങ്കിൽ നല്ല result ലഭിക്കും",
        )
        self.assertTrue(accepted)
        self.assertIn(reason, {"lexical_trust_improved", "readability_improved", "multi_signal_rescue_improved"})

    def test_transliteration_only_repaired_junk_does_not_become_clean_malayalam(self):
        classified = classify_malayalam_segment_type("ഇദ്ദുംദു ഓരോ ഉത്തരോ നേംഗല കലീആകുന പേഡീകീദന")
        self.assertIn(classified["type"], {"corrupted_malayalam_like", "low_trust_malayalam"})
        self.assertLess(float(classified.get("lexical_trust_score", 1.0)), 0.48)

    def test_choose_best_malayalam_candidate_prefers_more_readable_local_rescue(self):
        selected, reason, scored = choose_best_malayalam_segment_candidate({
            "groq": "ഇിദുംദു വാഇംകീനേംഗീല പേഡീകീദന",
            "repaired_groq": "ഇത് വാങ്ങിയെങ്കിൽ",
            "local_rescue": "നിങ്ങൾ ഭയപ്പെടേണ്ട. confidence ഉണ്ടെങ്കിൽ നല്ല result ലഭിക്കും. exam hall ൽ focus വേണം.",
        })
        self.assertIn(selected, {"local_rescue", "repaired_groq", "repaired_baseline"})
        self.assertGreater(scored["local_rescue"]["score"], scored["groq"]["score"])

    def test_quality_local_can_win_when_malayalam_trust_improves(self):
        selected, reason, scored = choose_best_malayalam_segment_candidate({
            "groq": "ഇദ്ദുംദു ഓരോ ഉത്തരോ നേംഗല വാഇംകീനേംഗീൽ",
            "repaired_groq": "ഇത് ഓരോ ഉത്തരം",
            "fast_local": "ഇത് ഓരോ ഉത്തരം നിങ്ങൾ result",
            "quality_local": "നിങ്ങൾ പറയുന്നത് ഇതാണ്. confidence ഉണ്ടെങ്കിൽ നല്ല result ലഭിക്കും.",
        })
        self.assertIn(selected, {"quality_local", "fast_local", "repaired_groq", "repaired_baseline"})
        if selected == "quality_local":
            self.assertEqual(reason, "malayalam_trust_improved")
            self.assertLess(float(scored["quality_local"]["wrong_script_ratio"]), float(scored["groq"]["wrong_script_ratio"]))

    @patch("videos.utils._extract_explicit_audio_window_for_asr", side_effect=["tight.wav"])
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils.os.path.exists", return_value=False)
    def test_corrupted_segment_rescue_runs_fast_and_quality_profiles(
        self,
        _exists,
        mock_pass,
        mock_get_model,
        _extract_window,
    ):
        mock_get_model.return_value = (object(), True, {"configured_model_name": "large-v3"})
        mock_pass.side_effect = [
            {"text": "ഇത് result", "segments": [], "word_timestamps": [], "language": "ml", "language_probability": 0.9},
            {"text": "ഇത് confidence result", "segments": [], "word_timestamps": [], "language": "ml", "language_probability": 0.9},
        ]
        result = rescue_malayalam_segment_with_local_large_v3(
            "audio.wav",
            {"id": 4, "start": 2.0, "end": 5.0, "text": "ഇദ്ദുംദു result"},
        )
        self.assertTrue(result["rescue_available"])
        self.assertEqual(result["fast_text"], "ഇത് result")
        self.assertEqual(result["quality_text"], "ഇത് confidence result")
        self.assertEqual(result["segments"][0]["start"], 2.0)
        self.assertEqual(result["segments"][0]["end"], 5.0)
        self.assertEqual(mock_pass.call_count, 2)

    @patch("videos.utils._extract_explicit_audio_window_for_asr", side_effect=["tight.wav", "medium.wav", "wide.wav"])
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils.os.path.exists", return_value=False)
    def test_wrong_script_local_rescue_candidate_is_rejected_early(
        self,
        _exists,
        mock_pass,
        mock_get_model,
        _extract_window,
    ):
        mock_get_model.return_value = (object(), True, {"configured_model_name": "large-v3"})
        mock_pass.side_effect = [
            {"text": "एक्साम हॉल confidence", "segments": [], "word_timestamps": [], "language": "ml", "language_probability": 0.9},
            {"text": "एक्साम हॉल confidence", "segments": [], "word_timestamps": [], "language": "ml", "language_probability": 0.9},
            {"text": "एक्साम हॉल confidence", "segments": [], "word_timestamps": [], "language": "ml", "language_probability": 0.9},
            {"text": "एक्साम हॉल confidence", "segments": [], "word_timestamps": [], "language": "ml", "language_probability": 0.9},
        ]
        result = rescue_malayalam_segment_with_local_large_v3(
            "audio.wav",
            {"id": 4, "start": 2.0, "end": 5.0, "text": "bad malayalam segment"},
        )
        self.assertFalse(result["rescue_available"])
        self.assertEqual(result.get("quality_text", ""), "")

    @patch("videos.utils._extract_explicit_audio_window_for_asr", side_effect=["tight.wav"])
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils.os.path.exists", return_value=False)
    def test_gurmukhi_like_local_rescue_candidate_is_rejected_early(
        self,
        _exists,
        mock_pass,
        mock_get_model,
        _extract_window,
    ):
        mock_get_model.return_value = (object(), True, {"configured_model_name": "large-v3"})
        mock_pass.side_effect = [
            {"text": "ਇਕਸਾਮ ਹਾਲ ਇਕਸਾਮ ਹਾਲ", "segments": [], "word_timestamps": [], "language": "ml", "language_probability": 0.9},
            {"text": "ਇਕਸਾਮ ਹਾਲ ਇਕਸാമ ਹਾਲ", "segments": [], "word_timestamps": [], "language": "ml", "language_probability": 0.9},
        ]
        result = rescue_malayalam_segment_with_local_large_v3(
            "audio.wav",
            {"id": 9, "start": 3.0, "end": 6.0, "text": "bad malayalam segment"},
        )
        self.assertFalse(result["rescue_available"])
        self.assertEqual(result.get("fast_text", ""), "")
        self.assertEqual(result.get("quality_text", ""), "")

    def test_mixed_malayalam_english_candidate_preserves_english_terms(self):
        selected, _reason, _scored = choose_best_malayalam_segment_candidate({
            "groq": "നിങ്ങൾ exam hall ൽ focus വേണം",
            "repaired_groq": "നിങ്ങൾ പരീക്ഷാഹാൾ ൽ focus വേണം",
            "fast_local": "നിങ്ങൾ exam hall ൽ focus വേണം",
            "quality_local": "നിങ്ങൾ exam hall ൽ confidence ഉം support ഉം വേണം",
        })
        self.assertIn(selected, {"groq", "baseline", "fast_local", "quality_local"})


    @override_settings(
        ASR_MALAYALAM_RESCUE_TIGHT_PAD_SECONDS=0.35,
        ASR_MALAYALAM_RESCUE_MAX_WINDOW_SECONDS=6.0,
    )
    def test_rescue_builds_bounded_tight_window(self):
        windows = _build_malayalam_rescue_windows(
            {"id": 1, "start": 10.0, "end": 12.0, "text": "bad malayalam segment"},
            segments=[
                {"id": 0, "start": 7.0, "end": 10.0, "text": "WhatsApp channel support and result"},
                {"id": 1, "start": 10.0, "end": 12.0, "text": "bad malayalam segment"},
                {"id": 2, "start": 12.0, "end": 14.0, "text": "another bad malayalam segment"},
            ],
        )
        self.assertEqual([w["name"] for w in windows], ["tight"])
        self.assertTrue(all((w["end"] - w["start"]) <= 6.0 for w in windows))
        self.assertGreaterEqual(windows[0]["duration"], 2.0)

    def test_hopeless_malayalam_transcript_skips_segment_rescue(self):
        should_skip, reason = should_skip_malayalam_segment_rescue(
            transcript_trust={
                "dominant_script_final": "other",
                "trusted_malayalam_intended_segments": 0,
                "lexical_trust_score": 0.09,
                "overall_readability": 0.15,
                "wrong_script_segments": 5,
                "corrupted_segments": 5,
                "total_segments": 5,
            },
            quality={"qa_metrics": {"garbled_detector_score": 0.127}},
        )
        self.assertTrue(should_skip)
        self.assertEqual(reason, "hopeless_low_trust_wrong_script_transcript")

    @patch("videos.utils._extract_explicit_audio_window_for_asr", side_effect=["tight.wav", "medium.wav", "wide.wav"])
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils.os.path.exists", return_value=False)
    def test_corrupted_segment_rescue_runs_fast_and_quality_profiles(
        self,
        _exists,
        mock_pass,
        mock_get_model,
        _extract_window,
    ):
        mock_get_model.return_value = (object(), True, {"configured_model_name": "large-v3"})
        mock_pass.side_effect = [
            {"text": "നിങ്ങൾ നല്ലത്", "segments": [], "word_timestamps": [], "language": "ml", "language_probability": 0.9},
            {"text": "നിങ്ങൾ വളരെ നല്ലത്", "segments": [], "word_timestamps": [], "language": "ml", "language_probability": 0.9},
        ]
        result = rescue_malayalam_segment_with_local_large_v3(
            "audio.wav",
            {"id": 4, "start": 2.0, "end": 5.0, "text": "bad segment result"},
            segments=[
                {"id": 3, "start": 0.0, "end": 2.0, "text": "WhatsApp channel support"},
                {"id": 4, "start": 2.0, "end": 5.0, "text": "bad segment result"},
                {"id": 5, "start": 5.0, "end": 7.0, "text": "another bad segment"},
            ],
        )
        self.assertTrue(result["rescue_available"])
        self.assertEqual(result["fast_text"], "നിങ്ങൾ നല്ലത്")
        self.assertEqual(result["quality_text"], "നിങ്ങൾ വളരെ നല്ലത്")
        self.assertEqual(result["medium_quality_text"], "")
        self.assertEqual(result["wide_quality_text"], "")
        self.assertEqual(result["segments"][0]["start"], 2.0)
        self.assertEqual(result["segments"][0]["end"], 5.0)
        self.assertEqual(mock_pass.call_count, 2)

    def test_medium_or_wide_quality_can_beat_tight_candidate_when_trust_improves(self):
        selected, reason, scored = choose_best_malayalam_segment_candidate({
            "groq": "ഇദ്ദുംദു വാഇംകീനേംഗീല result",
            "repaired_groq": "ഇത് weak result",
            "tight_fast": "ഇത് result",
            "tight_quality": {"text": "ഇത് confidence result", "trim": {"overrun_penalty": 0.18, "neighbor_drift_penalty": 0.14, "excessive_length_penalty": 0.08}},
            "medium_quality": {"text": "നിങ്ങൾ confidence ഉണ്ടെങ്കിൽ നല്ല result ലഭിക്കും", "trim": {"overrun_penalty": 0.04, "neighbor_drift_penalty": 0.02, "excessive_length_penalty": 0.0}},
            "wide_quality": {"text": "നിങ്ങൾ confidence ഉണ്ടെങ്കിൽ നല്ല result ലഭിക്കും exam hall focus വേണം", "trim": {"overrun_penalty": 0.05, "neighbor_drift_penalty": 0.03, "excessive_length_penalty": 0.0}},
        })
        self.assertIn(selected, {"medium_quality", "wide_quality"})
        self.assertEqual(reason, "malayalam_trust_improved")
        self.assertLess(scored[selected]["overrun_penalty"], 0.18)

    def test_overlong_wide_window_transcript_is_rejected_if_it_drifts(self):
        selected, _reason, _scored = choose_best_malayalam_segment_candidate({
            "groq": "confidence result",
            "repaired_groq": "confidence result",
            "tight_quality": {"text": "confidence result", "trim": {"overrun_penalty": 0.01, "neighbor_drift_penalty": 0.0, "excessive_length_penalty": 0.0}},
            "wide_quality": {"text": "listen everybody confidence result support exam hall whatsapp channel join now all of you", "trim": {"overrun_penalty": 0.42, "neighbor_drift_penalty": 0.35, "excessive_length_penalty": 0.40}},
        })
        self.assertNotEqual(selected, "wide_quality")

    @override_settings(
        ASR_MALAYALAM_ASSEMBLY_MAX_GAP_SECONDS=1.2,
        ASR_MALAYALAM_ASSEMBLY_MAX_UNIT_SECONDS=12.0,
        ASR_MALAYALAM_ASSEMBLY_MAX_SEGMENTS_PER_UNIT=4,
    )
    def test_short_malayalam_fragments_that_belong_together_are_assembled(self):
        assembled = assemble_malayalam_transcript_units([
            {"id": 0, "start": 0.0, "end": 1.8, "text": "നിങ്ങൾ confidence"},
            {"id": 1, "start": 1.9, "end": 3.4, "text": "ഉണ്ടെങ്കിൽ നല്ല result ലഭിക്കും"},
        ])
        self.assertEqual(len(assembled["units"]), 1)
        self.assertEqual(assembled["units"][0]["source_segment_indices"], [0, 1])
        self.assertIn("confidence", assembled["units"][0]["text"])
        self.assertIn("result", assembled["units"][0]["text"])

    def test_clean_english_only_segments_are_not_merged_unnecessarily(self):
        assembled = assemble_malayalam_transcript_units([
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Think about it."},
            {"id": 1, "start": 2.1, "end": 4.2, "text": "People who are not prepared need support."},
        ])
        self.assertEqual(len(assembled["units"]), 2)

    def test_low_trust_noisy_segment_does_not_contaminate_high_trust_neighbor(self):
        assembled = assemble_malayalam_transcript_units([
            {"id": 0, "start": 0.0, "end": 2.4, "text": "നിങ്ങൾ exam hall ൽ focus വേണം"},
            {"id": 1, "start": 2.5, "end": 3.3, "text": "ഇദ്ദുംദു വാഇംകീനേംഗീല"},
        ])
        self.assertEqual(len(assembled["units"]), 2)
        self.assertEqual(assembled["units"][0]["source_segment_indices"], [0])

    def test_assembled_units_preserve_original_timestamp_traceability(self):
        assembled = assemble_malayalam_transcript_units([
            {"id": 0, "start": 10.0, "end": 11.2, "text": "നിങ്ങൾ confidence"},
            {"id": 1, "start": 11.3, "end": 13.0, "text": "ഉണ്ടെങ്കിൽ നല്ല result ലഭിക്കും"},
        ])
        unit = assembled["units"][0]
        self.assertEqual(unit["display_start"], 10.0)
        self.assertEqual(unit["display_end"], 13.0)
        self.assertEqual(unit["source_ranges"][0]["start"], 10.0)
        self.assertEqual(unit["source_ranges"][1]["end"], 13.0)


    def test_recoverable_internal_evidence_does_not_become_visible_display_transcript(self):
        payload = build_malayalam_display_transcript_units(
            raw_segments=[
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.5,
                    "text": "ഇത് exam hall ൽ confidence നിലനിറുത്തണം",
                    "segment_type": "low_trust_malayalam",
                },
            ],
            assembled_units=[],
            transcript_state="degraded",
        )
        self.assertEqual(payload["units"], [])
        self.assertEqual(payload["readable_transcript"], "")
        self.assertEqual(len(payload.get("internal_evidence_units") or []), 1)
        self.assertTrue(bool((payload.get("internal_evidence_units") or [])[0].get("evidence_only", False)))

    def test_display_refinement_keeps_trusted_malayalam_with_light_polish(self):
        assembled = [{
            "id": 0,
            "start": 0.0,
            "end": 4.0,
            "display_start": 0.0,
            "display_end": 4.0,
            "text": "നിങ്ങൾ   തയ്യാറാകണം  .",
            "unit_readability": 0.72,
            "segment_type": "clean_malayalam",
            "source_segment_indices": [0],
            "source_ranges": [{"id": 0, "start": 0.0, "end": 4.0}],
        }]
        result = build_malayalam_display_transcript_units([], assembled, transcript_state="cleaned")
        self.assertEqual(len(result["units"]), 1)
        self.assertIn("നിങ്ങൾ തയ്യാറാകണം.", result["readable_transcript"])
        self.assertEqual(result["units"][0]["action"], "light_normalize")

    def test_display_refinement_preserves_trusted_mixed_english_terms(self):
        assembled = [{
            "id": 0,
            "start": 0.0,
            "end": 5.0,
            "display_start": 0.0,
            "display_end": 5.0,
            "text": "നിങ്ങൾ confidence exam hall strategy പാലിക്കണം",
            "unit_readability": 0.68,
            "segment_type": "mixed_malayalam_english",
            "source_segment_indices": [0],
            "source_ranges": [{"id": 0, "start": 0.0, "end": 5.0}],
        }]
        result = build_malayalam_display_transcript_units([], assembled, transcript_state="cleaned")
        rendered = result["readable_transcript"].lower()
        self.assertIn("confidence", rendered)
        self.assertIn("exam hall", rendered)
        self.assertIn("strategy", rendered)

    def test_display_refinement_blocks_standalone_english_leakage_for_malayalam(self):
        assembled = [
            {
                "id": 0,
                "start": 0.0,
                "end": 4.0,
                "display_start": 0.0,
                "display_end": 4.0,
                "text": "When you exit the exam hall you will feel satisfied",
                "unit_readability": 0.61,
                "segment_type": "clean_english",
                "source_segment_indices": [0],
                "source_ranges": [{"id": 0, "start": 0.0, "end": 4.0}],
            },
            {
                "id": 1,
                "start": 4.0,
                "end": 8.0,
                "display_start": 4.0,
                "display_end": 8.0,
                "text": "Check the result website and answer key",
                "unit_readability": 0.63,
                "segment_type": "clean_english",
                "source_segment_indices": [1],
                "source_ranges": [{"id": 1, "start": 4.0, "end": 8.0}],
            },
        ]
        result = build_malayalam_display_transcript_units([], assembled, transcript_state="degraded")
        self.assertEqual(result["readable_transcript"], "")
        self.assertEqual(result["units"], [])
        self.assertEqual(result["metadata"]["english_only_units"], 2)
        self.assertEqual(result["metadata"]["malayalam_like_units"], 0)

    def test_display_refinement_suppresses_obvious_low_trust_garbage(self):
        assembled = [{
            "id": 0,
            "start": 0.0,
            "end": 3.0,
            "display_start": 0.0,
            "display_end": 3.0,
            "text": "ഇദ്ദുംദു കളീആകുന പേഡീകീദന",
            "unit_readability": 0.10,
            "segment_type": "corrupted_malayalam_like",
            "source_segment_indices": [0],
            "source_ranges": [{"id": 0, "start": 0.0, "end": 3.0}],
        }]
        result = build_malayalam_display_transcript_units([], assembled, transcript_state="degraded")
        self.assertEqual(result["readable_transcript"], "")
        self.assertEqual(result["metadata"]["suppressed_low_trust_units"], 1)

    def test_display_refinement_keeps_clean_first_pass_accepted_malayalam_visible_when_degraded(self):
        assembled = [{
            "id": 0,
            "start": 0.0,
            "end": 6.1,
            "display_start": 0.0,
            "display_end": 6.1,
            "text": "ഇദു ഒരു മലയാളം ടിക്സ്ടു സ്പീച് ടിസ്ട് ആണ് നമ്മള ഇന്നു ഒരു പുധിയ കാരിയം പഡിക്കുകേ ആണ്.",
            "unit_readability": 0.287,
            "segment_type": "low_trust_malayalam",
            "source_segment_indices": [0],
            "source_ranges": [{"id": 0, "start": 0.0, "end": 6.1}],
        }]
        result = build_malayalam_display_transcript_units(
            [],
            assembled,
            transcript_state="degraded",
            allow_accepted_first_pass_visible=True,
        )
        self.assertTrue(result["readable_transcript"])
        self.assertEqual(len(result["units"]), 1)
        self.assertEqual(result["units"][0]["trace_action"], "accepted_first_pass_visible")
        self.assertGreater(result["metadata"]["final_visible_word_count"], 0)

    def test_display_refinement_does_not_unsuppress_degraded_malayalam_without_first_pass_acceptance(self):
        assembled = [{
            "id": 0,
            "start": 0.0,
            "end": 6.1,
            "display_start": 0.0,
            "display_end": 6.1,
            "text": "ഇദു ഒരു മലയാളം ടിക്സ്ടു സ്പീച് ടിസ്ട് ആണ് നമ്മള ഇന്നു ഒരു പുധിയ കാരിയം പഡിക്കുകേ ആണ്.",
            "unit_readability": 0.287,
            "segment_type": "low_trust_malayalam",
            "source_segment_indices": [0],
            "source_ranges": [{"id": 0, "start": 0.0, "end": 6.1}],
        }]
        result = build_malayalam_display_transcript_units([], assembled, transcript_state="degraded")
        self.assertEqual(result["readable_transcript"], "")
        self.assertEqual(result["units"], [])

    def test_wrong_script_gurmukhi_collapse_never_becomes_visible_malayalam_display(self):
        assembled = [{
            "id": 0,
            "start": 0.0,
            "end": 4.0,
            "display_start": 0.0,
            "display_end": 4.0,
            "text": "ਇਹ ਗਲਤ ਸਕ੍ਰിപ്റ്റ ഔട്ട്പുട്ടാണ് exam result",
            "unit_readability": 0.12,
            "segment_type": "corrupted_malayalam_like",
            "source_segment_indices": [0],
            "source_ranges": [{"id": 0, "start": 0.0, "end": 4.0}],
        }]
        result = build_malayalam_display_transcript_units([], assembled, transcript_state="degraded")
        self.assertEqual(result["readable_transcript"], "")
        self.assertEqual(result["units"], [])

    def test_contaminated_english_heavy_segment_does_not_improve_display_visibility(self):
        assembled = [
            {
                "id": 0,
                "start": 0.0,
                "end": 4.0,
                "display_start": 0.0,
                "display_end": 4.0,
                "text": "Fall of all of your please mute dublin",
                "source_segment_indices": [0],
                "source_ranges": [{"id": 0, "start": 0.0, "end": 4.0}],
            },
            {
                "id": 1,
                "start": 4.0,
                "end": 8.0,
                "display_start": 4.0,
                "display_end": 8.0,
                "text": "à´‡à´¦àµà´¦àµà´‚à´¦àµ à´µà´¾à´‡à´‚à´•àµ€à´¨àµ‡à´‚à´—àµ€à´²",
                "source_segment_indices": [1],
                "source_ranges": [{"id": 1, "start": 4.0, "end": 8.0}],
            },
        ]
        result = build_malayalam_display_transcript_units([], assembled, transcript_state="degraded")
        self.assertEqual(result["readable_transcript"], "")
        self.assertEqual(result["units"], [])

    def test_display_refinement_improves_degraded_readable_transcript_over_raw_assembled(self):
        assembled = [
            {
                "id": 0,
                "start": 0.0,
                "end": 4.0,
                "display_start": 0.0,
                "display_end": 4.0,
                "text": "നിങ്ങൾ confidence confidence exam hall",
                "unit_readability": 0.51,
                "segment_type": "mixed_malayalam_english",
                "source_segment_indices": [0],
                "source_ranges": [{"id": 0, "start": 0.0, "end": 4.0}],
            },
            {
                "id": 1,
                "start": 4.0,
                "end": 6.0,
                "display_start": 4.0,
                "display_end": 6.0,
                "text": "ഇദ്ദുംദു കളീആകുന",
                "unit_readability": 0.08,
                "segment_type": "corrupted_malayalam_like",
                "source_segment_indices": [1],
                "source_ranges": [{"id": 1, "start": 4.0, "end": 6.0}],
            },
        ]
        raw_text = " ".join(unit["text"] for unit in assembled)
        result = build_malayalam_display_transcript_units([], assembled, transcript_state="degraded")
        self.assertLess(len(result["readable_transcript"]), len(raw_text))
        self.assertIn("confidence", result["readable_transcript"].lower())

    def test_display_units_preserve_timestamp_traceability(self):
        assembled = [{
            "id": 0,
            "start": 1.0,
            "end": 3.0,
            "display_start": 1.0,
            "display_end": 3.0,
            "text": "നിങ്ങൾ confidence പാലിക്കണം",
            "unit_readability": 0.61,
            "segment_type": "mixed_malayalam_english",
            "source_segment_indices": [4],
            "source_ranges": [{"id": 4, "start": 1.0, "end": 3.0}],
        }]
        result = build_malayalam_display_transcript_units([], assembled, transcript_state="cleaned")
        unit = result["units"][0]
        self.assertEqual(unit["display_start"], 1.0)
        self.assertEqual(unit["display_end"], 3.0)
        self.assertEqual(unit["source_segment_indices"], [4])
        self.assertEqual(unit["source_ranges"][0]["start"], 1.0)

    def test_cleaned_malayalam_display_remains_close_to_assembled_text(self):
        assembled = [{
            "id": 0,
            "start": 0.0,
            "end": 4.0,
            "display_start": 0.0,
            "display_end": 4.0,
            "text": "നിങ്ങൾ തയ്യാറാകണം.",
            "unit_readability": 0.78,
            "segment_type": "clean_malayalam",
            "source_segment_indices": [0],
            "source_ranges": [{"id": 0, "start": 0.0, "end": 4.0}],
        }]
        result = build_malayalam_display_transcript_units([], assembled, transcript_state="cleaned")
        self.assertEqual(result["readable_transcript"], "നിങ്ങൾ തയ്യാറാകണം.")


class SummaryFaithfulnessTests(SimpleTestCase):
    def test_unfaithful_summary_falls_back_to_grounded_semantic_summary(self):
        transcript = (
            "Christopher Nolan explains that he prefers practical effects because they feel more real on camera. "
            "He says CGI can work, but practical effects often create a stronger result."
        )
        unstable = "The video says James Cameron used purple drones and quantum rendering in this workflow."
        stabilized = _stabilize_summary_faithfulness(unstable, transcript, "short")
        self.assertNotIn("James Cameron", stabilized)
        self.assertTrue(stabilized.strip())
        self.assertNotIn("quantum rendering", stabilized.lower())

    @override_settings(
        ENABLE_ABSTRACTIVE_SUMMARY=True,
        SUMMARIZATION_PROVIDER="hf",
        SUMMARIZATION_MODEL="facebook/bart-large-cnn",
    )
    def test_hf_summary_unknown_task_falls_back_to_extractive_safely(self):
        result = summarize_text(
            "This transcript explains a product workflow, key steps, and the final result in enough detail for testing.",
            summary_type="short",
            source_language="en",
            output_language="en",
        )

        self.assertTrue(result["content"].strip())
        self.assertEqual(result["model_used"], "extractive-hf-task-fallback")


class HybridAsrRouterTests(SimpleTestCase):
    def test_malayalam_groq_prompt_contains_anchor_and_lexicon_bias(self):
        prompt = build_malayalam_groq_prompt()
        self.assertIn("Malayalam script", prompt)
        self.assertIn("Do not translate Malayalam speech into English", prompt)
        self.assertIn("exam", prompt.lower())

    def test_malayalam_groq_request_path_uses_anchor_prompt_and_lexicon_bias(self):
        from videos import utils as videos_utils

        captured = {}

        class _DummyTranscriptions:
            def create(self, file=None, **kwargs):
                captured.update(kwargs)
                return {"text": "പരീക്ഷ ഫലം", "segments": [], "language": "ml", "language_probability": 0.98}

        class _DummyAudio:
            transcriptions = _DummyTranscriptions()

        class _DummyClient:
            audio = _DummyAudio()

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = object()
            payload = videos_utils._transcribe_single_with_groq(_DummyClient(), "demo.wav", "whisper-large-v3", "ml")

        self.assertEqual(captured.get("language"), "ml")
        self.assertIn("prompt", captured)
        self.assertIn("Malayalam script", captured.get("prompt", ""))
        self.assertTrue(payload.get("metadata", {}).get("malayalam_prompt_bias_used"))

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=180.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._transcribe_with_local_whisper")
    @patch("videos.asr_router.transcribe_prerecorded_audio")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_LONGFORM_LOCAL_THRESHOLD_SECONDS=900,
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_short_english_routes_to_groq_whisper(
        self,
        _exists,
        _garbled,
        mock_deepgram,
        mock_local,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_groq.return_value = {
            "text": "This is an English transcript.",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "This is an English transcript."}],
            "language": "en",
            "language_probability": 0.92,
            "metadata": {"asr_provider_used": "groq_whisper"},
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="en")
        self.assertEqual(result.get("language"), "en")
        self.assertEqual(result.get("asr_engine_used"), "groq_whisper")
        self.assertEqual(result.get("metadata", {}).get("asr_route_reason"), "english_fast_cloud")
        self.assertTrue(result.get("metadata", {}).get("transcript_quality_gate_passed"))
        self.assertEqual(result.get("metadata", {}).get("selected_model"), "whisper-large-v3-turbo")
        self.assertIsInstance(result.get("metadata", {}).get("fallback_chain"), list)
        mock_groq.assert_called_once()
        mock_local.assert_not_called()
        mock_deepgram.assert_not_called()

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=3600.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._transcribe_with_local_whisper")
    @patch("videos.asr_router.transcribe_prerecorded_audio")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(ASR_LOW_CONTENT_WPM_MIN=0, ASR_LOW_CONTENT_MIN_WORDS_LONG=1)
    @override_settings(
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_long_english_still_routes_to_groq_whisper(
        self,
        _exists,
        _garbled,
        _mock_deepgram,
        mock_local,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_groq.return_value = {
            "text": "Long-form English transcript routed through Groq first.",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "Long-form English transcript routed through Groq first."}],
            "language": "en",
            "language_probability": 0.90,
            "metadata": {"asr_provider_used": "groq_whisper"},
        }
        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="en")
        self.assertEqual(result.get("asr_engine_used"), "groq_whisper")
        self.assertEqual(result.get("metadata", {}).get("asr_route_reason"), "english_fast_cloud")
        mock_groq.assert_called_once()
        mock_local.assert_not_called()

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=120.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._transcribe_with_local_whisper")
    @patch("videos.asr_router.transcribe_prerecorded_audio")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    @override_settings(
        ASR_MALAYALAM_STRATEGY="quality_first",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_malayalam_routes_to_local_primary_model_first(
        self,
        _exists,
        _garbled,
        _mock_deepgram,
        mock_local,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_local.return_value = {
            "text": "മലയാളം ട്രാൻസ്ക്രിപ്റ്റ് വിശ്വസനീയമായി ലഭിച്ചു.",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "മലയാളം ട്രാൻസ്ക്രിപ്റ്റ് വിശ്വസനീയമായി ലഭിച്ചു."}],
            "language": "ml",
            "language_probability": 0.91,
            "metadata": {"asr_provider_used": "faster_whisper", "actual_local_model_name": "large-v2"},
        }
        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")
        self.assertEqual(result.get("asr_engine_used"), "whisper_local")
        self.assertEqual(result.get("metadata", {}).get("asr_route_reason"), "malayalam_quality_primary")
        self.assertEqual(result.get("metadata", {}).get("selected_model"), "large-v2")
        self.assertEqual(result.get("metadata", {}).get("fallback_chain"), ["whisper_local"])
        mock_local.assert_called_once_with("tmp.wav", "youtube", "ml")
        mock_groq.assert_not_called()

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=120.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._transcribe_with_malayalam_local")
    @patch("videos.asr_router._transcribe_with_local_whisper")
    @patch("videos.asr_router.transcribe_prerecorded_audio")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        ASR_USE_DEEPGRAM_FOR_NON_ENGLISH=True,
        DEEPGRAM_API_KEY="dg",
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    @override_settings(
        ASR_MALAYALAM_STRATEGY="quality_first",
        ASR_USE_GROQ_FALLBACK=True,
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_malayalam_does_not_fall_back_to_groq_when_local_path_fails(
        self,
        _exists,
        _garbled,
        mock_deepgram,
        mock_local,
        mock_ml_local,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_local.side_effect = RuntimeError("Local Malayalam model failed")
        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")

        mock_local.assert_called_once_with("tmp.wav", "youtube", "ml")
        mock_ml_local.assert_not_called()
        mock_groq.assert_not_called()
        mock_deepgram.assert_not_called()
        self.assertEqual(result.get("language"), "ml")
        self.assertEqual(result.get("text"), "")
        self.assertTrue(result.get("metadata", {}).get("source_language_fidelity_failed"))
        self.assertEqual(
            result.get("metadata", {}).get("transcript_fidelity_state"),
            "source_language_fidelity_failed",
        )
        self.assertTrue(result.get("metadata", {}).get("terminal_malayalam_failure"))
        self.assertEqual(
            result.get("metadata", {}).get("terminal_malayalam_failure_reason"),
            "Local Malayalam model failed",
        )

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=120.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._transcribe_with_local_whisper")
    @patch("videos.asr_router.transcribe_prerecorded_audio")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        ASR_ENABLE_DEEPGRAM_MALAYALAM_EXPERIMENT=True,
        ASR_USE_DEEPGRAM_FOR_NON_ENGLISH=True,
        DEEPGRAM_API_KEY="dg",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    @override_settings(
        ASR_MALAYALAM_STRATEGY="quality_first",
        ASR_ENABLE_DEEPGRAM_MALAYALAM_EXPERIMENT=True,
        ASR_USE_DEEPGRAM_FOR_NON_ENGLISH=True,
        DEEPGRAM_API_KEY="dg",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_malayalam_does_not_use_deepgram_even_when_experiment_flag_is_enabled(
        self,
        _exists,
        _garbled,
        mock_deepgram,
        mock_local,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_local.return_value = {
            "text": "മലയാളം ഉള്ളടക്കം സുരക്ഷിതമായി ലഭിച്ചു.",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "മലയാളം ഉള്ളടക്കം സുരക്ഷിതമായി ലഭിച്ചു."}],
            "language": "ml",
            "language_probability": 0.91,
            "metadata": {"asr_provider_used": "faster_whisper", "actual_local_model_name": "large-v3"},
        }
        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")
        self.assertEqual(result.get("asr_engine_used"), "whisper_local")
        self.assertEqual(result.get("metadata", {}).get("asr_route_reason"), "malayalam_quality_primary")
        mock_deepgram.assert_not_called()
        mock_local.assert_called_once_with("tmp.wav", "youtube", "ml")
        mock_groq.assert_not_called()

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=60.0)
    @patch("videos.asr_router._transcribe_with_local_whisper")
    @patch("videos.asr_router.transcribe_prerecorded_audio")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(ASR_USE_DEEPGRAM_FOR_NON_ENGLISH=True, DEEPGRAM_API_KEY="dg")
    def test_hindi_routes_to_deepgram_when_supported(
        self,
        _exists,
        _garbled,
        mock_deepgram,
        mock_local,
        _duration,
        _preprocess,
    ):
        mock_deepgram.return_value = {
            "text": "Hindi transcript from Deepgram with enough words for quality.",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "Hindi transcript from Deepgram with enough words for quality."}],
            "language": "hi",
            "confidence": 0.88,
            "metadata": {"asr_provider_used": "deepgram"},
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="hi")
        self.assertEqual(result.get("language"), "hi")
        self.assertEqual(result.get("asr_engine_used"), "deepgram")
        self.assertEqual(result.get("metadata", {}).get("asr_route_reason"), "deepgram_supported_language")
        self.assertEqual(result.get("metadata", {}).get("provider_name"), "deepgram")
        mock_deepgram.assert_called_once()
        mock_local.assert_not_called()

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=60.0)
    @patch("videos.asr_router._transcribe_with_local_whisper")
    @patch("videos.asr_router.transcribe_prerecorded_audio", side_effect=RuntimeError("Deepgram timeout"))
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(ASR_USE_DEEPGRAM_FOR_NON_ENGLISH=True, DEEPGRAM_API_KEY="dg")
    def test_hindi_falls_back_to_local_if_deepgram_fails(
        self,
        _exists,
        _garbled,
        _mock_deepgram,
        mock_local,
        _duration,
        _preprocess,
    ):
        mock_local.return_value = {
            "text": "Hindi local fallback transcript with enough words to pass quality gate.",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "Hindi local fallback transcript with enough words to pass quality gate."}],
            "language": "hi",
            "language_probability": 0.88,
            "metadata": {"asr_provider_used": "faster_whisper", "model_reused": True},
        }
        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="hi")
        self.assertEqual(result.get("asr_engine_used"), "whisper_local")
        self.assertEqual(result.get("metadata", {}).get("asr_route_reason"), "deepgram_fallback_to_local")
        self.assertTrue(result.get("metadata", {}).get("fallback_triggered"))
        mock_local.assert_called_once()

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=180.0)
    @patch("videos.asr_router._transcribe_with_local_whisper")
    @patch("videos.asr_router.transcribe_prerecorded_audio")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        ASR_USE_DEEPGRAM_FOR_NON_ENGLISH=True,
        DEEPGRAM_API_KEY="dg",
        ASR_USE_GROQ_FALLBACK=False,
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=5,
    )
    def test_hindi_falls_back_to_local_if_deepgram_quality_gate_fails(
        self,
        _exists,
        _garbled,
        mock_deepgram,
        mock_local,
        _duration,
        _preprocess,
    ):
        mock_deepgram.return_value = {
            "text": "too short",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "too short"}],
            "language": "hi",
            "confidence": 0.91,
            "metadata": {"asr_provider_used": "deepgram"},
        }
        mock_local.return_value = {
            "text": "Hindi local quality fallback transcript with enough words to pass the duration gate cleanly.",
            "segments": [{"id": 0, "start": 0.0, "end": 6.0, "text": "Hindi local quality fallback transcript with enough words to pass the duration gate cleanly."}],
            "language": "hi",
            "language_probability": 0.9,
            "metadata": {"asr_provider_used": "faster_whisper", "model_reused": True},
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="hi")
        self.assertEqual(result.get("asr_engine_used"), "whisper_local")
        self.assertEqual(result.get("metadata", {}).get("asr_route_reason"), "deepgram_fallback_to_local")
        self.assertEqual(result.get("metadata", {}).get("fallback_reason"), "deepgram_low_content")
        self.assertTrue(result.get("metadata", {}).get("transcript_quality_gate_passed"))

    @override_settings(
        ASR_REJECT_ON_GARBLE=True,
        ASR_MAX_RETRIES=2,
        ASR_USE_DEEPGRAM_FOR_NON_ENGLISH=True,
        DEEPGRAM_API_KEY="dg",
        ASR_USE_GROQ_FALLBACK=False,
    )
    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=60.0)
    @patch("videos.asr_router.transcribe_prerecorded_audio")
    @patch("videos.asr_router._is_garbled", return_value=True)
    @patch("videos.asr_router.candidate_languages_for_script", return_value=["hi", "ta", "te"])
    @patch("videos.asr_router.os.path.exists", return_value=False)
    def test_garbled_rejected_when_flag_enabled(
        self,
        _exists,
        _candidates,
        _garbled,
        mock_deepgram,
        _duration,
        _preprocess,
    ):
        mock_deepgram.return_value = {
            "text": "garbled text",
            "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "garbled text"}],
            "language": "hi",
            "confidence": 0.8,
            "metadata": {"asr_provider_used": "deepgram"},
        }

        with self.assertRaisesRegex(ValueError, "remained garbled"):
            transcribe_video_router("demo.wav", source_type="youtube", requested_language="hi")

    @override_settings(
        ASR_REJECT_ON_GARBLE=False,
        ASR_MAX_RETRIES=1,
        ASR_USE_DEEPGRAM_FOR_NON_ENGLISH=True,
        DEEPGRAM_API_KEY="dg",
        ASR_USE_GROQ_FALLBACK=False,
    )
    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=60.0)
    @patch("videos.asr_router.transcribe_prerecorded_audio")
    @patch("videos.asr_router._is_garbled", return_value=True)
    @patch("videos.asr_router.candidate_languages_for_script", return_value=["hi"])
    @patch("videos.asr_router.os.path.exists", return_value=False)
    def test_garbled_allowed_when_flag_disabled(
        self,
        _exists,
        _candidates,
        _garbled,
        mock_deepgram,
        _duration,
        _preprocess,
    ):
        mock_deepgram.return_value = {
            "text": "garbled but kept transcript with enough words to avoid the low content gate for this test.",
            "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "garbled but kept transcript with enough words to avoid the low content gate for this test."}],
            "language": "hi",
            "confidence": 0.8,
            "metadata": {"asr_provider_used": "deepgram"},
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="hi")
        self.assertEqual(result.get("asr_engine_used"), "deepgram")
        self.assertEqual(result.get("language"), "hi")

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=90.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._run_malayalam_local_model")
    @patch("videos.asr_router._should_attempt_malayalam_second_pass")
    @patch("videos.asr_router._analyze_malayalam_asr_payload")
    @patch("videos.asr_router._is_malayalam_second_pass_better", return_value=True)
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_MALAYALAM_STRATEGY="hybrid_retry",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_recoverable_weak_malayalam_can_trigger_bounded_second_pass_asr(
        self,
        _exists,
        _garbled,
        _better,
        mock_analyze,
        mock_second_pass_select,
        mock_retry_model,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_groq.return_value = {
            "text": "പാഠം question answer model",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "പാഠം question answer model"}],
            "language": "ml",
            "language_probability": 0.81,
            "metadata": {"asr_provider_used": "groq_whisper"},
        }
        mock_second_pass_select.return_value = {
            "attempt_second_pass": True,
            "reason": "bounded_second_pass_candidate",
            "blocked_reason": "",
            "analysis": {
                "lexical_trust": 0.26,
                "readability": 0.24,
                "wrong_script_burden": 0.18,
                "contamination_burden": 0.42,
                "trusted_visible_word_count": 3,
                "trusted_display_unit_count": 1,
            },
        }
        mock_retry_model.return_value = {
            "text": "കേരള പഠന ചോദ്യത്തിനുള്ള ശരിയായ ഉത്തരം ഇവിടെ വ്യക്തമായി പറയുന്നു",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "കേരള പഠന ചോദ്യത്തിനുള്ള ശരിയായ ഉത്തരം ഇവിടെ വ്യക്തമായി പറയുന്നു"}],
            "language": "ml",
            "language_probability": 0.93,
            "metadata": {"asr_provider_used": "faster_whisper", "actual_local_model_name": "large-v3"},
        }
        mock_analyze.return_value = {
            "quality_class": "clearly_good",
            "lexical_trust": 0.53,
            "readability": 0.48,
            "wrong_script_burden": 0.04,
            "contamination_burden": 0.14,
            "trusted_visible_word_count": 9,
            "trusted_display_unit_count": 1,
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")
        self.assertEqual(result.get("asr_engine_used"), "whisper_local")
        self.assertTrue(result.get("metadata", {}).get("second_pass_asr_attempted"))
        self.assertTrue(result.get("metadata", {}).get("second_pass_asr_improved"))
        mock_retry_model.assert_called_once()

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=90.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._run_malayalam_local_model")
    @patch("videos.asr_router._should_attempt_malayalam_second_pass")
    @patch("videos.asr_router._analyze_malayalam_asr_payload")
    @patch("videos.asr_router._is_malayalam_second_pass_better", return_value=True)
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_MALAYALAM_STRATEGY="hybrid_retry",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_confusion_retry_execution_is_visible_in_metadata(
        self,
        _exists,
        _garbled,
        _better,
        mock_analyze,
        mock_second_pass_select,
        mock_retry_model,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_groq.return_value = {
            "text": "corrupted mixed script transcript",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "corrupted mixed script transcript"}],
            "language": "ml",
            "language_probability": 0.93,
            "metadata": {"asr_provider_used": "groq_whisper"},
        }
        mock_second_pass_select.return_value = {
            "attempt_second_pass": True,
            "reason": "malayalam_script_confusion_retry_candidate",
            "blocked_reason": "",
            "analysis": {
                "lexical_trust": 0.057,
                "readability": 0.144,
                "wrong_script_burden": 0.31,
                "contamination_burden": 0.58,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "dominant_script_final": "other",
                "detected_language": "ml",
                "detected_language_confidence": 0.93,
                "confusion_candidate": True,
            },
        }
        mock_retry_model.return_value = {
            "text": "improved retry transcript",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "improved retry transcript"}],
            "language": "ml",
            "language_probability": 0.93,
            "metadata": {"asr_provider_used": "faster_whisper", "actual_local_model_name": "large-v3"},
        }
        mock_analyze.return_value = {
            "quality_class": "recoverable_but_weak",
            "lexical_trust": 0.26,
            "readability": 0.24,
            "wrong_script_burden": 0.14,
            "contamination_burden": 0.21,
            "trusted_visible_word_count": 4,
            "trusted_display_unit_count": 1,
            "dominant_script_final": "malayalam",
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")
        metadata = result.get("metadata", {})
        self.assertTrue(metadata.get("confusion_retry_candidate"))
        self.assertTrue(metadata.get("confusion_retry_executed"))
        self.assertTrue(metadata.get("confusion_retry_improved"))
        self.assertEqual(metadata.get("confusion_retry_model"), "large-v3")
        self.assertIn("before", metadata.get("confusion_retry_before_after", {}))

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=90.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._run_malayalam_local_model")
    @patch("videos.asr_router._should_attempt_malayalam_second_pass")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_MALAYALAM_STRATEGY="hybrid_retry",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_hopeless_malayalam_still_skips_heavy_second_pass_asr(
        self,
        _exists,
        _garbled,
        mock_second_pass_select,
        mock_retry_model,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_groq.return_value = {
            "text": "xxxx zzzz qqqq",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "xxxx zzzz qqqq"}],
            "language": "ml",
            "language_probability": 0.70,
            "metadata": {"asr_provider_used": "groq_whisper"},
        }
        mock_second_pass_select.return_value = {
            "attempt_second_pass": False,
            "reason": "",
            "blocked_reason": "hopeless_malayalam_skips_second_pass",
            "analysis": {
                "lexical_trust": 0.08,
                "readability": 0.07,
                "wrong_script_burden": 0.52,
                "contamination_burden": 0.84,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
            },
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")
        self.assertFalse(result.get("metadata", {}).get("second_pass_asr_attempted"))
        self.assertEqual(
            result.get("metadata", {}).get("second_pass_asr_blocked_reason"),
            "hopeless_malayalam_skips_second_pass",
        )
        mock_retry_model.assert_not_called()

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=90.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._run_malayalam_local_model")
    @patch("videos.asr_router._should_attempt_malayalam_second_pass")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_MALAYALAM_STRATEGY="hybrid_retry",
        ASR_MALAYALAM_PRIMARY_MODEL="large-v2",
        ASR_MALAYALAM_FAST_PRIMARY_MODEL="large-v2",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_second_pass_skips_when_retry_model_matches_primary(
        self,
        _exists,
        _garbled,
        mock_second_pass_select,
        mock_retry_model,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_groq.return_value = {
            "text": "weak malayalam transcript candidate",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "weak malayalam transcript candidate"}],
            "language": "ml",
            "language_probability": 0.81,
            "metadata": {"asr_provider_used": "groq_whisper"},
        }
        mock_second_pass_select.return_value = {
            "attempt_second_pass": True,
            "reason": "low_trust_malayalam_script_retry_candidate",
            "blocked_reason": "",
            "analysis": {
                "lexical_trust": 0.02,
                "readability": 0.09,
                "wrong_script_burden": 0.0,
                "contamination_burden": 1.0,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
            },
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")
        metadata = result.get("metadata", {})

        self.assertFalse(metadata.get("second_pass_asr_attempted"))
        self.assertEqual(metadata.get("second_pass_asr_blocked_reason"), "retry_model_same_as_primary")
        mock_retry_model.assert_not_called()

    @patch("videos.utils.classify_malayalam_segment_type", return_value={"type": "corrupted_or_non_malayalam", "score": {}})
    @patch("videos.utils._should_accept_usable_malayalam_first_pass", return_value={"first_pass_accepted": False, "first_pass_accept_reason": "", "quality_score": 0.11, "garbled_detector_score": 0.40})
    @patch("videos.utils.build_malayalam_transcript_trust")
    @patch("videos.utils._extract_asr_metrics", return_value={"other_indic_ratio": 0.58})
    @override_settings(ASR_MALAYALAM_SECOND_PASS_ENABLED=True)
    def test_malayalam_script_confusion_case_can_trigger_bounded_second_pass_selection(
        self,
        _metrics,
        mock_trust,
        _first_pass,
        _classify,
    ):
        mock_trust.return_value = {
            "total_segments": 1,
            "wrong_script_segments": 1,
            "malayalam_token_coverage": 0.18,
            "lexical_trust_score": 0.057,
            "overall_readability": 0.144,
            "dominant_script_final": "other",
        }
        payload = {
            "text": "exam class channel confidence support",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "exam class channel confidence support"}],
            "language": "ml",
            "language_probability": 0.93,
            "metadata": {"detected_language_confidence": 0.93},
        }

        decision = _should_attempt_malayalam_second_pass(
            payload,
            {"malayalam_asr_strategy": "hybrid_retry", "second_pass_enabled": True},
        )

        self.assertTrue(decision["attempt_second_pass"])
        self.assertEqual(decision["reason"], "malayalam_mixed_script_confusion_retry_candidate")
        self.assertEqual(decision["analysis"]["dominant_script_final"], "other")

    @patch("videos.utils.classify_malayalam_segment_type", return_value={"type": "corrupted_or_non_malayalam", "score": {}})
    @patch(
        "videos.utils._should_accept_usable_malayalam_first_pass",
        return_value={"first_pass_accepted": False, "first_pass_accept_reason": "", "quality_score": 0.07, "garbled_detector_score": 0.56},
    )
    @patch("videos.utils.build_malayalam_transcript_trust")
    @patch("videos.utils._extract_asr_metrics", return_value={"other_indic_ratio": 0.10})
    @override_settings(ASR_MALAYALAM_SECOND_PASS_ENABLED=True)
    def test_high_garble_malayalam_confusion_case_can_trigger_bounded_second_pass_selection(
        self,
        _metrics,
        mock_trust,
        _first_pass,
        _classify,
    ):
        mock_trust.return_value = {
            "total_segments": 2,
            "wrong_script_segments": 0,
            "malayalam_token_coverage": 0.78,
            "lexical_trust_score": 0.047,
            "overall_readability": 0.155,
            "dominant_script_final": "other",
        }
        payload = {
            "text": "topic model answer revision class",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "topic model"},
                {"id": 1, "start": 2.0, "end": 4.0, "text": "answer revision class"},
            ],
            "language": "ml",
            "language_probability": 0.98,
            "metadata": {"detected_language_confidence": 0.98},
        }

        decision = _should_attempt_malayalam_second_pass(
            payload,
            {"malayalam_asr_strategy": "hybrid_retry", "second_pass_enabled": True},
        )

        self.assertTrue(decision["attempt_second_pass"])
        self.assertEqual(decision["reason"], "high_garble_malayalam_confusion_retry_candidate")
        self.assertEqual(decision["analysis"]["dominant_script_final"], "other")
        self.assertGreaterEqual(decision["analysis"]["garble_score"], 0.56)

    @patch("videos.utils.classify_malayalam_segment_type", return_value={"type": "corrupted_or_non_malayalam", "score": {}})
    @patch(
        "videos.utils._should_accept_usable_malayalam_first_pass",
        return_value={"first_pass_accepted": False, "first_pass_accept_reason": "", "quality_score": 0.09, "garbled_detector_score": 0.129},
    )
    @patch("videos.utils.build_malayalam_transcript_trust")
    @patch("videos.utils._extract_asr_metrics", return_value={"other_indic_ratio": 0.12})
    @override_settings(ASR_MALAYALAM_SECOND_PASS_ENABLED=True)
    def test_dominant_other_script_zero_trusted_evidence_case_can_trigger_bounded_second_pass_selection(
        self,
        _metrics,
        mock_trust,
        _first_pass,
        _classify,
    ):
        mock_trust.return_value = {
            "total_segments": 2,
            "wrong_script_segments": 0,
            "malayalam_token_coverage": 0.82,
            "lexical_trust_score": 0.100,
            "overall_readability": 0.150,
            "dominant_script_final": "other",
        }
        payload = {
            "text": "module class answer lesson support",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "module class"},
                {"id": 1, "start": 2.0, "end": 4.0, "text": "answer lesson support"},
            ],
            "language": "ml",
            "language_probability": 0.98,
            "metadata": {"detected_language_confidence": 0.98},
        }

        decision = _should_attempt_malayalam_second_pass(
            payload,
            {"malayalam_asr_strategy": "hybrid_retry", "second_pass_enabled": True},
        )

        self.assertTrue(decision["attempt_second_pass"])
        self.assertEqual(decision["reason"], "dominant_script_other_zero_trusted_evidence_retry_candidate")
        self.assertEqual(decision["analysis"]["dominant_script_final"], "other")
        self.assertEqual(decision["analysis"]["trusted_visible_word_count"], 0)
        self.assertEqual(decision["analysis"]["trusted_display_unit_count"], 0)

    @patch("videos.utils.analyze_malayalam_source_fidelity", return_value={
        "source_language_fidelity_failed": True,
        "transcript_fidelity_state": "source_language_fidelity_failed",
        "suspicious_substitution_burden": 0.66,
        "dominant_non_malayalam_visible_ratio": 1.0,
    })
    @patch("videos.utils.classify_malayalam_segment_type", return_value={"type": "clean_english", "score": {"score": 0.15, "wrong_script_ratio": 0.0}})
    @patch(
        "videos.utils._should_accept_usable_malayalam_first_pass",
        return_value={"first_pass_accepted": False, "first_pass_accept_reason": "", "quality_score": 0.08, "garbled_detector_score": 0.22},
    )
    @patch("videos.utils.build_malayalam_transcript_trust")
    @patch("videos.utils._extract_asr_metrics", return_value={"other_indic_ratio": 0.04})
    @override_settings(ASR_MALAYALAM_SECOND_PASS_ENABLED=True)
    def test_full_clip_malayalam_fidelity_collapse_can_trigger_bounded_second_pass_selection(
        self,
        _metrics,
        mock_trust,
        _first_pass,
        _classify,
        _fidelity,
    ):
        mock_trust.return_value = {
            "total_segments": 2,
            "wrong_script_segments": 0,
            "malayalam_token_coverage": 0.10,
            "lexical_trust_score": 0.10,
            "overall_readability": 0.15,
            "dominant_script_final": "other",
        }
        payload = {
            "text": "When you exit the exam hall you will feel satisfied after checking the result site",
            "segments": [
                {"id": 0, "start": 0.0, "end": 3.0, "text": "When you exit the exam hall you will feel satisfied"},
                {"id": 1, "start": 3.0, "end": 6.0, "text": "after checking the result site"},
            ],
            "language": "ml",
            "language_probability": 0.98,
            "metadata": {"detected_language_confidence": 0.98},
        }

        decision = _should_attempt_malayalam_second_pass(
            payload,
            {"malayalam_asr_strategy": "hybrid_retry", "second_pass_enabled": True},
        )

        self.assertTrue(decision["attempt_second_pass"])
        self.assertEqual(decision["reason"], "malayalam_full_clip_fidelity_retry_candidate")
        self.assertTrue(decision["analysis"]["source_language_fidelity_failed"])

    def test_constrained_correction_never_converts_malayalam_source_to_english_output(self):
        payload = {
            "text": "പരീക്ഷാ ഫലം check result soon",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "പരീക്ഷാ ഫലം check result soon"}],
            "language": "ml",
            "metadata": {},
        }
        with patch("videos.asr_router._analyze_malayalam_asr_payload", side_effect=[
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "full_clip_fidelity_retry_candidate": False,
                "confusion_candidate": True,
                "suspicious_substitution_burden": 0.28,
                "wrong_script_burden": 0.12,
                "contamination_burden": 0.18,
            },
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "contamination_burden": 0.34,
            },
        ]), patch("videos.utils.repair_malayalam_degraded_transcript", return_value={
            "text": "When the result comes soon",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "When the result comes soon"}],
            "metadata": {},
        }):
            corrected, decision = _apply_bounded_malayalam_faithfulness_recovery(payload)
        self.assertFalse(decision["applied"])
        self.assertEqual(decision["blocked_reason"], "recovery_increased_english_substitution")
        self.assertEqual(corrected["text"], payload["text"])

    def test_constrained_correction_can_adopt_recoverable_malayalam_improvement(self):
        payload = {
            "text": "exam result support class",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "exam result support class"}],
            "language": "ml",
            "metadata": {},
        }
        with patch("videos.asr_router._analyze_malayalam_asr_payload", side_effect=[
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "full_clip_fidelity_retry_candidate": False,
                "confusion_candidate": True,
                "suspicious_substitution_burden": 0.18,
                "wrong_script_burden": 0.24,
                "contamination_burden": 0.30,
                "lexical_trust": 0.18,
                "readability": 0.20,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
            },
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "contamination_burden": 0.12,
                "lexical_trust": 0.32,
                "readability": 0.31,
                "trusted_visible_word_count": 5,
                "trusted_display_unit_count": 1,
                "wrong_script_burden": 0.08,
            },
        ]), patch("videos.utils.repair_malayalam_degraded_transcript", return_value={
            "text": "എക്സാം result support ക്ലാസ്",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "എക്സാം result support ക്ലാസ്"}],
            "metadata": {"degraded_repair_applied": True},
        }):
            corrected, decision = _apply_bounded_malayalam_faithfulness_recovery(payload)
        self.assertTrue(decision["applied"])
        self.assertEqual(decision["reason"], "bounded_malayalam_faithfulness_recovery_improved")
        self.assertIn("ക്ലാസ്", corrected["text"])

    def test_readability_only_improvement_is_rejected_when_faithfulness_does_not_improve(self):
        payload = {
            "text": "exam result support class",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "exam result support class"}],
            "language": "ml",
            "metadata": {},
        }
        with patch("videos.asr_router._analyze_malayalam_asr_payload", side_effect=[
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "full_clip_fidelity_retry_candidate": False,
                "confusion_candidate": True,
                "suspicious_substitution_burden": 0.20,
                "wrong_script_burden": 0.22,
                "contamination_burden": 0.24,
                "lexical_trust": 0.18,
                "readability": 0.20,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
            },
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "suspicious_substitution_burden": 0.20,
                "wrong_script_burden": 0.22,
                "contamination_burden": 0.24,
                "lexical_trust": 0.31,
                "readability": 0.34,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
            },
        ]), patch("videos.utils.repair_malayalam_degraded_transcript", return_value={
            "text": "exam result support class improved",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "exam result support class improved"}],
            "metadata": {"degraded_repair_applied": True},
        }):
            corrected, decision = _apply_bounded_malayalam_faithfulness_recovery(payload)
        self.assertFalse(decision["applied"])
        self.assertEqual(decision["blocked_reason"], "recovery_did_not_improve_faithfulness")
        self.assertEqual(corrected["text"], payload["text"])

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=120.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._should_attempt_malayalam_second_pass")
    @patch("videos.asr_router._apply_bounded_malayalam_faithfulness_recovery")
    @patch("videos.asr_router._analyze_malayalam_asr_payload")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_malayalam_groq_draft_is_marked_as_draft_until_final_faithfulness_choice(
        self,
        _exists,
        _garbled,
        mock_analyze,
        mock_recovery,
        mock_second_pass,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_groq.return_value = {
            "text": "à´ªà´°àµ€à´•àµà´· à´«à´²à´‚ à´‡à´µà´¿à´Ÿàµ† à´ªà´±à´¯àµà´¨àµà´¨àµ",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "à´ªà´°àµ€à´•àµà´· à´«à´²à´‚ à´‡à´µà´¿à´Ÿàµ† à´ªà´±à´¯àµà´¨àµà´¨àµ"}],
            "language": "ml",
            "language_probability": 0.97,
            "metadata": {"asr_provider_used": "groq_whisper"},
        }
        mock_second_pass.return_value = {
            "attempt_second_pass": False,
            "reason": "",
            "blocked_reason": "primary_result_already_good",
            "analysis": {
                "quality_class": "clearly_good",
                "confusion_candidate": False,
                "lexical_trust": 0.52,
                "readability": 0.44,
                "wrong_script_burden": 0.04,
                "contamination_burden": 0.08,
                "trusted_visible_word_count": 8,
                "trusted_display_unit_count": 1,
            },
        }
        mock_recovery.side_effect = lambda payload: (payload, {"attempted": False, "applied": False, "reason": "", "blocked_reason": "primary_result_already_faithful"})
        mock_analyze.return_value = {
            "source_language_fidelity_failed": False,
            "transcript_fidelity_state": "degraded_but_informative_malayalam",
            "catastrophic_wrong_script_failure": False,
            "nonempty_signal": True,
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")
        metadata = result.get("metadata", {})
        self.assertTrue(metadata.get("malayalam_draft_candidate"))
        self.assertEqual(metadata.get("malayalam_candidate_stage"), "fast_draft_candidate")
        self.assertEqual(metadata.get("malayalam_final_candidate_source"), "fast_draft_candidate")
        self.assertTrue(metadata.get("malayalam_draft_candidate_accepted_as_final"))

    def test_specialist_recovery_candidate_is_attempted_only_on_recoverable_malayalam(self):
        payload = {
            "text": "exam result support class",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "exam result support class"}],
            "language": "ml",
            "metadata": {},
        }
        with patch("videos.asr_router._analyze_malayalam_asr_payload", side_effect=[
            {
                "quality_class": "recoverable_but_weak",
                "detected_language": "ml",
                "detected_language_confidence": 0.97,
                "recoverable_malayalam_fidelity_gap": True,
                "confusion_candidate": True,
                "wrong_script_burden": 0.24,
                "suspicious_substitution_burden": 0.22,
            },
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "transcript_fidelity_state": "degraded_but_informative_malayalam",
                "trusted_visible_word_count": 6,
                "trusted_display_unit_count": 1,
                "visible_malayalam_char_ratio": 0.52,
                "wrong_script_burden": 0.08,
                "suspicious_substitution_burden": 0.08,
            },
        ]), patch("videos.asr_router._run_malayalam_local_model", return_value={
            "text": "à´ªà´°àµ€à´•àµà´· à´«à´²à´‚ à´‡à´µà´¿à´Ÿàµ† à´ªà´±à´¯àµà´¨àµà´¨àµ",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "à´ªà´°àµ€à´•àµà´· à´«à´²à´‚ à´‡à´µà´¿à´Ÿàµ† à´ªà´±à´¯àµà´¨àµà´¨àµ"}],
            "language": "ml",
            "metadata": {},
        }):
            candidate, decision = build_malayalam_specialist_candidate(
                audio_path="tmp.wav",
                source_type="youtube",
                current_payload=payload,
                route_decision={"retry_model": "large-v3"},
            )
        self.assertTrue(decision["attempted"])
        self.assertTrue(decision["applied"])
        self.assertEqual(decision["reason"], "specialist_candidate_improved_faithfulness")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.get("metadata", {}).get("malayalam_candidate_stage"), "specialist_recovery_candidate")

    def test_specialist_recovery_is_skipped_on_already_faithful_malayalam(self):
        payload = {"text": "good", "segments": [{"text": "good"}], "language": "ml", "metadata": {}}
        with patch("videos.asr_router._analyze_malayalam_asr_payload", return_value={
            "quality_class": "clearly_good",
            "detected_language": "ml",
            "detected_language_confidence": 0.98,
        }), patch("videos.asr_router._run_malayalam_local_model") as mock_run:
            candidate, decision = build_malayalam_specialist_candidate(
                audio_path="tmp.wav",
                source_type="youtube",
                current_payload=payload,
                route_decision={},
            )
        self.assertIsNone(candidate)
        self.assertFalse(decision["attempted"])
        self.assertEqual(decision["blocked_reason"], "already_faithful_malayalam")
        mock_run.assert_not_called()

    def test_specialist_recovery_is_skipped_on_hopeless_no_signal_malayalam(self):
        payload = {"text": "junk", "segments": [{"text": "junk"}], "language": "ml", "metadata": {}}
        with patch("videos.asr_router._analyze_malayalam_asr_payload", return_value={
            "quality_class": "clearly_hopeless",
            "detected_language": "ml",
            "detected_language_confidence": 0.98,
        }), patch("videos.asr_router._run_malayalam_local_model") as mock_run:
            candidate, decision = build_malayalam_specialist_candidate(
                audio_path="tmp.wav",
                source_type="youtube",
                current_payload=payload,
                route_decision={},
            )
        self.assertIsNone(candidate)
        self.assertFalse(decision["attempted"])
        self.assertEqual(decision["blocked_reason"], "hopeless_malayalam_no_specialist_signal")
        mock_run.assert_not_called()

    def test_specialist_recovery_is_rejected_when_readability_improves_but_faithfulness_does_not(self):
        payload = {"text": "exam result support class", "segments": [{"text": "exam result support class"}], "language": "ml", "metadata": {}}
        with patch("videos.asr_router._analyze_malayalam_asr_payload", side_effect=[
            {
                "quality_class": "recoverable_but_weak",
                "detected_language": "ml",
                "detected_language_confidence": 0.97,
                "recoverable_malayalam_fidelity_gap": True,
                "confusion_candidate": True,
                "wrong_script_burden": 0.20,
                "suspicious_substitution_burden": 0.18,
                "lexical_trust": 0.16,
                "readability": 0.18,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "visible_malayalam_char_ratio": 0.0,
            },
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "transcript_fidelity_state": "uncertain_malayalam_fidelity",
                "wrong_script_burden": 0.20,
                "suspicious_substitution_burden": 0.18,
                "lexical_trust": 0.34,
                "readability": 0.38,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "visible_malayalam_char_ratio": 0.0,
            },
        ]), patch("videos.asr_router._run_malayalam_local_model", return_value={
            "text": "exam result support class improved",
            "segments": [{"text": "exam result support class improved"}],
            "language": "ml",
            "metadata": {},
        }):
            candidate, decision = build_malayalam_specialist_candidate(
                audio_path="tmp.wav",
                source_type="youtube",
                current_payload=payload,
                route_decision={},
            )
        self.assertIsNone(candidate)
        self.assertTrue(decision["attempted"])
        self.assertFalse(decision["applied"])
        self.assertEqual(decision["blocked_reason"], "specialist_candidate_not_faithfully_better")

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=120.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._should_attempt_malayalam_second_pass")
    @patch("videos.asr_router._apply_bounded_malayalam_faithfulness_recovery")
    @patch("videos.asr_router.build_malayalam_specialist_candidate")
    @patch("videos.asr_router._analyze_malayalam_asr_payload")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_catastrophic_fidelity_failed_malayalam_still_suppresses_after_specialist_attempt(
        self,
        _exists,
        _garbled,
        mock_analyze,
        mock_specialist,
        mock_recovery,
        mock_second_pass,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_groq.return_value = {
            "text": "wrong script collapse",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "wrong script collapse"}],
            "language": "ml",
            "language_probability": 0.98,
            "metadata": {"asr_provider_used": "groq_whisper"},
        }
        mock_second_pass.return_value = {
            "attempt_second_pass": False,
            "reason": "",
            "blocked_reason": "hopeless_malayalam_skips_second_pass",
            "analysis": {
                "quality_class": "recoverable_but_weak",
                "confusion_candidate": False,
                "lexical_trust": 0.08,
                "readability": 0.10,
                "wrong_script_burden": 0.88,
                "contamination_burden": 0.22,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
            },
        }
        mock_recovery.side_effect = lambda payload: (payload, {"attempted": False, "applied": False, "reason": "", "blocked_reason": "fidelity_failed_without_recoverable_signal"})
        mock_specialist.return_value = (
            None,
            {
                "attempted": True,
                "applied": False,
                "reason": "recoverable_malayalam_specialist_candidate",
                "blocked_reason": "specialist_candidate_not_faithfully_better",
                "backend": "local_whisper:large-v3",
                "candidate_source": "specialist_recovery_candidate",
            },
        )
        mock_analyze.return_value = {
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "catastrophic_wrong_script_failure",
            "catastrophic_wrong_script_failure": True,
            "nonempty_signal": True,
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")
        metadata = result.get("metadata", {})
        self.assertTrue(metadata.get("malayalam_specialist_recovery_attempted"))
        self.assertFalse(metadata.get("malayalam_specialist_recovery_applied"))
        self.assertEqual(metadata.get("malayalam_specialist_backend"), "local_whisper:large-v3")
        self.assertEqual(metadata.get("malayalam_final_candidate_source"), "suppressed_fidelity_failed")

    def test_linguistic_correction_is_attempted_only_on_recoverable_malayalam(self):
        payload = {
            "text": "exam result support class",
            "segments": [{"text": "exam result support class"}],
            "language": "ml",
            "metadata": {},
        }
        with patch("videos.asr_router._analyze_malayalam_asr_payload", side_effect=[
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "recoverable_malayalam_fidelity_gap": True,
                "confusion_candidate": True,
                "wrong_script_burden": 0.24,
                "suspicious_substitution_burden": 0.18,
                "word_count": 4,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "visible_malayalam_char_ratio": 0.0,
            },
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "transcript_fidelity_state": "degraded_but_informative_malayalam",
                "wrong_script_burden": 0.05,
                "suspicious_substitution_burden": 0.05,
                "trusted_visible_word_count": 5,
                "trusted_display_unit_count": 1,
                "visible_malayalam_char_ratio": 0.48,
            },
        ]), patch("videos.utils.repair_malayalam_degraded_transcript", return_value={
            "text": "പരീക്ഷ result support ക്ലാസ്",
            "segments": [{"text": "പരീക്ഷ result support ക്ലാസ്"}],
            "metadata": {"degraded_repair_applied": True},
        }):
            candidate, decision = build_malayalam_linguistic_correction_candidate(current_payload=payload)
        self.assertTrue(decision["attempted"])
        self.assertTrue(decision["applied"])
        self.assertEqual(decision["reason"], "linguistic_correction_improved_faithfulness")
        self.assertIsNotNone(candidate)

    def test_linguistic_correction_is_skipped_on_already_faithful_malayalam(self):
        payload = {"text": "good", "segments": [{"text": "good"}], "language": "ml", "metadata": {}}
        with patch("videos.asr_router._analyze_malayalam_asr_payload", return_value={
            "quality_class": "clearly_good",
        }), patch("videos.utils.repair_malayalam_degraded_transcript") as mock_repair:
            candidate, decision = build_malayalam_linguistic_correction_candidate(current_payload=payload)
        self.assertIsNone(candidate)
        self.assertFalse(decision["attempted"])
        self.assertEqual(decision["blocked_reason"], "already_faithful_malayalam")
        mock_repair.assert_not_called()

    def test_linguistic_correction_is_skipped_on_hopeless_no_signal_malayalam(self):
        payload = {"text": "junk", "segments": [{"text": "junk"}], "language": "ml", "metadata": {}}
        with patch("videos.asr_router._analyze_malayalam_asr_payload", return_value={
            "quality_class": "clearly_hopeless",
        }), patch("videos.utils.repair_malayalam_degraded_transcript") as mock_repair:
            candidate, decision = build_malayalam_linguistic_correction_candidate(current_payload=payload)
        self.assertIsNone(candidate)
        self.assertFalse(decision["attempted"])
        self.assertEqual(decision["blocked_reason"], "hopeless_malayalam_no_linguistic_signal")
        mock_repair.assert_not_called()

    def test_linguistic_correction_never_converts_malayalam_source_into_english_output(self):
        payload = {
            "text": "à´ªà´°àµ€à´•àµà´· à´«à´²à´‚ result support",
            "segments": [{"text": "à´ªà´°àµ€à´•àµà´· à´«à´²à´‚ result support"}],
            "language": "ml",
            "metadata": {},
        }
        with patch("videos.asr_router._analyze_malayalam_asr_payload", return_value={
            "quality_class": "recoverable_but_weak",
            "source_language_fidelity_failed": False,
            "recoverable_malayalam_fidelity_gap": True,
            "confusion_candidate": True,
            "wrong_script_burden": 0.21,
            "suspicious_substitution_burden": 0.18,
            "word_count": 4,
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "visible_malayalam_char_ratio": 0.16,
        }), patch("videos.utils.repair_malayalam_degraded_transcript", return_value={
            "text": "When the exam result is available",
            "segments": [{"text": "When the exam result is available"}],
            "metadata": {},
        }):
            candidate, decision = build_malayalam_linguistic_correction_candidate(current_payload=payload)
        self.assertIsNone(candidate)
        self.assertTrue(decision["attempted"])
        self.assertEqual(decision["blocked_reason"], "linguistic_correction_increased_english_output")

    def test_linguistic_correction_never_turns_transcript_into_summary_like_fluent_text(self):
        payload = {
            "text": "à´ªà´°àµ€à´•àµà´· result support",
            "segments": [{"text": "à´ªà´°àµ€à´•àµà´· result support"}],
            "language": "ml",
            "metadata": {},
        }
        with patch("videos.asr_router._analyze_malayalam_asr_payload", return_value={
            "quality_class": "recoverable_but_weak",
            "source_language_fidelity_failed": False,
            "recoverable_malayalam_fidelity_gap": True,
            "confusion_candidate": True,
            "wrong_script_burden": 0.21,
            "suspicious_substitution_burden": 0.18,
            "word_count": 3,
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "visible_malayalam_char_ratio": 0.16,
        }), patch("videos.utils.repair_malayalam_degraded_transcript", return_value={
            "text": "ഈ വീഡിയോയിൽ overall summary ആയി പരീക്ഷ ഫലം വിശദമായി പറയുന്നു",
            "segments": [{"text": "ഈ വീഡിയോയിൽ overall summary ആയി പരീക്ഷ ഫലം വിശദമായി പറയുന്നു"}],
            "metadata": {},
        }):
            candidate, decision = build_malayalam_linguistic_correction_candidate(current_payload=payload)
        self.assertIsNone(candidate)
        self.assertTrue(decision["attempted"])
        self.assertEqual(decision["blocked_reason"], "summary_like_rewrite")

    def test_linguistic_correction_is_rejected_when_readability_improves_but_faithfulness_does_not(self):
        payload = {
            "text": "exam result support class",
            "segments": [{"text": "exam result support class"}],
            "language": "ml",
            "metadata": {},
        }
        with patch("videos.asr_router._analyze_malayalam_asr_payload", side_effect=[
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "recoverable_malayalam_fidelity_gap": True,
                "confusion_candidate": True,
                "wrong_script_burden": 0.22,
                "suspicious_substitution_burden": 0.20,
                "lexical_trust": 0.18,
                "readability": 0.20,
                "word_count": 4,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "visible_malayalam_char_ratio": 0.0,
            },
            {
                "quality_class": "recoverable_but_weak",
                "source_language_fidelity_failed": False,
                "transcript_fidelity_state": "uncertain_malayalam_fidelity",
                "wrong_script_burden": 0.22,
                "suspicious_substitution_burden": 0.20,
                "lexical_trust": 0.36,
                "readability": 0.40,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "visible_malayalam_char_ratio": 0.0,
            },
        ]), patch("videos.utils.repair_malayalam_degraded_transcript", return_value={
            "text": "exam result support class improved",
            "segments": [{"text": "exam result support class improved"}],
            "metadata": {},
        }):
            candidate, decision = build_malayalam_linguistic_correction_candidate(current_payload=payload)
        self.assertIsNone(candidate)
        self.assertEqual(decision["blocked_reason"], "linguistic_correction_not_faithfully_better")

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=120.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._should_attempt_malayalam_second_pass")
    @patch("videos.asr_router._apply_bounded_malayalam_faithfulness_recovery")
    @patch("videos.asr_router.build_malayalam_specialist_candidate")
    @patch("videos.asr_router.build_malayalam_linguistic_correction_candidate")
    @patch("videos.asr_router._analyze_malayalam_asr_payload")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        USE_GROQ_WHISPER=True,
        GROQ_API_KEY="x",
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_catastrophic_fidelity_failed_malayalam_still_suppresses_after_linguistic_correction_attempt(
        self,
        _exists,
        _garbled,
        mock_analyze,
        mock_linguistic,
        mock_specialist,
        mock_recovery,
        mock_second_pass,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_groq.return_value = {
            "text": "wrong script collapse",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "wrong script collapse"}],
            "language": "ml",
            "language_probability": 0.98,
            "metadata": {"asr_provider_used": "groq_whisper"},
        }
        mock_second_pass.return_value = {
            "attempt_second_pass": False,
            "reason": "",
            "blocked_reason": "hopeless_malayalam_skips_second_pass",
            "analysis": {
                "quality_class": "recoverable_but_weak",
                "confusion_candidate": False,
                "lexical_trust": 0.08,
                "readability": 0.10,
                "wrong_script_burden": 0.88,
                "contamination_burden": 0.22,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
            },
        }
        mock_recovery.side_effect = lambda payload: (payload, {"attempted": False, "applied": False, "reason": "", "blocked_reason": "fidelity_failed_without_recoverable_signal"})
        mock_specialist.return_value = (None, {"attempted": False, "applied": False, "reason": "", "blocked_reason": "hopeless_malayalam_no_specialist_signal", "backend": "", "candidate_source": ""})
        mock_linguistic.return_value = (
            None,
            {
                "attempted": True,
                "applied": False,
                "reason": "recoverable_malayalam_linguistic_correction_candidate",
                "blocked_reason": "linguistic_correction_not_faithfully_better",
                "backend": "deterministic_malayalam_repair",
                "candidate_source": "linguistic_correction_candidate",
            },
        )
        mock_analyze.return_value = {
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "catastrophic_wrong_script_failure",
            "catastrophic_wrong_script_failure": True,
            "nonempty_signal": True,
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")
        metadata = result.get("metadata", {})
        self.assertTrue(metadata.get("malayalam_linguistic_correction_attempted"))
        self.assertFalse(metadata.get("malayalam_linguistic_correction_applied"))
        self.assertEqual(metadata.get("malayalam_linguistic_correction_backend"), "deterministic_malayalam_repair")
        self.assertEqual(metadata.get("malayalam_final_candidate_source"), "suppressed_fidelity_failed")

    def test_downstream_blocked_reasons_persist_clearly_after_confusion_rerun_path(self):
        observability = _build_malayalam_observability(
            transcript_payload={
                "language": "ml",
                "metadata": {
                    "confusion_retry_candidate": True,
                    "confusion_retry_executed": True,
                    "confusion_retry_model": "large-v3",
                    "confusion_retry_improved": False,
                    "confusion_retry_improvement_reason": "not_materially_better",
                    "confusion_retry_before_after": {
                        "before": {"lexical_trust": 0.057, "readability": 0.144, "trusted_visible_word_count": 0, "trusted_display_unit_count": 0, "dominant_script_final": "other"},
                        "after": {"lexical_trust": 0.061, "readability": 0.149, "trusted_visible_word_count": 0, "trusted_display_unit_count": 0, "dominant_script_final": "other"},
                    },
                },
            },
            transcript_state="degraded",
            processing_metrics={
                "summary_blocked_reason": "low_evidence_malayalam_gate",
                "chatbot_blocked_reason": "low_evidence_malayalam_gate",
                "downstream_suppression_reason": "no_trusted_display_units",
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "lexical_trust_score": 0.057,
                "overall_readability": 0.144,
                "downstream_suppressed": True,
            },
        )
        self.assertTrue(observability["confusion_retry_candidate"])
        self.assertTrue(observability["confusion_retry_executed"])
        self.assertEqual(observability["summary_blocked_reason"], "low_evidence_malayalam_gate")
        self.assertEqual(observability["chatbot_blocked_reason"], "low_evidence_malayalam_gate")
        self.assertEqual(observability["downstream_suppression_reason"], "no_trusted_display_units")
        self.assertEqual(observability["trusted_visible_word_count"], 0)
        self.assertEqual(observability["trusted_display_unit_count"], 0)

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=90.0)
    @patch("videos.asr_router._run_malayalam_local_model")
    @patch("videos.asr_router._transcribe_with_malayalam_local_large_v3")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        ASR_MALAYALAM_STRATEGY="quality_first",
        ASR_MALAYALAM_MODEL_OVERRIDE_ENABLED=True,
        ASR_MALAYALAM_MODEL_OVERRIDE="custom-ml-model",
        ASR_MALAYALAM_SECOND_PASS_ENABLED=False,
        ASR_LOW_CONTENT_WPM_MIN=0,
        ASR_LOW_CONTENT_MIN_WORDS_LONG=1,
    )
    def test_malayalam_model_override_fails_safely_back_to_current_path(
        self,
        _exists,
        _garbled,
        mock_local_large_v3,
        mock_override_model,
        _duration,
        _preprocess,
    ):
        mock_override_model.side_effect = RuntimeError("override unavailable")
        mock_local_large_v3.return_value = {
            "text": "മലയാളം fallback transcript with enough grounded words to continue safely here.",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "മലയാളം fallback transcript with enough grounded words to continue safely here."}],
            "language": "ml",
            "language_probability": 0.91,
            "metadata": {"asr_provider_used": "faster_whisper", "actual_local_model_name": "large-v3"},
        }

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")
        self.assertEqual(result.get("asr_engine_used"), "whisper_local")
        self.assertTrue(result.get("metadata", {}).get("fallback_triggered"))
        self.assertEqual(result.get("metadata", {}).get("fallback_model_used"), "large-v3")
        self.assertEqual(result.get("metadata", {}).get("primary_model_used"), "custom-ml-model")
        mock_local_large_v3.assert_called_once_with("tmp.wav", "youtube")


class LocalWhisperReuseTests(SimpleTestCase):
    def _workspace_tmp_dir(self, name: str) -> Path:
        path = Path("backend") / "test_tmp" / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @patch("videos.utils._load_whisper_model_instance")
    @override_settings(ASR_LOCAL_MODEL_REUSE=True)
    def test_local_whisper_model_reuse_works(self, mock_loader):
        _WHISPER_MODEL_CACHE.clear()
        mock_loader.return_value = object()

        first, first_reused = _get_local_whisper_model("large-v3", "cpu", "int8")
        second, second_reused = _get_local_whisper_model("large-v3", "cpu", "int8")

        self.assertIs(first, second)
        self.assertFalse(first_reused)
        self.assertTrue(second_reused)
        mock_loader.assert_called_once_with("large-v3", "cpu", "int8")

    @patch("videos.utils._load_whisper_model_instance")
    @override_settings(ASR_LOCAL_MODEL_REUSE=True)
    def test_malayalam_local_model_reuse_works(self, mock_loader):
        _WHISPER_MODEL_CACHE.clear()
        mock_loader.return_value = object()

        first, first_reused = _get_local_whisper_model("smcproject/vegam-whisper-medium-ml-int8", "cpu", "int8")
        second, second_reused = _get_local_whisper_model("smcproject/vegam-whisper-medium-ml-int8", "cpu", "int8")

        self.assertIs(first, second)
        self.assertFalse(first_reused)
        self.assertTrue(second_reused)
        mock_loader.assert_called_once_with("smcproject/vegam-whisper-medium-ml-int8", "cpu", "int8")

    @patch("videos.utils._get_local_whisper_model")
    @patch("videos.utils._ensure_malayalam_ctranslate2_model")
    @override_settings(
        ASR_MALAYALAM_PRIMARY_MODEL="vrclc/Whisper-small-Malayalam",
        ASR_MALAYALAM_MODEL_FAMILY="transformers",
    )
    def test_malayalam_primary_model_uses_conversion_aware_resolution(self, mock_resolve, mock_get_model):
        mock_resolve.return_value = (
            "C:\\cached\\ctranslate2\\vrclc-Whisper-small-Malayalam",
            {
                "configured_model_name": "vrclc/Whisper-small-Malayalam",
                "resolved_model_name": "C:\\cached\\ctranslate2\\vrclc-Whisper-small-Malayalam",
                "model_family": "transformers",
                "model_converted": True,
                "converted_model_cached": True,
            },
        )
        mock_get_model.return_value = (object(), True)

        _, reused, meta = _get_local_whisper_model_with_meta("vrclc/Whisper-small-Malayalam", "cpu", "int8")

        self.assertTrue(reused)
        self.assertEqual(meta["configured_model_name"], "vrclc/Whisper-small-Malayalam")
        self.assertEqual(meta["resolved_model_name"], "C:\\cached\\ctranslate2\\vrclc-Whisper-small-Malayalam")
        self.assertEqual(meta["model_family"], "transformers")
        self.assertTrue(meta["model_converted"])
        self.assertTrue(meta["converted_model_cached"])
        mock_resolve.assert_called_once_with("vrclc/Whisper-small-Malayalam", "int8")
        mock_get_model.assert_called_once_with("C:\\cached\\ctranslate2\\vrclc-Whisper-small-Malayalam", "cpu", "int8")

    @patch("videos.utils._ensure_windows_safe_hf_snapshot")
    @patch("ctranslate2.converters.TransformersConverter")
    def test_malayalam_conversion_cache_rebuilt_when_incomplete(self, mock_converter_cls, mock_snapshot):
        broken_dir = self._workspace_tmp_dir("broken-cache")
        try:
            broken_dir.mkdir(parents=True, exist_ok=True)
            (broken_dir / "config.json").write_text("{}", encoding="utf-8")

            mock_snapshot.return_value = ("C:\\downloaded\\vrclc-Whisper-small-Malayalam", {"model_download_started": True})
            mock_converter = mock_converter_cls.return_value
            def _fake_convert(output_path, quantization=None):
                out = Path(output_path)
                out.mkdir(parents=True, exist_ok=True)
                (out / "model.bin").write_text("bin", encoding="utf-8")
                (out / "tokenizer.json").write_text("{}", encoding="utf-8")
            mock_converter.convert.side_effect = _fake_convert

            with patch("videos.utils._ctranslate2_cache_path_for_model", return_value=broken_dir):
                resolved, meta = _ensure_malayalam_ctranslate2_model("vrclc/Whisper-small-Malayalam", "int8")

            self.assertEqual(resolved, str(broken_dir))
            self.assertTrue(meta["conversion_rebuilt"])
            self.assertTrue(meta["model_converted"])
            self.assertTrue(meta["conversion_succeeded"])
            self.assertTrue(meta["converted_model_valid"])
            mock_converter.convert.assert_called_once()
        finally:
            shutil.rmtree(broken_dir.parent, ignore_errors=True)

    @patch("videos.utils._ensure_windows_safe_hf_snapshot", side_effect=RuntimeError("WinError 1314: A required privilege is not held by the client"))
    def test_malayalam_windows_privilege_issue_fails_clearly(self, _mock_snapshot):
        clean_target = self._workspace_tmp_dir("ctranslate2-cache")
        try:
            with patch("videos.utils._ctranslate2_cache_path_for_model", return_value=clean_target):
                with self.assertRaisesRegex(RuntimeError, "failed on Windows due to a privilege/symlink issue"):
                    _ensure_malayalam_ctranslate2_model("vrclc/Whisper-small-Malayalam", "int8")
        finally:
            shutil.rmtree(clean_target.parent, ignore_errors=True)

    @override_settings(ASR_MALAYALAM_MODEL_FAMILY="transformers")
    def test_raw_transformers_checkpoint_is_rejected_as_direct_faster_whisper_input(self):
        fake_snapshot = self._workspace_tmp_dir("raw-transformers-snapshot")
        try:
            (fake_snapshot / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Direct Transformers checkpoints are blocked"):
                _ensure_malayalam_ctranslate2_model(str(fake_snapshot), "int8")
        finally:
            shutil.rmtree(fake_snapshot.parent, ignore_errors=True)


class MalayalamLocalFallbackTests(SimpleTestCase):
    @override_settings(ASR_MALAYALAM_STRATEGY="quality_first")
    def test_malayalam_router_defaults_to_local_quality_first_path(self):
        route = _choose_primary_engine(
            requested_lang="ml",
            chosen_lang="ml",
            duration_seconds=180.0,
            deepgram_supported=set(),
            file_size_bytes=5 * 1024 * 1024,
            detection_confidence=0.98,
        )

        self.assertEqual(route["engine"], "whisper_local")
        self.assertEqual(route["reason"], "malayalam_quality_primary")
        self.assertEqual(route["malayalam_asr_strategy"], "quality_first")

    @override_settings(
        ASR_MALAYALAM_STRATEGY="quality_first",
        ASR_MALAYALAM_FAST_PRIMARY_MODEL="medium",
        ASR_MALAYALAM_PRIMARY_MODEL="large-v2",
        ASR_MALAYALAM_LONGFORM_SECONDS=480,
    )
    def test_malayalam_router_uses_fast_model_for_longform_uploads(self):
        route = _choose_primary_engine(
            requested_lang="ml",
            chosen_lang="ml",
            duration_seconds=600.0,
            deepgram_supported=set(),
            file_size_bytes=20 * 1024 * 1024,
            detection_confidence=0.98,
        )

        self.assertEqual(route["engine"], "whisper_local")
        self.assertEqual(route["model"], "medium")
        self.assertTrue(route["speed_optimized_for_longform"])

    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._audio_duration_seconds", return_value=600.0)
    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._transcribe_with_malayalam_local")
    @patch("videos.asr_router._run_malayalam_local_model")
    @patch("videos.asr_router._is_garbled", return_value=False)
    @patch("videos.asr_router.os.path.exists", return_value=False)
    @override_settings(
        ASR_MALAYALAM_STRATEGY="quality_first",
        ASR_MALAYALAM_FAST_PRIMARY_MODEL="medium",
        ASR_MALAYALAM_PRIMARY_MODEL="large-v2",
        ASR_MALAYALAM_LONGFORM_SECONDS=480,
    )
    def test_longform_fast_malayalam_skips_heavy_local_fallback_after_primary_failure(
        self,
        _exists,
        _garbled,
        mock_run_model,
        mock_ml_local,
        mock_groq,
        _duration,
        _preprocess,
    ):
        mock_run_model.side_effect = RuntimeError("Transcript appears empty or repeated. Audio may be silent or blocked.")

        result = transcribe_video_router("demo.wav", source_type="youtube", requested_language="ml")

        mock_run_model.assert_called_once_with("tmp.wav", "youtube", "medium")
        mock_ml_local.assert_not_called()
        mock_groq.assert_not_called()
        self.assertEqual(result.get("language"), "ml")
        self.assertEqual(result.get("text"), "")
        self.assertTrue(result.get("metadata", {}).get("terminal_malayalam_failure"))
        self.assertEqual(
            result.get("metadata", {}).get("terminal_malayalam_failure_reason"),
            "Transcript appears empty or repeated. Audio may be silent or blocked.",
        )

    def test_malayalam_local_prompt_keeps_transcription_not_translation_policy(self):
        prompt = build_malayalam_local_prompt()

        self.assertIn("Malayalam script", prompt)
        self.assertIn("not translation", prompt)
        self.assertIn("Do not rewrite Malayalam speech into English", prompt)

    @patch("videos.utils._transcribe_with_faster_whisper_model")
    @override_settings(
        ASR_MALAYALAM_FAST_PRIMARY_MODEL="small",
        ASR_MALAYALAM_PRIMARY_MODEL="vrclc/Whisper-small-Malayalam",
        WHISPER_MODEL_MALAYALAM_FALLBACK="large-v3",
    )
    def test_model_format_confusion_does_not_fall_back_to_large_v3(self, mock_core):
        mock_core.side_effect = RuntimeError(
            "Malayalam model conversion required for Faster-Whisper runtime but failed for 'vrclc/Whisper-small-Malayalam'."
        )

        with self.assertRaisesRegex(RuntimeError, "Pre-convert it to CTranslate2"):
            _transcribe_with_faster_whisper("demo.wav", "youtube", "ml")

        self.assertEqual(mock_core.call_count, 1)

    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @override_settings(
        ASR_MALAYALAM_PRIMARY_MODEL="medium",
        ASR_MALAYALAM_DEVICE="cpu",
        ASR_MALAYALAM_COMPUTE_TYPE="int8",
    )
    def test_malayalam_primary_model_uses_configured_compute_type(
        self,
        mock_model,
        mock_pass,
    ):
        mock_model.return_value = (
            object(),
            True,
            {
                "configured_model_name": "vrclc/Whisper-small-Malayalam",
                "resolved_model_name": "C:\\cached\\ctranslate2\\vrclc-Whisper-small-Malayalam",
                "model_family": "transformers",
                "model_load_seconds": 0.1,
            },
        )
        mock_pass.return_value = {
            "text": "Malayalam transcript with enough words to pass the quality gate safely for compute type coverage.",
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "Malayalam transcript with enough words to pass the quality gate safely for compute type coverage."}],
            "word_timestamps": [{"word": "Malayalam", "probability": 0.55}] * 16,
            "language": "ml",
            "language_probability": 0.95,
            "timing": {"first_token_latency_seconds": 0.1, "decode_seconds": 1.0},
        }

        result = _transcribe_with_faster_whisper_model("demo.wav", "youtube", "ml", "vrclc/Whisper-small-Malayalam", allow_force_large_v3=False)

        self.assertEqual(mock_model.call_args[0][2], "int8")
        self.assertEqual(result.get("metadata", {}).get("actual_local_compute_type"), "int8")
        self.assertEqual(result.get("metadata", {}).get("forced_transcription_language"), "ml")
        self.assertEqual(result.get("metadata", {}).get("asr_task_used"), "transcribe")

    @patch("videos.utils._get_local_whisper_model")
    @patch("videos.utils._ensure_malayalam_ctranslate2_model")
    @override_settings(
        ASR_MALAYALAM_PRIMARY_MODEL="medium",
        ASR_MALAYALAM_MODEL_FAMILY="auto",
    )
    def test_builtin_medium_primary_does_not_trigger_malayalam_model_conversion(self, mock_convert, mock_get_model):
        mock_get_model.return_value = (object(), True)
        _, _, meta = _get_local_whisper_model_with_meta("medium", "cpu", "int8")
        mock_convert.assert_not_called()
        self.assertEqual(meta.get("resolved_model_name"), "medium")

    @patch("videos.utils._transcribe_with_faster_whisper_model")
    @override_settings(
        ASR_MALAYALAM_PRIMARY_MODEL="",
        ASR_MALAYALAM_FAST_PRIMARY_MODEL="small",
        WHISPER_MODEL_MALAYALAM_SECONDARY="smcproject/vegam-whisper-medium-ml",
        WHISPER_MODEL_MALAYALAM_FALLBACK="large-v3",
    )
    def test_malayalam_fast_primary_model_is_used_before_fallback(self, mock_core):
        mock_core.return_value = {
            "text": "ഇത് പരീക്ഷയ്ക്ക് തയ്യാറാകാൻ സഹായിക്കുന്ന വ്യക്തമായ മലയാളം വിശദീകരണമാണ്.",
            "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "ഇത് പരീക്ഷയ്ക്ക് തയ്യാറാകാൻ സഹായിക്കുന്ന വ്യക്തമായ മലയാളം വിശദീകരണമാണ്."}],
            "word_timestamps": [{"word": "ഇത്", "probability": 0.86}, {"word": "മലയാളം", "probability": 0.84}] * 6,
            "language": "ml",
            "metadata": {
                "actual_local_model_name": "small",
                "model_reused": True,
            },
        }

        result = _transcribe_with_faster_whisper("demo.wav", "youtube", "ml")

        self.assertEqual(mock_core.call_args_list[0].kwargs["model_size"], "small")
        self.assertEqual(result.get("metadata", {}).get("actual_local_model_name"), "small")
        self.assertFalse(result.get("metadata", {}).get("fallback_triggered"))
        self.assertFalse(result.get("metadata", {}).get("large_fallback_executed"))

    @patch("videos.utils._transcribe_with_faster_whisper_model")
    @override_settings(
        ASR_MALAYALAM_PRIMARY_MODEL="vrclc/Whisper-small-Malayalam",
        ASR_MALAYALAM_FAST_PRIMARY_MODEL="",
        WHISPER_MODEL_MALAYALAM_SECONDARY="",
        WHISPER_MODEL_MALAYALAM_FALLBACK="large-v3",
        ASR_MALAYALAM_ENABLE_LARGE_FALLBACK=True,
    )
    def test_malayalam_falls_back_to_large_v3_only_if_quality_gate_fails(self, mock_core):
        weak_payload = {
            "text": "too short",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "too short"}],
            "language": "ml",
            "metadata": {
                "actual_local_model_name": "smcproject/vegam-whisper-medium-ml-int8",
                "configured_model_name": "vrclc/Whisper-small-Malayalam",
                "model_reused": True,
            },
        }
        strong_payload = {
            "text": "This Malayalam fallback transcript has enough useful content to pass quality safely.",
            "segments": [{"id": 0, "start": 0.0, "end": 3.0, "text": "This Malayalam fallback transcript has enough useful content to pass quality safely."}],
            "language": "ml",
            "metadata": {
                "actual_local_model_name": "large-v3",
                "model_reused": False,
            },
        }
        mock_core.side_effect = [weak_payload, strong_payload]

        result = _transcribe_with_faster_whisper("demo.wav", "youtube", "ml")

        self.assertEqual(mock_core.call_args_list[0].kwargs["model_size"], "vrclc/Whisper-small-Malayalam")
        self.assertEqual(mock_core.call_args_list[1].kwargs["model_size"], "large-v3")
        self.assertTrue(result.get("metadata", {}).get("fallback_triggered"))
        self.assertEqual(result.get("metadata", {}).get("fallback_reason"), "quality_gate_failed")
        self.assertEqual(result.get("metadata", {}).get("actual_local_model_name"), "large-v3")

    @patch("videos.utils._transcribe_with_faster_whisper_model")
    @override_settings(
        ASR_MALAYALAM_PRIMARY_MODEL="large-v2",
        ASR_MALAYALAM_FAST_PRIMARY_MODEL="",
        WHISPER_MODEL_MALAYALAM_SECONDARY="",
        WHISPER_MODEL_MALAYALAM_FALLBACK="large-v2",
        ASR_MALAYALAM_ENABLE_LARGE_FALLBACK=True,
    )
    def test_failed_malayalam_chunk_does_not_retry_same_model(self, mock_core):
        mock_core.side_effect = RuntimeError("Transcript appears empty or repeated. Audio may be silent or blocked.")

        with self.assertRaisesRegex(RuntimeError, "empty or repeated"):
            _transcribe_with_faster_whisper("demo.wav", "youtube", "ml")

        self.assertEqual(mock_core.call_count, 1)
        self.assertEqual(mock_core.call_args_list[0].kwargs["model_size"], "large-v2")

    @patch("videos.utils._transcribe_with_faster_whisper_model")
    @override_settings(
        ASR_MALAYALAM_PRIMARY_MODEL="vrclc/Whisper-small-Malayalam",
        WHISPER_MODEL_MALAYALAM_SECONDARY="",
        WHISPER_MODEL_MALAYALAM_FALLBACK="large-v3",
        ASR_MALAYALAM_ENABLE_LARGE_FALLBACK=False,
    )
    def test_malayalam_large_fallback_disabled_keeps_first_pass(self, mock_core):
        weak_payload = {
            "text": "too short",
            "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "too short"}],
            "word_timestamps": [{"word": "too", "probability": 0.2}] * 2,
            "language": "ml",
            "metadata": {
                "actual_local_model_name": "smcproject/vegam-whisper-medium-ml-int8",
                "configured_model_name": "vrclc/Whisper-small-Malayalam",
                "model_reused": True,
            },
        }
        mock_core.return_value = weak_payload

        result = _transcribe_with_faster_whisper("demo.wav", "youtube", "ml")

        self.assertEqual(mock_core.call_count, 1)
        self.assertEqual(result.get("metadata", {}).get("actual_local_model_name"), "vrclc/Whisper-small-Malayalam")
        self.assertFalse(result.get("metadata", {}).get("large_fallback_executed"))
        self.assertEqual(result.get("metadata", {}).get("large_fallback_skipped_reason"), "large_fallback_disabled")

    @patch("videos.utils._transcribe_with_faster_whisper_model")
    @override_settings(
        ASR_MALAYALAM_PRIMARY_MODEL="vrclc/Whisper-small-Malayalam",
        WHISPER_MODEL_MALAYALAM_SECONDARY="smcproject/vegam-whisper-medium-ml",
        ASR_MALAYALAM_FAST_MODE=True,
        ASR_MALAYALAM_ENABLE_LARGE_FALLBACK=False,
        ENABLE_ABSTRACTIVE_SUMMARY=False,
    )
    def test_malayalam_fast_path_transcript_flows_into_summary(self, mock_core):
        mock_core.return_value = {
            "text": "Malayalam tutorial transcript with enough usable content to continue into summary generation without waiting for a heavier fallback pass.",
            "segments": [{"id": 0, "start": 0.0, "end": 6.0, "text": "Malayalam tutorial transcript with enough usable content to continue into summary generation without waiting for a heavier fallback pass."}],
            "word_timestamps": [{"word": "Malayalam", "probability": 0.35}] * 18,
            "language": "ml",
            "metadata": {
                "actual_local_model_name": "smcproject/vegam-whisper-medium-ml-int8",
                "configured_model_name": "vrclc/Whisper-small-Malayalam",
                "model_reused": True,
            },
        }

        result = _transcribe_with_faster_whisper("demo.wav", "youtube", "ml")
        summary = summarize_text(
            text=result["text"],
            summary_type="short",
            output_language="ml",
            source_language="ml",
        )

        self.assertEqual(mock_core.call_count, 1)
        self.assertTrue(result.get("metadata", {}).get("first_pass_accepted"))
        self.assertTrue(summary.get("content") or summary.get("summary_out"))

    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @override_settings(
        ASR_MALAYALAM_PROMPT_ENABLED=True,
        ASR_MALAYALAM_LOCAL_INITIAL_PROMPT_ENABLED=True,
        ASR_MALAYALAM_ENABLE_FULL_RETRY=False,
        ASR_MALAYALAM_FAST_MODE=False,
    )
    def test_malayalam_first_pass_acceptable_quality_skips_retry(
        self,
        mock_model,
        mock_pass,
    ):
        mock_model.return_value = (object(), True, {"model_load_seconds": 0.2, "model_reused": True})
        mock_pass.return_value = {
            "text": "ഈ മലയാളം ട്രാൻസ്ക്രിപ്റ്റ് മതിയായ ഉള്ളടക്കത്തോടെ വ്യക്തമായി തുടരുന്നു, അതിനാൽ വീണ്ടും പൂർണ്ണ റൺ വേണ്ട.",
            "segments": [{"id": 0, "start": 0.0, "end": 10.0, "text": "ഈ മലയാളം ട്രാൻസ്ക്രിപ്റ്റ് മതിയായ ഉള്ളടക്കത്തോടെ വ്യക്തമായി തുടരുന്നു, അതിനാൽ വീണ്ടും പൂർണ്ണ റൺ വേണ്ട."}],
            "word_timestamps": [{"word": "ഈ", "probability": 0.55}, {"word": "ട്രാൻസ്ക്രിപ്റ്റ്", "probability": 0.56}],
            "language": "ml",
            "language_probability": 0.95,
            "timing": {"first_token_latency_seconds": 0.1, "decode_seconds": 1.0},
        }

        result = _transcribe_with_faster_whisper_model("demo.wav", "youtube", "ml", "small", allow_force_large_v3=False)

        self.assertEqual(mock_pass.call_count, 1)
        self.assertEqual(mock_pass.call_args.kwargs["language"], "ml")
        self.assertEqual(mock_pass.call_args.kwargs["task"], "transcribe")
        self.assertEqual(mock_pass.call_args.kwargs["initial_prompt"], build_malayalam_local_prompt())
        self.assertTrue(result["metadata"]["retry_considered"])
        self.assertFalse(result["metadata"]["retry_executed"])
        self.assertEqual(result["metadata"]["total_asr_passes"], 1)


class MalayalamHardeningTests(SimpleTestCase):
    def test_malayalam_router_raises_if_groq_selected(self):
        with self.assertRaises(AssertionError) as ctx:
            _assert_malayalam_provider_is_not_groq("groq")
        self.assertIn("ML_POLICY_VIOLATION", str(ctx.exception))
        self.assertIn("local Faster-Whisper", str(ctx.exception))

    def test_startup_rejects_purity_threshold_below_minimum(self):
        from django.core.exceptions import ImproperlyConfigured
        from videoiq import settings as app_settings

        with self.assertRaises(ImproperlyConfigured) as ctx:
            with override_settings(ASR_MALAYALAM_MIN_SCRIPT_PURITY=0.70):
                app_settings.validate_malayalam_settings()
        self.assertIn("0.85", str(ctx.exception))
        self.assertIn("English leakage", str(ctx.exception))

    def test_startup_accepts_purity_threshold_at_minimum(self):
        from videoiq import settings as app_settings

        with override_settings(ASR_MALAYALAM_MIN_SCRIPT_PURITY=0.85):
            try:
                app_settings.validate_malayalam_settings()
            except Exception as exc:
                self.fail(f"validate_malayalam_settings() raised unexpectedly at 0.85: {exc}")


class MalayalamModelConsistencyTests(TestCase):
    @override_settings(ASR_MALAYALAM_PRIMARY_MODEL="large-v2", ASR_MALAYALAM_SECOND_PASS_MODEL="")
    def test_malayalam_retry_model_follows_primary_setting(self):
        self.assertEqual(router_get_malayalam_model(), "large-v2")
        self.assertEqual(_get_malayalam_retry_model(), "large-v2")
        self.assertEqual(_malayalam_second_pass_model(), "large-v2")

    @patch("videos.utils._transcribe_with_faster_whisper_model")
    @override_settings(ASR_MALAYALAM_PRIMARY_MODEL="large-v2")
    def test_hardcoded_large_v3_helper_now_reads_from_setting(self, mock_transcribe):
        mock_transcribe.return_value = {
            "text": "ok",
            "segments": [],
            "language": "ml",
            "metadata": {},
        }

        _transcribe_with_malayalam_local("demo.wav", "youtube")

        self.assertEqual(mock_transcribe.call_args.kwargs["model_size"], "large-v2")
        self.assertFalse(mock_transcribe.call_args.kwargs["allow_force_large_v3"])

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    @override_settings(ASR_MALAYALAM_PRIMARY_MODEL="large-v3")
    def test_vram_guard_blocks_large_v3_on_low_memory_gpu(self, _mock_cuda_available, mock_device_props):
        from django.core.exceptions import ImproperlyConfigured
        from videoiq import settings as app_settings

        mock_device_props.return_value.total_memory = int(4 * 1e9)

        with self.assertRaises(ImproperlyConfigured) as ctx:
            app_settings.validate_malayalam_settings()

        self.assertIn("large-v2", str(ctx.exception))

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    @override_settings(ASR_MALAYALAM_PRIMARY_MODEL="large-v2")
    def test_vram_guard_passes_large_v2_on_low_memory_gpu(self, _mock_cuda_available, mock_device_props):
        from videoiq import settings as app_settings

        mock_device_props.return_value.total_memory = int(4 * 1e9)

        try:
            app_settings.validate_malayalam_settings()
        except Exception as exc:
            self.fail(f"validate_malayalam_settings() raised unexpectedly for large-v2 on low VRAM: {exc}")

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    @override_settings(ASR_MALAYALAM_PRIMARY_MODEL="large-v3")
    def test_vram_guard_passes_large_v3_on_sufficient_vram(self, _mock_cuda_available, mock_device_props):
        from videoiq import settings as app_settings

        mock_device_props.return_value.total_memory = int(8 * 1e9)

        try:
            app_settings.validate_malayalam_settings()
        except Exception as exc:
            self.fail(f"validate_malayalam_settings() raised unexpectedly for large-v3 on sufficient VRAM: {exc}")


class MalayalamFastFailTests(SimpleTestCase):
    def test_zero_malayalam_evidence_skips_second_pass(self):
        payload = {
            "text": "English leakage only.",
            "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "English leakage only."}],
            "language": "ml",
            "language_probability": 0.98,
            "metadata": {
                "first_pass_accepted": False,
                "first_pass_accept_reason": "malayalam_script_evidence_too_low",
                "quality_score": 0.42,
                "lexical_trust": 0.0,
                "readability": 0.3,
                "wrong_script_burden": 0.0,
                "contamination_burden": 1.0,
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "dominant_script_final": "latin",
                "detected_language": "ml",
                "detected_language_confidence": 0.98,
                "visible_malayalam_char_ratio": 0.0,
                "word_count": 3,
                "garble_score": 0.0,
            },
        }
        route_decision = {
            "malayalam_asr_strategy": "quality_first",
            "second_pass_enabled": True,
        }

        decision = _should_attempt_malayalam_second_pass(payload, route_decision)

        self.assertFalse(decision["attempt_second_pass"])
        self.assertEqual(decision["blocked_reason"], "zero_malayalam_evidence_skips_second_pass")

    @patch("videos.asr_router._transcribe_with_groq_whisper")
    @patch("videos.asr_router._choose_primary_engine")
    @patch("videos.asr_router._audio_file_size_bytes", return_value=1024)
    @patch("videos.asr_router._audio_duration_seconds", return_value=30.0)
    @patch("videos.asr_router._preprocess_audio_for_asr", return_value="tmp.wav")
    @patch("videos.asr_router._transcribe_with_local_whisper")
    def test_cuda_oom_on_malayalam_skips_groq_fallback(
        self,
        mock_local,
        _mock_preprocess,
        _mock_duration,
        _mock_size,
        mock_choose,
        mock_groq,
    ):
        mock_choose.return_value = {
            "engine": "whisper_local",
            "reason": "malayalam_quality_primary",
            "model": "large-v2",
            "fallback_chain": ["whisper_local"],
            "score": 0.9,
        }
        mock_local.side_effect = RuntimeError("CUDA failed with error out of memory")

        with self.assertRaises(RuntimeError) as ctx:
            transcribe_video_router(
                audio_path="demo.wav",
                source_type="youtube",
                requested_language="ml",
            )

        self.assertIn("out of memory", str(ctx.exception).lower())
        mock_groq.assert_not_called()
        mock_local.assert_called_once_with("tmp.wav", "youtube", "ml")

    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @override_settings(
        ASR_MALAYALAM_PROMPT_ENABLED=False,
        ASR_MALAYALAM_LOCAL_INITIAL_PROMPT_ENABLED=True,
        ASR_MALAYALAM_ENABLE_FULL_RETRY=False,
        ASR_MALAYALAM_FAST_MODE=False,
    )
    def test_prompt_disabled_when_flag_false(
        self,
        mock_model,
        mock_pass,
    ):
        mock_model.return_value = (object(), True, {"model_load_seconds": 0.2, "model_reused": True})
        mock_pass.return_value = {
            "text": "à´ˆ à´®à´²à´¯à´¾à´³à´‚ à´Ÿàµà´°à´¾àµ»à´¸àµà´•àµà´°à´¿à´ªàµà´±àµà´±àµ à´®à´¤à´¿à´¯à´¾à´¯ à´‰à´³àµà´³à´Ÿà´•àµà´•à´¤àµà´¤àµ‹à´Ÿàµ† à´µàµà´¯à´•àµà´¤à´®à´¾à´¯à´¿ à´¤àµà´Ÿà´°àµà´¨àµà´¨àµ, à´…à´¤à´¿à´¨à´¾àµ½ à´µàµ€à´£àµà´Ÿàµà´‚ à´ªàµ‚àµ¼à´£àµà´£ à´±àµº à´µàµ‡à´£àµà´Ÿ.",
            "segments": [{"id": 0, "start": 0.0, "end": 10.0, "text": "à´ˆ à´®à´²à´¯à´¾à´³à´‚ à´Ÿàµà´°à´¾àµ»à´¸àµà´•àµà´°à´¿à´ªàµà´±àµà´±àµ à´®à´¤à´¿à´¯à´¾à´¯ à´‰à´³àµà´³à´Ÿà´•àµà´•à´¤àµà´¤àµ‹à´Ÿàµ† à´µàµà´¯à´•àµà´¤à´®à´¾à´¯à´¿ à´¤àµà´Ÿà´°àµà´¨àµà´¨àµ, à´…à´¤à´¿à´¨à´¾àµ½ à´µàµ€à´£àµà´Ÿàµà´‚ à´ªàµ‚àµ¼à´£àµà´£ à´±àµº à´µàµ‡à´£àµà´Ÿ."}],
            "word_timestamps": [{"word": "à´ˆ", "probability": 0.55}, {"word": "à´Ÿàµà´°à´¾àµ»à´¸àµà´•àµà´°à´¿à´ªàµà´±àµà´±àµ", "probability": 0.56}],
            "language": "ml",
            "language_probability": 0.95,
            "timing": {"first_token_latency_seconds": 0.1, "decode_seconds": 1.0},
        }

        _transcribe_with_faster_whisper_model("demo.wav", "youtube", "ml", "small", allow_force_large_v3=False)

        self.assertIsNone(mock_pass.call_args.kwargs["initial_prompt"])


class ChunkManifestRoutingTests(SimpleTestCase):
    @patch("videos.asr_router._transcribe_video_router_single")
    def test_chunk_manifest_locks_confident_malayalam_after_first_chunk(
        self,
        mock_single,
    ):
        mock_single.side_effect = [
            {
                "text": "segment one",
                "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "segment one"}],
                "word_timestamps": [],
                "language": "ml",
                "language_probability": 0.97,
                "metadata": {},
            },
            {
                "text": "segment two",
                "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "segment two"}],
                "word_timestamps": [],
                "language": "ml",
                "language_probability": 0.95,
                "metadata": {},
            },
        ]
        chunks = [
            ChunkMetadata(chunk_id=0, start_s=0.0, end_s=10.0, path="chunk0.wav", duration_s=10.0),
            ChunkMetadata(chunk_id=1, start_s=10.0, end_s=20.0, path="chunk1.wav", duration_s=10.0),
        ]

        transcribe_video_router(
            source_type="upload",
            requested_language="auto",
            chunks=chunks,
        )

        first_call = mock_single.call_args_list[0].kwargs
        second_call = mock_single.call_args_list[1].kwargs
        self.assertEqual(first_call["requested_language"], "auto")
        self.assertEqual(second_call["requested_language"], "ml")

    @patch("videos.asr_router._transcribe_video_router_single")
    def test_chunk_manifest_passes_total_duration_for_routing(
        self,
        mock_single,
    ):
        mock_single.side_effect = [
            {
                "text": "segment one",
                "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "segment one"}],
                "word_timestamps": [],
                "language": "ml",
                "language_probability": 0.97,
                "metadata": {},
            },
            {
                "text": "segment two",
                "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "segment two"}],
                "word_timestamps": [],
                "language": "ml",
                "language_probability": 0.95,
                "metadata": {},
            },
        ]
        chunks = [
            ChunkMetadata(chunk_id=0, start_s=0.0, end_s=30.0, path="chunk0.wav", duration_s=30.0),
            ChunkMetadata(chunk_id=1, start_s=30.0, end_s=540.0, path="chunk1.wav", duration_s=510.0),
        ]

        transcribe_video_router(
            source_type="upload",
            requested_language="auto",
            chunks=chunks,
        )

        first_call = mock_single.call_args_list[0].kwargs
        second_call = mock_single.call_args_list[1].kwargs
        self.assertEqual(first_call["routing_duration_seconds"], 540.0)
        self.assertEqual(second_call["routing_duration_seconds"], 540.0)

    @patch("videos.asr_router._transcribe_with_local_whisper")
    @patch("videos.asr_router._choose_primary_engine")
    @patch("videos.asr_router._audio_file_size_bytes", return_value=1024)
    @patch("videos.asr_router._audio_duration_seconds", return_value=12.0)
    @patch("videos.asr_router._preprocess_audio_for_asr")
    def test_chunk_manifest_path_skips_ffmpeg_reencode(
        self,
        mock_preprocess,
        _mock_duration,
        _mock_size,
        mock_choose,
        mock_local,
    ):
        mock_preprocess.side_effect = lambda audio_path, language_hint="", already_preprocessed=False: audio_path
        mock_choose.return_value = {
            "engine": "whisper_local",
            "reason": "forced_test",
            "model": "large-v3",
            "fallback_chain": [],
            "score": 1.0,
            "latency_budget_seconds": 0.0,
            "detection_confidence": 1.0,
            "file_size_bytes": 1024,
        }
        mock_local.return_value = {
            "text": "മലയാളം ഉള്ളടക്കം",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "മലയാളം ഉള്ളടക്കം"}],
            "language": "ml",
            "metadata": {},
        }
        chunks = [ChunkMetadata(chunk_id=0, start_s=0.0, end_s=10.0, path="chunk0.wav", duration_s=10.0)]

        transcribe_video_router(
            source_type="upload",
            requested_language="ml",
            chunks=chunks,
        )

        self.assertTrue(mock_preprocess.called)
        self.assertTrue(mock_preprocess.call_args.kwargs["already_preprocessed"])

        mock_preprocess.reset_mock()
        transcribe_video_router(
            audio_path="single.wav",
            source_type="upload",
            requested_language="ml",
        )
        self.assertFalse(mock_preprocess.call_args.kwargs["already_preprocessed"])

    @patch("videos.asr_router._transcribe_with_local_whisper")
    @patch("videos.asr_router._choose_primary_engine")
    @patch("videos.asr_router._audio_file_size_bytes", return_value=1024)
    @patch("videos.asr_router._audio_duration_seconds", return_value=12.0)
    @patch("videos.asr_router._preprocess_audio_for_asr")
    def test_single_file_path_still_preprocesses_unconditionally(
        self,
        mock_preprocess,
        _mock_duration,
        _mock_size,
        mock_choose,
        mock_local,
    ):
        mock_preprocess.side_effect = lambda audio_path, language_hint="", already_preprocessed=False: audio_path
        mock_choose.return_value = {
            "engine": "whisper_local",
            "reason": "forced_test",
            "model": "large-v3",
            "fallback_chain": [],
            "score": 1.0,
            "latency_budget_seconds": 0.0,
            "detection_confidence": 1.0,
            "file_size_bytes": 1024,
        }
        mock_local.return_value = {
            "text": "മലയാളം ഉള്ളടക്കം",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "മലയാളം ഉള്ളടക്കം"}],
            "language": "ml",
            "metadata": {},
        }

        transcribe_video_router(
            audio_path="single.wav",
            source_type="upload",
            requested_language="ml",
        )

        self.assertTrue(mock_preprocess.called)
        self.assertFalse(mock_preprocess.call_args.kwargs["already_preprocessed"])

    @override_settings(
        ASR_MALAYALAM_SEGMENT_LOCAL_RESCUE_ALLOW_CPU=False,
        ASR_MALAYALAM_COMPUTE_TYPE="float16",
    )
    def test_cpu_segment_local_rescue_is_disabled_by_default(self):
        should_attempt, reason = should_attempt_malayalam_local_segment_rescue()
        self.assertFalse(should_attempt)
        self.assertEqual(reason, "cpu_segment_local_rescue_disabled")

    @patch("videos.asr_router._analyze_malayalam_asr_payload")
    @patch("videos.asr_router._transcribe_video_router_single")
    def test_per_chunk_fidelity_gate_applied_independently(
        self,
        mock_single,
        mock_analyze,
    ):
        chunks = [
            ChunkMetadata(chunk_id=0, start_s=0.0, end_s=10.0, path="chunk0.wav", duration_s=10.0),
            ChunkMetadata(chunk_id=1, start_s=10.0, end_s=20.0, path="chunk1.wav", duration_s=10.0),
            ChunkMetadata(chunk_id=2, start_s=20.0, end_s=28.0, path="chunk2.wav", duration_s=8.0),
        ]
        mock_single.side_effect = [
            {"text": "ആദ്യ ഭാഗം", "segments": [{"id": 0, "start": 0.5, "end": 2.0, "text": "ആദ്യ ഭാഗം"}], "language": "ml", "metadata": {}},
            {"text": "english leakage", "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "english leakage"}], "language": "ml", "metadata": {}},
            {"text": "മൂന്നാം ഭാഗം", "segments": [{"id": 0, "start": 1.0, "end": 3.0, "text": "മൂന്നാം ഭാഗം"}], "language": "ml", "metadata": {}},
        ]
        mock_analyze.side_effect = [
            {"source_language_fidelity_failed": False},
            {"source_language_fidelity_failed": True},
            {"source_language_fidelity_failed": False},
        ]

        result = transcribe_video_router(
            source_type="upload",
            requested_language="ml",
            chunks=chunks,
        )

        self.assertEqual(len(result["segments"]), 2)
        self.assertIn("ആദ്യ ഭാഗം", result["text"])
        self.assertIn("മൂന്നാം ഭാഗം", result["text"])
        self.assertNotIn("english leakage", result["text"])
        self.assertTrue(result["has_fidelity_gaps"])
        self.assertTrue(result["metadata"]["chunk_manifest_used"])

    @patch("videos.utils.analyze_malayalam_source_fidelity")
    @patch("videos.utils.classify_malayalam_segment_type")
    @patch("videos.utils._should_accept_usable_malayalam_first_pass")
    @patch("videos.utils.build_malayalam_transcript_trust")
    @patch("videos.utils._extract_asr_metrics", return_value={"other_indic_ratio": 0.05})
    def test_low_trust_malayalam_script_chunk_is_marked_recoverable(
        self,
        _metrics,
        mock_trust,
        mock_first_pass,
        mock_classify,
        mock_fidelity,
    ):
        mock_trust.return_value = {
            "total_segments": 1,
            "wrong_script_segments": 0,
            "malayalam_token_coverage": 0.22,
            "lexical_trust_score": 0.06,
            "overall_readability": 0.09,
            "dominant_script_final": "malayalam",
        }
        mock_first_pass.return_value = {
            "first_pass_accepted": False,
            "first_pass_accept_reason": "",
            "quality_score": 0.12,
            "garbled_detector_score": 0.13,
        }
        mock_classify.return_value = {"type": "corrupted_or_non_malayalam", "score": {}}
        mock_fidelity.return_value = {
            "source_language_fidelity_failed": False,
            "transcript_fidelity_state": "uncertain_malayalam_fidelity",
            "catastrophic_wrong_script_failure": False,
            "recoverable_malayalam_fidelity_gap": False,
            "suspicious_substitution_burden": 0.0,
            "dominant_non_malayalam_visible_ratio": 0.0,
            "visible_malayalam_char_ratio": 0.69,
            "nonempty_signal": True,
        }

        analysis = _analyze_malayalam_asr_payload(
            {
                "text": "അപിക വ്യ്യു ര്യു ര്യു ര്യു",
                "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "അപിക വ്യ്യു ര്യു ര്യു ര്യു"}],
                "language": "ml",
                "language_probability": 0.95,
                "metadata": {"detected_language_confidence": 0.95},
            }
        )

        self.assertEqual(analysis["quality_class"], "recoverable_but_weak")
        self.assertEqual(analysis["reason"], "low_trust_malayalam_script_retry_candidate")
        self.assertTrue(analysis["low_trust_malayalam_script_candidate"])

    @override_settings(ASR_MALAYALAM_CHUNK_MAX_DURATION_SECONDS=18)
    @patch("videos.tasks.chunk_on_silence_boundaries")
    @patch("videos.tasks.condition_audio_for_asr")
    @patch("videos.tasks.normalize_to_lufs")
    def test_prepare_audio_pipeline_uses_shorter_malayalam_chunk_budget(
        self,
        mock_normalize,
        mock_condition,
        mock_chunk,
    ):
        mock_normalize.return_value = type(
            "NormalizationResultStub",
            (),
            {"output_path": "normalized.wav", "input_lufs": -16.0, "output_lufs": -16.0, "was_reencoded": False},
        )()
        mock_condition.return_value = type(
            "ConditioningResultStub",
            (),
            {
                "output_path": "conditioned.wav",
                "was_noop": True,
                "dynaudnorm_applied": True,
                "speech_band_filter_applied": True,
            },
        )()
        mock_chunk.return_value = []

        _, prep_meta = _prepare_audio_for_pipeline("input.wav", transcription_language="ml")

        self.assertEqual(prep_meta["chunk_max_duration_seconds"], 18.0)
        self.assertEqual(mock_chunk.call_args.kwargs["max_chunk_duration_s"], 18.0)

    @patch("videos.asr_router._analyze_malayalam_asr_payload", return_value={"source_language_fidelity_failed": False})
    @patch("videos.asr_router._transcribe_video_router_single")
    def test_segment_timestamps_offset_correctly_to_full_timeline(
        self,
        mock_single,
        _mock_analyze,
    ):
        mock_single.return_value = {
            "text": "ചങ്ക് ടെക്സ്റ്റ്",
            "segments": [{"id": 0, "start": 2.5, "end": 4.0, "text": "ചങ്ക് ടെക്സ്റ്റ്"}],
            "language": "ml",
            "metadata": {},
        }
        result = transcribe_video_router(
            source_type="upload",
            requested_language="ml",
            chunks=[ChunkMetadata(chunk_id=0, start_s=30.0, end_s=40.0, path="chunk.wav", duration_s=10.0)],
        )

        self.assertEqual(result["segments"][0]["start"], 32.5)
        self.assertEqual(result["segments"][0]["end"], 34.0)

    @patch("videos.asr_router._analyze_malayalam_asr_payload", return_value={"source_language_fidelity_failed": True})
    @patch("videos.asr_router._transcribe_video_router_single")
    def test_all_chunks_failed_produces_empty_transcript_not_exception(
        self,
        mock_single,
        _mock_analyze,
    ):
        mock_single.side_effect = [
            {"text": "english one", "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "english one"}], "language": "ml", "metadata": {}},
            {"text": "english two", "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "english two"}], "language": "ml", "metadata": {}},
        ]
        result = transcribe_video_router(
            source_type="upload",
            requested_language="ml",
            chunks=[
                ChunkMetadata(chunk_id=0, start_s=0.0, end_s=10.0, path="chunk0.wav", duration_s=10.0),
                ChunkMetadata(chunk_id=1, start_s=10.0, end_s=20.0, path="chunk1.wav", duration_s=10.0),
            ],
        )

        self.assertEqual(result["segments"], [])
        self.assertEqual(result["text"], "")
        self.assertTrue(result["has_fidelity_gaps"])
        self.assertTrue(result["metadata"]["source_language_fidelity_failed"])

    def test_terminal_malayalam_failure_payload_is_not_considered_valid_chunk(self):
        payload = _build_terminal_malayalam_failure_payload(
            route_decision={"model": "medium"},
            route_reason="malayalam_quality_primary",
            failure_reason="Transcript appears empty or repeated. Audio may be silent or blocked.",
            detection_confidence=1.0,
        )

        self.assertFalse(_is_valid_chunk_payload(payload, "ml"))
        self.assertEqual(payload["metadata"]["actual_local_model_name"], "medium")

    @patch(
        "videos.asr_router._analyze_malayalam_asr_payload",
        return_value={
            "source_language_fidelity_failed": False,
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "readability": 0.05,
            "lexical_trust": 0.0,
        },
    )
    def test_low_evidence_malayalam_chunk_payload_is_rejected_from_stitching(self, _mock_analyze):
        payload = {
            "text": "അപിക വ്യ്യു ര്യു ര്യു ര്യു",
            "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "അപിക വ്യ്യു ര്യു ര്യു ര്യു"}],
            "language": "ml",
            "metadata": {},
        }

        self.assertFalse(_is_valid_chunk_payload(payload, "ml"))

    @patch("videos.asr_router._analyze_malayalam_asr_payload", return_value={"source_language_fidelity_failed": False})
    @patch("videos.asr_router._transcribe_video_router_single")
    def test_continuous_speech_fallback_chunk_stitches_correctly(
        self,
        mock_single,
        _mock_analyze,
    ):
        mock_single.return_value = {
            "text": "തുടർച്ചയായ പ്രസംഗം",
            "segments": [{"id": 0, "start": 0.0, "end": 29.5, "text": "തുടർച്ചയായ പ്രസംഗം"}],
            "language": "ml",
            "metadata": {},
        }
        result = transcribe_video_router(
            source_type="upload",
            requested_language="ml",
            chunks=[ChunkMetadata(chunk_id=0, start_s=0.0, end_s=30.0, path="chunk0.wav", duration_s=30.0)],
        )

        self.assertEqual(len(result["segments"]), 1)
        self.assertFalse(result["has_fidelity_gaps"])
        self.assertEqual(result["segments"][0]["end"], 29.5)

    def test_existing_malayalam_fidelity_gates_unchanged(self):
        from videoiq import settings as app_settings

        with self.assertRaises(AssertionError):
            _assert_malayalam_provider_is_not_groq("groq")
        with override_settings(ASR_MALAYALAM_MIN_SCRIPT_PURITY=0.85):
            app_settings.validate_malayalam_settings()

    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @patch("videos.utils._resolve_malayalam_runtime_device", return_value="cuda")
    @override_settings(
        ASR_MALAYALAM_FAST_MODE=True,
        ASR_MALAYALAM_FIRST_PASS_BEAM_SIZE=2,
        ASR_MALAYALAM_FIRST_PASS_BEST_OF=2,
    )
    def test_malayalam_fast_mode_uses_reduced_first_pass_decode_profile(
        self,
        _runtime_device,
        mock_model,
        mock_pass,
    ):
        mock_model.return_value = (object(), True, {"model_load_seconds": 0.2, "configured_model_name": "large-v3"})
        mock_pass.return_value = {
            "text": "Malayalam transcript with enough useful content to finish on the fast first pass without escalation.",
            "segments": [{"id": 0, "start": 0.0, "end": 9.0, "text": "Malayalam transcript with enough useful content to finish on the fast first pass without escalation."}],
            "word_timestamps": [{"word": "Malayalam", "probability": 0.55}] * 18,
            "language": "ml",
            "language_probability": 0.95,
            "timing": {"first_token_latency_seconds": 0.1, "decode_seconds": 1.0},
        }

        _transcribe_with_faster_whisper_model("demo.wav", "upload", "ml", "large-v3", allow_force_large_v3=False)

        self.assertEqual(mock_pass.call_args.kwargs["beam_size"], 2)
        self.assertEqual(mock_pass.call_args.kwargs["best_of"], 2)

    @patch("videos.utils._enhance_audio_for_speech", return_value="enhanced.wav")
    @patch("videos.utils.os.path.exists", return_value=False)
    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @override_settings(
        ASR_MALAYALAM_ENABLE_FULL_RETRY=True,
        ASR_MALAYALAM_FAST_MODE=False,
        ASR_MALAYALAM_RETRY_MIN_QUALITY=0.5,
    )
    def test_malayalam_retry_keeps_forced_malayalam_language(
        self,
        mock_model,
        mock_pass,
        _exists,
        _enhance,
    ):
        mock_model.return_value = (object(), False, {"model_load_seconds": 0.2})
        mock_pass.side_effect = [
            {
                "text": "Malayalam first pass transcript has enough tokens to avoid low content but still remains weak and repetitive for retry.",
                "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "Malayalam first pass transcript has enough tokens to avoid low content but still remains weak and repetitive for retry."}],
                "word_timestamps": [{"word": "Malayalam", "probability": 0.2}] * 14,
                "language": "ml",
                "language_probability": 0.7,
                "timing": {"first_token_latency_seconds": 0.1, "decode_seconds": 1.0},
            },
            {
                "text": "Malayalam retry transcript with enough useful content to justify the second pass safely.",
                "segments": [{"id": 0, "start": 0.0, "end": 8.0, "text": "Malayalam retry transcript with enough useful content to justify the second pass safely."}],
                "word_timestamps": [{"word": "Malayalam", "probability": 0.6}] * 14,
                "language": "ml",
                "language_probability": 0.95,
                "timing": {"first_token_latency_seconds": 0.1, "decode_seconds": 1.0},
            },
        ]

        _transcribe_with_faster_whisper_model("demo.wav", "upload", "ml", "large-v3", allow_force_large_v3=False)

        self.assertEqual(mock_pass.call_count, 2)
        self.assertEqual(mock_pass.call_args_list[1].kwargs["language"], "ml")

    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @override_settings(
        ASR_MALAYALAM_ENABLE_FULL_RETRY=True,
        ASR_MALAYALAM_FAST_MODE=True,
        ASR_MALAYALAM_RETRY_MIN_QUALITY=0.42,
    )
    def test_malayalam_low_confidence_but_above_fast_mode_threshold_skips_retry(
        self,
        mock_model,
        mock_pass,
    ):
        mock_model.return_value = (object(), False, {"model_load_seconds": 0.2})
        mock_pass.return_value = {
            "text": "Malayalam transcript with enough useful content to stay usable despite lower confidence from the first pass alone.",
            "segments": [{"id": 0, "start": 0.0, "end": 12.0, "text": "Malayalam transcript with enough useful content to stay usable despite lower confidence from the first pass alone."}],
            "word_timestamps": [{"word": "Malayalam", "probability": 0.3}] * 20,
            "language": "ml",
            "language_probability": 0.9,
            "timing": {"first_token_latency_seconds": 0.1, "decode_seconds": 1.0},
        }

        result = _transcribe_with_faster_whisper_model("demo.wav", "youtube", "ml", "small", allow_force_large_v3=False)

        self.assertEqual(mock_pass.call_count, 1)
        self.assertTrue(result["metadata"]["retry_considered"])
        self.assertFalse(result["metadata"]["retry_executed"])
        self.assertEqual(result["metadata"]["retry_skipped_reason"], "fast_mode_first_pass_acceptable")

    @patch("videos.utils._enhance_audio_for_speech", return_value="enhanced.wav")
    @patch("videos.utils.os.path.exists", return_value=False)
    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @override_settings(
        ASR_MALAYALAM_ENABLE_FULL_RETRY=True,
        ASR_MALAYALAM_FAST_MODE=False,
        ASR_MALAYALAM_RETRY_MIN_QUALITY=0.5,
    )
    def test_malayalam_critically_bad_transcript_executes_retry(
        self,
        mock_model,
        mock_pass,
        _exists,
        _enhance,
    ):
        mock_model.return_value = (object(), False, {"model_load_seconds": 0.2})
        mock_pass.side_effect = [
            {
                "text": "bad repeat bad repeat bad repeat",
                "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "bad repeat bad repeat bad repeat"}],
                "word_timestamps": [{"word": "bad", "probability": 0.1}] * 6,
                "language": "ml",
                "language_probability": 0.8,
                "timing": {"first_token_latency_seconds": 0.1, "decode_seconds": 1.0},
            },
            {
                "text": "This retried Malayalam transcript now contains enough useful content to justify using the second pass safely.",
                "segments": [{"id": 0, "start": 0.0, "end": 8.0, "text": "This retried Malayalam transcript now contains enough useful content to justify using the second pass safely."}],
                "word_timestamps": [{"word": "This", "probability": 0.6}] * 18,
                "language": "ml",
                "language_probability": 0.9,
                "timing": {"first_token_latency_seconds": 0.1, "decode_seconds": 1.2},
            },
        ]

        result = _transcribe_with_faster_whisper_model("demo.wav", "youtube", "ml", "small", allow_force_large_v3=False)

        self.assertEqual(mock_pass.call_count, 2)
        self.assertTrue(result["metadata"]["retry_considered"])
        self.assertTrue(result["metadata"]["retry_executed"])
        self.assertEqual(result["metadata"]["retry_decision_reason"], "critical_quality_failure")
        self.assertEqual(result["metadata"]["total_asr_passes"], 2)

    @patch("videos.utils._get_audio_duration_seconds", return_value=500.0)
    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @override_settings(
        ASR_MALAYALAM_ENABLE_FULL_RETRY=True,
        ASR_MALAYALAM_FAST_MODE=False,
        ASR_MALAYALAM_RETRY_MIN_QUALITY=0.5,
        ASR_MALAYALAM_SKIP_RETRY_ABOVE_DURATION_SECONDS=360,
    )
    def test_malayalam_long_video_skips_retry_due_to_duration_guardrail(
        self,
        mock_model,
        mock_pass,
        _duration,
    ):
        mock_model.return_value = (object(), False, {"model_load_seconds": 0.2})
        mock_pass.return_value = {
            "text": "This Malayalam transcript stays weak and repetitive enough to consider retry, but the duration guardrail should skip a second full pass safely.",
            "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "This Malayalam transcript stays weak and repetitive enough to consider retry, but the duration guardrail should skip a second full pass safely."}],
            "word_timestamps": [{"word": "This", "probability": 0.1}] * 20,
            "language": "ml",
            "language_probability": 0.8,
            "timing": {"first_token_latency_seconds": 0.1, "decode_seconds": 1.0},
        }

        result = _transcribe_with_faster_whisper_model("demo.wav", "youtube", "ml", "small", allow_force_large_v3=False)

        self.assertEqual(mock_pass.call_count, 1)
        self.assertFalse(result["metadata"]["retry_executed"])
        self.assertEqual(result["metadata"]["retry_skipped_reason"], "duration_guardrail")

    @patch("videos.utils._run_transcription_pass")
    @patch("videos.utils._get_local_whisper_model_with_meta")
    @override_settings(
        ASR_MALAYALAM_ENABLE_FULL_RETRY=False,
        ASR_MALAYALAM_FAST_MODE=True,
    )
    def test_malayalam_retry_metadata_is_recorded(
        self,
        mock_model,
        mock_pass,
    ):
        mock_model.return_value = (object(), True, {"model_load_seconds": 0.4})
        mock_pass.return_value = {
            "text": "short weak transcript words but still enough to trigger retry consideration for metadata coverage",
            "segments": [{"id": 0, "start": 0.0, "end": 8.0, "text": "short weak transcript words but still enough to trigger retry consideration for metadata coverage"}],
            "word_timestamps": [{"word": "short", "probability": 0.2}] * 14,
            "language": "ml",
            "language_probability": 0.9,
            "timing": {"first_token_latency_seconds": 0.2, "decode_seconds": 1.4},
        }

        result = _transcribe_with_faster_whisper_model("demo.wav", "youtube", "ml", "small", allow_force_large_v3=False)
        metadata = result["metadata"]

        self.assertIn("retry_considered", metadata)
        self.assertIn("retry_executed", metadata)
        self.assertIn("retry_skipped_reason", metadata)
        self.assertIn("malayalam_fast_mode", metadata)
        self.assertIn("total_asr_passes", metadata)
        self.assertIn("total_asr_seconds", metadata)

    @patch("videos.utils._garble_debug_snapshot")
    def test_high_quality_mixed_script_malayalam_can_bypass_over_strict_garble_rejection(self, mock_snapshot):
        mock_snapshot.return_value = {
            "garbled_score": 0.58,
            "script_distribution": {"malayalam": 0.46, "latin": 0.31, "devanagari": 0.23},
            "dominant_script": "malayalam",
            "dominant_script_ratio": 0.46,
            "active_scripts": ["malayalam", "latin", "devanagari"],
            "odd_token_ratio": 0.03,
            "preview": "safe mixed-script preview",
        }
        decision = _should_accept_malayalam_mixed_script_override(
            "ignored",
            {
                "quality_score": 0.95,
                "dominant_script_ratio": 0.46,
                "repeated_token_ratio": 0.0,
                "info_density": 0.9,
            },
        )
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "high_quality_mixed_script_override")

    @patch("videos.utils._garble_debug_snapshot")
    def test_truly_garbled_malayalam_mixed_script_still_rejected(self, mock_snapshot):
        mock_snapshot.return_value = {
            "garbled_score": 0.91,
            "script_distribution": {"malayalam": 0.21, "latin": 0.34, "devanagari": 0.29, "other": 0.16},
            "dominant_script": "latin",
            "dominant_script_ratio": 0.34,
            "active_scripts": ["malayalam", "latin", "devanagari", "other"],
            "odd_token_ratio": 0.37,
            "preview": "unsafe garbled preview",
        }
        decision = _should_accept_malayalam_mixed_script_override(
            "ignored",
            {
                "quality_score": 0.72,
                "dominant_script_ratio": 0.21,
                "repeated_token_ratio": 0.22,
                "info_density": 0.41,
            },
        )
        self.assertFalse(decision["accepted"])


class MultilingualSummaryFlowTests(SimpleTestCase):
    @override_settings(ENABLE_ABSTRACTIVE_SUMMARY=False)
    @patch("videos.utils._translate_text_with_groq")
    def test_non_english_transcript_summarized_from_canonical_and_returned_in_source_language(
        self,
        mock_translate,
    ):
        # Route all translation calls through deterministic markers.
        def _fake_translate(text, target_language, source_language="en", preserve_bullets=False):
            if target_language == source_language:
                return text
            return f"[{source_language}->{target_language}] {text}"

        mock_translate.side_effect = _fake_translate

        result = summarize_text(
            text="à´®àµ‚à´²à´­à´¾à´· à´Ÿàµ†à´•àµà´¸àµà´±àµà´±àµ",
            summary_type="full",
            output_language="auto",
            source_language="ml",
            canonical_text="This video explains exam strategy and confidence building.",
            canonical_language="en",
            summary_language_mode="same_as_transcript",
        )

        self.assertEqual(result.get("summary_language"), "ml")
        self.assertEqual(result.get("summary_source_language"), "ml")
        self.assertIn("[en->ml]", result.get("summary_out", ""))
        self.assertTrue(result.get("summary_en"))
        self.assertTrue(result.get("translation_used"))

    @override_settings(ENABLE_ABSTRACTIVE_SUMMARY=False)
    def test_bullet_summary_is_real_bullets(self):
        text = (
            "First the speaker explains the goal of the workflow. "
            "Then they generate an initial asset with an AI tool. "
            "Next they refine the output and export media assets. "
            "After that they assemble the website structure and visuals. "
            "They optimize loading and interaction behavior for usability. "
            "Finally they deploy the project and validate the result."
        )
        result = summarize_text(text=text, summary_type="bullet", output_language="en", source_language="en")
        lines = [ln.strip() for ln in (result.get("content") or "").splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(all(re.match(r"^[•-]\s", ln) for ln in lines))

    @override_settings(ENABLE_ABSTRACTIVE_SUMMARY=False)
    def test_short_summary_is_capped(self):
        text = " ".join(["This transcript describes a practical workflow and clear outcomes."] * 80)
        result = summarize_text(text=text, summary_type="short", output_language="en", source_language="en")
        words = len((result.get("content") or "").split())
        self.assertLessEqual(words, 60)

    @override_settings(SUMMARIZATION_PROVIDER="groq")
    @patch("videos.utils._summarize_full_density_with_groq", side_effect=AssertionError("full summary should not run for short summary"))
    @patch("videos.utils._build_summary_outline", return_value=(None, False))
    @patch("videos.utils._summarize_short_density_with_groq", return_value=("A concise summary of the workflow and outcome.", False))
    def test_short_summary_does_not_recurse_to_full_summary(
        self,
        _mock_short,
        _mock_outline,
        _mock_full,
    ):
        result = summarize_text(
            text="This transcript explains a workflow, the tools used, and the final result in clear steps.",
            summary_type="short",
            output_language="en",
            source_language="en",
        )
        self.assertIn("concise summary", result.get("content", "").lower())

    @override_settings(SUMMARIZATION_PROVIDER="hf", ENABLE_ABSTRACTIVE_SUMMARY=True, HF_BULLET_MAX_INPUT_WORDS=100)
    @patch("transformers.pipeline")
    def test_long_bullet_summary_skips_hf_pipeline_and_uses_safe_fallback(self, mock_pipeline):
        text = " ".join([
            "This transcript describes a long workflow with repeated steps, tools, deployment notes, and outcomes."
        ] * 80)
        result = summarize_text(text=text, summary_type="bullet", output_language="en", source_language="en")
        mock_pipeline.assert_not_called()
        lines = [ln.strip() for ln in (result.get("content") or "").splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(all(re.match(r"^[•-]\s", ln) for ln in lines))

    @override_settings(SUMMARIZATION_PROVIDER="hf", ENABLE_ABSTRACTIVE_SUMMARY=True, HF_SHORT_MAX_INPUT_WORDS=100)
    @patch("videos.utils._generate_semantic_short_summary", return_value="This tutorial explains the complete workflow and final deployment outcome.")
    @patch("transformers.pipeline")
    def test_long_short_summary_skips_hf_pipeline(self, mock_pipeline, _mock_semantic_short):
        text = " ".join([
            "This transcript describes a long workflow with multiple steps, tools, and final outcomes."
        ] * 90)
        result = summarize_text(text=text, summary_type="short", output_language="en", source_language="en")
        mock_pipeline.assert_not_called()
        self.assertTrue(result.get("content"))

    @override_settings(SUMMARIZATION_PROVIDER="hf", ENABLE_ABSTRACTIVE_SUMMARY=True, SUMMARIZATION_MODEL="facebook/bart-large-cnn")
    @patch("transformers.pipelines.pipeline")
    def test_hf_summary_model_uses_summarization_task_for_bart(self, mock_pipeline):
        _SUMMARY_PIPELINE_CACHE.clear()
        summarizer = lambda *args, **kwargs: [{"summary_text": "A clean summary of the lesson and the final advice."}]
        mock_pipeline.return_value = summarizer
        summarizer_obj, task_name, fallback_used, runtime_error = _load_hf_summary_pipeline("facebook/bart-large-cnn")
        self.assertIs(summarizer_obj, summarizer)
        self.assertEqual(task_name, "summarization")
        self.assertFalse(fallback_used)
        self.assertEqual(runtime_error, "")
        first_call = mock_pipeline.call_args_list[0]
        self.assertEqual(first_call.args[0], "summarization")
        self.assertEqual(first_call.kwargs["model"], "facebook/bart-large-cnn")

    @override_settings(SUMMARIZATION_PROVIDER="hf", ENABLE_ABSTRACTIVE_SUMMARY=True, SUMMARIZATION_MODEL="facebook/bart-large-cnn")
    @patch("videos.utils._load_hf_summary_pipeline", side_effect=RuntimeError("pipeline boot failed"))
    def test_hf_summary_model_load_failure_uses_explicit_bounded_fallback(self, _mock_loader):
        _SUMMARY_PIPELINE_CACHE.clear()
        result = summarize_text(
            text="This transcript explains preparation, confidence, and how to approach the final exam with clarity and practice.",
            summary_type="short",
            output_language="en",
            source_language="en",
        )
        self.assertEqual(result.get("summary_generation_mode"), "hf_pipeline_load_fallback")
        self.assertEqual(result.get("model_used"), "extractive-hf-task-fallback")
        self.assertTrue(result.get("summary_model_fallback_used"))
        self.assertIn("pipeline boot failed", result.get("summary_runtime_error", ""))

    @override_settings(SUMMARIZATION_PROVIDER="hf", ENABLE_ABSTRACTIVE_SUMMARY=True, HF_SHORT_MAX_INPUT_WORDS=100, SUMMARY_SHORT_MAX_WORDS=18)
    @patch("videos.utils._summarize_short_density_with_groq", return_value=(None, False))
    @patch("videos.utils._summarize_full_density_with_groq", return_value=(None, False))
    @patch("videos.utils._hierarchical_summarize", return_value="This tutorial explains the complete workflow from layout planning through prompt design, animation tuning, export, quality review, and final deployment for a finished result.")
    def test_short_summary_respects_word_cap_on_semantic_fallback(self, _mock_hier, _mock_full_density, _mock_short_density):
        text = " ".join([
            "This transcript explains layout planning, prompt design, animation polish, export, and deployment."
        ] * 90)
        result = summarize_text(text=text, summary_type="short", output_language="en", source_language="en")
        words = len((result.get("content") or "").split())
        self.assertLessEqual(words, 18)

    @override_settings(SUMMARIZATION_PROVIDER="hf", ENABLE_ABSTRACTIVE_SUMMARY=True, HF_SHORT_MAX_INPUT_WORDS=100)
    @patch("videos.utils._generate_short_summary_from_full", return_value="This tutorial explains the full workflow from layout planning to deployment.")
    @patch("videos.utils._hierarchical_summarize", return_value="The tutorial covers layout planning, prompt design, animation tuning, export, and deployment.")
    @patch("videos.utils._summarize_short_density_with_groq", return_value=(None, False))
    @patch("transformers.pipeline")
    def test_short_summary_is_generated_from_full_summary_compression(
        self,
        mock_pipeline,
        mock_short_density,
        mock_hier,
        mock_generate_from_full,
    ):
        text = " ".join([
            "First open the builder and create the layout.",
            "Then connect the prompt generator and adjust the hero section.",
            "After that tune the animation and export the result.",
            "Finally deploy the site and review the final output.",
        ] * 40)
        result = summarize_text(text=text, summary_type="short", output_language="en", source_language="en")
        mock_pipeline.assert_not_called()
        mock_hier.assert_called()
        mock_generate_from_full.assert_called_once()
        first_arg = mock_generate_from_full.call_args.args[0]
        self.assertIn("layout planning", first_arg.lower())
        self.assertIn("workflow", result.get("content", "").lower())
        self.assertNotIn("First open the builder", result.get("content", ""))

    @override_settings(SUMMARIZATION_PROVIDER="hf", ENABLE_ABSTRACTIVE_SUMMARY=True, HF_SHORT_MAX_INPUT_WORDS=100)
    @patch("videos.utils._summarize_short_density_with_groq", return_value=(None, False))
    @patch("videos.utils._hierarchical_summarize", return_value="This tutorial explains the complete workflow from layout planning to final deployment.")
    def test_short_summary_avoids_raw_transcript_fragments_on_fallback(self, _mock_hier, _mock_short_density):
        text = " ".join([
            "First open the builder and create the layout.",
            "Then connect the prompt generator so the hero section updates correctly.",
            "After that refine the animation timing and export the final version.",
            "Finally review the page and prepare it for deployment.",
        ] * 40)
        result = summarize_text(text=text, summary_type="short", output_language="en", source_language="en")
        content = result.get("content", "")
        self.assertIn("workflow", content.lower())
        self.assertNotIn("First open the builder", content)
        self.assertNotIn("Then connect the prompt generator", content)


class TranslationProviderTests(SimpleTestCase):
    @override_settings(TRANSLATION_PROVIDER="none")
    def test_translation_provider_none_skips_groq_path(self):
        text = "hola mundo"
        out = translate_text(text, source_language="es", target_language="en")
        self.assertEqual(out, text)


class TranscriptArtifactsTests(SimpleTestCase):
    def test_transcript_json_payload_contains_readable_and_captions(self):
        transcript_payload = {
            "text": "Hello world. This is a test transcript.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world."},
                {"id": 1, "start": 2.0, "end": 5.0, "text": "This is a test transcript."},
            ],
            "transcript_quality_score": 0.81,
            "metadata": {"asr_provider_used": "whisper_local"},
        }
        canonical_payload = {
            "canonical_segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world.", "original_text": "Hello world."}
            ]
        }
        payload = _build_transcript_json_payload(transcript_payload, canonical_payload)
        self.assertIn("readable_transcript", payload)
        self.assertIn("captions", payload)
        self.assertIn("transcript_state", payload)
        self.assertIn("draft_transcript", payload)
        self.assertTrue(payload["captions"]["srt"])
        self.assertTrue(payload["captions"]["vtt"].startswith("WEBVTT"))
        self.assertEqual(payload.get("quality_badge"), "high")

    @patch("videos.utils._looks_garbled_multiscript", return_value=False)
    def test_transcript_state_flags_low_confidence(self, _garbled):
        result = _compute_transcript_state(
            cleaned_text="hello hello hello hello hello hello hello hello",
            cleaned_segments=[{"start": 0.0, "end": 4.0, "text": "hello hello hello hello hello hello hello hello"}],
            transcript_payload={"transcript_quality_score": 0.82, "confidence": 0.32},
            audio_duration_seconds=240.0,
            transcript_language="en",
        )
        self.assertEqual(result["state"], "low_confidence")
        self.assertIn("low_confidence_span", result["warnings"])

    @patch("videos.utils._looks_garbled_multiscript", return_value=True)
    def test_transcript_state_flags_failed_for_garble(self, _garbled):
        result = _compute_transcript_state(
            cleaned_text="garbled mixed script",
            cleaned_segments=[{"start": 0.0, "end": 4.0, "text": "garbled mixed script"}],
            transcript_payload={"transcript_quality_score": 0.61, "confidence": 0.61},
            audio_duration_seconds=45.0,
            transcript_language="hi",
        )
        self.assertEqual(result["state"], "failed")
        self.assertIn("garbled_multiscript_detected", result["warnings"])

    @patch("videos.tasks.detect_script_type", return_value="mixed")
    @patch("videos.utils._should_accept_usable_malayalam_first_pass", return_value={"first_pass_accepted": False, "first_pass_accept_reason": "failed_quality_gate"})
    @patch("videos.utils._garbled_detector_score", return_value=0.58)
    @patch("videos.utils._looks_garbled_multiscript", return_value=True)
    def test_malayalam_groq_usable_mixed_script_is_accepted(
        self,
        _garbled,
        _garbled_score,
        _first_pass,
        _script_type,
    ):
        text = (
            "ഈ ഭാഗത്തിൽ അദ്ദേഹം വീഡിയോ നിർമ്മാണത്തിനുള്ള പ്രധാന ഘട്ടങ്ങളും ടൂൾ തിരഞ്ഞെടുപ്പും വിശദീകരിക്കുന്നു. "
            "Lovable, Mux, CTA, landing page എന്നീ പദങ്ങൾക്കൊപ്പം workflow സംബന്ധിച്ച വിശദീകരണവും കാണാം."
        )
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 120.0, "end": 136.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.914,
                "confidence": 0.0,
                "metadata": {"asr_provider_used": "groq_whisper"},
            },
            audio_duration_seconds=506.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "cleaned")
        self.assertTrue(result["qa_metrics"]["malayalam_post_asr_accepted"])
        self.assertEqual(result["qa_metrics"]["malayalam_post_asr_mode"], "accepted")
        self.assertEqual(result["qa_metrics"]["malayalam_post_asr_accept_reason"], "groq_malayalam_usable_mixed_script")
        self.assertEqual(result["summary_blocked_reason"], "")
        self.assertEqual(result["chatbot_blocked_reason"], "")
        self.assertIn("malayalam_mixed_script_accepted", result["warnings"])

    @patch("videos.tasks.detect_script_type", return_value="mixed")
    @patch("videos.utils._should_accept_usable_malayalam_first_pass", return_value={"first_pass_accepted": False, "first_pass_accept_reason": "failed_quality_gate"})
    @patch("videos.utils._garbled_detector_score", return_value=0.62)
    @patch("videos.utils._looks_garbled_multiscript", return_value=True)
    def test_malayalam_groq_fallback_transcript_keeps_summary_and_chat_open(
        self,
        _garbled,
        _garbled_score,
        _first_pass,
        _script_type,
    ):
        text = (
            "ഇവിടെ presenter UI section, hero section, CTA placement, testimonial section എന്നിവയെക്കുറിച്ച് വ്യക്തമാക്കുന്നു. "
            "ചുറ്റുപാടിലുള്ള mixed tokens ഉണ്ടെങ്കിലും ആശയം വായിക്കാൻ കഴിയുന്ന വിധത്തിലാണ് transcript."
        )
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 40.0, "end": 58.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.88,
                "confidence": 0.0,
                "metadata": {"asr_provider_used": "groq_whisper_chunked"},
            },
            audio_duration_seconds=420.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "cleaned")
        self.assertEqual(result["qa_metrics"]["summary_blocked_reason"], "")
        self.assertEqual(result["qa_metrics"]["chatbot_blocked_reason"], "")

    @patch("videos.tasks.detect_script_type", return_value="mixed")
    @patch("videos.utils._should_accept_usable_malayalam_first_pass", return_value={"first_pass_accepted": False, "first_pass_accept_reason": "failed_quality_gate"})
    @patch("videos.utils._garbled_detector_score", return_value=1.0)
    @patch("videos.utils._looks_garbled_multiscript", return_value=True)
    def test_malayalam_partially_usable_transcript_is_degraded_not_failed(
        self,
        _garbled,
        _garbled_score,
        _first_pass,
        _script_type,
    ):
        text = (
            "ഈ transcript പൂർണ്ണമായും perfect അല്ല, പക്ഷേ landing page, workflow, hero section, CTA placement "
            "എന്നിവയെക്കുറിച്ചുള്ള ആശയം വായിച്ച് മനസ്സിലാക്കാൻ കഴിയുന്ന രീതിയിൽ ഉള്ളടക്കം ഇപ്പോഴും ഉണ്ട്. "
            "tool usage, design steps, prompt refinement എന്നിവയും context-ൽ കാണാം."
        )
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[
                {"start": 10.0, "end": 24.0, "text": text[:120]},
                {"start": 24.0, "end": 40.0, "text": text[121:]},
            ],
            transcript_payload={
                "transcript_quality_score": 0.81,
                "confidence": 0.0,
                "metadata": {"asr_provider_used": "groq_whisper"},
            },
            audio_duration_seconds=506.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertEqual(result["malayalam_post_asr_mode"], "degraded")
        self.assertEqual(result["summary_blocked_reason"], "")
        self.assertEqual(result["chatbot_blocked_reason"], "")
        self.assertTrue(result["transcript_warning_message"])
        self.assertIn("malayalam_script_acceptance_blocked", result["warnings"])

    def test_malayalam_source_english_substitution_leakage_triggers_fidelity_failure(self):
        text = (
            "When you exit the exam hall you will feel satisfied. "
            "After checking the result site you can understand the final result clearly."
        )
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[
                {"start": 0.0, "end": 5.0, "text": "When you exit the exam hall you will feel satisfied."},
                {"start": 5.0, "end": 10.0, "text": "After checking the result site you can understand the final result clearly."},
            ],
            transcript_payload={
                "transcript_quality_score": 0.44,
                "confidence": 0.98,
                "language": "ml",
                "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
            },
            audio_duration_seconds=120.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertTrue(result["qa_metrics"]["source_language_fidelity_failed"])
        self.assertEqual(result["qa_metrics"]["transcript_fidelity_state"], "source_language_fidelity_failed")
        self.assertEqual(result["summary_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["chatbot_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["malayalam_post_asr_mode"], "source_language_fidelity_failed")

    @patch("videos.tasks.detect_script_type", return_value="latin")
    @patch(
        "videos.utils._extract_asr_metrics",
        return_value={
            "word_count": 24,
            "words_per_minute": 12.0,
            "dominant_script": "latin",
            "dominant_script_ratio": 0.96,
            "malayalam_ratio": 0.0,
            "other_indic_ratio": 0.0,
            "malayalam_token_coverage": 0.0,
            "repeated_token_ratio": 0.02,
            "info_density": 0.38,
        },
    )
    def test_malayalam_source_latin_collapse_with_zero_trusted_evidence_triggers_hard_fidelity_fail(self, _metrics, _script_type):
        text = (
            "When you leave the exam hall you will feel satisfied and ready for the next step. "
            "Check the result website and follow the official update for the answer key."
        )
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[
                {"start": 0.0, "end": 6.0, "text": "When you leave the exam hall you will feel satisfied and ready for the next step."},
                {"start": 6.0, "end": 12.0, "text": "Check the result website and follow the official update for the answer key."},
            ],
            transcript_payload={
                "transcript_quality_score": 0.41,
                "confidence": 0.98,
                "language": "ml",
                "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
            },
            audio_duration_seconds=120.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "source_language_fidelity_failed")
        self.assertTrue(result["qa_metrics"]["source_language_fidelity_failed"])
        self.assertEqual(result["qa_metrics"]["transcript_fidelity_state"], "catastrophic_latin_substitution_failure")
        self.assertEqual(result["qa_metrics"]["final_malayalam_fidelity_decision"], "source_language_fidelity_failed")
        self.assertAlmostEqual(result["qa_metrics"]["malayalam_token_coverage"], 0.0)
        self.assertEqual(result["summary_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["chatbot_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["malayalam_post_asr_mode"], "source_language_fidelity_failed")
        self.assertFalse(result["qa_metrics"]["meaningful_malayalam_evidence"])

    @patch("videos.tasks.detect_script_type", return_value="latin")
    @patch(
        "videos.utils._extract_asr_metrics",
        return_value={
            "word_count": 18,
            "words_per_minute": 10.0,
            "dominant_script": "latin",
            "dominant_script_ratio": 0.97,
            "malayalam_ratio": 0.0,
            "other_indic_ratio": 0.0,
            "malayalam_token_coverage": 0.0,
            "repeated_token_ratio": 0.01,
            "info_density": 0.52,
        },
    )
    def test_malayalam_educational_english_keywords_do_not_validate_fake_transcript(self, _metrics, _script_type):
        text = "exam result question answer key website update check result date"
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 0.0, "end": 6.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.63,
                "confidence": 0.98,
                "language": "ml",
                "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
            },
            audio_duration_seconds=60.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "source_language_fidelity_failed")
        self.assertTrue(result["qa_metrics"]["source_language_fidelity_failed"])
        self.assertFalse(result["qa_metrics"]["meaningful_malayalam_evidence"])
        self.assertEqual(result["summary_blocked_reason"], "malayalam_source_fidelity_failed")

    @patch("videos.tasks.detect_script_type", return_value="gurmukhi")
    @patch(
        "videos.utils._extract_asr_metrics",
        return_value={
            "word_count": 22,
            "words_per_minute": 5.5,
            "dominant_script": "gurmukhi",
            "dominant_script_ratio": 0.93,
            "malayalam_ratio": 0.01,
            "other_indic_ratio": 0.93,
            "malayalam_token_coverage": 0.0,
            "repeated_token_ratio": 0.04,
            "info_density": 0.21,
        },
    )
    def test_catastrophic_gurmukhi_collapse_triggers_hard_fidelity_failure(self, _metrics, _script_type):
        text = (
            "ਇਹ ਗਲਤ ਸਕ്രിപ്റ്റ് ഔട്ട്പുട്ടാണ് result site answer key "
            "ਤുടർന്ന് ഗർബിള് collapse ആയി exam date check ചെയ്യൂ"
        )
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[
                {"start": 0.0, "end": 5.0, "text": "ਇਹ ਗਲਤ ਸਕ്രിപ്റ്റ് ഔട്ട്പുട്ടാണ് result site answer key"},
                {"start": 5.0, "end": 10.0, "text": "ਤുടർന്ന് ഗർബിള് collapse ആയി exam date check ചെയ്യൂ"},
            ],
            transcript_payload={
                "transcript_quality_score": 0.22,
                "confidence": 0.98,
                "language": "ml",
                "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
            },
            audio_duration_seconds=120.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertTrue(result["qa_metrics"]["source_language_fidelity_failed"])
        self.assertEqual(result["qa_metrics"]["transcript_fidelity_state"], "catastrophic_wrong_script_failure")
        self.assertTrue(result["qa_metrics"]["catastrophic_wrong_script_failure"])
        self.assertEqual(result["summary_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["chatbot_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["malayalam_post_asr_mode"], "source_language_fidelity_failed")

    def test_genuine_malayalam_with_real_english_terms_does_not_trigger_fidelity_failure(self):
        text = (
            "നിങ്ങൾ result site ൽ date നോക്കി exam result പരിശോധിക്കാം. "
            "ഇത് students ന് സഹായകരമായ മാർഗമാണ്."
        )
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 0.0, "end": 8.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.82,
                "confidence": 0.98,
                "language": "ml",
                "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
            },
            audio_duration_seconds=90.0,
            transcript_language="ml",
        )
        self.assertFalse(result["qa_metrics"]["source_language_fidelity_failed"])
        self.assertTrue(result["qa_metrics"]["meaningful_malayalam_evidence"])
        self.assertNotEqual(result["qa_metrics"]["transcript_fidelity_state"], "source_language_fidelity_failed")
        self.assertFalse(result["qa_metrics"].get("catastrophic_wrong_script_failure", False))

    def test_transcript_json_payload_blocks_english_view_for_malayalam_fidelity_failure(self):
        transcript_payload = {
            "text": "When you exit the exam hall you will feel satisfied.",
            "language": "ml",
            "transcript_quality_score": 0.24,
            "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
        }
        payload = _build_transcript_json_payload(
            transcript_payload,
            {"canonical_segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "Canonical English"}]},
            transcript_state="degraded",
            qa_metrics={
                "source_language_fidelity_failed": True,
                "transcript_fidelity_state": "source_language_fidelity_failed",
            },
            readable_transcript="When you exit the exam hall you will feel satisfied.",
            display_readable_transcript="When you exit the exam hall you will feel satisfied.",
        )
        self.assertFalse(payload["english_view_available"])
        self.assertEqual(payload["translation_state"], "blocked")
        self.assertEqual(payload["translation_blocked_reason"], "source_language_fidelity_failed")
        self.assertEqual(payload["transcript_display_mode"], "suppressed_unfaithful_source")

    @patch("videos.tasks.detect_script_type", return_value="other")
    @patch(
        "videos.utils._extract_asr_metrics",
        return_value={
            "word_count": 46,
            "words_per_minute": 6.1,
            "dominant_script": "other",
            "dominant_script_ratio": 0.86,
            "malayalam_ratio": 0.04,
            "other_indic_ratio": 0.61,
            "malayalam_token_coverage": 0.02,
            "repeated_token_ratio": 0.01,
            "info_density": 0.73,
        },
    )
    @patch("videos.utils._should_accept_usable_malayalam_first_pass", return_value={"first_pass_accepted": True, "first_pass_accept_reason": "usable_first_pass_allowed"})
    @patch("videos.utils._garbled_detector_score", return_value=0.12)
    @patch("videos.utils._looks_garbled_multiscript", return_value=False)
    def test_wrong_script_malayalam_never_becomes_cleaned(
        self,
        _garbled,
        _garbled_score,
        _first_pass,
        _metrics,
        _script_type,
    ):
        text = (
            "ਇੱਿਦੁਂਦੁ ਓਰੋ ਉਤਰੋ ਨੇਂਗਲ ਕਲੀ ਆകਿਨ ਵੇਰਿਟ ਮോഗਤ ਤਡੀਕੀਨ ਰീധੀല ആലിയീകിൻ "
            "exam coaching motivation class WhatsApp channel."
        )
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 0.0, "end": 16.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.91,
                "confidence": 0.0,
                "metadata": {"asr_provider_used": "groq_whisper"},
            },
            audio_duration_seconds=452.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertEqual(result["malayalam_post_asr_mode"], "degraded")
        self.assertNotEqual(result["qa_metrics"]["malayalam_post_asr_mode"], "accepted")
        self.assertEqual(result["qa_metrics"]["script_detector_result"], "other")
        self.assertEqual(result["qa_metrics"]["dominant_script"], "other")
        self.assertIn("malayalam_script_acceptance_blocked", result["warnings"])
        self.assertEqual(result["qa_metrics"]["malayalam_post_asr_accept_reason"], "dominant_script_not_malayalam")
        self.assertEqual(result["summary_blocked_reason"], "")
        self.assertEqual(result["chatbot_blocked_reason"], "")

    def test_catastrophic_wrong_script_malayalam_never_uses_structurally_usable_degraded_path(self):
        text = (
            "ਇਹ ਗਲਤ ਸਕ്രിപ്റ്റ് ഔട്ട്പുട്ടാണ് answer key result site "
            "ਇത് മുഴുവൻ collapse ആയി wrong script garbage ആയി"
        )
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 0.0, "end": 12.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.58,
                "confidence": 0.98,
                "language": "ml",
                "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
            },
            audio_duration_seconds=150.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertEqual(result["malayalam_post_asr_mode"], "source_language_fidelity_failed")
        self.assertEqual(result["malayalam_post_asr_reason"], "malayalam_source_fidelity_failed")
        self.assertNotEqual(result["malayalam_post_asr_reason"], "structurally_usable_low_trust_malayalam")
        self.assertEqual(result["summary_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["chatbot_blocked_reason"], "malayalam_source_fidelity_failed")

    @patch("videos.tasks.detect_script_type", return_value="malayalam")
    @patch("videos.utils._should_accept_usable_malayalam_first_pass", return_value={"first_pass_accepted": True, "first_pass_accept_reason": "usable_first_pass_allowed"})
    @patch("videos.utils._garble_debug_snapshot", return_value={
        "garbled_score": 0.558,
        "dominant_script": "malayalam",
        "dominant_script_ratio": 0.79,
        "active_scripts": ["malayalam"],
        "raw_active_scripts": ["malayalam"],
        "odd_token_ratio": 0.0,
        "script_penalty": 0.0,
        "odd_token_penalty": 0.0,
        "replacement_penalty": 0.0,
        "malayalam_adjustment": 0.0,
        "replacement_chars": 0,
        "suspicious_count": 0,
        "suspicious_codepoints": [],
        "normalization_changed": False,
        "preview": "",
        "escaped_preview": "",
    })
    @patch("videos.utils._looks_garbled_multiscript", return_value=False)
    def test_high_garble_high_script_ratio_malayalam_stays_degraded(
        self,
        _garbled,
        _snapshot,
        _first_pass,
        _script_type,
    ):
        text = "ഇദ്ദുംദു ഓരോ ഉത്തരോ നേംഗല കലീആകുന പേഡീകീദന മീരീ മുരീ ചേരീദ"
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 0.0, "end": 8.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.91,
                "confidence": 0.0,
                "metadata": {"asr_provider_used": "groq_whisper"},
            },
            audio_duration_seconds=220.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertEqual(result["qa_metrics"]["malayalam_post_asr_accept_reason"], "final_garble_above_cleaned_threshold")

    @patch("videos.tasks.detect_script_type", return_value="malayalam")
    @patch("videos.utils._should_accept_usable_malayalam_first_pass", return_value={"first_pass_accepted": True, "first_pass_accept_reason": "usable_first_pass_allowed"})
    @patch("videos.utils._looks_garbled_multiscript", return_value=False)
    def test_semantically_bad_script_normalized_malayalam_stays_degraded(
        self,
        _garbled,
        _first_pass,
        _script_type,
    ):
        text = "ഇദ്ദുംദു ഓരോ ഉത്തരോ നേംഗല കലീആകുന പേഡീകീദന മീരീ മുരീ ചേരീദ"
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 0.0, "end": 8.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.88,
                "confidence": 0.0,
                "metadata": {"asr_provider_used": "groq_whisper"},
            },
            audio_duration_seconds=180.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertIn(result["qa_metrics"]["malayalam_post_asr_accept_reason"], {"semantic_trust_too_low", "pseudo_phonetic_ratio_too_high", "low_trust_malayalam_segment_share_too_high", "corrupted_segment_share_too_high"})

    @patch("videos.tasks.detect_script_type", return_value="other")
    @patch("videos.utils._looks_garbled_multiscript", return_value=False)
    def test_low_trust_malayalam_but_structurally_usable_becomes_degraded_not_failed(
        self,
        _garbled,
        _script_type,
    ):
        segments = [
            {"start": 0.0, "end": 7.0, "text": "ഇദ്ദുംദു ഓരോ ഉത്തരോ നേംഗല exam hall support result"},
            {"start": 7.0, "end": 14.0, "text": "confidence WhatsApp channel strategy focus preparation"},
            {"start": 14.0, "end": 22.0, "text": "കളീആകുന പേഡീകീദന മീരീ മുരീ ചേറീദ hard work"},
        ]
        text = " ".join(seg["text"] for seg in segments)
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=segments,
            transcript_payload={
                "transcript_quality_score": 0.82,
                "confidence": 0.0,
                "metadata": {"asr_provider_used": "groq_whisper"},
            },
            audio_duration_seconds=180.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertEqual(result["malayalam_post_asr_mode"], "degraded")
        self.assertIn("low-trust", result["transcript_warning_message"].lower())
        self.assertEqual(result["summary_blocked_reason"], "")
        self.assertEqual(result["chatbot_blocked_reason"], "")

    @patch("videos.tasks.detect_script_type", return_value="other")
    @patch("videos.utils._looks_garbled_multiscript", return_value=False)
    def test_low_trust_malayalam_and_structurally_unusable_stays_failed(
        self,
        _garbled,
        _script_type,
    ):
        text = "ഇദ്ദുംദു കളീആകുന പേഡീകീദന മീരീ മുരീ ചേറീദ"
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 0.0, "end": 5.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.14,
                "confidence": 0.0,
                "metadata": {"asr_provider_used": "groq_whisper"},
            },
            audio_duration_seconds=320.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["malayalam_post_asr_mode"], "failed")
        self.assertEqual(result["summary_blocked_reason"], "transcript_failed_garble_gate")
        self.assertEqual(result["chatbot_blocked_reason"], "transcript_failed_garble_gate")

    @patch("videos.tasks.detect_script_type", return_value="malayalam")
    @patch("videos.utils._should_accept_usable_malayalam_first_pass", return_value={"first_pass_accepted": True, "first_pass_accept_reason": "usable_first_pass_allowed"})
    @patch("videos.utils._looks_garbled_multiscript", return_value=False)
    def test_valid_english_segments_do_not_unfairly_penalize_malayalam_acceptance(
        self,
        _garbled,
        _first_pass,
        _script_type,
    ):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "നിങ്ങൾ confidence നിലനിറുത്തണം."},
            {"start": 5.0, "end": 10.0, "text": "WhatsApp channel support and exam hall strategy."},
            {"start": 10.0, "end": 15.0, "text": "പരീക്ഷയ്ക്ക് തയ്യാറാകാൻ നല്ല focus വേണം."},
        ]
        text = " ".join(seg["text"] for seg in segments)
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=segments,
            transcript_payload={
                "transcript_quality_score": 0.91,
                "confidence": 0.0,
                "metadata": {"asr_provider_used": "groq_whisper"},
            },
            audio_duration_seconds=180.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "cleaned")
        self.assertIn(result["qa_metrics"]["malayalam_post_asr_accept_reason"], {"usable_first_pass_allowed", "english_segments_excluded_from_malayalam_ratio"})
        trust = result["qa_metrics"]["transcript_trust"]
        self.assertGreaterEqual(int(trust.get("english_only_segments", 0) or 0) + int(trust.get("mixed_lang_segments", 0) or 0), 1)

    @patch("videos.tasks.detect_script_type", return_value="mixed")
    @patch("videos.utils._should_accept_usable_malayalam_first_pass", return_value={"first_pass_accepted": False, "first_pass_accept_reason": "failed_quality_gate"})
    @patch("videos.utils._garbled_detector_score", return_value=0.93)
    @patch("videos.utils._looks_garbled_multiscript", return_value=True)
    def test_truly_garbled_malayalam_transcript_is_blocked(
        self,
        _garbled,
        _garbled_score,
        _first_pass,
        _script_type,
    ):
        text = "### ### ask ai about this moment ### ### ??? ??? ???"
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 0.0, "end": 5.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.21,
                "confidence": 0.0,
                "metadata": {"asr_provider_used": "groq_whisper"},
            },
            audio_duration_seconds=300.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["malayalam_post_asr_mode"], "failed")
        self.assertEqual(result["summary_blocked_reason"], "transcript_failed_garble_gate")
        self.assertEqual(result["chatbot_blocked_reason"], "transcript_failed_garble_gate")


    @patch("videos.tasks.detect_script_type", return_value="gurmukhi")
    @patch(
        "videos.utils._extract_asr_metrics",
        return_value={
            "word_count": 22,
            "words_per_minute": 5.5,
            "dominant_script": "gurmukhi",
            "dominant_script_ratio": 0.93,
            "malayalam_ratio": 0.01,
            "other_indic_ratio": 0.93,
            "malayalam_token_coverage": 0.0,
            "repeated_token_ratio": 0.04,
            "info_density": 0.21,
        },
    )
    @patch(
        "videos.utils.analyze_malayalam_source_fidelity",
        return_value={
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "catastrophic_wrong_script_failure",
            "catastrophic_wrong_script_failure": True,
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "wrong_script_burden": 0.93,
            "visible_malayalam_char_ratio": 0.0,
            "substantial_visible_malayalam": False,
            "suspicious_substitution_burden": 0.0,
        },
    )
    def test_catastrophic_gurmukhi_collapse_triggers_hard_fidelity_failure(self, _fidelity, _metrics, _script_type):
        text = "catastrophic wrong script collapse result site answer key garbage exam date check"
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[
                {"start": 0.0, "end": 5.0, "text": "catastrophic wrong script collapse result site"},
                {"start": 5.0, "end": 10.0, "text": "answer key garbage exam date check"},
            ],
            transcript_payload={
                "transcript_quality_score": 0.22,
                "confidence": 0.98,
                "language": "ml",
                "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
            },
            audio_duration_seconds=120.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertTrue(result["qa_metrics"]["source_language_fidelity_failed"])
        self.assertEqual(result["qa_metrics"]["transcript_fidelity_state"], "catastrophic_wrong_script_failure")
        self.assertTrue(result["qa_metrics"]["catastrophic_wrong_script_failure"])
        self.assertEqual(result["summary_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["chatbot_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["malayalam_post_asr_mode"], "source_language_fidelity_failed")

    @patch("videos.tasks.detect_script_type", return_value="gurmukhi")
    @patch(
        "videos.utils._extract_asr_metrics",
        return_value={
            "word_count": 18,
            "words_per_minute": 4.8,
            "dominant_script": "gurmukhi",
            "dominant_script_ratio": 0.91,
            "malayalam_ratio": 0.02,
            "other_indic_ratio": 0.91,
            "malayalam_token_coverage": 0.0,
            "repeated_token_ratio": 0.02,
            "info_density": 0.18,
        },
    )
    @patch(
        "videos.utils.analyze_malayalam_source_fidelity",
        return_value={
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "catastrophic_wrong_script_failure",
            "catastrophic_wrong_script_failure": True,
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "wrong_script_burden": 0.91,
            "visible_malayalam_char_ratio": 0.0,
            "substantial_visible_malayalam": False,
            "suspicious_substitution_burden": 0.0,
        },
    )
    def test_catastrophic_wrong_script_malayalam_never_uses_structurally_usable_degraded_path(self, _fidelity, _metrics, _script_type):
        text = "catastrophic wrong script collapse answer key result site garbage"
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 0.0, "end": 12.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.18,
                "confidence": 0.98,
                "language": "ml",
                "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
            },
            audio_duration_seconds=150.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertEqual(result["malayalam_post_asr_mode"], "source_language_fidelity_failed")
        self.assertEqual(result["malayalam_post_asr_reason"], "malayalam_source_fidelity_failed")
        self.assertNotEqual(result["malayalam_post_asr_reason"], "structurally_usable_low_trust_malayalam")
        self.assertEqual(result["summary_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["chatbot_blocked_reason"], "malayalam_source_fidelity_failed")


class StructuredSummarySchemaTests(SimpleTestCase):
    def test_wired_style_qa_transcript_is_classified_as_interview(self):
        transcript_text = (
            "What is the first thing you do in the morning? The guest laughs and answers that coffee comes first. "
            "Who is the most likely to text back first? The guest says Robert Downey Jr. usually replies quickly. "
            "Why does Christopher Nolan prefer practical effects? He explains that real texture helps the performance. "
            "WIRED autocomplete interview continues with quick questions and short answers about cast, habits, and projects."
        )
        self.assertEqual(_classify_video_type(transcript_text), "interview")

    def test_tutorial_transcript_is_classified_as_tutorial(self):
        transcript_text = (
            "This tutorial shows how to build the landing page step by step. "
            "First open the builder, then create the layout and configure the prompt. "
            "Next generate the assets, export the frames, and finally deploy the site."
        )
        self.assertEqual(_classify_video_type(transcript_text), "tutorial")

    def test_non_english_transcript_is_not_forced_into_interview_by_english_summaries(self):
        transcript_text = (
            "सच बताना, सपने तो तुमने भी बहुत बड़े देखे थे ना. अपने parents को proud feel कराना है, "
            "लेकिन daily routine और discipline कैसे maintain करना है, यही discussion का main point है."
        )
        full_summary = (
            "The speaker explains how to stay disciplined, avoid procrastination, and improve exam preparation."
        )
        bullet_summary = (
            "- Build a daily study routine.\n"
            "- Reduce distractions and procrastination.\n"
            "- Focus on consistent preparation for exams."
        )
        self.assertNotEqual(
            _classify_video_type(transcript_text, full_summary=full_summary, bullet_summary=bullet_summary),
            "interview",
        )

    def test_malayalam_mixed_language_summary_uses_semantic_points_and_preserves_english_terms(self):
        transcript_text = (
            "ഈ ക്ലാസിൽ exam hall preparation, confidence, hard work, WhatsApp channel support എന്നിവയെക്കുറിച്ച് സംസാരിക്കുന്നു. "
            "പഠനരീതി, result, focus, coaching strategy എന്നിവയും വിശദീകരിക്കുന്നു."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 8.0, "text": "ഈ ക്ലാസിൽ exam hall preparation, confidence, hard work, WhatsApp channel support എന്നിവയെക്കുറിച്ച് സംസാരിക്കുന്നു."},
            {"id": 1, "start": 8.0, "end": 16.0, "text": "പഠനരീതി, result, focus, coaching strategy എന്നിവയും വിശദീകരിക്കുന്നു."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            transcript_state="cleaned",
            transcript_language="ml",
            full_summary="confidence, hard work, exam hall strategy, WhatsApp channel support.",
            bullet_summary="- confidence and hard work\n- exam hall preparation\n- WhatsApp channel support",
            short_summary="The video explains confidence, hard work, and exam hall preparation.",
        )
        self.assertTrue(result["key_points"])
        joined = " ".join(result["key_points"]).lower()
        self.assertNotIn("àµ€à´•àµ€", joined)
        self.assertTrue(any(term in joined for term in ["confidence", "exam hall", "hard work", "whatsapp channel"]))

    def test_summary_grounding_prefers_trusted_assembled_units_over_raw_fragments(self):
        raw_segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "ഇദ്ദുംദു confidence"},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "ഉണ്ടെങ്കിൽ നല്ല result ലഭിക്കും"},
        ]
        assembled_units = [
            {
                "id": 0,
                "start": 0.0,
                "end": 4.0,
                "display_start": 0.0,
                "display_end": 4.0,
                "text": "നിങ്ങൾ confidence ഉണ്ടെങ്കിൽ നല്ല result ലഭിക്കും",
                "unit_readability": 0.72,
                "segment_type": "mixed_malayalam_english",
            }
        ]
        result = build_structured_summary(
            transcript_text="നിങ്ങൾ confidence ഉണ്ടെങ്കിൽ നല്ല result ലഭിക്കും",
            segments=assembled_units,
            raw_segments=raw_segments,
            assembled_units=assembled_units,
            transcript_state="cleaned",
            transcript_language="ml",
            full_summary="confidence and result are discussed.",
            bullet_summary="- confidence and result",
            short_summary="The video explains confidence and result.",
        )
        rendered = json.dumps(result).lower()
        self.assertIn("confidence", rendered)
        self.assertIn("result", rendered)
        self.assertNotIn("order workers developers", rendered)

    def test_degraded_malayalam_summary_rejects_fake_participants_and_interview_bias(self):
        transcript_text = (
            "enda va confidence exam hall WhatsApp channel support result motivation coaching class. "
            "ഈ സംസാരത്തിൽ confidence, result, hard work എന്നിവയെക്കുറിച്ച് പറയുന്നു."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 6.0, "text": "Enda Va confidence exam hall WhatsApp channel support result motivation coaching class."},
            {"id": 1, "start": 6.0, "end": 12.0, "text": "ഈ സംസാരത്തിൽ confidence, result, hard work എന്നിവയെക്കുറിച്ച് പറയുന്നു."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            transcript_state="degraded",
            transcript_language="ml",
            full_summary="A discussion of confidence, result, and hard work.",
            bullet_summary="",
            short_summary="Confidence, result, and hard work are discussed.",
        )
        joined = " ".join(result.get("key_points", [])).lower()
        self.assertNotIn("enda va", joined)
        self.assertEqual(result.get("action_items"), [])
        self.assertNotEqual(result.get("tldr"), "The degraded transcript covers the main discussion topics, but the audio remains noisy.")
        rendered = json.dumps(result).lower()
        self.assertNotIn("discussion on us join if", rendered)
        self.assertNotIn("order workers developers", rendered)
        self.assertEqual(result.get("participants"), [])
        self.assertNotEqual(result.get("video_type"), "interview")

    def test_degraded_malayalam_summary_rejects_prinzip_ok_and_stays_neutral(self):
        transcript_text = (
            "Prinzip Ok confidence result support WhatsApp channel exam hall. "
            "This is a noisy degraded Malayalam mixed-language transcript about preparation and motivation."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "Prinzip Ok confidence result support WhatsApp channel exam hall."},
            {"id": 1, "start": 5.0, "end": 10.0, "text": "motivation coaching class hard work preparation."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            raw_segments=segments,
            assembled_units=segments,
            transcript_state="degraded",
            transcript_language="ml",
            full_summary="A degraded mixed-language discussion about confidence, preparation, and support.",
            bullet_summary="",
            short_summary="Confidence, preparation, and support are discussed.",
        )
        rendered = json.dumps(result).lower()
        self.assertEqual(result.get("participants"), [])
        self.assertNotEqual(result.get("video_type"), "interview")
        self.assertNotIn("prinzip ok", rendered)
        self.assertEqual(result.get("action_items"), [])

    def test_low_trust_degraded_malayalam_summary_stays_safe(self):
        transcript_text = (
            "ഇദ്ദുംദു confidence exam hall support WhatsApp channel. "
            "കളീആകുന പേഡീകീദന result hard work preparation."
        )
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 6.0,
                "text": "ഇദ്ദുംദു confidence exam hall support WhatsApp channel.",
                "unit_readability": 0.31,
                "segment_type": "mixed_malayalam_english",
            },
            {
                "id": 1,
                "start": 6.0,
                "end": 12.0,
                "text": "കളീആകുന പേഡീകീദന result hard work preparation.",
                "unit_readability": 0.28,
                "segment_type": "corrupted_malayalam_like",
            },
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            raw_segments=segments,
            assembled_units=segments,
            transcript_state="degraded",
            transcript_language="ml",
        )
        rendered = json.dumps(result).lower()
        self.assertNotIn("the degraded transcript covers", rendered)
        self.assertEqual(result.get("action_items"), [])

    def test_low_trust_degraded_malayalam_uses_safe_summary_when_no_trusted_units_exist(self):
        garbage_text = "ਇੱ੆മാലു ਪੋਰതികരિന്ദു fall of all of your please mute dublin"
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 7.0,
                "text": garbage_text,
                "unit_readability": 0.08,
                "segment_type": "corrupted_malayalam_like",
            },
            {
                "id": 1,
                "start": 7.0,
                "end": 14.0,
                "text": "Prinzip Ok noisy mixed fragment",
                "unit_readability": 0.11,
                "segment_type": "uncertain_noise",
            },
        ]
        result = build_structured_summary(
            transcript_text=f"{garbage_text} Prinzip Ok noisy mixed fragment",
            segments=segments,
            raw_segments=segments,
            assembled_units=segments,
            transcript_state="degraded",
            transcript_language="ml",
            full_summary="order workers developers",
            bullet_summary="- order workers developers\n- noisy fragment",
            short_summary="The degraded transcript covers important content.",
        )
        self.assertEqual(
            result.get("tldr"),
            "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.",
        )
        self.assertEqual(result.get("key_points"), [])
        self.assertEqual(result.get("action_items"), [])
        self.assertEqual(result.get("chapters"), [])
        self.assertEqual(result.get("participants"), [])
        self.assertEqual(result.get("_trace", {}).get("mode"), "degraded_safe")

    def test_degraded_malayalam_english_fragments_do_not_cause_overconfident_summary(self):
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 6.0,
                "text": "confidence exam hall",
                "unit_readability": 0.33,
                "segment_type": "clean_english",
            },
            {
                "id": 1,
                "start": 6.0,
                "end": 12.0,
                "text": "ਇੱ੆മാലു noisy corrupted chunk",
                "unit_readability": 0.09,
                "segment_type": "corrupted_malayalam_like",
            },
        ]
        result = build_structured_summary(
            transcript_text="confidence exam hall ഇੱ੆മാലു noisy corrupted chunk",
            segments=segments,
            raw_segments=segments,
            assembled_units=segments,
            transcript_state="degraded",
            transcript_language="ml",
            full_summary="A detailed lecture about exam strategy, scheduling, confidence, and score improvement.",
            bullet_summary="- detailed strategy\n- scheduling system\n- score improvement",
            short_summary="A detailed lecture on exam strategy.",
        )
        rendered = json.dumps(result).lower()
        self.assertIn("transcript quality is too low", result.get("tldr", "").lower())
        self.assertEqual(result.get("action_items"), [])
        self.assertEqual(result.get("chapters"), [])
        self.assertNotIn("detailed lecture", rendered)

    def test_degraded_malayalam_chapters_avoid_noisy_fragment_titles(self):
        result = build_structured_summary(
            transcript_text="നിങ്ങൾ confidence ഉണ്ടെങ്കിൽ result ലഭിക്കും WhatsApp channel support ഉണ്ട്",
            segments=[
                {"id": 0, "start": 0.0, "end": 5.0, "text": "order workers developers"},
                {"id": 1, "start": 5.0, "end": 10.0, "text": "നിങ്ങൾ confidence ഉണ്ടെങ്കിൽ result ലഭിക്കും"},
            ],
            transcript_state="degraded",
            transcript_language="ml",
            short_summary="confidence and result are discussed.",
        )
        titles = " ".join(chapter.get("title", "") for chapter in result.get("chapters", [])).lower()
        self.assertNotIn("order workers developers", titles)

    def test_degraded_malayalam_reconstruction_avoids_old_noisy_tldr_fallback(self):
        assembled_units = [
            {
                "id": 0,
                "start": 0.0,
                "end": 8.0,
                "display_start": 0.0,
                "display_end": 8.0,
                "text": "exam preparation confidenceine kurich samsarikkunnu",
                "unit_readability": 0.72,
                "segment_type": "mixed_malayalam_english",
            },
            {
                "id": 1,
                "start": 8.0,
                "end": 16.0,
                "display_start": 8.0,
                "display_end": 16.0,
                "text": "WhatsApp channel support exam hall readinessine kurich parayunnu",
                "unit_readability": 0.69,
                "segment_type": "mixed_malayalam_english",
            },
        ]
        result = build_structured_summary(
            transcript_text="exam preparation confidence WhatsApp channel support exam hall",
            segments=assembled_units,
            raw_segments=assembled_units,
            assembled_units=assembled_units,
            transcript_state="degraded",
            transcript_language="ml",
        )
        self.assertNotIn("The degraded transcript covers", result.get("tldr", ""))
        self.assertEqual(result.get("_trace", {}).get("mode"), "degraded_reconstruction")

    def test_degraded_malayalam_reconstruction_preserves_trusted_english_terms(self):
        assembled_units = [
            {
                "id": 0,
                "start": 0.0,
                "end": 7.0,
                "display_start": 0.0,
                "display_end": 7.0,
                "text": "confidence hard work result samsarikkunnu",
                "unit_readability": 0.74,
                "segment_type": "mixed_malayalam_english",
            },
            {
                "id": 1,
                "start": 7.0,
                "end": 14.0,
                "display_start": 7.0,
                "display_end": 14.0,
                "text": "WhatsApp channel support exam hall readinessine kurich parayunnu",
                "unit_readability": 0.70,
                "segment_type": "mixed_malayalam_english",
            },
        ]
        result = build_structured_summary(
            transcript_text="confidence hard work result WhatsApp channel support exam hall",
            segments=assembled_units,
            raw_segments=assembled_units,
            assembled_units=assembled_units,
            transcript_state="degraded",
            transcript_language="ml",
        )
        rendered = json.dumps(result).lower()
        self.assertIn("confidence", rendered)
        self.assertIn("whatsapp channel", rendered)
        self.assertIn("exam hall", rendered)

    def test_degraded_malayalam_reconstruction_keeps_action_items_empty_and_traceable(self):
        assembled_units = [
            {
                "id": 0,
                "start": 0.0,
                "end": 6.0,
                "display_start": 0.0,
                "display_end": 6.0,
                "text": "exam preparation confidenceine kurich samsarikkunnu",
                "unit_readability": 0.66,
                "segment_type": "mixed_malayalam_english",
            },
            {
                "id": 1,
                "start": 6.0,
                "end": 12.0,
                "display_start": 6.0,
                "display_end": 12.0,
                "text": "support guidance WhatsApp channel vazhi kittum",
                "unit_readability": 0.62,
                "segment_type": "mixed_malayalam_english",
            },
        ]
        result = build_structured_summary(
            transcript_text="preparation confidence support guidance WhatsApp channel",
            segments=assembled_units,
            raw_segments=assembled_units,
            assembled_units=assembled_units,
            transcript_state="degraded",
            transcript_language="ml",
        )
        self.assertEqual(result.get("action_items"), [])
        trace = result.get("_trace", {})
        self.assertTrue(trace.get("source_unit_indices"))
        self.assertTrue(trace.get("source_time_ranges"))
        self.assertGreaterEqual(trace.get("trust_count", 0), 1)

    def test_clean_malayalam_summary_path_is_not_replaced_by_degraded_reconstruction(self):
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 8.0,
                "text": "exam preparationum confidenceum visadikarikkunnu",
            },
        ]
        result = build_structured_summary(
            transcript_text=segments[0]["text"],
            segments=segments,
            transcript_state="cleaned",
            transcript_language="ml",
            short_summary="The video explains exam preparation and confidence.",
        )
        self.assertNotIn("_trace", result)

    def test_rhetorical_question_led_prompt_video_stays_tutorial(self):
        transcript_text = (
            "What if I told you this UI was generated using only prompts without any image reference? "
            "This tutorial walks through prompt writing, section generation, animation tuning, and deployment. "
            "First open Gemini, then generate the layout, refine each section, and finally deploy the site."
        )
        full_summary = (
            "This tutorial explains how to generate a premium UI with prompts, refine sections, and deploy the final website."
        )
        bullet_summary = (
            "- Write the base prompt.\n"
            "- Generate the layout and sections.\n"
            "- Tune the animation and deploy the website."
        )
        self.assertEqual(
            _classify_video_type(transcript_text, full_summary=full_summary, bullet_summary=bullet_summary),
            "tutorial",
        )

    def test_trailer_style_video_without_interview_cues_is_not_classified_as_interview(self):
        transcript_text = (
            "There was an idea to bring together a group of remarkable people to see if they could become something more. "
            "So when they needed us, we could fight the battles that they never could. "
            "Dread it. Run from it. Destiny still arrives. Evacuate the city. Engage all defenses. "
            "And get this man a shield."
        )
        full_summary = (
            "The trailer sets up a large-scale conflict, introduces the threat, and highlights the urgency of the coming battle."
        )
        bullet_summary = (
            "- A major threat is approaching.\n"
            "- The heroes prepare for a large-scale battle.\n"
            "- The tone is urgent and cinematic."
        )
        self.assertNotEqual(
            _classify_video_type(transcript_text, full_summary=full_summary, bullet_summary=bullet_summary),
            "interview",
        )

    def test_commentary_transcript_is_classified_as_commentary(self):
        transcript_text = (
            "In my opinion this review works because the commentary stays focused on the design choices. "
            "The speaker reacts to each section, gives analysis, and explains why the pacing feels strong."
        )
        self.assertEqual(_classify_video_type(transcript_text), "commentary")

    def test_structured_summary_schema_and_bounds(self):
        transcript_text = (
            "ആദ്യം വീഡിയോയുടെ ലക്ഷ്യം വിശദീകരിക്കുന്നു. ശേഷം ഇമേജ് ജനറേഷൻ നടത്തുന്നു. "
            "അടുത്തതായി ഫ്രെയിംസ് എക്സ്പോർട്ട് ചെയ്യുന്നു. പിന്നീട് വെബ്സൈറ്റ് അസംബിൾ ചെയ്യുന്നു. "
            "അവസാനത്തിൽ സൈറ്റ് ഡിപ്ലോയ് ചെയ്ത് ഫലം പരിശോധന നടത്തുന്നു."
        )
        segments = [
            {"id": i, "start": float(i * 35), "end": float(i * 35 + 20), "text": f"ഘട്ടം {i+1} വിശദീകരണം"}
            for i in range(20)
        ]
        full_summary = (
            "ഈ വീഡിയോയിൽ മുഴുവൻ പ്രവഹനം ഘട്ടംഘട്ടമായി വിശദീകരിക്കുന്നു. "
            "ഉപകരണങ്ങൾ ഉപയോഗിച്ച് മീഡിയ നിർമ്മിച്ച് അവസാനം വെബ്സൈറ്റ് ഡിപ്ലോയ് ചെയ്യുന്നു."
        )
        bullet_summary = (
            "- ലക്ഷ്യം നിർവചിക്കുക\n"
            "- ഇമേജ് ജനറേഷൻ നടത്തുക\n"
            "- ഫ്രെയിംസ് എക്സ്പോർട്ട് ചെയ്യുക\n"
            "- വെബ്സൈറ്റ് നിർമ്മിക്കുക\n"
            "- ഡിപ്ലോയ് ചെയ്യുക\n"
            "- ഫലം പരിശോധിക്കുക"
        )
        short_summary = "വീഡിയോ മുഴുവൻ workflow വേഗത്തിൽ കാണിച്ച് അവസാനം deploy ചെയ്യുന്നു."

        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary=full_summary,
            bullet_summary=bullet_summary,
            short_summary=short_summary,
        )

        self.assertEqual(set(result.keys()), {"tldr", "key_points", "action_items", "chapters"})
        self.assertLessEqual(len((result["tldr"] or "").splitlines()), 2)
        self.assertGreaterEqual(len(result["key_points"]), 4)
        self.assertLessEqual(len(result["key_points"]), 6)
        self.assertGreaterEqual(len(result["chapters"]), 5)
        self.assertLessEqual(len(result["chapters"]), 12)
        self.assertLessEqual(len(result["action_items"]), 6)
        self.assertTrue(all("title" in c and "timestamp" in c for c in result["chapters"]))
        self.assertTrue(any("\u0d00" <= ch <= "\u0d7f" for ch in (result["tldr"] or "")))

    def test_structured_summary_safe_default_on_failure(self):
        result = build_structured_summary(
            transcript_text="",
            segments=None,
            full_summary="",
            bullet_summary="",
            short_summary="",
        )
        self.assertEqual(result, default_structured_summary())

    def test_interview_summary_returns_empty_action_items_without_explicit_advice(self):
        transcript_text = (
            "The host asks the guest about the new film and how the role changed his routine. "
            "They discuss preparation, on-set experiences, and audience reactions. "
            "The host follows up with questions about the cast and future projects."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 25.0, "text": "The host asks about the new film and the guest explains the role."},
            {"id": 1, "start": 25.0, "end": 50.0, "text": "They discuss preparation, cast chemistry, and audience reactions."},
            {"id": 2, "start": 50.0, "end": 75.0, "text": "The conversation turns to future projects and personal reflections."},
            {"id": 3, "start": 75.0, "end": 100.0, "text": "The host closes with a final question about upcoming plans."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="The interview covers the film, the guest's preparation, and what the experience meant personally.",
            bullet_summary=(
                "- The interview focuses on the guest's new film and performance.\n"
                "- The guest explains how preparation shaped the role.\n"
                "- The conversation moves to cast dynamics and audience reactions.\n"
                "- The host asks about future projects and long-term goals."
            ),
            short_summary="An interview about the guest's film, preparation, and future plans.",
        )
        self.assertEqual(result["action_items"], [])

    def test_key_points_are_semantic_not_raw_transcript_fragments(self):
        transcript_text = (
            "First open the builder and create the layout. "
            "Then connect the prompt generator so the hero section updates correctly. "
            "After that refine the animation timing and export the final version."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "First open the builder and create the layout."},
            {"id": 1, "start": 20.0, "end": 40.0, "text": "Then connect the prompt generator so the hero section updates correctly."},
            {"id": 2, "start": 40.0, "end": 60.0, "text": "After that refine the animation timing and export the final version."},
            {"id": 3, "start": 60.0, "end": 80.0, "text": "Finally review the page and prepare it for deployment."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="The tutorial explains how the interface is built, refined, and prepared for final delivery.",
            bullet_summary=(
                "- First open the builder and create the layout.\n"
                "- Then connect the prompt generator so the hero section updates correctly.\n"
                "- After that refine the animation timing and export the final version.\n"
                "- Finally review the page and prepare it for deployment."
            ),
            short_summary="A tutorial on building, refining, and preparing the interface for release.",
        )
        self.assertTrue(result["key_points"])
        self.assertTrue(all("First open the builder" not in point for point in result["key_points"]))
        self.assertTrue(all("Then connect the prompt generator" not in point for point in result["key_points"]))

    def test_chapter_titles_are_topic_labels_not_broken_fragments(self):
        transcript_text = (
            "The speaker introduces the workflow. Then the speaker explains prompt design. "
            "Next the video covers export steps and deployment checks."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 15.0, "text": "00:00What if I told you this UI was generated with only prompts"},
            {"id": 1, "start": 15.0, "end": 30.0, "text": "and now we move into prompt design for the hero section"},
            {"id": 2, "start": 30.0, "end": 45.0, "text": "then export settings and deployment checks are discussed"},
            {"id": 3, "start": 45.0, "end": 60.0, "text": "the speaker closes with review and final adjustments"},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="The video moves from workflow overview to prompt design, export, and review.",
            bullet_summary="- Workflow overview\n- Prompt design\n- Export settings\n- Review and final adjustments",
            short_summary="A walkthrough of workflow setup, prompt design, export, and review.",
        )
        self.assertTrue(result["chapters"])
        self.assertTrue(all("What if I told you" not in chapter["title"] for chapter in result["chapters"]))
        self.assertTrue(all(not chapter["title"].startswith("and ") for chapter in result["chapters"]))

    def test_tldr_reflects_full_video_scope(self):
        transcript_text = (
            "The tutorial starts by explaining the layout strategy. "
            "It then covers prompt writing, section generation, animation tuning, and final deployment."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "The tutorial starts with the layout strategy."},
            {"id": 1, "start": 20.0, "end": 40.0, "text": "It moves into prompt writing and section generation."},
            {"id": 2, "start": 40.0, "end": 60.0, "text": "Animation tuning and deployment are handled near the end."},
            {"id": 3, "start": 60.0, "end": 80.0, "text": "The speaker reviews the final outcome and checks the result."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="The tutorial shows how to plan the layout, write prompts, generate sections, tune motion, and deploy the final result.",
            bullet_summary="- Layout strategy\n- Prompt writing\n- Section generation\n- Animation tuning\n- Deployment review",
            short_summary="This tutorial explains the full workflow from layout planning to final deployment.",
        )
        tldr = result["tldr"].lower()
        self.assertIn("tutorial", tldr)
        self.assertTrue("layout" in tldr or "planning" in tldr)
        self.assertTrue("deploy" in tldr or "outcome" in tldr)

    def test_interview_tldr_contains_natural_language_and_main_participants(self):
        transcript_text = (
            "WIRED asks Robert Downey Jr. and Christopher Nolan autocomplete questions about Oppenheimer, "
            "their careers, practical effects, and memorable film moments. "
            "They answer each prompt with short stories about filmmaking and personal experiences."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "WIRED asks Robert Downey Jr. and Christopher Nolan the first autocomplete question."},
            {"id": 1, "start": 20.0, "end": 40.0, "text": "Robert Downey Jr. answers about Oppenheimer and his career."},
            {"id": 2, "start": 40.0, "end": 60.0, "text": "Christopher Nolan explains his filmmaking process and practical effects."},
            {"id": 3, "start": 60.0, "end": 80.0, "text": "They discuss films, personal anecdotes, and audience questions."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Oppenheimer, filmmaking, and career moments.",
            bullet_summary=(
                "- Robert Downey Jr. answers questions about Oppenheimer and his career.\n"
                "- Christopher Nolan explains his filmmaking process and practical effects.\n"
                "- The interview covers film memories, creative choices, and personal anecdotes.\n"
                "- WIRED uses autocomplete prompts to guide the discussion."
            ),
            short_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Oppenheimer, filmmaking, and career moments.",
        )
        self.assertRegex(result["tldr"], r"Robert Downey Jr\.?")
        self.assertIn("Christopher Nolan", result["tldr"])
        self.assertNotRegex(result["tldr"].lower(), r"\b(?:hi on|bit on|my on|does got|got does)\b")

    def test_malformed_token_cluster_phrases_are_rejected(self):
        transcript_text = (
            "Robert Downey Jr. talks about tattoos, hobbies, and Oppenheimer. "
            "Christopher Nolan answers questions about practical effects and inspiration."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "Robert Downey Jr. talks about tattoos, hobbies, and Oppenheimer."},
            {"id": 1, "start": 20.0, "end": 40.0, "text": "Christopher Nolan answers questions about practical effects and inspiration."},
            {"id": 2, "start": 40.0, "end": 60.0, "text": "The host keeps the conversation moving with autocomplete prompts."},
            {"id": 3, "start": 60.0, "end": 80.0, "text": "They discuss film memories and creative choices."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Oppenheimer, filmmaking, and personal anecdotes.",
            bullet_summary=(
                "- hi on robert downey got does\n"
                "- bit on film christopher nolan oppenheimer\n"
                "- my on accent american both were\n"
                "- The interview covers filmmaking and personal anecdotes."
            ),
            short_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Oppenheimer and filmmaking.",
        )
        rendered = " ".join([result["tldr"]] + result["key_points"] + [c["title"] for c in result["chapters"]]).lower()
        self.assertNotRegex(rendered, r"\b(?:hi on|bit on|my on|does got|got does|think want|does type|film nolan going first|looking downey robert)\b")

    def test_fallback_templates_produce_readable_output(self):
        transcript_text = (
            "Christopher Nolan and Robert Downey Jr. answer autocomplete prompts about Oppenheimer, practical effects, "
            "career milestones, and personal habits."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 18.0, "text": "Christopher Nolan and Robert Downey Jr. answer autocomplete prompts about Oppenheimer."},
            {"id": 1, "start": 18.0, "end": 36.0, "text": "They discuss practical effects, career milestones, and personal habits."},
            {"id": 2, "start": 36.0, "end": 54.0, "text": "The host keeps the questions moving quickly."},
            {"id": 3, "start": 54.0, "end": 72.0, "text": "The guests respond with short, conversational answers."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="",
            bullet_summary="- hi on robert downey got does\n- bit on film christopher nolan oppenheimer",
            short_summary="",
        )
        self.assertTrue(result["tldr"])
        self.assertTrue(all(len(point.split()) >= 4 for point in result["key_points"]))
        self.assertTrue(all(
            chapter["title"] in SAFE_INTERVIEW_CHAPTER_TITLES
            or 3 <= len(chapter["title"].split()) <= 10
            for chapter in result["chapters"]
        ))

    def test_chapter_titles_are_natural_topic_phrases(self):
        transcript_text = (
            "Robert Downey Jr. discusses Oppenheimer, tattoos, and hobbies. "
            "Christopher Nolan explains practical effects, filmmaking choices, and inspiration."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 15.0, "text": "Robert Downey Jr. discusses Oppenheimer and his tattoos."},
            {"id": 1, "start": 15.0, "end": 30.0, "text": "He also talks about hobbies and personal habits."},
            {"id": 2, "start": 30.0, "end": 45.0, "text": "Christopher Nolan explains practical effects and filmmaking choices."},
            {"id": 3, "start": 45.0, "end": 60.0, "text": "He describes the inspiration behind the film."},
            {"id": 4, "start": 60.0, "end": 75.0, "text": "The interview closes with fast personal questions."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Oppenheimer, practical effects, and personal questions.",
            bullet_summary=(
                "- Robert Downey Jr. discusses Oppenheimer, tattoos, and hobbies.\n"
                "- Christopher Nolan explains practical effects and filmmaking choices.\n"
                "- The interview also covers inspiration and fast personal questions.\n"
                "- The discussion moves between film craft and personal anecdotes."
            ),
            short_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Oppenheimer, filmmaking, and personal questions.",
        )
        self.assertTrue(result["chapters"])
        self.assertTrue(all(
            chapter["title"] in SAFE_INTERVIEW_CHAPTER_TITLES
            or 3 <= len(chapter["title"].split()) <= 10
            for chapter in result["chapters"]
        ))
        self.assertTrue(all(not re.search(r"\b(?:hi on|bit on|my on|does got|got does)\b", chapter["title"].lower()) for chapter in result["chapters"]))

    def test_interview_tldr_rejects_fake_participants(self):
        transcript_text = (
            "Hi, Robert Downey Jr. and Christopher Nolan answer autocomplete questions about Oppenheimer, "
            "filmmaking, and career highlights. Very early prompts lead into a broader interview."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 15.0, "text": "Hi, Robert Downey Jr. and Christopher Nolan answer the first autocomplete question."},
            {"id": 1, "start": 15.0, "end": 30.0, "text": "Robert Downey Jr. talks about Oppenheimer and career highlights."},
            {"id": 2, "start": 30.0, "end": 45.0, "text": "Christopher Nolan discusses filmmaking and practical effects."},
            {"id": 3, "start": 45.0, "end": 60.0, "text": "Robert Downey Jr. and Christopher Nolan answer more personal questions."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Oppenheimer, filmmaking, and career highlights.",
            bullet_summary="- Hi and Robert Downey Jr.\n- Very on Robert Downey Jr.\n- Nolan on filmmaking and practical effects.",
            short_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Oppenheimer and filmmaking.",
        )
        self.assertIn("Robert Downey Jr.", result["tldr"])
        self.assertIn("Christopher Nolan", result["tldr"])
        self.assertNotIn("Hi and Robert", result["tldr"])
        self.assertNotIn("Very", result["tldr"])

    def test_interview_chapters_use_safe_topic_labels(self):
        transcript_text = (
            "Robert Downey Jr. and Christopher Nolan answer autocomplete questions about Oppenheimer, practical effects, "
            "personal habits, and career moments."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 18.0, "text": "Robert Downey Jr. and Christopher Nolan answer autocomplete questions."},
            {"id": 1, "start": 18.0, "end": 36.0, "text": "Christopher Nolan explains practical effects and filmmaking choices."},
            {"id": 2, "start": 36.0, "end": 54.0, "text": "Robert Downey Jr. talks about Iron Man, hobbies, and personal habits."},
            {"id": 3, "start": 54.0, "end": 72.0, "text": "The interview closes with faster personal questions."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Oppenheimer, practical effects, and personal topics.",
            bullet_summary="- Pointless Pointless Question If\n- Nolan Film Nolan Going First\n- Looking Downey Robert Does Type",
            short_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Oppenheimer and filmmaking.",
        )
        titles = [chapter["title"] for chapter in result["chapters"]]
        self.assertTrue(any(title in {
            "Interview Introduction",
            "Filmmaking And Oppenheimer",
            "Marvel / Iron Man",
            "Hobbies And Personal Life",
            "Career Discussion",
            "Closing Questions",
            "Nolan On Practical Effects",
            "Downey On Iron Man",
        } for title in titles))
        rendered = " ".join(titles).lower()
        self.assertNotRegex(rendered, r"\b(?:pointless|tick unique|does type|looking downey robert|film nolan going first)\b")

    def test_interview_tldr_uses_safe_semantic_topics(self):
        transcript_text = (
            "Christopher Nolan and Robert Downey Jr. answer autocomplete questions about Oppenheimer, practical effects, "
            "Iron Man, hobbies, and career milestones."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 18.0, "text": "The host opens the interview with autocomplete questions."},
            {"id": 1, "start": 18.0, "end": 36.0, "text": "Christopher Nolan explains practical effects and filmmaking choices for Oppenheimer."},
            {"id": 2, "start": 36.0, "end": 54.0, "text": "Robert Downey Jr. talks about Iron Man, hobbies, and personal life."},
            {"id": 3, "start": 54.0, "end": 72.0, "text": "The interview closes with career highlights and personal questions."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, Iron Man, and personal topics.",
            bullet_summary=(
                "- Christopher Nolan explains practical effects and Oppenheimer.\n"
                "- Robert Downey Jr. discusses Iron Man, hobbies, and personal life.\n"
                "- The conversation includes career highlights and closing questions."
            ),
            short_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, and career highlights.",
        )
        lowered = result["tldr"].lower()
        self.assertIn("christopher nolan", lowered)
        self.assertIn("robert downey jr", lowered)
        self.assertRegex(lowered, r"(filmmaking|oppenheimer)")
        self.assertNotRegex(lowered, r"\b(?:hi|very|pointless|tick unique|does type)\b")

    def test_interview_duplicate_or_weak_labels_are_regenerated(self):
        transcript_text = (
            "Christopher Nolan discusses practical effects and Oppenheimer. "
            "Christopher Nolan also answers a second question about practical effects. "
            "Robert Downey Jr. talks about Iron Man, hobbies, and personal questions."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 18.0, "text": "Christopher Nolan discusses practical effects and Oppenheimer."},
            {"id": 1, "start": 18.0, "end": 36.0, "text": "Christopher Nolan answers another question about practical effects."},
            {"id": 2, "start": 36.0, "end": 54.0, "text": "Robert Downey Jr. talks about Iron Man and hobbies."},
            {"id": 3, "start": 54.0, "end": 72.0, "text": "The interview closes with personal and career questions."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, Iron Man, and personal questions.",
            bullet_summary=(
                "- Nolan Film Nolan Going First\n"
                "- Nolan Film Nolan Going First\n"
                "- Looking Downey Robert Does Type\n"
                "- Pointless Pointless Question If"
            ),
            short_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking and personal questions.",
        )
        titles = [chapter["title"] for chapter in result["chapters"]]
        self.assertEqual(len(titles), len(set(titles)))
        self.assertTrue(all(title in SAFE_INTERVIEW_CHAPTER_TITLES for title in titles))

    def test_interview_key_points_are_natural_semantic_sentences(self):
        transcript_text = (
            "Christopher Nolan explains why he prefers practical effects over CGI in Oppenheimer. "
            "Robert Downey Jr. talks about Iron Man, Marvel, tattoos, and hobbies. "
            "The interview also covers career highlights and personal anecdotes."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "Christopher Nolan explains why he prefers practical effects over CGI in Oppenheimer."},
            {"id": 1, "start": 20.0, "end": 40.0, "text": "Robert Downey Jr. talks about Iron Man, Marvel, tattoos, and hobbies."},
            {"id": 2, "start": 40.0, "end": 60.0, "text": "The interview also covers career highlights and personal anecdotes."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, Marvel, and personal anecdotes.",
            bullet_summary=(
                "- Christopher Nolan explains practical effects and Oppenheimer.\n"
                "- Robert Downey Jr. discusses Marvel, Iron Man, and personal interests.\n"
                "- The interview covers career highlights and personal anecdotes."
            ),
            short_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, and personal topics.",
        )
        self.assertTrue(result["key_points"])
        self.assertTrue(all(point.endswith(".") for point in result["key_points"]))
        self.assertTrue(all(len(point.split()) >= 6 for point in result["key_points"]))
        rendered = " ".join(result["key_points"]).lower()
        self.assertNotRegex(rendered, r"\b(?:gwyneth paltrow i do not|think want|does type)\b")

    def test_transcript_like_interview_bullets_are_rejected(self):
        transcript_text = (
            "Robert Downey Jr. says Gwyneth Paltrow is a longtime co-star and friend. "
            "Christopher Nolan discusses practical effects and Oppenheimer."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "Robert Downey Jr. says Gwyneth Paltrow is a longtime co-star and friend."},
            {"id": 1, "start": 20.0, "end": 40.0, "text": "Christopher Nolan discusses practical effects and Oppenheimer."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing co-stars, filmmaking, and Oppenheimer.",
            bullet_summary=(
                "- Gwyneth Paltrow I do not Gwyneth Paltrow\n"
                "- Robert Downey Jr. says Gwyneth Paltrow is a longtime co-star and friend.\n"
                "- Christopher Nolan discusses practical effects and Oppenheimer."
            ),
            short_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing filmmaking and co-stars.",
        )
        rendered = " ".join(result["key_points"]).lower()
        self.assertNotIn("gwyneth paltrow i do not", rendered)
        self.assertNotIn("longtime co-star and friend", rendered)

    def test_interview_dialogue_fragments_are_rejected_from_key_points(self):
        transcript_text = (
            "Robert Downey Jr. answers a question about Marvel and Iron Man. "
            "Christopher Nolan discusses practical effects and Oppenheimer."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "How did Robert Downey Jr. become Iron Man?"},
            {"id": 1, "start": 20.0, "end": 40.0, "text": "I probably should have taken my blood pressure medication before this interview."},
            {"id": 2, "start": 40.0, "end": 60.0, "text": "Christopher Nolan discusses practical effects and Oppenheimer."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Marvel, Iron Man, and Oppenheimer.",
            bullet_summary=(
                "- The interview covers how did Robert Downey Jr.\n"
                "- The interview covers i probably should have taken my blood pressure medication.\n"
                "- The interview covers why Christopher Nolan.\n"
            ),
            short_summary="This interview features Robert Downey Jr. and Christopher Nolan discussing Marvel, Iron Man, and filmmaking.",
        )
        rendered = " ".join(result["key_points"]).lower()
        self.assertNotIn("how did robert downey jr", rendered)
        self.assertNotIn("blood pressure medication", rendered)
        self.assertNotIn("why christopher nolan", rendered)
        self.assertTrue(all(point.endswith(".") for point in result["key_points"]))
        self.assertTrue(all(len(point.split()) >= 6 for point in result["key_points"]))

    def test_interview_tldr_does_not_repeat_same_topic_twice(self):
        transcript_text = (
            "Christopher Nolan and Robert Downey Jr. discuss filmmaking, Oppenheimer, and career highlights. "
            "The interview returns to filmmaking and Oppenheimer in later questions."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "Christopher Nolan and Robert Downey Jr. discuss filmmaking and Oppenheimer."},
            {"id": 1, "start": 20.0, "end": 40.0, "text": "The interview returns to career highlights and personal questions."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, and career highlights.",
            bullet_summary="- The interview covers filmmaking and Oppenheimer.\n- The interview covers career highlights and personal questions.",
            short_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, and career highlights.",
        )
        lowered = result["tldr"].lower()
        self.assertLessEqual(lowered.count("filmmaking"), 1)
        self.assertLessEqual(lowered.count("oppenheimer"), 1)

    def test_generic_interview_fallback_does_not_leak_nolan_downey_names(self):
        transcript_text = (
            "The host speaks with two startup founders about product decisions, customer feedback, "
            "team growth, and lessons from building their company."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "The host opens with quick background questions about the company."},
            {"id": 1, "start": 20.0, "end": 40.0, "text": "The guests discuss product decisions, team growth, and customer feedback."},
            {"id": 2, "start": 40.0, "end": 60.0, "text": "They reflect on hiring, execution, and lessons learned."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="",
            bullet_summary="- bad token cluster\n- another malformed phrase",
            short_summary="",
        )
        rendered = " ".join([result["tldr"]] + result["key_points"] + [chapter["title"] for chapter in result["chapters"]]).lower()
        self.assertNotIn("christopher nolan", rendered)
        self.assertNotIn("robert downey jr", rendered)
        self.assertNotIn("oppenheimer", rendered)
        self.assertNotIn("iron man", rendered)
        self.assertNotIn("marvel", rendered)
        self.assertTrue(result["key_points"])
        self.assertTrue(any(term in rendered for term in ("product", "customer", "team", "company", "project", "career")))

    def test_structured_summary_ignores_foreign_summary_support_when_transcript_differs(self):
        transcript_text = (
            "This interview features two startup founders discussing product strategy, hiring, "
            "customer feedback, and execution challenges."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "The guests discuss product strategy and customer feedback."},
            {"id": 1, "start": 20.0, "end": 40.0, "text": "They talk about hiring, team growth, and execution challenges."},
            {"id": 2, "start": 40.0, "end": 60.0, "text": "The host asks about lessons learned while building the company."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing Oppenheimer and filmmaking.",
            bullet_summary="- Christopher Nolan explains practical effects.\n- Robert Downey Jr. discusses Iron Man.",
            short_summary="This interview features Christopher Nolan and Robert Downey Jr.",
        )
        rendered = " ".join([result["tldr"]] + result["key_points"] + [chapter["title"] for chapter in result["chapters"]]).lower()
        self.assertNotIn("christopher nolan", rendered)
        self.assertNotIn("robert downey jr", rendered)
        self.assertNotIn("oppenheimer", rendered)
        self.assertNotIn("iron man", rendered)
        self.assertNotIn("marvel", rendered)
        self.assertTrue(any(term in rendered for term in ("product", "customer", "hiring", "team", "execution")))

    def test_near_duplicate_interview_chapters_are_deduplicated(self):
        transcript_text = (
            "Christopher Nolan discusses practical effects and Oppenheimer. "
            "Christopher Nolan returns to practical effects again later. "
            "Robert Downey Jr. talks about Iron Man and Marvel."
        )
        segments = [
            {"id": 0, "start": 0.0, "end": 18.0, "text": "Christopher Nolan discusses practical effects and Oppenheimer."},
            {"id": 1, "start": 18.0, "end": 36.0, "text": "Christopher Nolan returns to practical effects and Oppenheimer later."},
            {"id": 2, "start": 36.0, "end": 54.0, "text": "Robert Downey Jr. talks about Iron Man and Marvel."},
        ]
        result = build_structured_summary(
            transcript_text=transcript_text,
            segments=segments,
            full_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, and Marvel.",
            bullet_summary=(
                "- Christopher Nolan explains practical effects and Oppenheimer.\n"
                "- Christopher Nolan explains practical effects and Oppenheimer again.\n"
                "- Robert Downey Jr. discusses Marvel and Iron Man."
            ),
            short_summary="This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking and Marvel.",
        )
        titles = [chapter["title"] for chapter in result["chapters"]]
        self.assertEqual(len(titles), len(set(titles)))


class MalayalamWorkflowRegressionTests(TestCase):
    def setUp(self):
        self.video = Video.objects.create(
            title="Malayalam Workflow Regression",
            status="processing",
            processing_progress=20,
            duration=180.0,
        )

    def _draft_transcript(self):
        return Transcript.objects.create(
            video=self.video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="groq_whisper",
            asr_engine_used="groq_whisper",
            transcript_quality_score=0.61,
            full_text="draft text",
            transcript_original_text="draft text",
            transcript_canonical_text="",
            transcript_canonical_en_text="",
            json_data={},
            word_timestamps=[],
        )

    def _quality_payload(self, state: str, reason: str, warning: str = ""):
        return {
            "state": state,
            "warnings": [],
            "qa_metrics": {
                "malayalam_post_asr_accepted": False,
                "malayalam_post_asr_mode": state if state != "cleaned" else "accepted",
                "malayalam_post_asr_accept_reason": reason,
                "garbled_detector_score": 0.0,
                "script_detector_result": "other",
                "transcript_trust": {},
            },
            "quality_score": 0.61,
            "message": warning if state != "failed" else "Transcript quality failed the garble checks. Summaries and chatbot were not generated.",
            "malayalam_post_asr_mode": state if state != "cleaned" else "accepted",
            "malayalam_post_asr_reason": reason,
            "transcript_warning_message": warning,
            "summary_blocked_reason": "" if state == "degraded" else "transcript_failed_garble_gate" if state == "failed" else "",
            "chatbot_blocked_reason": "" if state == "degraded" else "transcript_failed_garble_gate" if state == "failed" else "",
        }

    def _set_malayalam_downstream_gate_payload(
        self,
        transcript,
        *,
        display_units,
        visible_word_count,
        readability,
        lexical_trust,
        wrong_script_segments,
        corrupted_segments,
        total_segments,
        state="degraded",
    ):
        transcript.json_data = {
            "language": "ml",
            "transcript_state": state,
            "display_transcript_units": display_units,
            "quality_metrics": {
                "overall_readability": readability,
                "lexical_trust_score": lexical_trust,
            },
            "asr_metadata": {
                "malayalam_display_refinement": {
                    "final_visible_word_count": visible_word_count,
                },
                "transcript_trust": {
                    "overall_readability": readability,
                    "lexical_trust_score": lexical_trust,
                    "wrong_script_segments": wrong_script_segments,
                    "corrupted_segments": corrupted_segments,
                    "total_segments": total_segments,
                },
            },
        }
        transcript.save(update_fields=["json_data"])

    @patch("videos.tasks._rebuild_chatbot_index")
    @patch("videos.tasks._rebuild_highlights")
    @patch("videos.tasks._upsert_all_summaries")
    @patch("videos.tasks._finalize_transcript_record")
    @patch("videos.tasks._create_draft_transcript_record")
    @patch("videos.tasks.build_canonical_text")
    @patch("videos.tasks.transcribe_video")
    def test_low_trust_structurally_usable_malayalam_does_not_raise_and_video_does_not_fail(
        self,
        mock_transcribe,
        mock_canonical,
        mock_create_draft,
        mock_finalize,
        mock_summaries,
        _highlights,
        _chatbot,
    ):
        draft = self._draft_transcript()
        mock_create_draft.return_value = draft
        mock_transcribe.return_value = {"text": "sample", "segments": [], "language": "ml", "metadata": {"asr_provider_used": "groq_whisper"}}
        mock_canonical.return_value = {"canonical_text": "sample", "canonical_segments": []}
        mock_finalize.return_value = (
            draft,
            self._quality_payload(
                "degraded",
                "semantic_trust_too_low",
                "Malayalam transcript is low-trust but structurally usable; continuing in degraded mode.",
            ),
        )
        mock_summaries.side_effect = KeyError("low_trust_malayalam")

        result = _run_audio_pipeline(
            self.video,
            audio_path="audio.wav",
            source_type="youtube",
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertNotEqual(self.video.status, "failed")
        self.assertEqual(self.video.status, "completed")

    @patch("videos.serializers.get_or_build_structured_summary", return_value={"summary_state": "degraded_safe"})
    @patch("videos.tasks._rebuild_chatbot_index")
    @patch("videos.tasks._rebuild_highlights")
    @patch("videos.tasks._upsert_all_summaries")
    @patch("videos.tasks._finalize_transcript_record")
    @patch("videos.tasks._create_draft_transcript_record")
    @patch("videos.tasks.build_canonical_text")
    @patch("videos.tasks.transcribe_video")
    def test_low_evidence_malayalam_skips_summaries_highlights_and_indexing(
        self,
        mock_transcribe,
        mock_canonical,
        mock_create_draft,
        mock_finalize,
        mock_summaries,
        mock_highlights,
        mock_chatbot,
        _structured,
    ):
        draft = self._draft_transcript()
        self._set_malayalam_downstream_gate_payload(
            draft,
            display_units=[],
            visible_word_count=0,
            readability=0.14,
            lexical_trust=0.08,
            wrong_script_segments=6,
            corrupted_segments=6,
            total_segments=6,
        )
        Summary.objects.create(video=self.video, summary_type="short", title="Short", content="stale")
        HighlightSegment.objects.create(
            video=self.video,
            start_time=0.0,
            end_time=2.0,
            importance_score=0.7,
            reason="stale",
            transcript_snippet="stale",
        )
        mock_create_draft.return_value = draft
        mock_transcribe.return_value = {"text": "sample", "segments": [], "language": "ml", "metadata": {"asr_provider_used": "groq_whisper"}}
        mock_canonical.return_value = {"canonical_text": "sample", "canonical_segments": []}
        mock_finalize.return_value = (
            draft,
            self._quality_payload(
                "degraded",
                "semantic_trust_too_low",
                "Malayalam transcript quality was too low for reliable summarization.",
            ),
        )

        result = _run_audio_pipeline(
            self.video,
            audio_path="audio.wav",
            source_type="youtube",
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        draft.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertFalse(mock_summaries.called)
        self.assertFalse(mock_highlights.called)
        self.assertFalse(mock_chatbot.called)
        self.assertEqual(Summary.objects.filter(video=self.video).count(), 0)
        self.assertEqual(HighlightSegment.objects.filter(video=self.video).count(), 0)
        self.assertTrue(draft.json_data.get("downstream_suppressed"))
        self.assertEqual(draft.json_data.get("downstream_suppression_reason"), "no_trusted_display_units")
        self.assertTrue(draft.json_data.get("low_evidence_malayalam"))
        self.assertEqual(draft.json_data.get("quality_metrics", {}).get("summary_blocked_reason"), "low_evidence_malayalam_gate")
        self.assertEqual(draft.json_data.get("quality_metrics", {}).get("chatbot_blocked_reason"), "low_evidence_malayalam_gate")

    @patch("videos.serializers.get_or_build_structured_summary", return_value={"summary_state": "degraded_safe"})
    @patch("videos.tasks._rebuild_chatbot_index")
    @patch("videos.tasks._rebuild_highlights")
    @patch("videos.tasks._upsert_all_summaries")
    @patch("videos.tasks._finalize_transcript_record")
    @patch("videos.tasks._create_draft_transcript_record")
    @patch("videos.tasks.build_canonical_text")
    @patch("videos.tasks.transcribe_video")
    def test_low_evidence_malayalam_stays_blocked_after_confusion_retry_without_material_improvement(
        self,
        mock_transcribe,
        mock_canonical,
        mock_create_draft,
        mock_finalize,
        mock_summaries,
        mock_highlights,
        mock_chatbot,
        _structured,
    ):
        draft = self._draft_transcript()
        self._set_malayalam_downstream_gate_payload(
            draft,
            display_units=[],
            visible_word_count=0,
            readability=0.155,
            lexical_trust=0.047,
            wrong_script_segments=3,
            corrupted_segments=4,
            total_segments=4,
        )
        mock_create_draft.return_value = draft
        mock_transcribe.return_value = {
            "text": "topic model answer revision class",
            "segments": [],
            "language": "ml",
            "metadata": {
                "asr_provider_used": "groq_whisper",
                "second_pass_asr_attempted": True,
                "second_pass_asr_reason": "high_garble_malayalam_confusion_retry_candidate",
                "second_pass_asr_model": "large-v3",
                "second_pass_asr_improved": False,
                "confusion_retry_candidate": True,
                "confusion_retry_executed": True,
                "confusion_retry_model": "large-v3",
                "confusion_retry_improved": False,
                "confusion_retry_improvement_reason": "not_materially_better",
            },
        }
        mock_canonical.return_value = {"canonical_text": "topic model answer revision class", "canonical_segments": []}
        mock_finalize.return_value = (
            draft,
            self._quality_payload(
                "degraded",
                "semantic_trust_too_low",
                "Malayalam transcript quality was too low for reliable summarization.",
            ),
        )

        result = _run_audio_pipeline(
            self.video,
            audio_path="audio.wav",
            source_type="youtube",
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        draft.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertFalse(mock_summaries.called)
        self.assertFalse(mock_highlights.called)
        self.assertFalse(mock_chatbot.called)
        self.assertTrue(draft.json_data.get("downstream_suppressed"))
        self.assertEqual(draft.json_data.get("downstream_suppression_reason"), "no_trusted_display_units")
        self.assertEqual(draft.json_data.get("quality_metrics", {}).get("summary_blocked_reason"), "low_evidence_malayalam_gate")
        self.assertEqual(draft.json_data.get("quality_metrics", {}).get("chatbot_blocked_reason"), "low_evidence_malayalam_gate")

    @patch("videos.serializers.get_or_build_structured_summary", return_value={"summary_state": "degraded_safe"})
    @patch("videos.tasks._rebuild_chatbot_index")
    @patch("videos.tasks._rebuild_highlights")
    @patch("videos.tasks._upsert_all_summaries")
    @patch("videos.tasks._finalize_transcript_record")
    @patch("videos.tasks._create_draft_transcript_record")
    @patch("videos.tasks.build_canonical_text")
    @patch("videos.tasks.transcribe_video")
    def test_degraded_malayalam_with_trusted_visible_content_does_not_trigger_downstream_gate(
        self,
        mock_transcribe,
        mock_canonical,
        mock_create_draft,
        mock_finalize,
        mock_summaries,
        mock_highlights,
        mock_chatbot,
        _structured,
    ):
        draft = self._draft_transcript()
        self._set_malayalam_downstream_gate_payload(
            draft,
            display_units=[
                {
                    "id": 0,
                    "text": "confidence exam hall support",
                    "unit_readability": 0.58,
                    "wrong_script_ratio": 0.0,
                    "contamination_score": 0.12,
                }
            ],
            visible_word_count=14,
            readability=0.31,
            lexical_trust=0.26,
            wrong_script_segments=1,
            corrupted_segments=1,
            total_segments=5,
        )
        mock_create_draft.return_value = draft
        mock_transcribe.return_value = {"text": "sample", "segments": [], "language": "ml", "metadata": {"asr_provider_used": "groq_whisper"}}
        mock_canonical.return_value = {"canonical_text": "sample", "canonical_segments": []}
        mock_finalize.return_value = (
            draft,
            self._quality_payload(
                "degraded",
                "semantic_trust_too_low",
                "Malayalam transcript quality was too low for reliable summarization.",
            ),
        )

        _run_audio_pipeline(
            self.video,
            audio_path="audio.wav",
            source_type="youtube",
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        draft.refresh_from_db()
        self.assertTrue(mock_summaries.called)
        self.assertTrue(mock_highlights.called)
        self.assertTrue(mock_chatbot.called)
        self.assertFalse(draft.json_data.get("downstream_suppressed"))

    @patch("videos.tasks._rebuild_chatbot_index")
    @patch("videos.tasks._rebuild_highlights")
    @patch("videos.tasks._upsert_all_summaries")
    @patch("videos.tasks._finalize_transcript_record")
    @patch("videos.tasks._create_draft_transcript_record")
    @patch("videos.tasks.build_canonical_text")
    @patch("videos.tasks.transcribe_video")
    def test_stale_draft_row_but_finalized_malayalam_quality_uses_final_qa_state(
        self,
        mock_transcribe,
        mock_canonical,
        mock_create_draft,
        mock_finalize,
        mock_summaries,
        mock_highlights,
        mock_chatbot,
    ):
        draft = self._draft_transcript()
        draft.json_data = {
            **(draft.json_data or {}),
            "language": "ml",
            "transcript_state": "draft",
        }
        draft.save(update_fields=["json_data"])
        self._set_malayalam_downstream_gate_payload(
            draft,
            display_units=[
                {
                    "id": 0,
                    "text": "confidence exam hall support",
                    "unit_readability": 0.58,
                    "wrong_script_ratio": 0.0,
                    "contamination_score": 0.12,
                }
            ],
            visible_word_count=14,
            readability=0.31,
            lexical_trust=0.26,
            wrong_script_segments=1,
            corrupted_segments=1,
            total_segments=5,
            state="draft",
        )
        mock_create_draft.return_value = draft
        mock_transcribe.return_value = {"text": "sample", "segments": [], "language": "ml", "metadata": {"asr_provider_used": "groq_whisper"}}
        mock_canonical.return_value = {"canonical_text": "sample", "canonical_segments": []}
        mock_finalize.return_value = (
            draft,
            self._quality_payload(
                "degraded",
                "semantic_trust_too_low",
                "Malayalam transcript quality was too low for reliable summarization.",
            ),
        )

        _run_audio_pipeline(
            self.video,
            audio_path="audio.wav",
            source_type="youtube",
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.assertTrue(mock_summaries.called)
        self.assertTrue(mock_highlights.called)
        self.assertTrue(mock_chatbot.called)

    def test_cleaned_malayalam_does_not_trigger_low_evidence_downstream_gate(self):
        transcript = self._draft_transcript()
        self._set_malayalam_downstream_gate_payload(
            transcript,
            display_units=[
                {
                    "id": 0,
                    "text": "നിങ്ങൾ confidence നിലനിർത്തണം",
                    "unit_readability": 0.64,
                    "wrong_script_ratio": 0.0,
                    "contamination_score": 0.08,
                }
            ],
            visible_word_count=18,
            readability=0.62,
            lexical_trust=0.58,
            wrong_script_segments=0,
            corrupted_segments=0,
            total_segments=3,
            state="cleaned",
        )
        self.assertEqual(_should_suppress_low_trust_malayalam_outputs(transcript), (False, ""))

    @patch("videos.tasks._rebuild_chatbot_index")
    @patch("videos.tasks._rebuild_highlights")
    @patch("videos.tasks._upsert_all_summaries")
    @patch("videos.tasks._finalize_transcript_record")
    @patch("videos.tasks._create_draft_transcript_record")
    @patch("videos.tasks.build_canonical_text")
    @patch("videos.tasks.transcribe_video")
    def test_low_trust_structurally_usable_malayalam_can_continue_degraded_summary_path(
        self,
        mock_transcribe,
        mock_canonical,
        mock_create_draft,
        mock_finalize,
        mock_summaries,
        _highlights,
        _chatbot,
    ):
        draft = self._draft_transcript()
        mock_create_draft.return_value = draft
        mock_transcribe.return_value = {"text": "sample", "segments": [], "language": "ml", "metadata": {"asr_provider_used": "groq_whisper"}}
        mock_canonical.return_value = {"canonical_text": "sample", "canonical_segments": []}
        mock_finalize.return_value = (
            draft,
            self._quality_payload(
                "degraded",
                "semantic_trust_too_low",
                "Malayalam transcript is low-trust but structurally usable; continuing in degraded mode.",
            ),
        )
        mock_summaries.return_value = None

        result = _run_audio_pipeline(
            self.video,
            audio_path="audio.wav",
            source_type="youtube",
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.video.status, "completed")
        self.assertGreaterEqual(mock_summaries.call_count, 2)

    @patch("videos.tasks._upsert_all_summaries")
    @patch("videos.tasks._finalize_transcript_record")
    @patch("videos.tasks._create_draft_transcript_record")
    @patch("videos.tasks.build_canonical_text")
    @patch("videos.tasks.transcribe_video")
    def test_low_trust_structurally_unusable_malayalam_still_fails(
        self,
        mock_transcribe,
        mock_canonical,
        mock_create_draft,
        mock_finalize,
        mock_summaries,
    ):
        draft = self._draft_transcript()
        mock_create_draft.return_value = draft
        mock_transcribe.return_value = {"text": "sample", "segments": [], "language": "ml", "metadata": {"asr_provider_used": "groq_whisper"}}
        mock_canonical.return_value = {"canonical_text": "sample", "canonical_segments": []}
        mock_finalize.return_value = (
            draft,
            self._quality_payload(
                "failed",
                "structurally_unusable_low_trust_malayalam",
            ),
        )

        result = _run_audio_pipeline(
            self.video,
            audio_path="audio.wav",
            source_type="youtube",
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.video.status, "failed")
        mock_summaries.assert_not_called()

    @patch("videos.tasks._rebuild_chatbot_index")
    @patch("videos.tasks._rebuild_highlights")
    @patch("videos.tasks._upsert_all_summaries")
    @patch("videos.tasks._finalize_transcript_record")
    @patch("videos.tasks._create_draft_transcript_record")
    @patch("videos.tasks.build_canonical_text")
    @patch("videos.tasks.transcribe_video")
    def test_no_failed_status_overwrite_after_degraded_low_trust_resolution(
        self,
        mock_transcribe,
        mock_canonical,
        mock_create_draft,
        mock_finalize,
        mock_summaries,
        _highlights,
        _chatbot,
    ):
        draft = self._draft_transcript()
        mock_create_draft.return_value = draft
        mock_transcribe.return_value = {"text": "sample", "segments": [], "language": "ml", "metadata": {"asr_provider_used": "groq_whisper"}}
        mock_canonical.return_value = {"canonical_text": "sample", "canonical_segments": []}
        mock_finalize.return_value = (
            draft,
            self._quality_payload(
                "degraded",
                "low_trust_malayalam_segment_share_too_high",
                "Malayalam transcript is low-trust but structurally usable; continuing in degraded mode.",
            ),
        )
        mock_summaries.side_effect = KeyError("low_trust_malayalam")

        result = _run_audio_pipeline(
            self.video,
            audio_path="audio.wav",
            source_type="youtube",
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertNotEqual(self.video.status, "failed")

    @patch("videos.tasks._rebuild_chatbot_index")
    @patch("videos.tasks._rebuild_highlights")
    @patch("videos.tasks._upsert_all_summaries")
    @patch("videos.tasks._finalize_transcript_record")
    @patch("videos.tasks._create_draft_transcript_record")
    @patch("videos.tasks.build_canonical_text")
    @patch("videos.tasks.transcribe_video")
    def test_nested_low_trust_exception_still_continues_degraded(
        self,
        mock_transcribe,
        mock_canonical,
        mock_create_draft,
        mock_finalize,
        mock_summaries,
        _highlights,
        _chatbot,
    ):
        draft = self._draft_transcript()
        mock_create_draft.return_value = draft
        mock_transcribe.return_value = {"text": "sample", "segments": [], "language": "ml", "metadata": {"asr_provider_used": "groq_whisper"}}
        mock_canonical.return_value = {"canonical_text": "sample", "canonical_segments": []}
        mock_finalize.return_value = (
            draft,
            self._quality_payload(
                "degraded",
                "structurally_usable_low_trust_malayalam",
                "Malayalam transcript is low-trust but structurally usable; continuing in degraded mode.",
            ),
        )
        wrapped = RuntimeError("summary_wrapper")
        wrapped.__cause__ = KeyError("low_trust_malayalam")
        mock_summaries.side_effect = wrapped

        result = _run_audio_pipeline(
            self.video,
            audio_path="audio.wav",
            source_type="youtube",
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.video.status, "completed")

    @patch("videos.tasks._run_audio_pipeline", side_effect=KeyError("low_trust_malayalam"))
    @patch("videos.tasks.get_video_duration", return_value=120.0)
    @patch("videos.tasks.os.path.getsize", return_value=100000)
    @patch("videos.tasks.os.path.exists", return_value=True)
    @patch("videos.tasks.subprocess.run")
    def test_low_trust_structurally_usable_malayalam_sync_wrapper_does_not_fail_video(
        self,
        mock_run,
        _exists,
        _getsize,
        _duration,
        _pipeline,
    ):
        self.video.youtube_url = "https://youtube.test/watch?v=abc"
        self.video.save(update_fields=["youtube_url", "updated_at"])
        Transcript.objects.create(
            video=self.video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="groq_whisper",
            asr_engine_used="groq_whisper",
            transcript_quality_score=0.31,
            full_text="sample",
            transcript_original_text="sample",
            transcript_canonical_text="sample",
            transcript_canonical_en_text="sample",
            json_data={
                "language": "ml",
                "transcript_state": "degraded",
                "transcript_warning_message": "Malayalam transcript is low-trust but structurally usable; continuing in degraded mode.",
                "malayalam_post_asr_reason": "structurally_usable_low_trust_malayalam",
                "quality_metrics": {},
            },
            word_timestamps=[],
        )
        mock_run.return_value = type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

        result = process_youtube_video_sync(
            self.video.id,
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.video.status, "completed")

    @override_settings(YTDLP_COOKIES_FROM_BROWSERS=["edge"])
    @patch("videos.tasks._run_audio_pipeline", return_value={"status": "success", "video_id": "demo"})
    @patch("videos.tasks.get_video_duration", return_value=120.0)
    @patch("videos.tasks.os.path.getsize", return_value=100000)
    @patch("videos.tasks.os.path.exists", return_value=True)
    @patch("videos.tasks.subprocess.run")
    def test_youtube_sync_download_retries_with_cookie_fallback_after_bot_protection(
        self,
        mock_run,
        _exists,
        _getsize,
        _duration,
        _pipeline,
    ):
        self.video.youtube_url = "https://youtube.test/watch?v=abc"
        self.video.save(update_fields=["youtube_url", "updated_at"])

        def _result(returncode, stderr=""):
            return type("Result", (), {"returncode": returncode, "stderr": stderr, "stdout": ""})()

        mock_run.side_effect = [
            _result(1, "ERROR: HTTP Error 429: Too Many Requests\nERROR: Sign in to confirm you're not a bot\nUse --cookies-from-browser"),
            _result(0, ""),
        ]

        result = process_youtube_video_sync(
            self.video.id,
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertEqual(mock_run.call_count, 2)
        first_args = mock_run.call_args_list[0].args[0]
        second_args = mock_run.call_args_list[1].args[0]
        self.assertNotIn("--cookies-from-browser", first_args)
        self.assertIn("--cookies-from-browser", second_args)
        self.assertIn("edge", second_args)

    @override_settings(YTDLP_COOKIES_FROM_BROWSERS=["edge"])
    @patch("videos.tasks.subprocess.run")
    def test_youtube_sync_download_surfaces_clear_bot_protection_failure(
        self,
        mock_run,
    ):
        self.video.youtube_url = "https://youtube.test/watch?v=abc"
        self.video.save(update_fields=["youtube_url", "updated_at"])
        mock_run.return_value = type(
            "Result",
            (),
            {
                "returncode": 1,
                "stderr": "WARNING: HTTP Error 429: Too Many Requests\nERROR: Sign in to confirm you're not a bot\nUse --cookies-from-browser",
                "stdout": "",
            },
        )()

        result = process_youtube_video_sync(
            self.video.id,
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.video.status, "failed")
        self.assertIn("YouTube blocked automated download", self.video.error_message)
        self.assertIn("authenticated browser-cookie path", self.video.error_message)

    @patch("videos.tasks.extract_audio", return_value="audio.wav")
    @patch("videos.tasks._run_audio_pipeline", side_effect=KeyError("low_trust_malayalam"))
    def test_low_trust_structurally_usable_file_sync_wrapper_does_not_fail_video(
        self,
        _pipeline,
        _extract_audio,
    ):
        self.video.original_file.save("dummy.mp4", SimpleUploadedFile("dummy.mp4", b"0"), save=True)
        Transcript.objects.create(
            video=self.video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="groq_whisper",
            asr_engine_used="groq_whisper",
            transcript_quality_score=0.31,
            full_text="sample",
            transcript_original_text="sample",
            transcript_canonical_text="sample",
            transcript_canonical_en_text="sample",
            json_data={
                "language": "ml",
                "transcript_state": "degraded",
                "transcript_warning_message": "Malayalam transcript is low-trust but structurally usable; continuing in degraded mode.",
                "malayalam_post_asr_reason": "structurally_usable_low_trust_malayalam",
                "quality_metrics": {"structurally_usable": True},
            },
            word_timestamps=[],
        )

        result = process_video_transcription_sync(
            self.video.id,
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.video.status, "completed")

    def test_file_sync_wrapper_skips_duplicate_processing_when_video_already_active(self):
        self.video.status = "transcribing"
        self.video.processing_progress = 30
        self.video.save(update_fields=["status", "processing_progress", "updated_at"])

        result = process_video_transcription_sync(
            self.video.id,
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.assertEqual(result["status"], "already_processing")

    def test_youtube_sync_wrapper_skips_duplicate_processing_when_video_already_active(self):
        self.video.status = "transcribing"
        self.video.processing_progress = 30
        self.video.youtube_url = "https://youtube.test/watch?v=abc"
        self.video.save(update_fields=["status", "processing_progress", "youtube_url", "updated_at"])

        result = process_youtube_video_sync(
            self.video.id,
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.assertEqual(result["status"], "already_processing")

    def test_degraded_low_trust_malayalam_outputs_are_marked_for_suppression(self):
        transcript = Transcript.objects.create(
            video=self.video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="groq_whisper",
            asr_engine_used="groq_whisper",
            transcript_quality_score=0.18,
            full_text="garbled text",
            transcript_original_text="garbled text",
            transcript_canonical_text="garbled text",
            transcript_canonical_en_text="garbled text",
            json_data={
                "language": "ml",
                "transcript_state": "degraded",
                "display_transcript_units": [],
                "asr_metadata": {
                    "malayalam_display_refinement": {
                        "final_visible_word_count": 0,
                    }
                },
            },
            word_timestamps=[],
        )
        suppress, reason = _should_suppress_low_trust_malayalam_outputs(transcript)
        self.assertTrue(suppress)
        self.assertEqual(reason, "no_trusted_display_units")

    @patch("videos.tasks.detect_highlights")
    def test_degraded_low_trust_malayalam_suppresses_highlights(self, mock_detect_highlights):
        transcript = Transcript.objects.create(
            video=self.video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="groq_whisper",
            asr_engine_used="groq_whisper",
            transcript_quality_score=0.18,
            full_text="garbled text",
            transcript_original_text="garbled text",
            transcript_canonical_text="garbled text",
            transcript_canonical_en_text="garbled text",
            json_data={
                "language": "ml",
                "transcript_state": "degraded",
                "display_transcript_units": [],
                "asr_metadata": {
                    "malayalam_display_refinement": {
                        "final_visible_word_count": 0,
                    }
                },
            },
            word_timestamps=[],
        )
        HighlightSegment.objects.create(
            video=self.video,
            start_time=0.0,
            end_time=3.0,
            importance_score=0.8,
            reason="Old highlight",
            transcript_snippet="stale",
        )

        _rebuild_highlights(self.video, transcript)

        self.assertEqual(self.video.highlight_segments.count(), 0)
        self.assertEqual(mock_detect_highlights.call_count, 0)

    @patch("videos.tasks.extract_audio", return_value="audio.wav")
    @patch("videos.tasks._run_audio_pipeline", side_effect=KeyError("low_trust_malayalam"))
    def test_low_trust_structurally_usable_file_async_wrapper_does_not_fail_video(
        self,
        _pipeline,
        _extract_audio,
    ):
        self.video.original_file.save("dummy.mp4", SimpleUploadedFile("dummy.mp4", b"0"), save=True)
        Transcript.objects.create(
            video=self.video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="groq_whisper",
            asr_engine_used="groq_whisper",
            transcript_quality_score=0.31,
            full_text="sample",
            transcript_original_text="sample",
            transcript_canonical_text="sample",
            transcript_canonical_en_text="sample",
            json_data={
                "language": "ml",
                "transcript_state": "degraded",
                "transcript_warning_message": "Malayalam transcript is low-trust but structurally usable; continuing in degraded mode.",
                "malayalam_post_asr_reason": "structurally_usable_low_trust_malayalam",
                "quality_metrics": {"structurally_usable": True},
            },
            word_timestamps=[],
        )

        result = process_video_transcription.run(
            str(self.video.id),
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.video.status, "completed")

    @patch("videos.tasks._run_audio_pipeline", side_effect=KeyError("low_trust_malayalam"))
    @patch("videos.tasks.get_video_duration", return_value=120.0)
    @patch("videos.tasks.os.path.getsize", return_value=100000)
    @patch("videos.tasks.os.path.exists", return_value=True)
    @patch("videos.tasks.subprocess.run")
    def test_low_trust_structurally_usable_youtube_async_wrapper_does_not_fail_video(
        self,
        mock_run,
        _exists,
        _getsize,
        _duration,
        _pipeline,
    ):
        self.video.youtube_url = "https://youtube.test/watch?v=abc"
        self.video.save(update_fields=["youtube_url", "updated_at"])
        Transcript.objects.create(
            video=self.video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="groq_whisper",
            asr_engine_used="groq_whisper",
            transcript_quality_score=0.31,
            full_text="sample",
            transcript_original_text="sample",
            transcript_canonical_text="sample",
            transcript_canonical_en_text="sample",
            json_data={
                "language": "ml",
                "transcript_state": "degraded",
                "transcript_warning_message": "Malayalam transcript is low-trust but structurally usable; continuing in degraded mode.",
                "malayalam_post_asr_reason": "structurally_usable_low_trust_malayalam",
                "quality_metrics": {"structurally_usable": True},
            },
            word_timestamps=[],
        )
        mock_run.return_value = type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

        result = process_youtube_video.run(
            str(self.video.id),
            transcription_language="ml",
            output_language="ml",
            summary_language_mode="same_as_transcript",
        )

        self.video.refresh_from_db()
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.video.status, "completed")


class ProcessingMetadataApiTests(TestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.video = Video.objects.create(
            title="Processing Metadata Demo",
            status="completed",
            processing_progress=100,
        )
        Transcript.objects.create(
            video=self.video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="whisper_local",
            asr_engine_used="whisper_local",
            transcript_quality_score=0.8721,
            full_text="??? ??? ???????? ???????????????????.",
            transcript_original_text="??? ??? ???????? ???????????????????.",
            transcript_canonical_text="This is a test transcript.",
            transcript_canonical_en_text="This is a test transcript.",
            json_data={
                "segments": [
                    {"id": 0, "start": 0.0, "end": 5.0, "text": "??? ??? ???????? ???????????????????."},
                    {"id": 1, "start": 5.0, "end": 10.0, "text": "?????? ????? ??????????."},
                    {"id": 2, "start": 10.0, "end": 15.0, "text": "??? structured summary ???? ??????????????."},
                    {"id": 3, "start": 15.0, "end": 20.0, "text": "???????? ???????????? ????????."},
                    {"id": 4, "start": 20.0, "end": 25.0, "text": "????? ??? ???????????."},
                ]
            },
        )
        Summary.objects.create(video=self.video, summary_type="full", title="Full", content="??? ??? summary ???.")
        Summary.objects.create(
            video=self.video,
            summary_type="bullet",
            title="Bullets",
            content="- ?????? ?????\n- ?????? ?????\n- ??????? ?????\n- ????? ?????\n- ?????? ?????",
        )
        Summary.objects.create(video=self.video, summary_type="short", title="Short", content="??? ???? summary ???.")

    def _assert_processing_metadata_schema(self, payload):
        self.assertIn("processing_metadata", payload)
        meta = payload["processing_metadata"]
        self.assertIn("asr_engine", meta)
        self.assertIn("language", meta)
        self.assertIn("processing_time_seconds", meta)
        self.assertIn("transcript_quality_score", meta)
        self.assertIn("summary_quality_score", meta)
        self.assertIn("transcript_state", meta)
        self.assertIn("transcript_warning_message", meta)
        self.assertIn("downstream_suppressed", meta)
        self.assertIn("downstream_suppression_reason", meta)
        self.assertIn("trusted_visible_word_count", meta)
        self.assertIn("trusted_display_unit_count", meta)
        self.assertIn("low_evidence_malayalam", meta)
        self.assertIn("detected_language", meta)
        self.assertIn("detected_language_confidence", meta)
        self.assertIn("is_multilingual_content", meta)
        self.assertIn("english_view_available", meta)
        self.assertIn("translation_state", meta)
        self.assertIn("translation_blocked_reason", meta)
        self.assertIn("summary_ready", meta)
        self.assertIn("chat_ready", meta)
        self.assertIn("malayalam_observability", meta)
        self.assertIn("processing_metrics", meta)
        self.assertIsInstance(meta["asr_engine"], str)
        self.assertIsInstance(meta["language"], str)
        self.assertIsInstance(meta["processing_time_seconds"], float)
        self.assertIsInstance(meta["transcript_quality_score"], float)
        self.assertIsInstance(meta["summary_quality_score"], float)
        self.assertIsInstance(meta["transcript_state"], str)
        self.assertIsInstance(meta["transcript_warning_message"], str)
        self.assertIsInstance(meta["downstream_suppressed"], bool)
        self.assertIsInstance(meta["downstream_suppression_reason"], str)
        self.assertIsInstance(meta["trusted_visible_word_count"], int)
        self.assertIsInstance(meta["trusted_display_unit_count"], int)
        self.assertIsInstance(meta["low_evidence_malayalam"], bool)
        self.assertIsInstance(meta["detected_language"], str)
        self.assertIsInstance(meta["detected_language_confidence"], float)
        self.assertIsInstance(meta["is_multilingual_content"], bool)
        self.assertIsInstance(meta["english_view_available"], bool)
        self.assertIsInstance(meta["translation_state"], str)
        self.assertIsInstance(meta["translation_blocked_reason"], str)
        self.assertIsInstance(meta["summary_ready"], bool)
        self.assertIsInstance(meta["chat_ready"], bool)
        self.assertIsInstance(meta["malayalam_observability"], dict)
        self.assertIsInstance(meta["processing_metrics"], dict)

    def test_processing_metadata_exposes_malayalam_observability_block(self):
        transcript = self.video.transcripts.first()
        transcript.json_data = {
            **(transcript.json_data or {}),
            "processing_metrics": {
                "transcription_seconds": 12.4,
                "malayalam_observability": {
                    "enabled": True,
                    "first_pass_accepted": True,
                    "first_pass_accept_reason": "usable_first_pass_allowed",
                    "retry_considered": True,
                    "retry_executed": False,
                    "retry_skipped_reason": "fast_mode_first_pass_acceptable",
                    "retry_decision_reason": "",
                    "transcript_state": "cleaned",
                    "state_bucket": "cleaned",
                    "fallback_triggered": False,
                    "fallback_reason": "",
                    "model_path": "malayalam_local_stable_fallback",
                    "selected_model": "large-v3",
                    "resolved_model": "large-v3",
                    "provider": "faster_whisper",
                    "total_asr_seconds": 12.4,
                    "first_pass_transcription_seconds": 12.0,
                    "retry_transcription_seconds": 0.0,
                    "total_asr_passes": 1,
                    "transcript_quality_gate_passed": True,
                    "garbled_detector_score": 0.08,
                    "language": "ml",
                },
            },
        }
        transcript.save(update_fields=["json_data"])

        view = VideoViewSet.as_view({"get": "retrieve"})
        req = self.factory.get(f"/api/v1/videos/{self.video.id}/")
        resp = view(req, pk=str(self.video.id))
        self.assertEqual(resp.status_code, 200)
        obs = resp.data["processing_metadata"]["malayalam_observability"]
        self.assertTrue(obs["enabled"])
        self.assertTrue(obs["first_pass_accepted"])
        self.assertEqual(obs["model_path"], "malayalam_local_stable_fallback")
        self.assertEqual(obs["selected_model"], "large-v3")
        self.assertEqual(resp.data["processing_metadata"]["transcript_warning_message"], "")

    def test_extract_structured_summary_inputs_prefers_internal_evidence_only_for_degraded_malayalam(self):
        transcript = self.video.transcripts.first()
        transcript.json_data = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 3.0, "text": "garbled raw fragment"},
            ],
            "assembled_transcript_units": [],
            "internal_evidence_units": [
                {"id": 0, "start": 0.0, "end": 3.0, "text": "exam hall confidence", "evidence_only": True},
            ],
            "transcript_state": "degraded",
            "readable_transcript": "",
        }
        transcript.transcript_language = "ml"
        transcript.save(update_fields=["json_data", "transcript_language"])

        inputs = _extract_structured_summary_inputs(self.video, transcript)
        self.assertEqual(inputs["segments"][0]["text"], "exam hall confidence")
        self.assertEqual(inputs["assembled_units"][0]["text"], "exam hall confidence")
        self.assertIn("exam hall confidence", inputs["transcript_text"])

    def test_extract_structured_summary_inputs_keeps_cleaned_malayalam_assembled_units(self):
        transcript = self.video.transcripts.first()
        transcript.json_data = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 3.0, "text": "raw cleaned segment"},
            ],
            "assembled_transcript_units": [
                {"id": 0, "start": 0.0, "end": 3.0, "text": "trusted assembled malayalam"},
            ],
            "internal_evidence_units": [
                {"id": 0, "start": 0.0, "end": 3.0, "text": "evidence only fragment", "evidence_only": True},
            ],
            "transcript_state": "cleaned",
            "readable_transcript": "trusted assembled malayalam",
        }
        transcript.transcript_language = "ml"
        transcript.save(update_fields=["json_data", "transcript_language"])

        inputs = _extract_structured_summary_inputs(self.video, transcript)
        self.assertEqual(inputs["segments"][0]["text"], "trusted assembled malayalam")
        self.assertEqual(inputs["assembled_units"][0]["text"], "trusted assembled malayalam")

    def test_video_detail_contains_processing_metadata(self):
        view = VideoViewSet.as_view({"get": "retrieve"})
        req = self.factory.get(f"/api/v1/videos/{self.video.id}/")
        resp = view(req, pk=str(self.video.id))
        self.assertEqual(resp.status_code, 200)
        self._assert_processing_metadata_schema(resp.data)

    def test_processing_metadata_exposes_low_evidence_downstream_flags(self):
        transcript = self.video.transcripts.first()
        transcript.json_data = {
            **(transcript.json_data or {}),
            "transcript_state": "degraded",
            "transcript_warning_message": "Malayalam transcript quality was too low for reliable summarization.",
            "downstream_suppressed": True,
            "downstream_suppression_reason": "no_trusted_display_units",
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "low_evidence_malayalam": True,
            "processing_metrics": {
                **((transcript.json_data or {}).get("processing_metrics", {}) if isinstance((transcript.json_data or {}).get("processing_metrics", {}), dict) else {}),
                "downstream_suppressed": True,
                "downstream_suppression_reason": "no_trusted_display_units",
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "low_evidence_malayalam": True,
            },
        }
        transcript.save(update_fields=["json_data"])

        view = VideoViewSet.as_view({"get": "retrieve"})
        resp = view(self.factory.get(f"/api/v1/videos/{self.video.id}/"), pk=str(self.video.id))
        self.assertEqual(resp.status_code, 200)
        meta = resp.data["processing_metadata"]
        self.assertTrue(meta["downstream_suppressed"])
        self.assertEqual(meta["downstream_suppression_reason"], "no_trusted_display_units")
        self.assertEqual(meta["trusted_visible_word_count"], 0)
        self.assertEqual(meta["trusted_display_unit_count"], 0)
        self.assertTrue(meta["low_evidence_malayalam"])

    def test_structured_summary_endpoint_contains_processing_metadata(self):
        view = VideoViewSet.as_view({"get": "structured_summary"})
        req = self.factory.get(f"/api/v1/videos/{self.video.id}/structured_summary/")
        resp = view(req, pk=str(self.video.id))
        self.assertEqual(resp.status_code, 200)
        self._assert_processing_metadata_schema(resp.data)
        self.assertIn("tldr", resp.data)
        self.assertIn("key_points", resp.data)
        self.assertIn("action_items", resp.data)
        self.assertIn("chapters", resp.data)

    def test_api_exposes_cleaned_vs_degraded_malayalam_summary_modes(self):
        cleaned_video = Video.objects.create(
            title="Malayalam Cleaned API",
            status="completed",
            processing_progress=100,
        )
        Transcript.objects.create(
            video=cleaned_video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="groq_whisper",
            asr_engine_used="groq_whisper",
            transcript_quality_score=0.84,
            full_text="confidence result exam hall support",
            transcript_original_text="confidence result exam hall support",
            transcript_canonical_text="confidence result exam hall support",
            transcript_canonical_en_text="confidence result exam hall support",
            json_data={
                "language": "ml",
                "transcript_state": "cleaned",
                "transcript_warning_message": "",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 6.0, "text": "confidence result exam hall support", "segment_type": "mixed_malayalam_english", "unit_readability": 0.74},
                    {"id": 1, "start": 6.0, "end": 12.0, "text": "preparation hard work guidance", "segment_type": "mixed_malayalam_english", "unit_readability": 0.71},
                ],
                "assembled_transcript_units": [
                    {"id": 0, "start": 0.0, "end": 6.0, "display_start": 0.0, "display_end": 6.0, "text": "confidence result exam hall support", "segment_type": "mixed_malayalam_english", "unit_readability": 0.74},
                    {"id": 1, "start": 6.0, "end": 12.0, "display_start": 6.0, "display_end": 12.0, "text": "preparation hard work guidance", "segment_type": "mixed_malayalam_english", "unit_readability": 0.71},
                ],
                "display_transcript_units": [
                    {"id": 0, "display_start": 0.0, "display_end": 12.0, "text": "confidence result exam hall support preparation hard work guidance", "unit_readability": 0.76, "wrong_script_ratio": 0.0},
                ],
                "readable_transcript": "confidence result exam hall support preparation hard work guidance",
            },
            word_timestamps=[],
        )
        Summary.objects.create(video=cleaned_video, summary_type="full", title="Full", content="confidence result exam hall support guidance")
        Summary.objects.create(video=cleaned_video, summary_type="bullet", title="Bullets", content="- confidence and result\n- exam hall support")
        Summary.objects.create(video=cleaned_video, summary_type="short", title="Short", content="Confidence and result are discussed.")

        degraded_video = Video.objects.create(
            title="Malayalam Degraded API",
            status="completed",
            processing_progress=100,
        )
        Transcript.objects.create(
            video=degraded_video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="groq_whisper",
            asr_engine_used="groq_whisper",
            transcript_quality_score=0.18,
            full_text="garbled evidence",
            transcript_original_text="garbled evidence",
            transcript_canonical_text="garbled evidence",
            transcript_canonical_en_text="garbled evidence",
            json_data={
                "language": "ml",
                "transcript_state": "degraded",
                "transcript_warning_message": "Malayalam transcript quality was too low for reliable summarization.",
                "segments": [
                    {"id": 0, "start": 0.0, "end": 6.0, "text": "ਇੱ੆മാലു noisy corrupted chunk", "segment_type": "corrupted_malayalam_like", "unit_readability": 0.08},
                ],
                "assembled_transcript_units": [
                    {"id": 0, "start": 0.0, "end": 6.0, "display_start": 0.0, "display_end": 6.0, "text": "ਇੱ੆മാലു noisy corrupted chunk", "segment_type": "corrupted_malayalam_like", "unit_readability": 0.08},
                ],
                "display_transcript_units": [],
                "readable_transcript": "",
            },
            word_timestamps=[],
        )
        Summary.objects.create(video=degraded_video, summary_type="full", title="Full", content="stale noisy full summary")
        Summary.objects.create(video=degraded_video, summary_type="bullet", title="Bullets", content="- stale bullet")
        Summary.objects.create(video=degraded_video, summary_type="short", title="Short", content="stale short")

        view = VideoViewSet.as_view({"get": "retrieve"})

        cleaned_resp = view(self.factory.get(f"/api/v1/videos/{cleaned_video.id}/"), pk=str(cleaned_video.id))
        self.assertEqual(cleaned_resp.status_code, 200)
        self.assertEqual(cleaned_resp.data["processing_metadata"]["transcript_state"], "cleaned")
        self.assertEqual(cleaned_resp.data["processing_metadata"]["transcript_warning_message"], "")
        self.assertTrue(cleaned_resp.data["structured_summary"]["tldr"])
        self.assertNotEqual(cleaned_resp.data["structured_summary"].get("summary_state"), "degraded_safe")
        self.assertTrue(cleaned_resp.data["structured_summary"]["key_points"])
        cleaned_transcript_payload = cleaned_resp.data["transcripts"][0]
        self.assertTrue(cleaned_transcript_payload["english_view_available"])
        self.assertTrue(cleaned_transcript_payload["english_view_text"])
        self.assertIn("summary_english_view_available", cleaned_resp.data["structured_summary"])

        degraded_resp = view(self.factory.get(f"/api/v1/videos/{degraded_video.id}/"), pk=str(degraded_video.id))
        self.assertEqual(degraded_resp.status_code, 200)
        self.assertEqual(degraded_resp.data["processing_metadata"]["transcript_state"], "degraded")
        self.assertEqual(
            degraded_resp.data["processing_metadata"]["transcript_warning_message"],
            "Malayalam transcript quality was too low for reliable summarization.",
        )
        self.assertEqual(degraded_resp.data["structured_summary"].get("summary_state"), "degraded_safe")
        self.assertEqual(
            degraded_resp.data["structured_summary"].get("warning_message"),
            "Malayalam transcript quality was too low for reliable summarization.",
        )
        self.assertEqual(degraded_resp.data["structured_summary"]["chapters"], [])
        degraded_transcript_payload = degraded_resp.data["transcripts"][0]
        self.assertFalse(degraded_transcript_payload["english_view_available"])
        self.assertEqual(degraded_transcript_payload["translation_blocked_reason"], "degraded_safe_translation_blocked")
        self.assertIn("summary_translation_state", degraded_resp.data["structured_summary"])

    def test_cleaned_malayalam_rebuild_overrides_stale_degraded_safe_cache(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.transcript_original_text = "Motivational class about exam preparation and confidence."
        transcript.full_text = transcript.transcript_original_text
        transcript.json_data = {
            "language": "ml",
            "transcript_state": "cleaned",
            "segments": [
                {"id": 0, "start": 0.0, "end": 8.0, "text": "raw noisy fragment"},
            ],
            "assembled_transcript_units": [
                {"id": 0, "start": 0.0, "end": 10.0, "text": "Motivational class guidance.", "unit_readability": 0.66},
                {"id": 1, "start": 10.0, "end": 20.0, "text": "Exam preparation and confidence discussion.", "unit_readability": 0.71},
            ],
            "readable_transcript": "Motivational class guidance. Exam preparation and confidence discussion.",
            "structured_summary_cache": {
                "cache_key": "stale-degraded",
                "payload": {
                    "summary_state": "degraded_safe",
                    "warning_message": "Malayalam transcript quality was too low for reliable summarization.",
                    "tldr": "This video appears to contain Malayalam speech, but the transcript quality is too low for reliable summarization.",
                    "key_points": [],
                    "action_items": [],
                    "chapters": [],
                },
                "english_view_cache": {
                    "english_view_source_hash": "stale-degraded-english",
                    "english_view_valid": True,
                    "payload": {
                        "summary_english_view_available": False,
                        "summary_translation_state": "blocked",
                        "summary_translation_blocked_reason": "low_evidence_source_language",
                    },
                },
            },
        }
        transcript.save(update_fields=["transcript_language", "transcript_original_text", "full_text", "json_data"])
        Summary.objects.update_or_create(video=self.video, summary_type="short", defaults={"title": "Short", "content": "Exam preparation and confidence are discussed."})

        view = VideoViewSet.as_view({"get": "retrieve"})
        resp = view(self.factory.get(f"/api/v1/videos/{self.video.id}/"), pk=str(self.video.id))
        self.assertEqual(resp.status_code, 200)
        self.assertNotEqual(resp.data["structured_summary"].get("summary_state"), "degraded_safe")
        self.assertTrue(resp.data["structured_summary"]["tldr"])
        self.assertTrue(resp.data["structured_summary"]["key_points"])
        self.assertIn("summary_translation_state", resp.data["structured_summary"])

    def test_english_transcript_does_not_trigger_unnecessary_english_view_generation(self):
        english_video = Video.objects.create(title="English Video", status="completed", processing_progress=100)
        Transcript.objects.create(
            video=english_video,
            language="en",
            transcript_language="en",
            canonical_language="en",
            transcript_quality_score=0.91,
            full_text="This is an English transcript.",
            transcript_original_text="This is an English transcript.",
            transcript_canonical_text="This is an English transcript.",
            transcript_canonical_en_text="This is an English transcript.",
            json_data={
                "language": "en",
                "transcript_state": "cleaned",
                "readable_transcript": "This is an English transcript.",
                "display_readable_transcript": "This is an English transcript.",
            },
            word_timestamps=[],
        )
        view = VideoViewSet.as_view({"get": "retrieve"})
        resp = view(self.factory.get(f"/api/v1/videos/{english_video.id}/"), pk=str(english_video.id))
        self.assertEqual(resp.status_code, 200)
        transcript_payload = resp.data["transcripts"][0]
        self.assertFalse(transcript_payload["english_view_available"])
        self.assertEqual(transcript_payload["translation_state"], "same_as_original")

    def test_fidelity_failed_malayalam_transcript_never_exposes_same_as_original_english_view(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.transcript_canonical_en_text = "When you exit the exam hall you will feel satisfied."
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "degraded",
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "source_language_fidelity_failed",
            "display_readable_transcript": "",
            "readable_transcript": "",
            "translation_state": "blocked",
            "translation_blocked_reason": "source_language_fidelity_failed",
        }
        transcript.save(update_fields=["transcript_language", "language", "transcript_canonical_en_text", "json_data"])

        meta = _safe_transcript_english_view(transcript)

        self.assertFalse(meta["english_view_available"])
        self.assertEqual(meta["translation_state"], "blocked")
        self.assertEqual(meta["translation_blocked_reason"], "source_language_fidelity_failed")
        self.assertEqual(meta["current_available_views"], ["original"])

    def test_draft_state_malayalam_never_generates_transcript_en_view(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.transcript_canonical_en_text = "Potential canonical English should not leak yet."
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "draft",
            "display_readable_transcript": "temporary draft text",
            "readable_transcript": "temporary draft text",
        }
        transcript.save(update_fields=["transcript_language", "language", "transcript_canonical_en_text", "json_data"])

        meta = _safe_transcript_english_view(transcript)

        self.assertFalse(meta["english_view_available"])
        self.assertEqual(meta["translation_state"], "pending")
        self.assertEqual(meta["translation_blocked_reason"], "pending_final_malayalam_state")
        self.assertNotEqual(meta["translation_state"], "same_as_original")

    def test_draft_state_malayalam_never_generates_summary_en_view(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "draft",
        }
        transcript.save(update_fields=["transcript_language", "language", "json_data"])

        summary_payload = _augment_structured_summary_with_english_view(
            {"tldr": "draft summary", "key_points": ["draft"], "action_items": [], "chapters": []},
            transcript,
        )

        self.assertFalse(summary_payload["summary_english_view_available"])
        self.assertEqual(summary_payload["summary_translation_state"], "pending")
        self.assertEqual(summary_payload["summary_translation_blocked_reason"], "pending_final_malayalam_state")

    def test_fidelity_failed_malayalam_never_builds_partial_summary_artifacts(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "degraded",
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "source_language_fidelity_failed",
            "transcript_warning_message": "Malayalam speech could not be transcribed faithfully enough for safe display.",
            "structured_summary_cache": {
                "cache_key": "stale-fidelity",
                "payload": {
                    "tldr": "stale summary should never survive",
                    "key_points": ["stale key point"],
                    "action_items": ["stale action"],
                    "chapters": [{"title": "Stale chapter", "timestamp": "00:00"}],
                },
            },
        }
        transcript.save(update_fields=["transcript_language", "language", "json_data"])

        payload = get_or_build_structured_summary(self.video, transcript)

        self.assertEqual(payload.get("summary_state"), "blocked_unfaithful_source")
        self.assertEqual(payload.get("key_points"), [])
        self.assertEqual(payload.get("chapters"), [])
        self.assertFalse(payload.get("summary_english_view_available", False))
        self.assertEqual(payload.get("summary_translation_blocked_reason"), "source_language_fidelity_failed")
        self.assertEqual(payload.get("warning_message"), "Malayalam speech could not be transcribed faithfully enough for safe display.")

    @patch("videos.serializers.build_structured_summary")
    def test_fidelity_failed_malayalam_skips_structured_summary_reconstruction_entirely(self, mock_build_summary):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "degraded",
            "source_language_fidelity_failed": True,
            "final_malayalam_fidelity_decision": "source_language_fidelity_failed",
            "trusted_display_unit_count": 0,
            "trusted_visible_word_count": 0,
            "quality_metrics": {
                "malayalam_token_coverage": 0.0,
            },
        }
        transcript.save(update_fields=["transcript_language", "language", "json_data"])

        payload = get_or_build_structured_summary(self.video, transcript)

        mock_build_summary.assert_not_called()
        self.assertEqual(payload.get("summary_state"), "blocked_unfaithful_source")
        self.assertEqual(payload.get("summary_blocked_reason"), "malayalam_source_fidelity_failed")

    @patch("videos.serializers._persist_transcript_english_view_cache")
    def test_fidelity_failed_malayalam_skips_transcript_english_view_rebuild_and_persist(self, mock_persist):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "degraded",
            "display_readable_transcript": "",
            "readable_transcript": "",
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "catastrophic_latin_substitution_failure",
        }
        transcript.save(update_fields=["transcript_language", "language", "json_data"])

        payload = _safe_transcript_english_view(transcript)

        mock_persist.assert_not_called()
        self.assertFalse(payload["english_view_available"])
        self.assertEqual(payload["translation_blocked_reason"], "source_language_fidelity_failed")

    @patch("videos.serializers._persist_structured_summary_english_view_cache")
    def test_fidelity_failed_malayalam_skips_summary_english_view_rebuild_and_persist(self, mock_persist):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "degraded",
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "catastrophic_latin_substitution_failure",
        }
        transcript.save(update_fields=["transcript_language", "language", "json_data"])

        payload = _augment_structured_summary_with_english_view(
            _fidelity_failed_malayalam_summary_payload(transcript),
            transcript,
        )

        mock_persist.assert_not_called()
        self.assertFalse(payload["summary_english_view_available"])
        self.assertEqual(payload["summary_translation_blocked_reason"], "source_language_fidelity_failed")

    @patch("videos.serializers.build_structured_summary")
    def test_pending_malayalam_serializer_returns_lightweight_payload_without_summary_rebuild(self, mock_build_summary):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "draft",
        }
        transcript.save(update_fields=["transcript_language", "language", "json_data"])

        payload = get_or_build_structured_summary(self.video, transcript)

        mock_build_summary.assert_not_called()
        self.assertFalse(payload.get("tldr"))
        self.assertEqual(payload.get("key_points"), [])

    def test_fidelity_failed_malayalam_invalidates_stale_transcript_english_view_cache(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.transcript_canonical_en_text = "stale english cache should not survive"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "degraded",
            "display_readable_transcript": "",
            "readable_transcript": "",
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "catastrophic_wrong_script_failure",
            "english_view_cache": {
                "english_view_source_hash": "stale-hash",
                "english_view_valid": True,
                "payload": {
                    "english_view_available": True,
                    "english_view_text": "stale leaked english",
                    "translation_state": "same_as_original",
                    "translation_blocked_reason": "",
                    "current_available_views": ["original", "english"],
                },
            },
        }
        transcript.save(update_fields=["transcript_language", "language", "transcript_canonical_en_text", "json_data"])

        payload = _safe_transcript_english_view(transcript)
        transcript.refresh_from_db()

        self.assertFalse(payload["english_view_available"])
        self.assertEqual(payload["translation_blocked_reason"], "source_language_fidelity_failed")
        self.assertEqual(transcript.json_data.get("translation_blocked_reason"), "source_language_fidelity_failed")
        self.assertFalse(transcript.json_data.get("english_view_available", False))

    def test_fidelity_failed_malayalam_invalidates_stale_summary_english_view_cache(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "degraded",
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "catastrophic_wrong_script_failure",
            "transcript_warning_message": "Malayalam speech could not be transcribed faithfully enough for safe display.",
            "structured_summary_cache": {
                "cache_key": "stale-summary-fidelity",
                "payload": default_structured_summary(),
                "english_view_cache": {
                    "english_view_source_hash": "stale-summary-hash",
                    "english_view_valid": True,
                    "payload": {
                        "summary_english_view_available": True,
                        "summary_translation_state": "same_as_original",
                        "summary_translation_blocked_reason": "",
                        "summary_current_available_views": ["original", "english"],
                        "english_view_structured_summary": {"tldr": "stale leaked summary"},
                    },
                },
            },
        }
        transcript.save(update_fields=["transcript_language", "language", "json_data"])

        payload = get_or_build_structured_summary(self.video, transcript)
        transcript.refresh_from_db()

        self.assertFalse(payload.get("summary_english_view_available", False))
        self.assertEqual(payload.get("summary_translation_blocked_reason"), "source_language_fidelity_failed")
        self.assertEqual(
            ((transcript.json_data.get("structured_summary_cache", {}) or {}).get("english_view_cache", {}) or {}).get("payload", {}).get("summary_translation_blocked_reason"),
            "source_language_fidelity_failed",
        )

    def test_stale_bad_malayalam_degraded_transcript_is_hidden_as_fidelity_failed(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.full_text = "When you exit the exam hall and check the result website"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "degraded",
            "readable_transcript": "When you exit the exam hall and check the result website",
            "display_readable_transcript": "When you exit the exam hall and check the result website",
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "quality_metrics": {
                "dominant_script": "latin",
                "malayalam_token_coverage": 0.0,
            },
        }
        transcript.save(update_fields=["transcript_language", "language", "full_text", "json_data"])

        payload = TranscriptSerializer(transcript).data

        self.assertEqual(payload["transcript_state"], "source_language_fidelity_failed")
        self.assertEqual(payload["readable_transcript"], "")

    def test_stale_bad_malayalam_transcript_serializer_scrubs_raw_text_fields(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.full_text = "When you exit the exam hall and check the result website"
        transcript.transcript_original_text = "When you exit the exam hall and check the result website"
        transcript.transcript_canonical_text = "When you exit the exam hall and check the result website"
        transcript.transcript_canonical_en_text = "When you exit the exam hall and check the result website"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "language": "ml",
            "transcript_state": "degraded",
            "readable_transcript": "When you exit the exam hall and check the result website",
            "display_readable_transcript": "When you exit the exam hall and check the result website",
            "evidence_readable_transcript": "When you exit the exam hall and check the result website",
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "quality_metrics": {
                "dominant_script": "latin",
                "malayalam_token_coverage": 0.0,
            },
        }
        transcript.save(
            update_fields=[
                "transcript_language",
                "language",
                "full_text",
                "transcript_original_text",
                "transcript_canonical_text",
                "transcript_canonical_en_text",
                "json_data",
            ]
        )

        payload = TranscriptSerializer(transcript).data

        self.assertEqual(payload["full_text"], "")
        self.assertEqual(payload["transcript_original_text"], "")
        self.assertEqual(payload["transcript_canonical_text"], "")
        self.assertEqual(payload["transcript_canonical_en_text"], "")
        self.assertEqual(payload["readable_transcript"], "")
        self.assertEqual(payload["json_data"]["display_readable_transcript"], "")
        self.assertEqual(payload["json_data"]["evidence_readable_transcript"], "")

    def test_video_detail_uses_latest_transcript_only_and_does_not_fallback_when_latest_failed(self):
        older = self.video.transcripts.first()
        older.transcript_language = "ml"
        older.language = "ml"
        older.full_text = "When you exit the exam hall and check the result website"
        older.json_data = {
            **(older.json_data or {}),
            "language": "ml",
            "transcript_state": "degraded",
            "readable_transcript": "When you exit the exam hall and check the result website",
            "display_readable_transcript": "When you exit the exam hall and check the result website",
            "trusted_visible_word_count": 0,
            "trusted_display_unit_count": 0,
            "quality_metrics": {
                "dominant_script": "latin",
                "malayalam_token_coverage": 0.0,
            },
        }
        older.save(update_fields=["transcript_language", "language", "full_text", "json_data"])

        latest = Transcript.objects.create(
            video=self.video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            asr_engine="groq_whisper",
            asr_engine_used="groq_whisper",
            transcript_quality_score=0.0,
            full_text="",
            transcript_original_text="",
            transcript_canonical_text="",
            transcript_canonical_en_text="",
            json_data={
                "language": "ml",
                "transcript_state": "failed",
                "readable_transcript": "",
                "display_readable_transcript": "",
            },
            word_timestamps=[],
        )

        payload = VideoDetailSerializer(self.video, context={"request": self.factory.get("/api/v1/videos/")}).data

        self.assertEqual(latest.json_data["transcript_state"], "failed")
        self.assertEqual(payload["transcripts"], [])

    def test_transcript_json_payload_promotes_final_malayalam_fidelity_failure_top_level(self):
        payload = _build_transcript_json_payload(
            {
                "text": "When you exit the exam",
                "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "When you exit the exam"}],
                "language": "ml",
                "metadata": {},
            },
            {"canonical_text": "", "canonical_segments": []},
            transcript_state="source_language_fidelity_failed",
            qa_metrics={
                "source_language_fidelity_failed": True,
                "transcript_fidelity_state": "catastrophic_latin_substitution_failure",
                "final_malayalam_fidelity_decision": "source_language_fidelity_failed",
                "catastrophic_latin_substitution_failure": True,
                "summary_blocked_reason": "malayalam_source_fidelity_failed",
                "chatbot_blocked_reason": "malayalam_source_fidelity_failed",
            },
            readable_transcript="When you exit the exam",
            display_readable_transcript="",
            transcript_warning_message="Malayalam speech could not be transcribed faithfully enough for safe display.",
            malayalam_post_asr_mode="source_language_fidelity_failed",
            malayalam_post_asr_reason="malayalam_source_fidelity_failed",
        )

        self.assertEqual(payload["transcript_state"], "source_language_fidelity_failed")
        self.assertTrue(payload["source_language_fidelity_failed"])
        self.assertEqual(payload["transcript_fidelity_state"], "catastrophic_latin_substitution_failure")
        self.assertEqual(payload["final_malayalam_fidelity_decision"], "source_language_fidelity_failed")
        self.assertTrue(payload["catastrophic_latin_substitution_failure"])
        self.assertEqual(payload["summary_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(payload["chatbot_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(payload["readable_transcript"], "")
        self.assertEqual(payload["transcript_display_mode"], "suppressed_unfaithful_source")

    def test_minimal_low_trust_checkpoint_suppresses_visible_text_for_fidelity_failed_malayalam(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.save(update_fields=["transcript_language", "language"])

        _persist_minimal_low_trust_malayalam_checkpoint(
            transcript,
            transcript_payload={"language": "ml", "word_timestamps": []},
            canonical_payload={"canonical_text": "", "canonical_language": "en"},
            cleaned_text="When you exit the exam",
            cleaned_segments=[{"id": 0, "start": 0.0, "end": 1.0, "text": "When you exit the exam"}],
            processing_seconds=0.3,
            quality={
                "state": "source_language_fidelity_failed",
                "quality_score": 0.0,
                "warnings": [],
                "transcript_warning_message": "Malayalam speech could not be transcribed faithfully enough for safe display.",
                "malayalam_post_asr_mode": "source_language_fidelity_failed",
                "malayalam_post_asr_reason": "malayalam_source_fidelity_failed",
                "summary_blocked_reason": "malayalam_source_fidelity_failed",
                "chatbot_blocked_reason": "malayalam_source_fidelity_failed",
                "qa_metrics": {
                    "source_language_fidelity_failed": True,
                    "transcript_fidelity_state": "catastrophic_latin_substitution_failure",
                    "final_malayalam_fidelity_decision": "source_language_fidelity_failed",
                    "catastrophic_latin_substitution_failure": True,
                },
            },
        )

        transcript.refresh_from_db()
        self.assertEqual(transcript.json_data["transcript_state"], "source_language_fidelity_failed")
        self.assertTrue(transcript.json_data["source_language_fidelity_failed"])
        self.assertEqual(transcript.json_data["final_malayalam_fidelity_decision"], "source_language_fidelity_failed")
        self.assertEqual(transcript.json_data["readable_transcript"], "")
        self.assertEqual(transcript.json_data["display_readable_transcript"], "")
        self.assertEqual(transcript.json_data["transcript_display_mode"], "suppressed_unfaithful_source")

    @patch("videos.serializers.build_safe_english_view_text")
    def test_cleaned_non_english_transcript_english_view_persists_and_reuses_cache(self, mock_build):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "transcript_state": "cleaned",
            "display_readable_transcript": "confidence result exam hall support",
            "readable_transcript": "confidence result exam hall support",
            "transcript_warning_message": "",
        }
        transcript.transcript_canonical_en_text = "confidence result exam hall support"
        transcript.save(update_fields=["transcript_language", "language", "json_data", "transcript_canonical_en_text"])
        mock_build.return_value = {
            "original_language": "ml",
            "translated_language": "en",
            "english_view_available": True,
            "translation_state": "translated",
            "translation_warning": "",
            "translation_blocked_reason": "",
            "current_available_views": ["original", "english"],
            "english_view_text": "confidence result exam hall support",
        }

        first = _safe_transcript_english_view(transcript)
        transcript.refresh_from_db()
        second = _safe_transcript_english_view(transcript)

        self.assertTrue(first["english_view_available"])
        self.assertEqual(second["english_view_text"], "confidence result exam hall support")
        self.assertEqual(mock_build.call_count, 1)
        self.assertIn("english_view_cache", transcript.json_data)
        self.assertTrue(transcript.json_data["english_view_cache"]["english_view_valid"])

    @patch("videos.serializers.build_safe_english_view_text")
    def test_transcript_english_view_invalidates_when_source_changes(self, mock_build):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "transcript_state": "cleaned",
            "display_readable_transcript": "old visible text",
            "readable_transcript": "old visible text",
        }
        transcript.save(update_fields=["transcript_language", "language", "json_data"])
        mock_build.side_effect = [
            {
                "original_language": "ml",
                "translated_language": "en",
                "english_view_available": True,
                "translation_state": "translated",
                "translation_warning": "",
                "translation_blocked_reason": "",
                "current_available_views": ["original", "english"],
                "english_view_text": "old english view",
            },
            {
                "original_language": "ml",
                "translated_language": "en",
                "english_view_available": True,
                "translation_state": "translated",
                "translation_warning": "",
                "translation_blocked_reason": "",
                "current_available_views": ["original", "english"],
                "english_view_text": "new english view",
            },
        ]

        _safe_transcript_english_view(transcript)
        transcript.refresh_from_db()
        old_hash = transcript.json_data["english_view_cache"]["english_view_source_hash"]
        transcript.json_data = {
            **(transcript.json_data or {}),
            "display_readable_transcript": "new visible text",
            "readable_transcript": "new visible text",
        }
        transcript.save(update_fields=["json_data"])
        transcript.refresh_from_db()
        rebuilt = _safe_transcript_english_view(transcript)

        self.assertEqual(rebuilt["english_view_text"], "new english view")
        self.assertEqual(mock_build.call_count, 2)
        self.assertNotEqual(old_hash, transcript.json_data["english_view_cache"]["english_view_source_hash"])

    def test_api_never_reports_transcript_english_view_available_without_payload(self):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "transcript_state": "cleaned",
            "display_readable_transcript": "reliable visible text",
            "readable_transcript": "reliable visible text",
            "english_view_cache": {
                "english_view_source_hash": "stale",
                "english_view_valid": True,
                "payload": {
                    "english_view_available": True,
                    "english_view_text": "",
                    "translation_state": "translated",
                    "translation_blocked_reason": "",
                },
            },
        }
        transcript.transcript_canonical_en_text = "reliable visible text"
        transcript.save(update_fields=["transcript_language", "language", "json_data", "transcript_canonical_en_text"])
        view = VideoViewSet.as_view({"get": "retrieve"})
        resp = view(self.factory.get(f"/api/v1/videos/{self.video.id}/"), pk=str(self.video.id))
        self.assertEqual(resp.status_code, 200)
        transcript_payload = resp.data["transcripts"][0]
        if transcript_payload["english_view_available"]:
            self.assertTrue(transcript_payload["english_view_text"])

    def test_degraded_safe_summary_english_view_preserves_sparse_honesty(self):
        degraded_video = Video.objects.create(title="Malayalam Degraded English View", status="completed", processing_progress=100)
        Transcript.objects.create(
            video=degraded_video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            transcript_quality_score=0.12,
            full_text="",
            transcript_original_text="",
            transcript_canonical_text="",
            transcript_canonical_en_text="",
            json_data={
                "language": "ml",
                "transcript_state": "degraded",
                "transcript_warning_message": "Malayalam transcript quality was too low for reliable summarization.",
                "low_evidence_malayalam": True,
                "display_transcript_units": [],
                "readable_transcript": "",
            },
            word_timestamps=[],
        )
        view = VideoViewSet.as_view({"get": "retrieve"})
        resp = view(self.factory.get(f"/api/v1/videos/{degraded_video.id}/"), pk=str(degraded_video.id))
        self.assertEqual(resp.status_code, 200)
        structured = resp.data["structured_summary"]
        self.assertEqual(structured.get("summary_state"), "degraded_safe")
        self.assertFalse(structured.get("summary_english_view_available"))
        self.assertIn(structured.get("summary_translation_state"), {"same_as_original", "blocked"})

    @patch("videos.serializers.build_safe_english_view_structured_summary")
    def test_degraded_safe_summary_english_view_invalidates_when_summary_source_changes(self, mock_build):
        transcript = self.video.transcripts.first()
        transcript.transcript_language = "ml"
        transcript.language = "ml"
        transcript.json_data = {
            **(transcript.json_data or {}),
            "transcript_state": "degraded",
            "low_evidence_malayalam": False,
            "structured_summary_cache": {},
        }
        transcript.save(update_fields=["transcript_language", "language", "json_data"])
        mock_build.side_effect = [
            {
                "summary_original_language": "ml",
                "summary_english_view_available": True,
                "summary_translation_state": "translated",
                "summary_translation_warning": "",
                "summary_translation_blocked_reason": "",
                "summary_current_available_views": ["original", "english"],
                "english_view_structured_summary": {"tldr": "old summary", "key_points": [], "action_items": [], "chapters": []},
            },
            {
                "summary_original_language": "ml",
                "summary_english_view_available": True,
                "summary_translation_state": "translated",
                "summary_translation_warning": "",
                "summary_translation_blocked_reason": "",
                "summary_current_available_views": ["original", "english"],
                "english_view_structured_summary": {"tldr": "new summary", "key_points": [], "action_items": [], "chapters": []},
            },
        ]
        old_payload = {"tldr": "old", "key_points": [], "action_items": [], "chapters": []}
        new_payload = {"tldr": "new", "key_points": [], "action_items": [], "chapters": []}

        _augment_structured_summary_with_english_view(old_payload, transcript)
        transcript.refresh_from_db()
        old_hash = transcript.json_data["structured_summary_cache"]["english_view_cache"]["english_view_source_hash"]
        rebuilt = _augment_structured_summary_with_english_view(new_payload, transcript)

        self.assertTrue(rebuilt["summary_english_view_available"])
        self.assertEqual(rebuilt["english_view_structured_summary"]["tldr"], "new summary")
        self.assertEqual(mock_build.call_count, 2)
        self.assertNotEqual(old_hash, transcript.json_data["structured_summary_cache"]["english_view_cache"]["english_view_source_hash"])

    @patch("videos.serializers.build_structured_summary")
    def test_video_detail_uses_cached_structured_summary_on_repeat_fetch(self, mock_build):
        mock_build.return_value = {
            "tldr": "Cached summary.",
            "key_points": ["Point one.", "Point two.", "Point three.", "Point four."],
            "action_items": [],
            "chapters": [{"title": "Discussion Topic", "timestamp": "00:00"}],
        }
        view = VideoViewSet.as_view({"get": "retrieve"})
        req1 = self.factory.get(f"/api/v1/videos/{self.video.id}/")
        resp1 = view(req1, pk=str(self.video.id))
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(mock_build.call_count, 1)

        req2 = self.factory.get(f"/api/v1/videos/{self.video.id}/")
        resp2 = view(req2, pk=str(self.video.id))
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(mock_build.call_count, 1)

    @patch("videos.serializers.build_structured_summary")
    def test_structured_summary_recomputes_when_summary_input_changes(self, mock_build):
        mock_build.return_value = {
            "tldr": "Fresh summary.",
            "key_points": ["Point one.", "Point two.", "Point three.", "Point four."],
            "action_items": [],
            "chapters": [{"title": "Discussion Topic", "timestamp": "00:00"}],
        }
        view = VideoViewSet.as_view({"get": "retrieve"})
        req1 = self.factory.get(f"/api/v1/videos/{self.video.id}/")
        resp1 = view(req1, pk=str(self.video.id))
        self.assertEqual(resp1.status_code, 200)
        self.assertEqual(mock_build.call_count, 1)

        summary = self.video.summaries.get(summary_type="full")
        summary.content = "Updated full summary content."
        summary.save(update_fields=["content"])

        req2 = self.factory.get(f"/api/v1/videos/{self.video.id}/")
        resp2 = view(req2, pk=str(self.video.id))
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(mock_build.call_count, 2)

    def test_video_detail_uses_legacy_summary_rows_when_cached_structured_summary_is_empty(self):
        transcript = self.video.transcripts.first()
        inputs = _extract_structured_summary_inputs(self.video, transcript)
        transcript.json_data = {
            **(transcript.json_data or {}),
            "structured_summary_cache": {
                "cache_key": structured_summary_cache_key(**inputs),
                "payload": default_structured_summary(),
            },
        }
        transcript.save(update_fields=["json_data"])

        view = VideoViewSet.as_view({"get": "retrieve"})
        req = self.factory.get(f"/api/v1/videos/{self.video.id}/")
        resp = view(req, pk=str(self.video.id))
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["structured_summary"]["tldr"])
        self.assertTrue(resp.data["structured_summary"]["key_points"])

    def test_video_detail_uses_legacy_summary_rows_when_built_structured_summary_is_empty(self):
        transcript = self.video.transcripts.first()
        transcript.json_data = {**(transcript.json_data or {}), "structured_summary_cache": {}}
        transcript.save(update_fields=["json_data"])

        with patch("videos.serializers.build_structured_summary", return_value=default_structured_summary()):
            view = VideoViewSet.as_view({"get": "retrieve"})
            req = self.factory.get(f"/api/v1/videos/{self.video.id}/")
            resp = view(req, pk=str(self.video.id))

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["structured_summary"]["tldr"])
        self.assertTrue(resp.data["structured_summary"]["key_points"])

    def test_video_detail_legacy_fallback_builds_chapters_from_highlights(self):
        transcript = self.video.transcripts.first()
        transcript.json_data = {**(transcript.json_data or {}), "structured_summary_cache": {}}
        transcript.save(update_fields=["json_data"])
        HighlightSegment.objects.create(
            video=self.video,
            start_time=12.0,
            end_time=24.0,
            importance_score=0.88,
            reason="Main discussion segment",
            transcript_snippet="Key part of the discussion.",
        )

        with patch("videos.serializers.build_structured_summary", return_value=default_structured_summary()):
            view = VideoViewSet.as_view({"get": "retrieve"})
            req = self.factory.get(f"/api/v1/videos/{self.video.id}/")
            resp = view(req, pk=str(self.video.id))

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["structured_summary"]["chapters"])
        self.assertTrue(resp.data["structured_summary"]["chapters"][0]["timestamp"])

    def test_video_detail_legacy_fallback_derives_action_items_from_existing_summary_text(self):
        transcript = self.video.transcripts.first()
        transcript.json_data = {**(transcript.json_data or {}), "structured_summary_cache": {}}
        transcript.save(update_fields=["json_data"])

        bullet = self.video.summaries.get(summary_type="bullet")
        bullet.content = (
            "• Review the main ideas before the exam\n"
            "• Practice with mock questions\n"
            "• Keep a short revision schedule"
        )
        bullet.save(update_fields=["content"])

        with patch("videos.serializers.build_structured_summary", return_value=default_structured_summary()):
            view = VideoViewSet.as_view({"get": "retrieve"})
            req = self.factory.get(f"/api/v1/videos/{self.video.id}/")
            resp = view(req, pk=str(self.video.id))

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["structured_summary"]["action_items"])
        self.assertIn("Practice with mock questions", resp.data["structured_summary"]["action_items"])

    def test_structured_summary_cache_key_is_video_scoped(self):
        segments = [{"id": 0, "start": 0.0, "end": 5.0, "text": "Same transcript content."}]
        key_a = structured_summary_cache_key(
            video_id=str(self.video.id),
            transcript_hash="hash-a",
            transcript_text="Same transcript content.",
            segments=segments,
            full_summary="Same full summary.",
            bullet_summary="- Same bullet",
            short_summary="Same short summary.",
        )
        other_video = Video.objects.create(title="Other video", status="completed")
        key_b = structured_summary_cache_key(
            video_id=str(other_video.id),
            transcript_hash="hash-a",
            transcript_text="Same transcript content.",
            segments=segments,
            full_summary="Same full summary.",
            bullet_summary="- Same bullet",
            short_summary="Same short summary.",
        )
        self.assertNotEqual(key_a, key_b)

    def test_structured_summary_cache_key_changes_with_transcript_state(self):
        segments = [{"id": 0, "start": 0.0, "end": 5.0, "text": "Same transcript content."}]
        degraded_key = structured_summary_cache_key(
            video_id=str(self.video.id),
            transcript_hash="hash-a",
            transcript_text="Same transcript content.",
            segments=segments,
            full_summary="Same full summary.",
            bullet_summary="- Same bullet",
            short_summary="Same short summary.",
            transcript_state="degraded",
        )
        cleaned_key = structured_summary_cache_key(
            video_id=str(self.video.id),
            transcript_hash="hash-a",
            transcript_text="Same transcript content.",
            segments=segments,
            full_summary="Same full summary.",
            bullet_summary="- Same bullet",
            short_summary="Same short summary.",
            transcript_state="cleaned",
        )
        self.assertNotEqual(degraded_key, cleaned_key)

    def test_structured_summary_cache_key_accepts_internal_evidence_units(self):
        key = structured_summary_cache_key(
            video_id=str(self.video.id),
            transcript_hash="hash-a",
            transcript_text="Same transcript content.",
            segments=[{"id": 0, "start": 0.0, "end": 5.0, "text": "raw degraded fragment"}],
            internal_evidence_units=[
                {"id": 0, "start": 0.0, "end": 5.0, "text": "recoverable evidence text", "evidence_only": True},
            ],
            transcript_state="degraded",
            transcript_language="ml",
        )
        self.assertTrue(key)

    def test_degraded_malayalam_internal_evidence_cache_key_does_not_throw(self):
        transcript = self.video.transcripts.first()
        transcript.json_data = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 3.0, "text": "garbled raw fragment"},
            ],
            "assembled_transcript_units": [],
            "internal_evidence_units": [
                {"id": 0, "start": 0.0, "end": 3.0, "text": "exam hall confidence", "evidence_only": True},
            ],
            "transcript_state": "degraded",
            "readable_transcript": "",
        }
        transcript.transcript_language = "ml"
        transcript.save(update_fields=["json_data", "transcript_language"])

        inputs = _extract_structured_summary_inputs(self.video, transcript)
        cache_key = structured_summary_cache_key(**inputs)
        self.assertTrue(cache_key)

    def test_cleaned_malayalam_cache_key_remains_stable_when_internal_evidence_changes(self):
        common_kwargs = {
            "video_id": str(self.video.id),
            "transcript_hash": "hash-a",
            "transcript_text": "trusted assembled malayalam",
            "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "trusted assembled malayalam"}],
            "assembled_units": [{"id": 0, "start": 0.0, "end": 5.0, "text": "trusted assembled malayalam"}],
            "transcript_state": "cleaned",
            "transcript_language": "ml",
        }
        key_a = structured_summary_cache_key(
            **common_kwargs,
            internal_evidence_units=[{"id": 0, "start": 0.0, "end": 5.0, "text": "evidence one", "evidence_only": True}],
        )
        key_b = structured_summary_cache_key(
            **common_kwargs,
            internal_evidence_units=[{"id": 0, "start": 0.0, "end": 5.0, "text": "different evidence", "evidence_only": True}],
        )
        self.assertEqual(key_a, key_b)

    def test_degraded_structured_summary_uses_safer_outputs(self):
        result = build_structured_summary(
            transcript_text="A noisy transcript still mentions planning, review, and release steps.",
            segments=[
                {"id": 0, "start": 0.0, "end": 30.0, "text": "Planning discussion with noisy fragments."},
                {"id": 1, "start": 30.0, "end": 60.0, "text": "Review stage with mixed terminology and repeated fragments."},
                {"id": 2, "start": 60.0, "end": 90.0, "text": "Release discussion and next steps."},
            ],
            full_summary="The discussion covers planning, review, and release, but the transcript remains noisy.",
            bullet_summary="- noisy fragment one\n- noisy fragment two\n- repeated repeated repeated",
            short_summary="Planning, review, and release are discussed in a noisy transcript.",
            transcript_state="degraded",
        )
        self.assertEqual(result["action_items"], [])
        self.assertLessEqual(len(result["key_points"]), 4)
        self.assertTrue(all(chapter.get("title") for chapter in result["chapters"]))

    def test_degraded_structured_summary_filters_suspicious_fake_entities(self):
        result = build_structured_summary(
            transcript_text=(
                "Announcer Dingen sounds like a noisy artifact in this degraded transcript. "
                "The video mostly discusses exam preparation, coaching advice, and motivation."
            ),
            segments=[
                {"id": 0, "start": 0.0, "end": 20.0, "text": "Announcer Dingen exam motivation noisy phrase."},
                {"id": 1, "start": 20.0, "end": 45.0, "text": "Coaching advice and exam preparation discussion."},
            ],
            full_summary="Announcer Dingen discusses exam preparation and coaching advice.",
            bullet_summary="- Announcer Dingen opens the talk\n- exam preparation advice\n- coaching and motivation",
            short_summary="A degraded transcript about exam preparation and coaching advice.",
            transcript_state="degraded",
        )
        rendered = json.dumps(result)
        self.assertNotIn("Announcer Dingen", rendered)
        self.assertEqual(result["action_items"], [])

    def test_malayalam_cleaned_structured_summary_prefers_semantic_points_over_noisy_fragments(self):
        result = build_structured_summary(
            transcript_text="ഇത് ഒരു motivational coaching talk ആണ്. നിങ്ങൾ പരീക്ഷയ്ക്ക് വേണ്ടി തയ്യാറാകണം.",
            segments=[
                {"id": 0, "start": 0.0, "end": 20.0, "text": "ഇിദുംദു ഓരോ ഉത്തരം നിങ്ങൾ കളിയാക്കുന്ന motivational talk."},
                {"id": 1, "start": 20.0, "end": 40.0, "text": "രീദീല പരീക്ഷ preparation കുറിച്ച് സംസാരിക്കുന്നു."},
            ],
            transcript_state="cleaned",
            transcript_language="ml",
            full_summary="ഇിദുംദു ഓരോ ഉത്തരം നിങ്ങൾ കളിയാക്കുന്ന motivational talk about exam preparation.",
            bullet_summary="- ഇിദുംദു ഓരോ ഉത്തരം\n- രീദീല preparation\n- exam guidance",
            short_summary="Motivational guidance about exam preparation.",
        )
        rendered = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("ഇിദുംദു", rendered)
        self.assertNotIn("രീദീല", rendered)
        self.assertTrue(result["key_points"])
        self.assertNotEqual(result.get("summary_state"), "degraded_safe")
        self.assertEqual(result.get("_trace", {}).get("structured_summary_route"), "normal_grounded")
        self.assertTrue(result.get("_trace", {}).get("structured_grounding_passed"))

    def test_cleaned_malayalam_with_valid_assembled_grounding_uses_normal_structured_route(self):
        result = build_structured_summary(
            transcript_text="Motivational coaching class about preparation and confidence.",
            segments=[
                {"id": 0, "start": 0.0, "end": 12.0, "text": "noisy raw fragment"},
            ],
            assembled_units=[
                {"id": 0, "start": 0.0, "end": 12.0, "text": "Motivational coaching class discussion.", "unit_readability": 0.66},
                {"id": 1, "start": 12.0, "end": 24.0, "text": "Preparation and confidence guidance for the exam.", "unit_readability": 0.69},
            ],
            transcript_state="cleaned",
            transcript_language="ml",
            short_summary="Guidance about preparation and confidence.",
        )
        self.assertTrue(result["tldr"])
        self.assertTrue(result["key_points"])
        self.assertTrue(result["chapters"])
        self.assertEqual(result.get("_trace", {}).get("structured_summary_route"), "normal_grounded")
        self.assertIn(result.get("_trace", {}).get("structured_input_source"), {"assembled_units", "trusted_assembled_units"})

    def test_degraded_malayalam_internal_evidence_stays_on_degraded_safe_route(self):
        result = build_structured_summary(
            transcript_text="confidence exam hall support",
            segments=[
                {"id": 0, "start": 0.0, "end": 4.0, "text": "garbled raw fragment"},
            ],
            assembled_units=[
                {"id": 0, "start": 0.0, "end": 4.0, "text": "confidence exam hall support", "evidence_only": True},
            ],
            transcript_state="degraded",
            transcript_language="ml",
            short_summary="confidence exam hall support",
        )
        self.assertEqual(result.get("summary_state"), "degraded_safe")
        self.assertEqual(result.get("_trace", {}).get("structured_summary_route"), "degraded_safe")

    @patch("videos.serializers.build_structured_summary")
    def test_failed_transcript_does_not_serve_or_build_structured_summary(self, mock_build):
        transcript = self.video.transcripts.first()
        transcript.json_data = {
            **(transcript.json_data or {}),
            "transcript_state": "failed",
            "structured_summary_cache": {
                "cache_key": "stale",
                "payload": {
                    "tldr": "This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking.",
                    "key_points": ["Robert Downey Jr. discusses Marvel and Iron Man."],
                    "action_items": [],
                    "chapters": [{"title": "Discussion Topic", "timestamp": "00:00"}],
                },
            },
        }
        transcript.save(update_fields=["json_data"])

        view = VideoViewSet.as_view({"get": "retrieve"})
        req = self.factory.get(f"/api/v1/videos/{self.video.id}/")
        resp = view(req, pk=str(self.video.id))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["structured_summary"], default_structured_summary())
        self.assertEqual(mock_build.call_count, 0)

    @patch("videos.tasks.detect_script_type", return_value="gurmukhi")
    @patch(
        "videos.utils._extract_asr_metrics",
        return_value={
            "word_count": 22,
            "words_per_minute": 5.5,
            "dominant_script": "gurmukhi",
            "dominant_script_ratio": 0.93,
            "malayalam_ratio": 0.01,
            "other_indic_ratio": 0.93,
            "malayalam_token_coverage": 0.0,
            "repeated_token_ratio": 0.04,
            "info_density": 0.21,
        },
    )
    def test_catastrophic_gurmukhi_collapse_triggers_hard_fidelity_failure(self, _metrics, _script_type):
        text = "wrong script collapse result site answer key garbage exam date check"
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[
                {"start": 0.0, "end": 5.0, "text": "wrong script collapse result site answer key"},
                {"start": 5.0, "end": 10.0, "text": "garbage exam date check"},
            ],
            transcript_payload={
                "transcript_quality_score": 0.22,
                "confidence": 0.98,
                "language": "ml",
                "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
            },
            audio_duration_seconds=120.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertTrue(result["qa_metrics"]["source_language_fidelity_failed"])
        self.assertEqual(result["qa_metrics"]["transcript_fidelity_state"], "catastrophic_wrong_script_failure")
        self.assertTrue(result["qa_metrics"]["catastrophic_wrong_script_failure"])
        self.assertEqual(result["summary_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["chatbot_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["malayalam_post_asr_mode"], "source_language_fidelity_failed")

    @patch("videos.tasks.detect_script_type", return_value="gurmukhi")
    @patch(
        "videos.utils._extract_asr_metrics",
        return_value={
            "word_count": 18,
            "words_per_minute": 4.8,
            "dominant_script": "gurmukhi",
            "dominant_script_ratio": 0.91,
            "malayalam_ratio": 0.02,
            "other_indic_ratio": 0.91,
            "malayalam_token_coverage": 0.0,
            "repeated_token_ratio": 0.02,
            "info_density": 0.18,
        },
    )
    def test_catastrophic_wrong_script_malayalam_never_uses_structurally_usable_degraded_path(self, _metrics, _script_type):
        text = "wrong script collapse answer key result site garbage"
        result = _compute_transcript_state(
            cleaned_text=text,
            cleaned_segments=[{"start": 0.0, "end": 12.0, "text": text}],
            transcript_payload={
                "transcript_quality_score": 0.18,
                "confidence": 0.98,
                "language": "ml",
                "metadata": {"asr_provider_used": "groq_whisper", "language_detection_confidence": 0.98},
            },
            audio_duration_seconds=150.0,
            transcript_language="ml",
        )
        self.assertEqual(result["state"], "degraded")
        self.assertEqual(result["malayalam_post_asr_mode"], "source_language_fidelity_failed")
        self.assertEqual(result["malayalam_post_asr_reason"], "malayalam_source_fidelity_failed")
        self.assertNotEqual(result["malayalam_post_asr_reason"], "structurally_usable_low_trust_malayalam")
        self.assertEqual(result["summary_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertEqual(result["chatbot_blocked_reason"], "malayalam_source_fidelity_failed")


class MalayalamEvaluationTests(SimpleTestCase):
    def test_evaluation_helper_classifies_cleaned_malayalam_success_case(self):
        transcript_json = {
            "language": "ml",
            "transcript_state": "cleaned",
            "display_readable_transcript": "confidence result exam hall support",
            "trusted_visible_word_count": 16,
            "trusted_display_unit_count": 2,
            "quality_metrics": {
                "lexical_trust_score": 0.62,
                "overall_readability": 0.58,
                "wrong_script_burden": 0.04,
                "contamination_burden": 0.10,
            },
            "processing_metrics": {
                "malayalam_observability": {
                    "retry_executed": True,
                    "retry_decision_reason": "recoverable_segments_ranked",
                }
            },
        }
        processing_metadata = {
            "language": "ml",
            "detected_language": "ml",
            "transcript_state": "cleaned",
            "trusted_visible_word_count": 16,
            "trusted_display_unit_count": 2,
            "downstream_suppressed": False,
            "english_view_available": True,
            "translation_state": "translated",
            "malayalam_observability": transcript_json["processing_metrics"]["malayalam_observability"],
        }
        structured_summary = {
            "tldr": "Exam guidance about confidence and results.",
            "key_points": ["Confidence matters.", "Exam hall support is discussed."],
            "_trace": {
                "structured_summary_route": "normal_grounded",
                "structured_summary_route_reason": "cleaned_malayalam_grounded_summary",
                "structured_input_source": "trusted_assembled_units",
            },
        }

        evaluation = build_multilingual_evaluation_result(
            clip_identifier="cleaned-ml-success",
            transcript_json=transcript_json,
            processing_metadata=processing_metadata,
            structured_summary=structured_summary,
        )

        self.assertEqual(evaluation["calibration_bucket"], "clearly_cleaned")
        self.assertEqual(evaluation["evaluation_status"], "cleaned")
        self.assertEqual(evaluation["rescue_effect"], "helped")
        self.assertEqual(evaluation["structured_summary_route"], "normal_grounded")

    def test_evaluation_helper_classifies_degraded_low_evidence_malayalam(self):
        evaluation = build_multilingual_evaluation_result(
            clip_identifier="degraded-low-evidence",
            transcript_json={
                "language": "ml",
                "transcript_state": "degraded",
                "display_readable_transcript": "",
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "low_evidence_malayalam": True,
                "quality_metrics": {
                    "lexical_trust_score": 0.09,
                    "overall_readability": 0.11,
                    "wrong_script_burden": 0.22,
                    "contamination_burden": 0.66,
                },
            },
            processing_metadata={
                "language": "ml",
                "detected_language": "ml",
                "transcript_state": "degraded",
                "downstream_suppressed": True,
                "downstream_suppression_reason": "no_trusted_display_units",
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "low_evidence_malayalam": True,
                "english_view_available": False,
                "translation_blocked_reason": "degraded_safe_translation_blocked",
                "malayalam_observability": {"retry_considered": True, "retry_skipped_reason": "hopeless_low_trust_wrong_script_transcript"},
            },
            structured_summary={
                "summary_state": "degraded_safe",
                "warning_message": "Malayalam transcript quality was too low for reliable summarization.",
                "_trace": {"structured_summary_route": "degraded_safe", "structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction"},
            },
        )

        self.assertEqual(evaluation["calibration_bucket"], "degraded_low_evidence")
        self.assertEqual(evaluation["downstream_decision"], "suppressed")
        self.assertEqual(evaluation["english_view_decision"], "blocked")

    def test_evaluation_helper_identifies_wrong_script_hopeless_case(self):
        evaluation = build_multilingual_evaluation_result(
            clip_identifier="hopeless-wrong-script",
            transcript_json={
                "language": "ml",
                "transcript_state": "failed",
                "quality_metrics": {
                    "lexical_trust_score": 0.05,
                    "overall_readability": 0.07,
                    "wrong_script_burden": 0.41,
                    "contamination_burden": 0.71,
                },
            },
            processing_metadata={
                "language": "ml",
                "detected_language": "ml",
                "transcript_state": "failed",
                "downstream_suppressed": True,
                "english_view_available": False,
                "malayalam_observability": {"retry_skipped_reason": "all_segments_wrong_script_without_trusted_malayalam"},
            },
            structured_summary={"summary_state": "degraded_safe"},
        )

        self.assertEqual(evaluation["calibration_bucket"], "hopeless_wrong_script")
        self.assertIn("verify_wrong_script_rejection", evaluation["recommendation_flags"])

    def test_evaluation_helper_identifies_mixed_malayalam_english_educational_case(self):
        evaluation = build_multilingual_evaluation_result(
            clip_identifier="mixed-educational",
            transcript_json={
                "language": "ml",
                "transcript_state": "degraded",
                "display_readable_transcript": "question paper answer sheet exam hall support",
                "trusted_visible_word_count": 11,
                "trusted_display_unit_count": 1,
                "quality_metrics": {
                    "lexical_trust_score": 0.24,
                    "overall_readability": 0.26,
                    "wrong_script_burden": 0.08,
                    "contamination_burden": 0.22,
                },
            },
            processing_metadata={
                "language": "ml",
                "detected_language": "ml",
                "transcript_state": "degraded",
                "downstream_suppressed": False,
                "trusted_visible_word_count": 11,
                "trusted_display_unit_count": 1,
                "english_view_available": True,
                "translation_state": "translated",
                "malayalam_observability": {"retry_executed": True},
            },
            structured_summary={
                "summary_state": "degraded_safe",
                "_trace": {"structured_summary_route": "degraded_safe", "structured_input_source": "internal_evidence_units"},
            },
        )

        self.assertEqual(evaluation["calibration_bucket"], "degraded_but_useful")
        self.assertEqual(evaluation["structured_input_source"], "internal_evidence_units")

    def test_evaluation_helper_leaves_english_path_correctly_classified(self):
        evaluation = build_multilingual_evaluation_result(
            clip_identifier="english-reference",
            transcript_json={
                "language": "en",
                "transcript_state": "cleaned",
                "display_readable_transcript": "This is an English transcript.",
                "quality_metrics": {"overall_readability": 0.91},
            },
            processing_metadata={
                "language": "en",
                "detected_language": "en",
                "transcript_state": "cleaned",
                "english_view_available": False,
                "translation_state": "same_as_original",
            },
            structured_summary={"tldr": "English summary."},
        )

        self.assertEqual(evaluation["calibration_bucket"], "english_stable")
        self.assertEqual(evaluation["english_view_decision"], "same_as_original")

    def test_calibration_helper_marks_borderline_cases_explicitly(self):
        bucket = classify_malayalam_calibration_bucket(
            {
                "clip_identifier": "borderline-case",
                "detected_language": "ml",
                "transcript_state": "degraded",
                "lexical_trust_score": 0.21,
                "overall_readability": 0.31,
                "wrong_script_burden": 0.24,
                "contamination_burden": 0.43,
                "trusted_visible_word_count": 5,
                "trusted_display_unit_count": 1,
                "downstream_suppressed": False,
                "low_evidence_malayalam": False,
                "english_view_decision": "blocked",
                "structured_summary_route": "degraded_safe",
            }
        )

        self.assertEqual(bucket["calibration_bucket"], "borderline_review")
        self.assertTrue(bucket["borderline_signals"])

    def test_decision_trace_includes_rescue_downstream_and_english_view_reasoning(self):
        evaluation = build_multilingual_evaluation_result(
            clip_identifier="decision-trace-case",
            transcript_json={
                "language": "ml",
                "transcript_state": "degraded",
                "display_readable_transcript": "",
                "quality_metrics": {
                    "lexical_trust_score": 0.10,
                    "overall_readability": 0.12,
                    "wrong_script_burden": 0.30,
                    "contamination_burden": 0.60,
                },
            },
            processing_metadata={
                "language": "ml",
                "detected_language": "ml",
                "transcript_state": "degraded",
                "downstream_suppressed": True,
                "downstream_suppression_reason": "no_trusted_display_units",
                "english_view_available": False,
                "translation_blocked_reason": "degraded_safe_translation_blocked",
                "malayalam_observability": {"retry_skipped_reason": "hopeless_low_trust_wrong_script_transcript"},
            },
            structured_summary={
                "summary_state": "degraded_safe",
                "_trace": {"structured_summary_route_reason": "degraded_malayalam_uses_degraded_safe_reconstruction"},
            },
        )

        self.assertIn("rescue", evaluation["decision_trace"])
        self.assertIn("downstream", evaluation["decision_trace"])
        self.assertIn("english_view", evaluation["decision_trace"])
        self.assertTrue(evaluation["decision_trace_short"])


class MalayalamBenchmarkSuiteTests(SimpleTestCase):
    def test_benchmark_runner_evaluates_multiple_cases_and_returns_aggregate_summary(self):
        cases = [
            BenchmarkCase(
                case_id="cleaned-1",
                label="Cleaned Malayalam",
                expected_language="ml",
                expected_quality_bucket="clearly_cleaned",
                expected_downstream_decision="allowed",
                expected_english_view_decision="available",
                benchmark_group="cleaned_malayalam",
                transcript_snapshot={
                    "language": "ml",
                    "transcript_state": "cleaned",
                    "display_readable_transcript": "confidence result exam hall support guidance",
                    "trusted_visible_word_count": 12,
                    "trusted_display_unit_count": 2,
                    "quality_metrics": {
                        "lexical_trust_score": 0.58,
                        "overall_readability": 0.55,
                        "wrong_script_burden": 0.06,
                        "contamination_burden": 0.12,
                    },
                },
                metadata_snapshot={
                    "language": "ml",
                    "detected_language": "ml",
                    "transcript_state": "cleaned",
                    "downstream_suppressed": False,
                    "trusted_visible_word_count": 12,
                    "trusted_display_unit_count": 2,
                    "english_view_available": True,
                    "translation_state": "translated",
                },
                summary_snapshot={
                    "_trace": {
                        "structured_summary_route": "normal_grounded",
                        "structured_summary_route_reason": "cleaned_malayalam_grounded_summary",
                    }
                },
            ),
            BenchmarkCase(
                case_id="degraded-1",
                label="Low Evidence Malayalam",
                expected_language="ml",
                expected_quality_bucket="degraded_low_evidence",
                expected_downstream_decision="suppressed",
                expected_english_view_decision="blocked",
                benchmark_group="degraded_low_evidence_malayalam",
                transcript_snapshot={
                    "language": "ml",
                    "transcript_state": "degraded",
                    "display_readable_transcript": "",
                    "trusted_visible_word_count": 0,
                    "trusted_display_unit_count": 0,
                    "low_evidence_malayalam": True,
                    "quality_metrics": {
                        "lexical_trust_score": 0.08,
                        "overall_readability": 0.10,
                        "wrong_script_burden": 0.24,
                        "contamination_burden": 0.67,
                    },
                },
                metadata_snapshot={
                    "language": "ml",
                    "detected_language": "ml",
                    "transcript_state": "degraded",
                    "downstream_suppressed": True,
                    "downstream_suppression_reason": "no_trusted_display_units",
                    "trusted_visible_word_count": 0,
                    "trusted_display_unit_count": 0,
                    "low_evidence_malayalam": True,
                    "english_view_available": False,
                    "translation_blocked_reason": "degraded_safe_translation_blocked",
                },
                summary_snapshot={
                    "summary_state": "degraded_safe",
                    "_trace": {"structured_summary_route": "degraded_safe"},
                },
            ),
        ]

        suite = run_multilingual_benchmark_suite(cases)

        self.assertEqual(suite["summary"]["total_cases"], 2)
        self.assertEqual(suite["summary"]["passed_expectations"], 2)
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)
        self.assertEqual(suite["summary"]["bucket_counts"]["degraded_low_evidence"], 1)

    def test_benchmark_runner_detects_expectation_mismatches_correctly(self):
        result = evaluate_benchmark_case(
            BenchmarkCase(
                case_id="mismatch-1",
                label="Mismatch Case",
                expected_quality_bucket="clearly_cleaned",
                expected_downstream_decision="allowed",
                expected_english_view_decision="available",
                transcript_snapshot={
                    "language": "ml",
                    "transcript_state": "degraded",
                    "display_readable_transcript": "",
                    "quality_metrics": {
                        "lexical_trust_score": 0.09,
                        "overall_readability": 0.10,
                        "wrong_script_burden": 0.22,
                        "contamination_burden": 0.60,
                    },
                },
                metadata_snapshot={
                    "language": "ml",
                    "detected_language": "ml",
                    "transcript_state": "degraded",
                    "downstream_suppressed": True,
                    "downstream_suppression_reason": "no_trusted_display_units",
                    "english_view_available": False,
                },
            )
        )

        self.assertFalse(result["passed_expectations"])
        self.assertTrue(result["mismatches"])
        self.assertTrue(result["decision_trace_excerpt"])

    def test_benchmark_summary_highlights_borderline_cases(self):
        suite = run_multilingual_benchmark_suite(
            [
                BenchmarkCase(
                    case_id="borderline-1",
                    label="Borderline Malayalam",
                    benchmark_group="borderline",
                    human_review_required=True,
                    transcript_snapshot={
                        "language": "ml",
                        "transcript_state": "degraded",
                        "trusted_visible_word_count": 5,
                        "trusted_display_unit_count": 1,
                        "quality_metrics": {
                            "lexical_trust_score": 0.21,
                            "overall_readability": 0.31,
                            "wrong_script_burden": 0.24,
                            "contamination_burden": 0.43,
                        },
                    },
                    metadata_snapshot={
                        "language": "ml",
                        "detected_language": "ml",
                        "transcript_state": "degraded",
                        "downstream_suppressed": False,
                        "english_view_available": False,
                    },
                )
            ]
        )

        self.assertIn("borderline-1", suite["summary"]["borderline_case_ids"])
        self.assertIn("borderline-1", suite["summary"]["review_required_cases"])

    def test_benchmark_suite_preserves_low_evidence_suppression_expectations(self):
        suite = run_multilingual_benchmark_suite(
            [
                BenchmarkCase(
                    case_id="suppressed-1",
                    label="Suppressed Malayalam",
                    expected_quality_bucket="degraded_low_evidence",
                    expected_downstream_decision="suppressed",
                    expected_english_view_decision="blocked",
                    benchmark_group="degraded_low_evidence_malayalam",
                    transcript_snapshot={
                        "language": "ml",
                        "transcript_state": "degraded",
                        "quality_metrics": {
                            "lexical_trust_score": 0.07,
                            "overall_readability": 0.09,
                            "wrong_script_burden": 0.26,
                            "contamination_burden": 0.69,
                        },
                    },
                    metadata_snapshot={
                        "language": "ml",
                        "detected_language": "ml",
                        "transcript_state": "degraded",
                        "downstream_suppressed": True,
                        "downstream_suppression_reason": "no_trusted_display_units",
                        "low_evidence_malayalam": True,
                        "english_view_available": False,
                        "translation_blocked_reason": "degraded_safe_translation_blocked",
                    },
                )
            ]
        )

        self.assertEqual(suite["summary"]["failed_expectations"], 0)
        self.assertEqual(suite["summary"]["downstream_decision_counts"]["suppressed"], 1)

    def test_benchmark_suite_preserves_wrong_script_rejection_expectations(self):
        suite = run_multilingual_benchmark_suite(
            [
                BenchmarkCase(
                    case_id="wrong-script-1",
                    label="Wrong Script Hopeless",
                    expected_quality_bucket="hopeless_wrong_script",
                    expected_downstream_decision="suppressed",
                    expected_english_view_decision="blocked",
                    benchmark_group="wrong_script_hopeless",
                    transcript_snapshot={
                        "language": "ml",
                        "transcript_state": "failed",
                        "quality_metrics": {
                            "lexical_trust_score": 0.05,
                            "overall_readability": 0.06,
                            "wrong_script_burden": 0.39,
                            "contamination_burden": 0.70,
                        },
                    },
                    metadata_snapshot={
                        "language": "ml",
                        "detected_language": "ml",
                        "transcript_state": "failed",
                        "downstream_suppressed": True,
                        "english_view_available": False,
                    },
                )
            ]
        )

        self.assertEqual(suite["summary"]["bucket_counts"]["hopeless_wrong_script"], 1)
        self.assertIn("wrong-script-1", suite["summary"]["persistent_failure_case_ids"])

    def test_benchmark_suite_preserves_cleaned_malayalam_grounded_success_expectations(self):
        suite = run_multilingual_benchmark_suite(
            [
                BenchmarkCase(
                    case_id="cleaned-grounded-1",
                    label="Cleaned Grounded",
                    expected_quality_bucket="clearly_cleaned",
                    expected_downstream_decision="allowed",
                    expected_english_view_decision="available",
                    benchmark_group="cleaned_malayalam",
                    transcript_snapshot={
                        "language": "ml",
                        "transcript_state": "cleaned",
                        "display_readable_transcript": "grounded cleaned transcript",
                        "trusted_visible_word_count": 14,
                        "trusted_display_unit_count": 2,
                        "quality_metrics": {
                            "lexical_trust_score": 0.55,
                            "overall_readability": 0.52,
                            "wrong_script_burden": 0.05,
                            "contamination_burden": 0.08,
                        },
                    },
                    metadata_snapshot={
                        "language": "ml",
                        "detected_language": "ml",
                        "transcript_state": "cleaned",
                        "downstream_suppressed": False,
                        "trusted_visible_word_count": 14,
                        "trusted_display_unit_count": 2,
                        "english_view_available": True,
                    },
                    summary_snapshot={"_trace": {"structured_summary_route": "normal_grounded"}},
                )
            ]
        )

        self.assertEqual(suite["summary"]["failed_expectations"], 0)
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)

    def test_benchmark_suite_preserves_english_stability_expectations(self):
        suite = run_multilingual_benchmark_suite(
            [
                BenchmarkCase(
                    case_id="english-1",
                    label="English Stable",
                    expected_language="en",
                    expected_quality_bucket="english_stable",
                    expected_downstream_decision="allowed",
                    expected_english_view_decision="same_as_original",
                    benchmark_group="english_reference",
                    transcript_snapshot={
                        "language": "en",
                        "transcript_state": "cleaned",
                        "display_readable_transcript": "English transcript",
                        "quality_metrics": {"overall_readability": 0.90},
                    },
                    metadata_snapshot={
                        "language": "en",
                        "detected_language": "en",
                        "transcript_state": "cleaned",
                        "downstream_suppressed": False,
                        "english_view_available": False,
                        "translation_state": "same_as_original",
                    },
                )
            ]
        )

        self.assertEqual(suite["summary"]["english_view_decision_counts"]["same_as_original"], 1)
        self.assertEqual(suite["summary"]["failed_expectations"], 0)

    def test_benchmark_output_includes_decision_trace_excerpts_for_mismatches(self):
        suite = run_multilingual_benchmark_suite(
            [
                BenchmarkCase(
                    case_id="mismatch-trace-1",
                    label="Mismatch Trace",
                    expected_downstream_decision="allowed",
                    transcript_snapshot={
                        "language": "ml",
                        "transcript_state": "degraded",
                        "quality_metrics": {
                            "lexical_trust_score": 0.08,
                            "overall_readability": 0.11,
                            "wrong_script_burden": 0.20,
                            "contamination_burden": 0.64,
                        },
                    },
                    metadata_snapshot={
                        "language": "ml",
                        "detected_language": "ml",
                        "transcript_state": "degraded",
                        "downstream_suppressed": True,
                        "downstream_suppression_reason": "no_trusted_display_units",
                        "english_view_available": False,
                    },
                )
            ]
        )

        self.assertEqual(suite["summary"]["failed_expectations"], 1)
        self.assertTrue(suite["summary"]["mismatched_cases"][0]["decision_trace_excerpt"])


class MalayalamBenchmarkCommandTests(SimpleTestCase):
    @patch("videos.management.commands.benchmark_malayalam_local_asr._prepare_audio_input", return_value=("sample.wav", False))
    @patch("videos.management.commands.benchmark_malayalam_local_asr._benchmark_profile")
    def test_command_outputs_two_profiles(self, mock_profile, _prepare):
        mock_profile.side_effect = [
            {
                "profile": "small_first_pass",
                "total_transcription_time_seconds": 10.0,
                "real_time_factor": 1.2,
                "transcript_state": "cleaned",
                "fallback_triggered": False,
                "selected_model_path": "small",
            },
            {
                "profile": "large_v3",
                "total_transcription_time_seconds": 20.0,
                "real_time_factor": 2.4,
                "transcript_state": "cleaned",
                "fallback_triggered": False,
                "selected_model_path": "large-v3",
            },
        ]
        out = StringIO()
        sample = Path("backend/videos/tests_multilingual.py")
        call_command("benchmark_malayalam_local_asr", str(sample), stdout=out)
        rendered = out.getvalue()
        self.assertIn('"profile": "small_first_pass"', rendered)
        self.assertIn('"profile": "large_v3"', rendered)


class MalayalamThresholdCalibrationTests(SimpleTestCase):
    def _build_cases(self):
        return [
            BenchmarkCase(
                case_id="cleaned-success",
                label="Cleaned Malayalam Success",
                benchmark_group="cleaned_malayalam",
                transcript_snapshot={
                    "language": "ml",
                    "transcript_state": "cleaned",
                    "display_readable_transcript": "grounded cleaned transcript with exam hall guidance and result support",
                    "trusted_visible_word_count": 12,
                    "trusted_display_unit_count": 2,
                    "quality_metrics": {
                        "lexical_trust_score": 0.56,
                        "overall_readability": 0.51,
                        "wrong_script_burden": 0.06,
                        "contamination_burden": 0.11,
                    },
                },
                metadata_snapshot={
                    "detected_language": "ml",
                    "transcript_state": "cleaned",
                    "downstream_suppressed": False,
                    "trusted_visible_word_count": 12,
                    "trusted_display_unit_count": 2,
                    "english_view_available": True,
                },
                summary_snapshot={"_trace": {"structured_summary_route": "normal_grounded"}},
            ),
            BenchmarkCase(
                case_id="borderline-useful",
                label="Borderline Useful",
                benchmark_group="degraded_useful_malayalam",
                transcript_snapshot={
                    "language": "ml",
                    "transcript_state": "degraded",
                    "display_readable_transcript": "exam hall result confidence support",
                    "trusted_visible_word_count": 7,
                    "trusted_display_unit_count": 1,
                    "quality_metrics": {
                        "lexical_trust_score": 0.20,
                        "overall_readability": 0.24,
                        "wrong_script_burden": 0.12,
                        "contamination_burden": 0.24,
                    },
                },
                metadata_snapshot={
                    "detected_language": "ml",
                    "transcript_state": "degraded",
                    "downstream_suppressed": False,
                    "trusted_visible_word_count": 7,
                    "trusted_display_unit_count": 1,
                    "english_view_available": False,
                },
                summary_snapshot={"_trace": {"structured_summary_route": "degraded_safe"}},
            ),
            BenchmarkCase(
                case_id="low-evidence",
                label="Low Evidence",
                benchmark_group="degraded_low_evidence_malayalam",
                transcript_snapshot={
                    "language": "ml",
                    "transcript_state": "degraded",
                    "trusted_visible_word_count": 0,
                    "trusted_display_unit_count": 0,
                    "quality_metrics": {
                        "lexical_trust_score": 0.08,
                        "overall_readability": 0.10,
                        "wrong_script_burden": 0.22,
                        "contamination_burden": 0.68,
                    },
                },
                metadata_snapshot={
                    "detected_language": "ml",
                    "transcript_state": "degraded",
                    "downstream_suppressed": True,
                    "downstream_suppression_reason": "no_trusted_display_units",
                    "low_evidence_malayalam": True,
                    "english_view_available": False,
                    "translation_blocked_reason": "degraded_safe_translation_blocked",
                },
                summary_snapshot={"summary_state": "degraded_safe"},
            ),
            BenchmarkCase(
                case_id="wrong-script",
                label="Wrong Script",
                benchmark_group="wrong_script_hopeless",
                transcript_snapshot={
                    "language": "ml",
                    "transcript_state": "failed",
                    "trusted_visible_word_count": 0,
                    "trusted_display_unit_count": 0,
                    "quality_metrics": {
                        "lexical_trust_score": 0.05,
                        "overall_readability": 0.06,
                        "wrong_script_burden": 0.42,
                        "contamination_burden": 0.73,
                    },
                },
                metadata_snapshot={
                    "detected_language": "ml",
                    "transcript_state": "failed",
                    "downstream_suppressed": True,
                    "english_view_available": False,
                },
            ),
            BenchmarkCase(
                case_id="english-contaminated",
                label="English Contaminated",
                benchmark_group="english_contaminated",
                transcript_snapshot={
                    "language": "ml",
                    "transcript_state": "degraded",
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
                    "detected_language": "ml",
                    "transcript_state": "degraded",
                    "downstream_suppressed": True,
                    "low_evidence_malayalam": True,
                    "english_view_available": False,
                },
            ),
        ]

    def test_threshold_profile_schema_can_represent_candidate_override_cleanly(self):
        profile = MalayalamThresholdProfile(
            profile_name="candidate_relaxed_degraded_useful",
            base_profile="runtime_default",
            overridden_thresholds={
                "degraded_useful_min_trusted_visible_words": 6.0,
                "degraded_useful_min_readability": 0.22,
            },
            notes="dry-run only",
        )
        self.assertEqual(profile.profile_name, "candidate_relaxed_degraded_useful")
        self.assertEqual(profile.overridden_thresholds["degraded_useful_min_trusted_visible_words"], 6.0)

    def test_dry_run_comparison_detects_changed_benchmark_outcomes(self):
        comparison = compare_threshold_profiles_on_benchmark(
            self._build_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="candidate_more_permissive_useful",
                overridden_thresholds={
                    "degraded_useful_min_trusted_visible_words": 7.0,
                    "degraded_useful_min_readability": 0.24,
                    "degraded_useful_min_lexical_trust": 0.20,
                },
            ),
        )
        self.assertIn("borderline-useful", comparison["summary"]["improved_case_ids"])
        self.assertIn("borderline-useful", comparison["summary"]["changed_bucket_cases"])

    def test_protected_low_evidence_suppression_regressions_are_flagged(self):
        comparison = compare_threshold_profiles_on_benchmark(
            self._build_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="candidate_low_evidence_regression",
                overridden_thresholds={
                    "degraded_low_evidence_max_trusted_visible_words": -1.0,
                    "low_evidence_max_trusted_display_units": -1.0,
                    "english_contamination_min_burden": 0.90,
                    "degraded_useful_min_trusted_visible_words": 0.0,
                    "degraded_useful_min_trusted_display_units": 0.0,
                    "degraded_useful_min_readability": 0.05,
                    "degraded_useful_min_lexical_trust": 0.05,
                },
            ),
        )
        self.assertIn("low_evidence_suppression_regressed", comparison["summary"]["safety_regression_flags"])
        self.assertTrue(comparison["summary"]["protected_case_regressions"])

    def test_wrong_script_rejection_regressions_are_flagged(self):
        comparison = compare_threshold_profiles_on_benchmark(
            self._build_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="candidate_wrong_script_softened",
                overridden_thresholds={
                    "hopeless_wrong_script_min_burden": 0.50,
                },
            ),
        )
        self.assertIn("wrong_script_rejection_regressed", comparison["summary"]["safety_regression_flags"])

    def test_english_contamination_regressions_are_flagged(self):
        comparison = compare_threshold_profiles_on_benchmark(
            self._build_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="candidate_contamination_softened",
                overridden_thresholds={
                    "english_contamination_min_burden": 0.80,
                    "degraded_useful_min_trusted_visible_words": 2.0,
                    "degraded_useful_min_trusted_display_units": 0.0,
                    "degraded_useful_min_readability": 0.18,
                    "degraded_useful_min_lexical_trust": 0.18,
                },
            ),
        )
        self.assertIn("english_contamination_regressed", comparison["summary"]["safety_regression_flags"])

    def test_cleaned_malayalam_useful_improvements_can_be_reported_as_helpful_changes(self):
        cases = self._build_cases()
        cases[0] = BenchmarkCase(
            case_id="cleaned-success",
            label="Cleaned Malayalam Success",
            benchmark_group="cleaned_malayalam",
            transcript_snapshot={
                "language": "ml",
                "transcript_state": "cleaned",
                "display_readable_transcript": "grounded cleaned transcript",
                "trusted_visible_word_count": 9,
                "trusted_display_unit_count": 1,
                "quality_metrics": {
                    "lexical_trust_score": 0.45,
                    "overall_readability": 0.40,
                    "wrong_script_burden": 0.08,
                    "contamination_burden": 0.10,
                },
            },
            metadata_snapshot={
                "detected_language": "ml",
                "transcript_state": "cleaned",
                "downstream_suppressed": False,
                "trusted_visible_word_count": 9,
                "trusted_display_unit_count": 1,
                "english_view_available": True,
            },
            summary_snapshot={"_trace": {"structured_summary_route": "normal_grounded"}},
        )
        comparison = compare_threshold_profiles_on_benchmark(
            cases,
            candidate_profile=MalayalamThresholdProfile(
                profile_name="candidate_cleaned_support",
                overridden_thresholds={
                    "cleaned_min_trusted_visible_words": 9.0,
                },
            ),
        )
        self.assertIn("cleaned-success", comparison["summary"]["improved_case_ids"])

    def test_unchanged_benchmark_cases_remain_stable_under_identical_profile(self):
        comparison = compare_threshold_profiles_on_benchmark(self._build_cases())
        self.assertEqual(comparison["summary"]["total_changed_cases"], 0)
        self.assertEqual(len(comparison["summary"]["unchanged_case_ids"]), len(self._build_cases()))

    def test_runtime_behavior_remains_unchanged_when_no_override_profile_is_supplied(self):
        base = run_multilingual_benchmark_suite(self._build_cases())
        comparison = compare_threshold_profiles_on_benchmark(self._build_cases())
        self.assertEqual(base["summary"]["bucket_counts"], comparison["base_suite"]["summary"]["bucket_counts"])
        self.assertEqual(comparison["summary"]["total_changed_cases"], 0)


class MalayalamFirstCandidateExperimentTests(SimpleTestCase):
    def _build_cases(self):
        return MalayalamThresholdCalibrationTests()._build_cases()

    def test_first_candidate_profile_is_represented_cleanly_and_is_bounded(self):
        profile = build_first_candidate_threshold_profile()
        self.assertEqual(profile["profile_name"], "candidate_borderline_recoverable_malayalam_v1")
        self.assertLessEqual(len(profile["overridden_thresholds"]), 3)
        self.assertIn("wrong_script_rejection_unchanged", profile["protected_constraints"])

    def test_candidate_experiment_runner_compares_default_vs_candidate_and_returns_decision_report(self):
        report = run_first_candidate_threshold_experiment(self._build_cases())
        self.assertIn("report", report)
        self.assertIn("experiment_decision", report["report"])
        self.assertIn("changed_case_summaries", report["report"])
        self.assertEqual(report["candidate_profile"]["profile_name"], "candidate_borderline_recoverable_malayalam_v1")

    def test_protected_safety_regressions_force_reject_candidate(self):
        report = run_first_candidate_threshold_experiment(
            self._build_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="unsafe_candidate",
                overridden_thresholds={
                    "degraded_low_evidence_max_trusted_visible_words": -1.0,
                    "low_evidence_max_trusted_display_units": -1.0,
                    "english_contamination_min_burden": 0.90,
                },
            ),
        )
        self.assertEqual(report["report"]["experiment_decision"], "reject_candidate")
        self.assertTrue(report["report"]["protected_case_regressions"])

    def test_borderline_only_improvements_can_produce_review_candidate(self):
        cases = self._build_cases()
        report = run_first_candidate_threshold_experiment(
            cases,
            candidate_profile=MalayalamThresholdProfile(
                profile_name="borderline_review_candidate",
                overridden_thresholds={
                    "degraded_useful_min_trusted_visible_words": 7.0,
                    "degraded_useful_min_readability": 0.24,
                    "degraded_useful_min_lexical_trust": 0.20,
                },
            ),
        )
        self.assertEqual(report["report"]["experiment_decision"], "review_candidate")
        self.assertIn("borderline-useful", report["report"]["borderline_case_movements"])

    def test_beneficial_changes_without_protected_regressions_can_be_cautiously_promising(self):
        cases = self._build_cases()
        cases[0] = BenchmarkCase(
            case_id="cleaned-success",
            label="Cleaned Malayalam Success",
            benchmark_group="cleaned_malayalam",
            transcript_snapshot={
                "language": "ml",
                "transcript_state": "cleaned",
                "display_readable_transcript": "grounded cleaned transcript",
                "trusted_visible_word_count": 9,
                "trusted_display_unit_count": 1,
                "quality_metrics": {
                    "lexical_trust_score": 0.45,
                    "overall_readability": 0.40,
                    "wrong_script_burden": 0.08,
                    "contamination_burden": 0.10,
                },
            },
            metadata_snapshot={
                "detected_language": "ml",
                "transcript_state": "cleaned",
                "downstream_suppressed": False,
                "trusted_visible_word_count": 9,
                "trusted_display_unit_count": 1,
                "english_view_available": True,
            },
            summary_snapshot={"_trace": {"structured_summary_route": "normal_grounded"}},
        )
        report = run_first_candidate_threshold_experiment(
            cases,
            candidate_profile=MalayalamThresholdProfile(
                profile_name="helpful_candidate",
                overridden_thresholds={"cleaned_min_trusted_visible_words": 9.0},
            ),
        )
        self.assertEqual(report["report"]["experiment_decision"], "cautiously_promising")
        self.assertIn("cleaned-success", report["report"]["changed_bucket_cases"])

    def test_identical_candidate_default_profile_produces_unchanged_or_neutral_report(self):
        report = run_first_candidate_threshold_experiment(
            self._build_cases(),
            candidate_profile=MalayalamThresholdProfile(profile_name="identical_runtime_default"),
        )
        self.assertEqual(report["report"]["total_changed_cases"], 0)
        self.assertEqual(report["report"]["experiment_decision"], "review_candidate")

    def test_runtime_behavior_remains_unchanged_when_experiment_helpers_are_unused(self):
        base = run_multilingual_benchmark_suite(self._build_cases())
        self.assertEqual(base["summary"]["bucket_counts"]["degraded_low_evidence"], 1)


class MalayalamExpandedBenchmarkFixtureTests(SimpleTestCase):
    def test_expanded_benchmark_fixture_pack_builds_successfully_and_returns_stable_case_ids(self):
        cases = build_expanded_malayalam_benchmark_cases()
        case_ids = [case.case_id for case in cases]
        self.assertGreaterEqual(len(cases), 10)
        self.assertIn("ml_cleaned_grounded_exam_case_01", case_ids)
        self.assertIn("ml_degraded_low_evidence_case_01", case_ids)
        self.assertIn("ml_wrong_script_hopeless_case_01", case_ids)
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_expanded_benchmark_suite_covers_protected_low_evidence_suppression_cases(self):
        suite = run_multilingual_benchmark_suite(build_expanded_malayalam_benchmark_cases())
        self.assertGreaterEqual(suite["summary"]["bucket_counts"]["degraded_low_evidence"], 2)

    def test_expanded_benchmark_suite_covers_wrong_script_rejection_cases(self):
        suite = run_multilingual_benchmark_suite(build_expanded_malayalam_benchmark_cases())
        self.assertGreaterEqual(suite["summary"]["bucket_counts"]["hopeless_wrong_script"], 1)

    def test_expanded_benchmark_suite_covers_english_contamination_cases(self):
        suite = run_multilingual_benchmark_suite(build_expanded_malayalam_benchmark_cases())
        self.assertGreaterEqual(suite["summary"]["bucket_counts"]["english_contaminated"], 1)

    def test_expanded_benchmark_suite_covers_mixed_malayalam_english_educational_cases(self):
        suite = run_multilingual_benchmark_suite(build_expanded_malayalam_benchmark_cases())
        groups = {row["benchmark_group"] for row in suite["results"]}
        self.assertIn("mixed_malayalam_english_educational", groups)

    def test_expanded_benchmark_suite_covers_cleaned_malayalam_grounded_success_cases(self):
        suite = run_multilingual_benchmark_suite(build_expanded_malayalam_benchmark_cases())
        self.assertGreaterEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 2)

    def test_first_candidate_experiment_runner_can_run_against_expanded_fixture_pack(self):
        report = run_first_candidate_threshold_experiment(build_expanded_malayalam_benchmark_cases())
        self.assertIn("report", report)
        self.assertIn("comparison", report)
        self.assertIn(report["report"]["experiment_decision"], {"reject_candidate", "review_candidate", "cautiously_promising"})

    def test_experiment_report_remains_structured_and_stable_with_expanded_fixture_pack(self):
        report = run_first_candidate_threshold_experiment(build_expanded_malayalam_benchmark_cases())
        self.assertIn("changed_case_summaries", report["report"])
        self.assertIn("dominant_improvement_reasons", report["report"])
        self.assertIn("dominant_regression_reasons", report["report"])
        self.assertIn("protected_case_regressions", report["report"])

    def test_default_fixture_registry_helper_returns_structured_benchmark_cases(self):
        cases = build_default_multilingual_benchmark_cases()
        self.assertTrue(all(isinstance(case, BenchmarkCase) for case in cases))


class MalayalamExperimentReportExportTests(SimpleTestCase):
    def test_experiment_report_export_helper_returns_structured_stable_payload(self):
        experiment = run_first_candidate_threshold_experiment(build_expanded_malayalam_benchmark_cases())
        exported = export_first_candidate_experiment_report(experiment)
        self.assertEqual(exported["report_version"], "ml_experiment_report_v1")
        self.assertIn("experiment_decision", exported)
        self.assertIn("changed_case_summaries", exported)

    def test_export_includes_executive_summary_and_protected_regression_summary(self):
        experiment = run_first_candidate_threshold_experiment(build_expanded_malayalam_benchmark_cases())
        exported = export_first_candidate_experiment_report(experiment)
        self.assertIn("executive_summary", exported)
        self.assertIn("protected_regression_summary", exported)

    def test_changed_case_entries_include_old_new_bucket_and_decision_fields(self):
        experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="review_candidate_export",
                overridden_thresholds={
                    "degraded_useful_min_trusted_visible_words": 7.0,
                    "degraded_useful_min_readability": 0.20,
                },
            ),
        )
        exported = export_first_candidate_experiment_report(experiment)
        if exported["changed_case_summaries"]:
            row = exported["changed_case_summaries"][0]
            self.assertIn("old_bucket", row)
            self.assertIn("new_bucket", row)
            self.assertIn("old_downstream_decision", row)
            self.assertIn("new_downstream_decision", row)

    def test_protected_regressions_are_clearly_represented_in_exported_report(self):
        experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="unsafe_export_candidate",
                overridden_thresholds={
                    "degraded_low_evidence_max_trusted_visible_words": -1.0,
                    "low_evidence_max_trusted_display_units": -1.0,
                    "english_contamination_min_burden": 0.90,
                },
            ),
        )
        exported = export_first_candidate_experiment_report(experiment)
        self.assertTrue(exported["protected_case_regressions"])
        self.assertTrue(exported["safety_regression_flags"])
        self.assertTrue(exported["protected_regression_summary"]["has_protected_regressions"])

    def test_unchanged_neutral_experiments_still_export_cleanly(self):
        experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(profile_name="runtime_default_clone"),
        )
        exported = export_first_candidate_experiment_report(experiment)
        self.assertEqual(exported["total_changed_cases"], 0)
        self.assertEqual(exported["changed_case_summaries"], [])

    def test_first_candidate_experiment_result_can_be_exported_from_expanded_benchmark_fixture_run(self):
        exported = export_first_candidate_experiment_report(
            run_first_candidate_threshold_experiment(build_expanded_malayalam_benchmark_cases())
        )
        self.assertGreaterEqual(exported["total_cases"], 10)

    def test_runtime_behavior_remains_unchanged(self):
        suite = run_multilingual_benchmark_suite(build_default_multilingual_benchmark_cases())
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)


class MalayalamFirstCandidateDecisionReviewTests(SimpleTestCase):
    def test_protected_regressions_produce_reject_and_archive(self):
        experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="unsafe_review_candidate",
                overridden_thresholds={
                    "degraded_low_evidence_max_trusted_visible_words": -1.0,
                    "low_evidence_max_trusted_display_units": -1.0,
                    "english_contamination_min_burden": 0.90,
                },
            ),
        )
        review = review_first_candidate_experiment(experiment)
        self.assertEqual(review["final_recommendation"], "reject_and_archive")
        self.assertTrue(review["protected_regression_present"])
        self.assertTrue(review["should_reject_current_candidate"])

    def test_borderline_only_movements_can_produce_manual_review_needed(self):
        experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="borderline_review_candidate",
                overridden_thresholds={
                    "degraded_useful_min_trusted_visible_words": 7.0,
                    "degraded_useful_min_readability": 0.24,
                    "degraded_useful_min_lexical_trust": 0.20,
                },
            ),
        )
        review = review_first_candidate_experiment(experiment)
        self.assertEqual(review["final_recommendation"], "manual_review_needed")
        self.assertTrue(review["manual_review_required"])

    def test_cautiously_promising_cleaned_only_promotions_can_produce_consider_narrow_followup_candidate(self):
        cases = MalayalamThresholdCalibrationTests()._build_cases()
        cases[0] = BenchmarkCase(
            case_id="cleaned-success",
            label="Cleaned Malayalam Success",
            benchmark_group="cleaned_malayalam",
            transcript_snapshot={
                "language": "ml",
                "transcript_state": "cleaned",
                "display_readable_transcript": "grounded cleaned transcript",
                "trusted_visible_word_count": 9,
                "trusted_display_unit_count": 1,
                "quality_metrics": {
                    "lexical_trust_score": 0.45,
                    "overall_readability": 0.40,
                    "wrong_script_burden": 0.08,
                    "contamination_burden": 0.10,
                },
            },
            metadata_snapshot={
                "detected_language": "ml",
                "transcript_state": "cleaned",
                "downstream_suppressed": False,
                "trusted_visible_word_count": 9,
                "trusted_display_unit_count": 1,
                "english_view_available": True,
            },
            summary_snapshot={"_trace": {"structured_summary_route": "normal_grounded"}},
        )
        experiment = run_first_candidate_threshold_experiment(
            cases,
            candidate_profile=MalayalamThresholdProfile(
                profile_name="helpful_review_candidate",
                overridden_thresholds={"cleaned_min_trusted_visible_words": 9.0},
            ),
        )
        review = review_first_candidate_experiment(experiment)
        self.assertEqual(review["final_recommendation"], "consider_narrow_followup_candidate")
        self.assertTrue(review["should_try_followup_candidate"])

    def test_unchanged_neutral_report_can_produce_keep_as_reference_only(self):
        experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(profile_name="runtime_default_clone"),
        )
        review = review_first_candidate_experiment(experiment)
        self.assertEqual(review["final_recommendation"], "keep_as_reference_only")
        self.assertTrue(review["should_keep_for_reference_only"])

    def test_next_step_guidance_includes_priority_case_ids(self):
        experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="review_priority_candidate",
                overridden_thresholds={
                    "degraded_useful_min_trusted_visible_words": 7.0,
                    "degraded_useful_min_readability": 0.24,
                    "degraded_useful_min_lexical_trust": 0.20,
                },
            ),
        )
        review = review_first_candidate_experiment(export_first_candidate_experiment_report(experiment))
        self.assertIn("priority_case_ids_for_manual_review", review)
        self.assertIsInstance(review["priority_case_ids_for_manual_review"], list)
        self.assertIn("priority_case_ids_for_audio_review", review)
        self.assertIn("priority_case_ids_for_threshold_focus", review)

    def test_decision_rationale_includes_strongest_positive_and_negative_signals(self):
        experiment = run_first_candidate_threshold_experiment(build_expanded_malayalam_benchmark_cases())
        review = review_first_candidate_experiment(experiment)
        self.assertIn("strongest_positive_signal", review)
        self.assertIn("strongest_negative_signal", review)
        self.assertIn("decision_confidence", review)

    def test_runtime_behavior_remains_unchanged(self):
        suite = run_multilingual_benchmark_suite(build_default_multilingual_benchmark_cases())
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)


class MalayalamSecondCandidateExperimentTests(SimpleTestCase):
    def _build_promising_first_review(self):
        cases = MalayalamThresholdCalibrationTests()._build_cases()
        cases[0] = BenchmarkCase(
            case_id="cleaned-success",
            label="Cleaned Malayalam Success",
            benchmark_group="cleaned_malayalam",
            transcript_snapshot={
                "language": "ml",
                "transcript_state": "cleaned",
                "display_readable_transcript": "grounded cleaned transcript",
                "trusted_visible_word_count": 9,
                "trusted_display_unit_count": 1,
                "quality_metrics": {
                    "lexical_trust_score": 0.45,
                    "overall_readability": 0.40,
                    "wrong_script_burden": 0.08,
                    "contamination_burden": 0.10,
                },
            },
            metadata_snapshot={
                "detected_language": "ml",
                "transcript_state": "cleaned",
                "downstream_suppressed": False,
                "trusted_visible_word_count": 9,
                "trusted_display_unit_count": 1,
                "english_view_available": True,
            },
            summary_snapshot={"_trace": {"structured_summary_route": "normal_grounded"}},
        )
        experiment = run_first_candidate_threshold_experiment(
            cases,
            candidate_profile=MalayalamThresholdProfile(
                profile_name="helpful_review_candidate",
                overridden_thresholds={"cleaned_min_trusted_visible_words": 9.0},
            ),
        )
        return review_first_candidate_experiment(experiment), cases

    def test_first_candidate_reject_and_archive_blocks_second_candidate_creation(self):
        experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="unsafe_review_candidate",
                overridden_thresholds={
                    "degraded_low_evidence_max_trusted_visible_words": -1.0,
                    "low_evidence_max_trusted_display_units": -1.0,
                    "english_contamination_min_burden": 0.90,
                },
            ),
        )
        gate = should_build_second_candidate(review_first_candidate_experiment(experiment))
        self.assertFalse(gate["allow_second_candidate"])
        self.assertEqual(gate["gate_reason"], "first_candidate_rejected_due_to_protected_regressions")

    def test_keep_as_reference_only_blocks_second_candidate_creation(self):
        experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(profile_name="runtime_default_clone"),
        )
        gate = should_build_second_candidate(review_first_candidate_experiment(experiment))
        self.assertFalse(gate["allow_second_candidate"])
        self.assertEqual(gate["gate_reason"], "first_candidate_archived_as_reference_only")

    def test_consider_narrow_followup_candidate_allows_second_candidate_creation(self):
        first_review, _cases = self._build_promising_first_review()
        gate = should_build_second_candidate(first_review)
        self.assertTrue(gate["allow_second_candidate"])
        self.assertEqual(gate["source_recommendation"], "consider_narrow_followup_candidate")

    def test_second_candidate_is_narrower_or_equally_bounded_relative_to_first_candidate(self):
        first_review, _cases = self._build_promising_first_review()
        first_profile = build_first_candidate_threshold_profile()
        second_profile = build_second_candidate_threshold_profile(first_review)
        self.assertLessEqual(len(second_profile["overridden_thresholds"]), len(first_profile["overridden_thresholds"]))

    def test_second_candidate_experiment_runner_returns_blocked_noop_result_when_gate_denies_creation(self):
        experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(profile_name="runtime_default_clone"),
        )
        second = run_second_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            first_candidate_review=review_first_candidate_experiment(experiment),
        )
        self.assertTrue(second["second_candidate_blocked"])
        self.assertFalse(second["second_candidate_built"])
        exported = export_second_candidate_experiment_report(second)
        self.assertEqual(exported["experiment_decision"], "blocked_no_second_candidate")

    def test_second_candidate_experiment_can_run_when_gate_allows_it(self):
        first_review, cases = self._build_promising_first_review()
        second = run_second_candidate_threshold_experiment(
            cases,
            first_candidate_review=first_review,
        )
        self.assertTrue(second["second_candidate_attempted"])
        self.assertTrue(second["second_candidate_built"])
        self.assertFalse(second["second_candidate_blocked"])
        review = review_second_candidate_experiment(second)
        self.assertIn(
            review["final_recommendation"],
            {"reject_and_archive", "manual_review_needed", "keep_as_reference_only", "consider_threshold_adoption_review"},
        )

    def test_runtime_behavior_remains_unchanged(self):
        suite = run_multilingual_benchmark_suite(build_default_multilingual_benchmark_cases())
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)


class MalayalamCalibrationCycleConclusionTests(SimpleTestCase):
    def _build_promising_first_and_second(self):
        review, cases = MalayalamSecondCandidateExperimentTests()._build_promising_first_review()
        second = run_second_candidate_threshold_experiment(cases, first_candidate_review=review)
        second_review = review_second_candidate_experiment(second)
        return review, second, second_review

    def test_when_both_candidates_are_weak_or_risky_final_conclusion_recommends_no_threshold_adoption(self):
        first_experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="borderline_review_candidate",
                overridden_thresholds={
                    "degraded_useful_min_trusted_visible_words": 7.0,
                    "degraded_useful_min_readability": 0.24,
                    "degraded_useful_min_lexical_trust": 0.20,
                },
            ),
        )
        first_review = review_first_candidate_experiment(first_experiment)
        second = run_second_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            first_candidate_review=first_review,
        )
        conclusion = conclude_malayalam_calibration_cycle(
            first_candidate_result=first_experiment,
            first_candidate_review=first_review,
            second_candidate_result=second,
        )
        self.assertEqual(conclusion["final_cycle_recommendation"], "no_threshold_adoption")
        self.assertTrue(conclusion["no_threshold_adoption_recommended"])

    def test_when_first_is_blocked_risky_and_second_is_noop_final_conclusion_archives_stops_cleanly(self):
        first_experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="unsafe_review_candidate",
                overridden_thresholds={
                    "degraded_low_evidence_max_trusted_visible_words": -1.0,
                    "low_evidence_max_trusted_display_units": -1.0,
                    "english_contamination_min_burden": 0.90,
                },
            ),
        )
        first_review = review_first_candidate_experiment(first_experiment)
        second = run_second_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            first_candidate_review=first_review,
        )
        conclusion = conclude_malayalam_calibration_cycle(
            first_candidate_result=first_experiment,
            first_candidate_review=first_review,
            second_candidate_result=second,
        )
        self.assertEqual(conclusion["final_cycle_recommendation"], "archive_and_stop")
        self.assertTrue(conclusion["archive_candidates_for_reference"])

    def test_when_candidate_is_only_useful_as_reference_final_conclusion_can_produce_keep_reference_and_stop(self):
        first_experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(profile_name="runtime_default_clone"),
        )
        first_review = review_first_candidate_experiment(first_experiment)
        second = run_second_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            first_candidate_review=first_review,
        )
        conclusion = conclude_malayalam_calibration_cycle(
            first_candidate_result=first_experiment,
            first_candidate_review=first_review,
            second_candidate_result=second,
        )
        self.assertEqual(conclusion["final_cycle_recommendation"], "keep_reference_and_stop")
        self.assertEqual(conclusion["winning_candidate"], "first_candidate")

    def test_when_one_candidate_is_cautiously_promising_without_protected_regressions_final_conclusion_can_produce_manual_adoption_review_only(self):
        first_review, second_result, second_review = self._build_promising_first_and_second()
        conclusion = conclude_malayalam_calibration_cycle(
            first_candidate_review=first_review,
            second_candidate_result=second_result,
            second_candidate_review=second_review,
        )
        self.assertEqual(conclusion["final_cycle_recommendation"], "manual_adoption_review_only")
        self.assertTrue(conclusion["adoption_review_justified"])

    def test_future_guidance_includes_fixture_expansion_and_audio_review_recommendations_when_adoption_is_not_justified(self):
        first_experiment = run_first_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            candidate_profile=MalayalamThresholdProfile(
                profile_name="borderline_review_candidate",
                overridden_thresholds={
                    "degraded_useful_min_trusted_visible_words": 7.0,
                    "degraded_useful_min_readability": 0.24,
                    "degraded_useful_min_lexical_trust": 0.20,
                },
            ),
        )
        first_review = review_first_candidate_experiment(first_experiment)
        second = run_second_candidate_threshold_experiment(
            build_expanded_malayalam_benchmark_cases(),
            first_candidate_review=first_review,
        )
        conclusion = conclude_malayalam_calibration_cycle(
            first_candidate_result=first_experiment,
            first_candidate_review=first_review,
            second_candidate_result=second,
        )
        self.assertIn("recommended_fixture_expansion_focus", conclusion)
        self.assertIn("recommended_audio_review_case_ids", conclusion)
        self.assertTrue(
            conclusion["recommended_future_work_type"] in {"fixture_expansion_then_audio_review", "fixture_expansion_and_source_asr_review"}
        )

    def test_runtime_behavior_remains_unchanged(self):
        suite = run_multilingual_benchmark_suite(build_default_multilingual_benchmark_cases())
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)


class MalayalamRealAudioReviewTests(SimpleTestCase):
    def _sample_case(self):
        return RealAudioReviewCase(
            case_id="real_audio_ml_case_01",
            label="Real Audio Malayalam Sample",
            media_path="tests/fixtures/ml_sample.wav",
            expected_language="ml",
            expected_quality_bucket="degraded_low_evidence",
            expected_downstream_decision="suppressed",
            expected_english_view_decision="blocked",
            benchmark_group="real_audio_malayalam",
            tags=["real_audio", "suppressed"],
            notes="Synthetic real-audio review fixture for test coverage.",
        )

    def _mock_runner(self, _case):
        return {
            "status": "completed",
            "transcript_snapshot": {
                "language": "ml",
                "transcript_state": "degraded",
                "display_readable_transcript": "",
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "quality_metrics": {
                    "lexical_trust_score": 0.08,
                    "overall_readability": 0.09,
                    "wrong_script_burden": 0.26,
                    "contamination_burden": 0.63,
                },
            },
            "metadata_snapshot": {
                "detected_language": "ml",
                "transcript_state": "degraded",
                "downstream_suppressed": True,
                "downstream_suppression_reason": "no_trusted_display_units",
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "english_view_available": False,
                "translation_blocked_reason": "degraded_safe_translation_blocked",
            },
            "summary_snapshot": {
                "summary_state": "degraded_safe",
                "_trace": {"structured_summary_route": "degraded_safe"},
            },
            "generated_artifact_paths": ["artifacts/ml_case_01.json"],
        }

    def test_real_audio_review_case_schema_builds_correctly(self):
        case = self._sample_case()
        self.assertEqual(case.case_id, "real_audio_ml_case_01")
        self.assertEqual(case.expected_language, "ml")
        self.assertEqual(case.benchmark_group, "real_audio_malayalam")

    def test_end_to_end_review_runner_returns_structured_result_shape(self):
        result = run_real_audio_review_case(self._sample_case(), pipeline_runner=self._mock_runner)
        self.assertEqual(result["case_id"], "real_audio_ml_case_01")
        self.assertIn("pipeline_result_summary", result)
        self.assertIn("evaluation_result", result)
        self.assertIn("expectation_mismatches", result)

    def test_review_summary_helper_includes_transcript_downstream_english_view_decisions(self):
        result = run_real_audio_review_case(self._sample_case(), pipeline_runner=self._mock_runner)
        summary = summarize_real_audio_review_run(result)
        self.assertIn("transcript_state", summary)
        self.assertIn("downstream_decision", summary)
        self.assertIn("english_view_decision", summary)

    def test_fixture_to_real_comparison_can_flag_divergence(self):
        result = run_real_audio_review_case(self._sample_case(), pipeline_runner=self._mock_runner)
        divergent = compare_real_audio_review_to_fixture(
            result,
            benchmark_case=BenchmarkCase(
                case_id="fixture_ml_case",
                label="Fixture Malayalam Case",
                expected_quality_bucket="degraded_but_useful",
                expected_downstream_decision="allowed",
                expected_english_view_decision="blocked",
            ),
        )
        self.assertEqual(divergent["fixture_alignment_status"], "divergent")
        self.assertTrue(divergent["should_add_or_update_fixture"])

    def test_runtime_behavior_remains_unchanged_when_review_helpers_are_unused(self):
        suite = run_multilingual_benchmark_suite(build_default_multilingual_benchmark_cases())
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)


class MalayalamCuratedRealAudioReviewPackTests(SimpleTestCase):
    def _mock_runner(self, case):
        language = "en" if "english" in case.benchmark_group else "ml"
        if case.case_id.endswith("cleaned_grounded_01"):
            return {
                "status": "completed",
                "transcript_snapshot": {
                    "language": language,
                    "transcript_state": "cleaned",
                    "display_readable_transcript": "grounded readable transcript",
                    "trusted_visible_word_count": 12,
                    "trusted_display_unit_count": 2,
                    "quality_metrics": {
                        "lexical_trust_score": 0.56,
                        "overall_readability": 0.51,
                        "wrong_script_burden": 0.06,
                        "contamination_burden": 0.10,
                    },
                },
                "metadata_snapshot": {
                    "detected_language": language,
                    "transcript_state": "cleaned",
                    "downstream_suppressed": False,
                    "trusted_visible_word_count": 12,
                    "trusted_display_unit_count": 2,
                    "english_view_available": language != "en",
                    "translation_state": "translated" if language != "en" else "same_as_original",
                },
                "summary_snapshot": {"summary_state": "normal_grounded"},
            }
        if "wrong_script" in case.benchmark_group:
            return {
                "status": "completed",
                "transcript_snapshot": {
                    "language": "ml",
                    "transcript_state": "failed",
                    "display_readable_transcript": "",
                    "trusted_visible_word_count": 0,
                    "trusted_display_unit_count": 0,
                    "quality_metrics": {
                        "lexical_trust_score": 0.05,
                        "overall_readability": 0.05,
                        "wrong_script_burden": 0.42,
                        "contamination_burden": 0.71,
                    },
                },
                "metadata_snapshot": {
                    "detected_language": "ml",
                    "transcript_state": "failed",
                    "downstream_suppressed": True,
                    "english_view_available": False,
                },
                "summary_snapshot": {},
            }
        return {
            "status": "completed",
            "transcript_snapshot": {
                "language": language,
                "transcript_state": "degraded",
                "display_readable_transcript": "",
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "quality_metrics": {
                    "lexical_trust_score": 0.08,
                    "overall_readability": 0.09,
                    "wrong_script_burden": 0.24,
                    "contamination_burden": 0.63,
                },
            },
            "metadata_snapshot": {
                "detected_language": language,
                "transcript_state": "degraded",
                "downstream_suppressed": True,
                "downstream_suppression_reason": "no_trusted_display_units",
                "trusted_visible_word_count": 0,
                "trusted_display_unit_count": 0,
                "english_view_available": False,
                "translation_blocked_reason": "degraded_safe_translation_blocked",
            },
            "summary_snapshot": {
                "summary_state": "degraded_safe",
                "_trace": {"structured_summary_route": "degraded_safe"},
            },
        }

    def test_curated_real_audio_review_registry_builds_successfully(self):
        ml_cases = build_malayalam_real_audio_review_cases()
        default_cases = build_default_real_audio_review_cases()
        self.assertGreaterEqual(len(ml_cases), 5)
        self.assertGreaterEqual(len(default_cases), 6)
        self.assertEqual(len({case.case_id for case in default_cases}), len(default_cases))

    def test_real_audio_suite_export_returns_structured_stable_payload(self):
        suite = run_real_audio_review_suite(build_default_real_audio_review_cases(), pipeline_runner=self._mock_runner)
        exported = export_real_audio_review_suite_report(suite)
        self.assertEqual(exported["report_version"], "ml_real_audio_review_report_v1")
        self.assertIn("case_reports", exported)
        self.assertIn("summary_by_transcript_state", exported)

    def test_export_groups_cases_by_transcript_downstream_and_english_view_decisions(self):
        suite = run_real_audio_review_suite(build_default_real_audio_review_cases(), pipeline_runner=self._mock_runner)
        exported = export_real_audio_review_suite_report(suite)
        self.assertIn("cleaned", exported["summary_by_transcript_state"])
        self.assertIn("suppressed", exported["summary_by_downstream_decision"])
        self.assertTrue(exported["summary_by_english_view_decision"])

    def test_real_audio_review_recommendation_helper_returns_actionable_guidance(self):
        suite = run_real_audio_review_suite(build_default_real_audio_review_cases(), pipeline_runner=self._mock_runner)
        exported = export_real_audio_review_suite_report(suite)
        review = review_real_audio_suite_report(exported)
        self.assertIn("recommended_next_action", review)
        self.assertIn("recommended_non_threshold_work", review)
        self.assertIn("recommended_audio_review_case_ids", review)

    def test_fixture_promotion_suggestions_are_included_for_divergent_real_cases(self):
        suite = run_real_audio_review_suite(build_default_real_audio_review_cases(), pipeline_runner=self._mock_runner)
        exported = export_real_audio_review_suite_report(suite)
        self.assertTrue(exported["changed_fixture_recommendations"])
        self.assertIn("recommended_fixture_case_id", exported["changed_fixture_recommendations"][0])

    def test_runtime_behavior_remains_unchanged_when_review_helpers_are_unused(self):
        suite = run_multilingual_benchmark_suite(build_default_multilingual_benchmark_cases())
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)


class MalayalamAsrStrategyComparisonTests(SimpleTestCase):
    def _case(self, *, case_id="real_audio_ml_strategy_case_01", group="cleaned_malayalam", expected_downstream="allowed", expected_english="blocked"):
        return RealAudioReviewCase(
            case_id=case_id,
            label="Malayalam Strategy Review Case",
            media_path="review_samples/ml_strategy_case.wav",
            expected_language="ml",
            expected_quality_bucket="degraded_but_useful",
            expected_downstream_decision=expected_downstream,
            expected_english_view_decision=expected_english,
            benchmark_group=group,
            tags=["real_audio"] + (["protected"] if expected_downstream == "suppressed" else []),
        )

    def _strategy_runner(self, case, strategy=None):
        selected_strategy = strategy or getattr(settings, "ASR_MALAYALAM_STRATEGY", "current_default")
        base = {
            "status": "completed",
            "transcript_snapshot": {
                "language": "ml",
                "transcript_state": "degraded",
                "display_readable_transcript": "",
                "trusted_visible_word_count": 2,
                "trusted_display_unit_count": 1,
                "quality_metrics": {
                    "lexical_trust_score": 0.20,
                    "overall_readability": 0.21,
                    "wrong_script_burden": 0.12,
                    "contamination_burden": 0.19,
                },
            },
            "metadata_snapshot": {
                "detected_language": "ml",
                "transcript_state": "degraded",
                "downstream_suppressed": False,
                "trusted_visible_word_count": 2,
                "trusted_display_unit_count": 1,
                "english_view_available": False,
                "malayalam_asr_strategy": selected_strategy,
                "primary_model_used": "primary-model",
                "fallback_model_used": "",
                "retry_model_used": "retry-model" if selected_strategy == "hybrid_retry" else "",
                "second_pass_asr_attempted": selected_strategy == "hybrid_retry",
                "second_pass_asr_improved": selected_strategy == "hybrid_retry",
                "second_pass_asr_reason": "recoverable_but_weak" if selected_strategy == "hybrid_retry" else "",
                "second_pass_asr_blocked_reason": "",
            },
            "summary_snapshot": {
                "summary_state": "degraded_safe",
                "_trace": {"structured_summary_route": "degraded_safe"},
            },
        }
        if "protected" in (case.tags or []):
            if selected_strategy == "quality_first":
                base["transcript_snapshot"]["transcript_state"] = "cleaned"
                base["transcript_snapshot"]["trusted_visible_word_count"] = 10
                base["transcript_snapshot"]["trusted_display_unit_count"] = 2
                base["transcript_snapshot"]["display_readable_transcript"] = "unsafe readable transcript"
                base["transcript_snapshot"]["quality_metrics"]["lexical_trust_score"] = 0.58
                base["transcript_snapshot"]["quality_metrics"]["overall_readability"] = 0.53
                base["metadata_snapshot"]["downstream_suppressed"] = False
                base["metadata_snapshot"]["english_view_available"] = True
                base["summary_snapshot"]["summary_state"] = "normal_grounded"
            else:
                base["metadata_snapshot"]["downstream_suppressed"] = True
                base["metadata_snapshot"]["trusted_visible_word_count"] = 0
                base["metadata_snapshot"]["trusted_display_unit_count"] = 0
                base["metadata_snapshot"]["english_view_available"] = False
                base["transcript_snapshot"]["trusted_visible_word_count"] = 0
                base["transcript_snapshot"]["trusted_display_unit_count"] = 0
            return base

        if selected_strategy == "fast_first":
            return base
        if selected_strategy == "quality_first":
            base["transcript_snapshot"]["transcript_state"] = "cleaned"
            base["transcript_snapshot"]["display_readable_transcript"] = "better grounded transcript"
            base["transcript_snapshot"]["trusted_visible_word_count"] = 11
            base["transcript_snapshot"]["trusted_display_unit_count"] = 2
            base["transcript_snapshot"]["quality_metrics"]["lexical_trust_score"] = 0.51
            base["transcript_snapshot"]["quality_metrics"]["overall_readability"] = 0.49
            base["summary_snapshot"]["summary_state"] = "normal_grounded"
            return base
        if selected_strategy == "hybrid_retry":
            base["transcript_snapshot"]["transcript_state"] = "cleaned"
            base["transcript_snapshot"]["display_readable_transcript"] = "retry improved transcript"
            base["transcript_snapshot"]["trusted_visible_word_count"] = 12
            base["transcript_snapshot"]["trusted_display_unit_count"] = 2
            base["transcript_snapshot"]["quality_metrics"]["lexical_trust_score"] = 0.54
            base["transcript_snapshot"]["quality_metrics"]["overall_readability"] = 0.50
            base["summary_snapshot"]["summary_state"] = "normal_grounded"
            return base
        return base

    def test_strategy_review_case_helper_returns_structured_per_strategy_results(self):
        result = run_malayalam_asr_strategy_review_case(
            self._case(),
            strategies=["current_default", "hybrid_retry"],
            pipeline_runner=self._strategy_runner,
        )
        self.assertIn("strategy_results", result)
        self.assertIn("hybrid_retry", result["strategy_results"])
        self.assertIn("trusted_visible_word_count", result["strategy_results"]["hybrid_retry"]["summary"])

    def test_suite_comparison_runner_aggregates_improved_regressed_cases_by_strategy(self):
        suite = run_malayalam_asr_strategy_review_suite(
            [self._case(), self._case(case_id="real_audio_ml_strategy_case_02", group="degraded_low_evidence_malayalam", expected_downstream="suppressed")],
            strategies=["current_default", "quality_first", "hybrid_retry"],
            pipeline_runner=self._strategy_runner,
        )
        self.assertIn("hybrid_retry", suite["improved_case_ids_by_strategy"])
        self.assertIn("real_audio_ml_strategy_case_01", suite["improved_case_ids_by_strategy"]["hybrid_retry"])
        self.assertIn("quality_first", suite["regressed_case_ids_by_strategy"])

    def test_summary_helper_can_recommend_keeping_current_default_when_no_strategy_clearly_wins(self):
        comparison = run_malayalam_asr_strategy_review_suite(
            [self._case(case_id="protected_case", group="degraded_low_evidence_malayalam", expected_downstream="suppressed")],
            strategies=["current_default", "quality_first"],
            pipeline_runner=self._strategy_runner,
        )
        summary = summarize_malayalam_asr_strategy_comparison(comparison)
        self.assertEqual(summary["recommended_default_strategy"], "current_default")

    def test_summary_helper_flags_safety_regressions_over_raw_gains(self):
        comparison = run_malayalam_asr_strategy_review_suite(
            [
                self._case(case_id="safe_case_01"),
                self._case(case_id="protected_case_01", group="degraded_low_evidence_malayalam", expected_downstream="suppressed"),
            ],
            strategies=["current_default", "quality_first"],
            pipeline_runner=self._strategy_runner,
        )
        summary = summarize_malayalam_asr_strategy_comparison(comparison)
        self.assertIn("protected_case_01", summary["safety_regression_case_ids_by_strategy"]["quality_first"])
        self.assertEqual(summary["recommended_default_strategy"], "current_default")

    def test_second_pass_helped_and_not_helpful_cases_are_tracked(self):
        def mixed_runner(case, strategy=None):
            result = self._strategy_runner(case, strategy=strategy)
            if case.case_id.endswith("_02") and (strategy or getattr(settings, "ASR_MALAYALAM_STRATEGY", "")) == "hybrid_retry":
                result["metadata_snapshot"]["second_pass_asr_improved"] = False
                result["transcript_snapshot"]["transcript_state"] = "degraded"
            return result

        comparison = run_malayalam_asr_strategy_review_suite(
            [self._case(case_id="real_audio_ml_strategy_case_01"), self._case(case_id="real_audio_ml_strategy_case_02")],
            strategies=["current_default", "hybrid_retry"],
            pipeline_runner=mixed_runner,
        )
        summary = summarize_malayalam_asr_strategy_comparison(comparison)
        self.assertIn("real_audio_ml_strategy_case_01", summary["second_pass_helped_case_ids"]["hybrid_retry"])
        self.assertIn("real_audio_ml_strategy_case_02", summary["second_pass_not_helpful_case_ids"]["hybrid_retry"])

    def test_runtime_behavior_remains_unchanged_when_strategy_comparison_helpers_are_unused(self):
        suite = run_multilingual_benchmark_suite(build_default_multilingual_benchmark_cases())
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)


class MalayalamAsrStrategyDecisionTests(SimpleTestCase):
    def _summary(
        self,
        *,
        candidates=None,
        improved=None,
        regressed=None,
        safety=None,
        second_pass_helped=None,
        second_pass_not_helpful=None,
        gains=None,
        losses=None,
    ):
        candidate_strategies = list(candidates or ["fast_first", "quality_first", "hybrid_retry"])
        return {
            "baseline_strategy": "current_default",
            "candidate_strategies": candidate_strategies,
            "total_cases": 4,
            "strategy_case_counts": {strategy: 4 for strategy in ["current_default"] + candidate_strategies},
            "improved_case_ids_by_strategy": dict(improved or {strategy: [] for strategy in candidate_strategies}),
            "regressed_case_ids_by_strategy": dict(regressed or {strategy: [] for strategy in candidate_strategies}),
            "unchanged_case_ids_by_strategy": {strategy: [] for strategy in candidate_strategies},
            "safety_regression_case_ids_by_strategy": dict(safety or {strategy: [] for strategy in candidate_strategies}),
            "second_pass_helped_case_ids": dict(second_pass_helped or {"current_default": [], **{strategy: [] for strategy in candidate_strategies}}),
            "second_pass_not_helpful_case_ids": dict(second_pass_not_helpful or {"current_default": [], **{strategy: [] for strategy in candidate_strategies}}),
            "dominant_improvement_reasons_by_strategy": {strategy: [] for strategy in candidate_strategies},
            "dominant_regression_reasons_by_strategy": {strategy: [] for strategy in candidate_strategies},
            "strategy_fixture_alignment_gain": dict(gains or {strategy: [] for strategy in candidate_strategies}),
            "strategy_fixture_alignment_loss": dict(losses or {strategy: [] for strategy in candidate_strategies}),
            "fixture_alignment_case_changes": {strategy: [] for strategy in candidate_strategies},
            "recommended_default_strategy": "current_default",
            "recommended_default_strategy_reason": "No candidate strategy clearly outperformed the current default without added risk.",
            "recommended_followup_cases_for_audio_review": [],
            "recommended_followup_cases_for_asr_investigation": [],
        }

    def test_when_no_candidate_clearly_wins_final_decision_keeps_current_default(self):
        decision = conclude_malayalam_asr_strategy_default(self._summary())
        self.assertEqual(decision["final_strategy_recommendation"], "keep_current_default")
        self.assertTrue(decision["keep_current_default"])

    def test_safety_regressing_candidate_strategies_are_rejected(self):
        decision = conclude_malayalam_asr_strategy_default(
            self._summary(
                safety={"fast_first": [], "quality_first": ["ml_case_02"], "hybrid_retry": []},
                regressed={"fast_first": [], "quality_first": ["ml_case_02"], "hybrid_retry": []},
            )
        )
        self.assertIn("quality_first", decision["rejected_candidate_strategies"])
        self.assertTrue(decision["reject_candidate_strategies"])

    def test_safe_improving_candidate_can_produce_manual_adoption_review_only(self):
        decision = conclude_malayalam_asr_strategy_default(
            self._summary(
                candidates=["hybrid_retry"],
                improved={"hybrid_retry": ["ml_case_01", "ml_case_02"]},
                regressed={"hybrid_retry": []},
                safety={"hybrid_retry": []},
                second_pass_helped={"current_default": [], "hybrid_retry": ["ml_case_01"]},
                gains={"hybrid_retry": ["ml_case_02"]},
            )
        )
        self.assertEqual(decision["final_strategy_recommendation"], "manual_adoption_review_only")
        self.assertEqual(decision["winning_strategy"], "hybrid_retry")

    def test_ranked_strategies_are_returned_in_stable_order(self):
        decision = conclude_malayalam_asr_strategy_default(
            self._summary(
                improved={"fast_first": [], "quality_first": ["ml_case_01"], "hybrid_retry": ["ml_case_01", "ml_case_02"]},
                regressed={"fast_first": [], "quality_first": [], "hybrid_retry": []},
                safety={"fast_first": [], "quality_first": [], "hybrid_retry": []},
            )
        )
        self.assertEqual(decision["ranked_strategies"][0], "current_default")
        self.assertEqual(decision["ranked_strategies"][1], "hybrid_retry")

    def test_follow_up_guidance_includes_case_ids_for_audio_review_and_investigation(self):
        decision = conclude_malayalam_asr_strategy_default(
            self._summary(
                regressed={"fast_first": ["ml_case_03"], "quality_first": [], "hybrid_retry": []},
                safety={"fast_first": ["ml_case_03"], "quality_first": [], "hybrid_retry": []},
                second_pass_not_helpful={"current_default": [], "fast_first": [], "quality_first": [], "hybrid_retry": ["ml_case_04"]},
            )
        )
        self.assertIn("ml_case_03", decision["recommended_audio_review_case_ids"])
        self.assertIn("ml_case_04", decision["recommended_asr_investigation_case_ids"])

    def test_export_helper_returns_structured_json_safe_output(self):
        exported = export_malayalam_asr_strategy_decision(self._summary())
        self.assertEqual(exported["export_version"], "ml_asr_strategy_decision_v1")
        self.assertIn("recommendation", exported)
        self.assertIn("ranking", exported)
        self.assertEqual(json.loads(json.dumps(exported))["recommendation"]["final_strategy_recommendation"], "keep_current_default")

    def test_runtime_behavior_remains_unchanged_when_decision_helpers_are_unused(self):
        suite = run_multilingual_benchmark_suite(build_default_multilingual_benchmark_cases())
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)


class MalayalamAsrStrategyDecisionFlowTests(SimpleTestCase):
    def _strategy_runner(self, case, strategy=None):
        selected_strategy = strategy or getattr(settings, "ASR_MALAYALAM_STRATEGY", "current_default")
        base = {
            "status": "completed",
            "transcript_snapshot": {
                "language": "ml",
                "transcript_state": "degraded",
                "display_readable_transcript": "",
                "trusted_visible_word_count": 1,
                "trusted_display_unit_count": 1,
                "quality_metrics": {
                    "lexical_trust_score": 0.21,
                    "overall_readability": 0.20,
                    "wrong_script_burden": 0.10,
                    "contamination_burden": 0.18,
                },
            },
            "metadata_snapshot": {
                "detected_language": "ml",
                "transcript_state": "degraded",
                "downstream_suppressed": "low_evidence" in case.case_id or "wrong_script" in case.case_id,
                "trusted_visible_word_count": 1,
                "trusted_display_unit_count": 1,
                "english_view_available": False,
                "malayalam_asr_strategy": selected_strategy,
                "second_pass_asr_attempted": selected_strategy == "hybrid_retry",
                "second_pass_asr_improved": selected_strategy == "hybrid_retry" and "cleaned_grounded" in case.case_id,
                "second_pass_asr_reason": "recoverable_but_weak" if selected_strategy == "hybrid_retry" else "",
            },
            "summary_snapshot": {
                "summary_state": "degraded_safe",
                "_trace": {"structured_summary_route": "degraded_safe"},
            },
        }
        if selected_strategy in {"quality_first", "hybrid_retry"} and "cleaned_grounded" in case.case_id:
            base["transcript_snapshot"]["transcript_state"] = "cleaned"
            base["transcript_snapshot"]["display_readable_transcript"] = "grounded transcript"
            base["transcript_snapshot"]["trusted_visible_word_count"] = 12
            base["transcript_snapshot"]["trusted_display_unit_count"] = 2
            base["transcript_snapshot"]["quality_metrics"]["lexical_trust_score"] = 0.55
            base["transcript_snapshot"]["quality_metrics"]["overall_readability"] = 0.50
            base["metadata_snapshot"]["downstream_suppressed"] = False
            base["summary_snapshot"]["summary_state"] = "normal_grounded"
        if selected_strategy == "quality_first" and "wrong_script" in case.case_id:
            base["transcript_snapshot"]["transcript_state"] = "cleaned"
            base["transcript_snapshot"]["display_readable_transcript"] = "unsafe transcript"
            base["metadata_snapshot"]["downstream_suppressed"] = False
            base["metadata_snapshot"]["english_view_available"] = True
            base["summary_snapshot"]["summary_state"] = "normal_grounded"
        return base

    def test_orchestration_helper_returns_all_expected_sections(self):
        result = run_default_malayalam_asr_strategy_decision_flow(pipeline_runner=self._strategy_runner)
        self.assertIn("suite_result", result)
        self.assertIn("comparison_summary", result)
        self.assertIn("final_decision", result)
        self.assertIn("exported_decision", result)

    def test_final_exported_decision_is_json_safe_and_structured(self):
        result = run_default_malayalam_asr_strategy_decision_flow(pipeline_runner=self._strategy_runner)
        exported = result["exported_decision"]
        self.assertIn("recommendation", exported)
        self.assertIn("rationale", exported)
        self.assertEqual(json.loads(json.dumps(exported))["recommendation"]["recommended_production_default"], exported["recommendation"]["recommended_production_default"])

    def test_runtime_behavior_remains_unchanged(self):
        suite = run_multilingual_benchmark_suite(build_default_multilingual_benchmark_cases())
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)


class MalayalamRealAudioReviewHistoryTests(SimpleTestCase):
    def _report(self, *, case_reports, changed_fixture_recommendations=None):
        return {
            "report_version": "ml_real_audio_review_report_v1",
            "total_cases": len(case_reports),
            "successful_runs": len(case_reports),
            "blocked_runs": 0,
            "failed_runs": 0,
            "cases_with_expectation_mismatches": [],
            "cases_aligned_with_fixture": [],
            "cases_diverging_from_fixture": [],
            "summary_by_transcript_state": {},
            "summary_by_downstream_decision": {},
            "summary_by_english_view_decision": {},
            "summary_by_benchmark_group": {},
            "changed_fixture_recommendations": list(changed_fixture_recommendations or []),
            "case_reports": case_reports,
        }

    def test_review_history_entries_build_from_suite_export_correctly(self):
        report = self._report(
            case_reports=[
                {
                    "case_id": "real_audio_ml_case_01",
                    "label": "Malayalam Case",
                    "benchmark_group": "cleaned_malayalam",
                    "input_reference": "video:1",
                    "transcript_state": "cleaned",
                    "rescue_ran": False,
                    "downstream_decision": "allowed",
                    "summary_state": "normal_grounded",
                    "english_view_decision": "available",
                    "dominant_safety_reason": "allowed",
                    "dominant_failure_reason": "",
                    "fixture_alignment_status": "aligned",
                    "mismatch_summary": [],
                }
            ]
        )
        entries = build_real_audio_review_history_entries(
            report,
            review_run_id="run_001",
            created_at="2026-03-18T10:00:00Z",
            suite_name="ml_real_audio",
        )
        self.assertEqual(entries[0]["review_run_id"], "run_001")
        self.assertEqual(entries[0]["case_id"], "real_audio_ml_case_01")
        self.assertEqual(entries[0]["suite_name"], "ml_real_audio")

    def test_history_snapshot_export_is_deterministic_and_json_safe(self):
        report = self._report(
            case_reports=[
                {"case_id": "b_case", "label": "B", "benchmark_group": "g2"},
                {"case_id": "a_case", "label": "A", "benchmark_group": "g1"},
            ]
        )
        snapshot = export_real_audio_review_history_snapshot(
            report,
            review_run_id="run_002",
            created_at="2026-03-18T10:05:00Z",
        )
        self.assertEqual(snapshot["entries"][0]["case_id"], "a_case")
        self.assertEqual(json.loads(json.dumps(snapshot))["review_run_id"], "run_002")

    def test_run_to_run_comparison_detects_improved_cases(self):
        previous = export_real_audio_review_history_snapshot(
            self._report(
                case_reports=[
                    {
                        "case_id": "real_audio_ml_case_01",
                        "label": "Case",
                        "transcript_state": "degraded",
                        "downstream_decision": "suppressed",
                        "english_view_decision": "blocked",
                        "fixture_alignment_status": "aligned",
                        "mismatch_summary": [],
                    }
                ]
            ),
            review_run_id="run_old",
            created_at="2026-03-18T10:00:00Z",
        )
        current = export_real_audio_review_history_snapshot(
            self._report(
                case_reports=[
                    {
                        "case_id": "real_audio_ml_case_01",
                        "label": "Case",
                        "transcript_state": "cleaned",
                        "downstream_decision": "allowed",
                        "english_view_decision": "available",
                        "fixture_alignment_status": "aligned",
                        "mismatch_summary": [],
                    }
                ]
            ),
            review_run_id="run_new",
            created_at="2026-03-18T10:10:00Z",
        )
        comparison = compare_real_audio_review_history(previous, current)
        self.assertIn("real_audio_ml_case_01", comparison["improved_case_ids"])

    def test_run_to_run_comparison_detects_regressed_cases(self):
        previous = export_real_audio_review_history_snapshot(
            self._report(
                case_reports=[
                    {
                        "case_id": "real_audio_ml_case_01",
                        "label": "Case",
                        "transcript_state": "cleaned",
                        "downstream_decision": "allowed",
                        "english_view_decision": "available",
                        "fixture_alignment_status": "aligned",
                        "mismatch_summary": [],
                    }
                ]
            ),
            review_run_id="run_old",
            created_at="2026-03-18T10:00:00Z",
        )
        current = export_real_audio_review_history_snapshot(
            self._report(
                case_reports=[
                    {
                        "case_id": "real_audio_ml_case_01",
                        "label": "Case",
                        "transcript_state": "failed",
                        "downstream_decision": "suppressed",
                        "english_view_decision": "blocked",
                        "fixture_alignment_status": "divergent",
                        "mismatch_summary": ["expected_quality_bucket"],
                    }
                ],
                changed_fixture_recommendations=[
                    {
                        "case_id": "real_audio_ml_case_01",
                        "recommended_fixture_case_id": "real_audio_ml_case_01",
                        "promotion_priority": "high",
                    }
                ],
            ),
            review_run_id="run_new",
            created_at="2026-03-18T10:10:00Z",
        )
        comparison = compare_real_audio_review_history(previous, current)
        self.assertTrue(
            "real_audio_ml_case_01" in comparison["regressed_case_ids"]
            or "real_audio_ml_case_01" in comparison["review_needed_case_ids"]
        )

    def test_run_to_run_comparison_detects_new_and_removed_cases(self):
        previous = export_real_audio_review_history_snapshot(
            self._report(case_reports=[{"case_id": "old_case", "label": "Old"}]),
            review_run_id="run_old",
            created_at="2026-03-18T10:00:00Z",
        )
        current = export_real_audio_review_history_snapshot(
            self._report(case_reports=[{"case_id": "new_case", "label": "New"}]),
            review_run_id="run_new",
            created_at="2026-03-18T10:10:00Z",
        )
        comparison = compare_real_audio_review_history(previous, current)
        self.assertIn("new_case", comparison["new_case_ids"])
        self.assertIn("old_case", comparison["removed_case_ids"])

    def test_history_comparison_summary_returns_actionable_guidance(self):
        comparison = {
            "improved_case_ids": [],
            "regressed_case_ids": ["real_audio_ml_case_01"],
            "unchanged_case_ids": [],
            "review_needed_case_ids": ["real_audio_ml_case_02"],
            "new_case_ids": [],
            "removed_case_ids": [],
            "change_summaries": [
                {
                    "case_id": "real_audio_ml_case_01",
                    "change_reasons": ["dominant_failure_reason_changed"],
                    "new_transcript_state": "failed",
                },
                {
                    "case_id": "real_audio_ml_case_02",
                    "change_reasons": ["fixture_alignment_changed", "new_mismatch_appeared"],
                    "new_transcript_state": "degraded",
                },
            ],
        }
        summary = summarize_real_audio_review_history_comparison(comparison)
        self.assertIn("recommended_next_action", summary)
        self.assertIn("real_audio_ml_case_01", summary["recommended_case_ids_for_manual_audio_review"])
        self.assertIn("real_audio_ml_case_02", summary["recommended_case_ids_for_fixture_update"])

    def test_runtime_behavior_remains_unchanged_when_history_helpers_are_unused(self):
        suite = run_multilingual_benchmark_suite(build_default_multilingual_benchmark_cases())
        self.assertEqual(suite["summary"]["bucket_counts"]["clearly_cleaned"], 1)





