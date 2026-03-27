import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from django.conf import settings
from django.test import TestCase, override_settings
from rest_framework.test import APIRequestFactory

from chatbot.models import ChatMessage, ChatSession, VideoIndex
from chatbot.rag_engine import ChatbotEngine, VideoRAGEngine, _EMBEDDING_MODEL_CACHE, prewarm_embedding_model
from chatbot.serializers import ChatMessageSerializer
from chatbot.views import ChatbotView, _build_voice_narration
from videos.models import Summary, Transcript, Video

class MultilingualChatFlowTests(TestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.video = Video.objects.create(title="Demo", status="completed")
        Transcript.objects.create(
            video=self.video,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            full_text="à´‡à´¤àµ à´’à´°àµ à´ªàµà´°à´­à´¾à´·à´£à´®à´¾à´£àµ.",
            transcript_original_text="à´‡à´¤àµ à´’à´°àµ à´ªàµà´°à´­à´¾à´·à´£à´®à´¾à´£àµ.",
            transcript_canonical_text="This is a speech.",
            json_data={
                "segments": [{"id": 0, "start": 0, "end": 5, "text": "à´‡à´¤àµ à´’à´°àµ à´ªàµà´°à´­à´¾à´·à´£à´®à´¾à´£àµ."}],
                "canonical_segments": [
                    {"id": 0, "start": 0, "end": 5, "text": "This is a speech.", "original_text": "à´‡à´¤àµ à´’à´°àµ à´ªàµà´°à´­à´¾à´·à´£à´®à´¾à´£àµ."}
                ],
            },
            word_timestamps=[],
        )

    @patch("chatbot.views.translate_text")
    @patch("chatbot.views.detect_text_language")
    @patch("chatbot.views.ChatbotEngine")
    def test_hindi_question_retrieval_en_answer_hi(self, mock_engine_cls, mock_detect_lang, mock_translate):
        mock_detect_lang.return_value = ("hi", 0.98, "devanagari", "script")
        mock_translate.side_effect = [
            "What is this video about?",  # user question hi -> en
            "à¤¯à¤¹ à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤à¤• à¤ªà¥à¤°à¥‡à¤°à¤£à¤¾à¤¦à¤¾à¤¯à¤• à¤­à¤¾à¤·à¤£ à¤•à¥‡ à¤¬à¤¾à¤°à¥‡ à¤®à¥‡à¤‚ à¤¹à¥ˆà¥¤",  # answer en -> hi
            "à¤¯à¤¹ à¤à¤• à¤­à¤¾à¤·à¤£ à¤¹à¥ˆà¥¤",  # citation en -> hi
        ]

        engine = mock_engine_cls.return_value
        engine.initialize.return_value = True
        engine.ask.return_value = {
            "answer": "This video is about a motivational speech.",
            "sources": [{"text": "This is a speech.", "timestamp": "0.0s - 5.0s", "relevance": 0.9}],
        }

        request = self.factory.post(
            "/api/v1/chatbot/chat/",
            {
                "video_id": str(self.video.id),
                "message": "à¤¯à¤¹ à¤µà¥€à¤¡à¤¿à¤¯à¥‹ à¤•à¤¿à¤¸ à¤¬à¤¾à¤°à¥‡ à¤®à¥‡à¤‚ à¤¹à¥ˆ?",
                "output_language": "hi",
            },
            format="json",
        )
        response = ChatbotView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["output_language"], "hi")
        self.assertEqual(response.data["retrieval_language"], "en")
        self.assertIn("à¤µà¥€à¤¡à¤¿à¤¯à¥‹", response.data["answer"])
        self.assertTrue(response.data.get("sources"))
        self.assertIn("timestamp", response.data["sources"][0])
        self.assertIn("processing_metadata", response.data)
        self.assertTrue(response.data["english_view_available"])
        self.assertIn("This video is about a motivational speech.", response.data["english_view_answer"])
        self.assertEqual(response.data["chatbot_answer_language"], "hi")
        meta = response.data["processing_metadata"]
        self.assertIsInstance(meta.get("asr_engine"), str)
        self.assertIsInstance(meta.get("language"), str)
        self.assertIsInstance(meta.get("processing_time_seconds"), float)
        self.assertIsInstance(meta.get("transcript_quality_score"), float)

        bot_msg = ChatMessage.objects.filter(sender="bot").latest("created_at")
        self.assertEqual(bot_msg.output_language, "hi")
        self.assertEqual(bot_msg.retrieval_language, "en")

    @patch("chatbot.views.translate_text", side_effect=lambda text, **kwargs: text)
    @patch("chatbot.views.detect_text_language", return_value=("en", 0.99, "latin", "script"))
    @patch("chatbot.views.ChatbotEngine")
    def test_low_trust_degraded_malayalam_blocks_polished_english_view(self, mock_engine_cls, _mock_detect_lang, _mock_translate):
        transcript = Transcript.objects.get(video=self.video)
        transcript.json_data = {
            "transcript_state": "degraded",
            "low_evidence_malayalam": True,
            "transcript_warning_message": "Malayalam transcript quality was too low for reliable summarization.",
            "segments": [{"id": 0, "start": 0, "end": 5, "text": "garbled"}],
        }
        transcript.save(update_fields=["json_data"])

        engine = mock_engine_cls.return_value
        engine.initialize.return_value = True
        engine.ask.return_value = {
            "answer": "This is not clearly stated in the transcript.",
            "sources": [],
        }

        request = self.factory.post(
            "/api/v1/chatbot/chat/",
            {
                "video_id": str(self.video.id),
                "message": "What does he say?",
                "output_language": "ml",
            },
            format="json",
        )
        response = ChatbotView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["english_view_available"])
        self.assertEqual(response.data["chatbot_translation_blocked_reason"], "degraded_safe_translation_blocked")

    @patch("chatbot.views.translate_text", side_effect=lambda text, **kwargs: text)
    @patch("chatbot.views.detect_text_language", return_value=("en", 0.99, "latin", "script"))
    @patch("chatbot.views.ChatbotEngine")
    def test_fidelity_failed_malayalam_blocks_chat_before_index_build(self, mock_engine_cls, _mock_detect_lang, _mock_translate):
        transcript = Transcript.objects.get(video=self.video)
        transcript.json_data = {
            "transcript_state": "degraded",
            "source_language_fidelity_failed": True,
            "transcript_fidelity_state": "source_language_fidelity_failed",
            "final_malayalam_fidelity_decision": "source_language_fidelity_failed",
            "transcript_warning_message": "Malayalam speech could not be transcribed faithfully enough for safe display.",
            "segments": [
                {"id": 0, "start": 0, "end": 5, "text": "When you leave the exam hall you will feel satisfied."}
            ],
            "canonical_segments": [
                {"id": 0, "start": 0, "end": 5, "text": "When you leave the exam hall you will feel satisfied."}
            ],
        }
        transcript.save(update_fields=["json_data"])

        request = self.factory.post(
            "/api/v1/chatbot/chat/",
            {
                "video_id": str(self.video.id),
                "message": "What does he say?",
                "output_language": "ml",
            },
            format="json",
        )
        response = ChatbotView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["chatbot_blocked_reason"], "malayalam_source_fidelity_failed")
        self.assertIn("faithful", response.data["error"].lower())
        self.assertFalse(response.data["english_view_available"])
        self.assertEqual(response.data["chatbot_translation_blocked_reason"], "source_language_fidelity_failed")
        mock_engine_cls.assert_not_called()

    def test_chatbot_engine_refuses_index_build_for_fidelity_failed_malayalam(self):
        transcript = Transcript.objects.get(video=self.video)
        transcript.json_data = {
            "transcript_state": "degraded",
            "source_language_fidelity_failed": True,
            "final_malayalam_fidelity_decision": "source_language_fidelity_failed",
            "segments": [{"id": 0, "start": 0, "end": 5, "text": "English leakage"}],
        }
        transcript.save(update_fields=["json_data"])

        engine = ChatbotEngine(str(self.video.id))
        initialized = engine.initialize()
        built = engine.build_from_transcript([{"id": 0, "start": 0, "end": 5, "text": "English leakage"}])

        self.assertFalse(initialized)
        self.assertFalse(built)
        self.assertEqual(engine.index_blocked_reason, "malayalam_source_fidelity_failed")

    def test_chat_message_english_view_marks_stale_when_answer_source_changes(self):
        session = ChatSession.objects.create(video_id=self.video.id)
        bot_msg = ChatMessage.objects.create(
            session=session,
            sender="bot",
            message="Final answer in Malayalam",
            user_language="en",
            output_language="ml",
            retrieval_language="en",
            referenced_segments={
                "sources": [{"text": "updated source"}],
                "_english_view_cache": {
                    "english_view_source_hash": "stale-hash",
                    "payload": {
                        "english_view_text": "Old English answer",
                        "english_view_available": True,
                        "translation_state": "translated",
                        "translation_warning": "",
                        "translation_blocked_reason": "",
                    },
                },
            },
        )
        serialized = ChatMessageSerializer(bot_msg).data
        self.assertFalse(serialized["english_view_available"])
        self.assertEqual(serialized["chatbot_translation_state"], "stale")
        self.assertEqual(serialized["chatbot_translation_blocked_reason"], "stale_cached_translation")

    @override_settings(EMBEDDING_MODEL="BAAI/bge-m3")
    def test_hindi_retrieval_uses_bge_m3_embedding_model(self):
        engine = VideoRAGEngine(str(self.video.id))
        self.assertEqual(engine.embedding_model, "BAAI/bge-m3")

    def test_retrieval_gate_accepts_score_above_new_threshold(self):
        engine = VideoRAGEngine(str(self.video.id))
        self.assertTrue(engine._passes_retrieval_gate([{"rank_score": 0.3514}, {"rank_score": 0.29}]))  # pylint: disable=protected-access

    @override_settings(RAG_RETRIEVAL_MIN_USEFUL_RESULTS=1, RAG_MIN_LEXICAL_OVERLAP=0.08, RAG_MIN_SOURCE_QUALITY=0.2)
    def test_retrieval_gate_rejects_thin_low_overlap_evidence(self):
        engine = VideoRAGEngine(str(self.video.id))
        self.assertFalse(engine._passes_retrieval_gate([  # pylint: disable=protected-access
            {"rank_score": 0.31, "lexical_overlap": 0.0, "source_quality": 0.05},
            {"rank_score": 0.27, "lexical_overlap": 0.01, "source_quality": 0.1},
        ]))

    def test_embedding_prewarm_reuses_process_cache(self):
        _EMBEDDING_MODEL_CACHE.clear()
        mock_sentence_transformer = Mock(return_value=object())

        with patch("chatbot.rag_engine._load_sentence_transformer_classes", return_value=(mock_sentence_transformer, Mock())):
            first = prewarm_embedding_model("BAAI/bge-m3")
            second = prewarm_embedding_model("BAAI/bge-m3")

        self.assertIs(first, second)
        mock_sentence_transformer.assert_called_once_with("BAAI/bge-m3")

    @override_settings(
        EMBEDDING_MODEL="BAAI/bge-m3",
        EMBEDDING_MODEL_FALLBACKS="sentence-transformers/all-MiniLM-L6-v2",
    )
    def test_embedding_loader_falls_back_when_preferred_model_fails(self):
        _EMBEDDING_MODEL_CACHE.clear()
        fallback_model = object()

        def fake_sentence_transformer(model_name):
            if model_name == "BAAI/bge-m3":
                raise RuntimeError("torch.load weights_only failure")
            if model_name == "sentence-transformers/all-MiniLM-L6-v2":
                return fallback_model
            raise AssertionError(f"unexpected model {model_name}")

        with patch("chatbot.rag_engine._load_sentence_transformer_classes", return_value=(fake_sentence_transformer, Mock())):
            engine = VideoRAGEngine(str(self.video.id))
            model = engine._load_embedding_model()  # pylint: disable=protected-access

        self.assertIs(model, fallback_model)
        self.assertEqual(engine.embedding_model_requested, "BAAI/bge-m3")
        self.assertEqual(engine.embedding_model_used, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertTrue(engine.embedding_model_fallback_used)
        self.assertIn("torch runtime", engine.embedding_runtime_error)

    @patch("chatbot.views.translate_text")
    @patch("chatbot.views.detect_text_language")
    @patch("chatbot.views.ChatbotEngine")
    def test_chat_can_be_scoped_to_specific_moment(self, mock_engine_cls, mock_detect_lang, mock_translate):
        mock_detect_lang.return_value = ("en", 0.99, "latin", "script")
        mock_translate.side_effect = lambda text, **kwargs: text

        transcript = Transcript.objects.get(video=self.video)
        transcript.json_data = {
            "segments": [
                {"id": 0, "start": 0, "end": 5, "text": "Intro segment."},
                {"id": 1, "start": 42, "end": 48, "text": "Fear of failure is discussed here."},
                {"id": 2, "start": 90, "end": 96, "text": "Closing remarks."},
            ],
            "canonical_segments": [
                {"id": 0, "start": 0, "end": 5, "text": "Intro segment.", "original_text": "Intro segment."},
                {"id": 1, "start": 42, "end": 48, "text": "Fear of failure is discussed here.", "original_text": "Fear of failure is discussed here."},
                {"id": 2, "start": 90, "end": 96, "text": "Closing remarks.", "original_text": "Closing remarks."},
            ],
        }
        transcript.save(update_fields=["json_data"])

        engine = mock_engine_cls.return_value
        engine.initialize.return_value = True
        engine.ask.return_value = {
            "answer": "The speaker is describing fear of failure.",
            "sources": [{"text": "Fear of failure is discussed here.", "timestamp": "42.0s - 48.0s", "relevance": 1.0}],
            "timestamp_context": "00:25 — 01:05",
        }

        request = self.factory.post(
            "/api/v1/chatbot/chat/",
            {
                "video_id": str(self.video.id),
                "message": "What does he mean here?",
                "context_timestamp": 45,
                "context_window_seconds": 20,
            },
            format="json",
        )
        response = ChatbotView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["timestamp_context"], "00:25 — 01:05")

        ask_kwargs = mock_engine_cls.return_value.ask.call_args.kwargs
        self.assertEqual(ask_kwargs["context_timestamp"], 45)
        self.assertEqual(ask_kwargs["context_window_seconds"], 20)
        self.assertEqual(len(ask_kwargs["moment_segments"]), 1)
        self.assertEqual(ask_kwargs["moment_segments"][0]["start"], 42)

    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    @patch.object(VideoRAGEngine, "get_relevant_context")
    def test_moment_mode_stays_local_for_tutorial_timestamp(self, mock_context, _mock_summary_support):
        mock_context.return_value = (
            "[180.0s - 210.0s] The broader video later discusses deployment and hosting.",
            [
                {"text": "The broader video later discusses deployment and hosting.", "source_text": "The broader video later discusses deployment and hosting.", "start": 180, "end": 210, "score": 0.91},
            ],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask(
            "What is happening at this moment?",
            use_llm=False,
            moment_segments=[
                {"text": "Here the tutorial shows how to build the hero section in Lovable and tighten the CTA copy.", "source_text": "Here the tutorial shows how to build the hero section in Lovable and tighten the CTA copy.", "start": 62, "end": 78, "score": 0.98},
                {"text": "It then adjusts the landing page layout and testimonial section.", "source_text": "It then adjusts the landing page layout and testimonial section.", "start": 78, "end": 92, "score": 0.95},
            ],
            context_timestamp=70,
        )
        lowered = response["answer"].lower()
        self.assertIn("around 01:02", lowered)
        self.assertIn("hero section", lowered)
        self.assertNotIn("deployment", lowered)
        self.assertTrue(response["sources"])
        self.assertTrue(response["timestamp_context"].startswith("00:50"))

    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    @patch.object(VideoRAGEngine, "get_relevant_context", return_value=("", []))
    def test_moment_mode_cleans_ui_heavy_local_transcript(self, _mock_context, _mock_summary_support):
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask(
            "Explain this moment",
            use_llm=False,
            moment_segments=[
                {
                    "text": "Ask AI about this moment. Here they are building the CTA and hero section in Lovable for the landing page.",
                    "source_text": "Ask AI about this moment. Here they are building the CTA and hero section in Lovable for the landing page.",
                    "start": 120,
                    "end": 136,
                    "score": 0.96,
                }
            ],
            context_timestamp=124,
        )
        lowered = response["answer"].lower()
        self.assertIn("cta", lowered)
        self.assertIn("hero section", lowered)
        self.assertNotIn("ask ai about this moment", lowered)
        self.assertNotIn("screen recording", lowered)

    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    @patch.object(VideoRAGEngine, "get_relevant_context", return_value=("", []))
    def test_weak_noisy_moment_context_abstains_cleanly(self, _mock_context, _mock_summary_support):
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask(
            "What is happening at this moment?",
            use_llm=False,
            moment_segments=[
                {"text": "uh yeah", "source_text": "uh yeah", "start": 40, "end": 42, "score": 0.4},
                {"text": "screen recording", "source_text": "screen recording", "start": 42, "end": 43, "score": 0.35},
            ],
            context_timestamp=41,
        )
        self.assertIn("not clearly explained in the transcript", response["answer"].lower())

    def test_general_about_question_is_not_classified_as_quote(self):
        engine = ChatbotEngine(str(self.video.id))
        intent, top_k = engine._detect_intent_and_top_k("What is this video about?")  # pylint: disable=protected-access
        self.assertEqual(intent, "global_summary")
        self.assertGreaterEqual(top_k, 6)

    def test_build_index_marks_degraded_when_embedding_load_is_blocked(self):
        temp_base = Path(settings.BASE_DIR) / "tmp_test_rag_status"
        shutil.rmtree(temp_base, ignore_errors=True)
        temp_base.mkdir(parents=True, exist_ok=True)
        try:
            with override_settings(BASE_DIR=str(temp_base)):
                engine = VideoRAGEngine(str(self.video.id))
                with patch.object(
                    VideoRAGEngine,
                    "_load_embedding_model",
                    side_effect=RuntimeError("Embedding model load blocked by the current torch runtime."),
                ):
                    built = engine.build_index(
                        [{"id": 0, "start": 0.0, "end": 8.0, "text": "This transcript contains enough text for indexing."}]
                    )

                self.assertFalse(built)
                self.assertEqual(engine.index_status, "degraded")
                self.assertIn("torch runtime", engine.index_error_reason)

                status_path = temp_base / "vector_indices" / str(self.video.id) / "status.json"
                self.assertTrue(status_path.exists())
                status_payload = json.loads(status_path.read_text(encoding="utf-8"))
                self.assertEqual(status_payload["status"], "degraded")
                self.assertIn("torch runtime", status_payload["reason"])

                video_index = VideoIndex.objects.get(video_id=self.video.id)
                self.assertFalse(video_index.is_indexed)
        finally:
            shutil.rmtree(temp_base, ignore_errors=True)

    @override_settings(RAG_TOP_K_SUMMARY=9, RAG_TOP_K_FACTUAL=7, RAG_TOP_K_QUOTE=5)
    def test_intent_specific_top_k_uses_configured_defaults(self):
        engine = ChatbotEngine(str(self.video.id))
        summary_intent, summary_top_k = engine._detect_intent_and_top_k("What is this video about?")  # pylint: disable=protected-access
        factual_intent, factual_top_k = engine._detect_intent_and_top_k("What does Robert Downey Jr say about Iron Man?")  # pylint: disable=protected-access
        quote_intent, quote_top_k = engine._detect_intent_and_top_k("Quote what he said about CGI")  # pylint: disable=protected-access
        self.assertEqual(summary_intent, "global_summary")
        self.assertEqual(summary_top_k, 9)
        self.assertEqual(factual_intent, "factual")
        self.assertEqual(factual_top_k, 7)
        self.assertEqual(quote_intent, "quote")
        self.assertEqual(quote_top_k, 5)

    def test_broad_named_question_is_classified_as_global_summary(self):
        engine = ChatbotEngine(str(self.video.id))
        intent, _top_k = engine._detect_intent_and_top_k(
            "What do Christopher Nolan and Robert Downey Jr talk about?"
        )  # pylint: disable=protected-access
        self.assertEqual(intent, "global_summary")

    @patch.object(VideoRAGEngine, "search", return_value=[])
    @patch.object(VideoRAGEngine, "get_relevant_context", return_value=("", []))
    def test_summary_and_chapter_fallback_used_when_retrieval_fails(self, _mock_context, _mock_search):
        Summary.objects.create(
            video=self.video,
            summary_type="full",
            title="Full",
            content="The video explains a motivational speech and the speaker's central message."
        )
        Summary.objects.create(
            video=self.video,
            summary_type="bullet",
            title="Bullet",
            content="• The speech focuses on confidence\n• The speaker explains preparation\n• The message centers on persistence\n• The talk closes with encouragement"
        )
        Summary.objects.create(
            video=self.video,
            summary_type="short",
            title="Short",
            content="A motivational speech about confidence, preparation, and persistence."
        )

        transcript = Transcript.objects.get(video=self.video)
        transcript.json_data = {
            "segments": [
                {"id": 0, "start": 0, "end": 20, "text": "The speech opens with the idea of confidence."},
                {"id": 1, "start": 20, "end": 40, "text": "The speaker then explains preparation and repetition."},
                {"id": 2, "start": 40, "end": 60, "text": "The final section focuses on persistence and effort."},
                {"id": 3, "start": 60, "end": 80, "text": "The talk closes with a motivational message."},
                {"id": 4, "start": 80, "end": 100, "text": "The audience takeaway is to keep practicing with purpose."},
            ]
        }
        transcript.save(update_fields=["json_data"])

        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("Explain the main discussion points.", use_llm=False)
        self.assertIn("Answer:", response["answer"])
        self.assertTrue(response["sources"])
        self.assertIn("timestamp", response["sources"][0])

    def test_chatbot_fallback_answer_keeps_required_structure(self):
        engine = ChatbotEngine(str(self.video.id))
        response = engine._fallback_answer_response("The transcript does not contain enough information to answer this question.")  # pylint: disable=protected-access
        self.assertIn("Answer:", response)
        self.assertIn("Key Points:", response)
        self.assertIn("The transcript does not contain enough information", response)

    @patch.object(ChatbotEngine, "_load_structured_summary_support")
    @patch.object(VideoRAGEngine, "get_relevant_context")
    def test_summary_question_returns_semantic_answer_not_quote(self, mock_context, mock_summary_support):
        mock_context.return_value = (
            "[0.0s - 30.0s] The video explains prompt writing and layout planning.",
            [
                {"text": "The video explains prompt writing and layout planning.", "source_text": "The video explains prompt writing and layout planning.", "start": 0, "end": 30, "score": 0.91},
                {"text": "It then covers animation tuning and deployment.", "source_text": "It then covers animation tuning and deployment.", "start": 30, "end": 60, "score": 0.88},
            ],
        )
        mock_summary_support.return_value = (
            {
                "tldr": "This tutorial covers layout planning, prompt writing, animation tuning, and final deployment.",
                "key_points": ["Layout planning", "Prompt writing", "Animation tuning"],
            },
            [
                {"text": "Layout Planning: Layout planning", "source_text": "Layout Planning", "start": 0, "end": 30, "score": 0.9},
                {"text": "Animation Tuning: Animation tuning", "source_text": "Animation Tuning", "start": 30, "end": 60, "score": 0.85},
            ],
        )

        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What is this video about?", use_llm=False)
        self.assertIn("Answer:", response["answer"])
        self.assertIn("Key Points:", response["answer"])
        self.assertNotIn("which line", response["answer"].lower())
        self.assertTrue(response["sources"])

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    def test_factual_question_retrieves_relevant_chunk(self, _mock_summary_support, mock_context):
        mock_context.return_value = (
            "[120.0s - 150.0s] Robert Downey Jr. says he enjoys drawing and restoring classic cars.",
            [
                {"text": "Robert Downey Jr. says he enjoys drawing and restoring classic cars.", "source_text": "Robert Downey Jr. says he enjoys drawing and restoring classic cars.", "start": 120, "end": 150, "score": 0.94},
            ],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What hobbies does RDJ mention?", use_llm=False)
        self.assertIn("cars", response["answer"].lower())
        self.assertTrue(response["sources"])
        self.assertEqual(response["sources"][0]["timestamp"], "02:00 — 02:30")

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    def test_quote_request_returns_exact_transcript_like_response(self, _mock_summary_support, mock_context):
        mock_context.return_value = (
            "[42.0s - 48.0s] Fear of failure is discussed here.",
            [
                {"text": "Fear of failure is discussed here.", "source_text": "Fear of failure is discussed here.", "start": 42, "end": 48, "score": 0.97},
            ],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("quote what he said about fear of failure", use_llm=False)
        self.assertIn("Fear of failure is discussed here.", response["answer"])
        self.assertTrue(response["sources"])

    @patch.object(VideoRAGEngine, "get_relevant_context", return_value=("", []))
    @patch.object(ChatbotEngine, "_answer_from_summary_and_chapters")
    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    def test_weak_retrieval_can_abstain_or_use_summary_fallback(self, _mock_summary_support, mock_fallback, _mock_context):
        mock_fallback.return_value = (
            "Answer:\nThe video discusses confidence and preparation.\n\nKey Points:\n• Confidence\n• Preparation\n• Persistence",
            [{"text": "Confidence", "source_text": "Confidence", "start": 0, "end": 30, "score": 0.5}],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What topics are discussed?", use_llm=False)
        self.assertIn("Answer:", response["answer"])
        self.assertTrue("sources" in response)

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support")
    def test_different_question_types_do_not_repeat_same_answer(self, mock_summary_support, mock_context):
        mock_summary_support.return_value = (
            {
                "tldr": "This tutorial explains the full workflow from layout planning to deployment.",
                "key_points": ["Layout planning", "Prompt writing", "Deployment"],
            },
            [{"text": "Layout Planning: Layout planning", "source_text": "Layout Planning", "start": 0, "end": 30, "score": 0.9}],
        )

        def _context_side_effect(question, top_k=None):
            if "about" in question.lower():
                return (
                    "[0.0s - 30.0s] The tutorial explains the full workflow from layout planning to deployment.",
                    [{"text": "The tutorial explains the full workflow from layout planning to deployment.", "source_text": "The tutorial explains the full workflow from layout planning to deployment.", "start": 0, "end": 30, "score": 0.92}],
                )
            return (
                "[60.0s - 90.0s] The speaker says the deployment step validates the final interface.",
                [{"text": "The speaker says the deployment step validates the final interface.", "source_text": "The speaker says the deployment step validates the final interface.", "start": 60, "end": 90, "score": 0.89}],
            )

        mock_context.side_effect = _context_side_effect
        engine = ChatbotEngine(str(self.video.id))
        summary_response = engine.ask("What is this video about?", use_llm=False)
        factual_response = engine.ask("What does he say about deployment?", use_llm=False)
        self.assertNotEqual(summary_response["answer"], factual_response["answer"])

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    def test_source_timestamps_remain_clickable_compatible(self, _mock_summary_support, mock_context):
        mock_context.return_value = (
            "[75.0s - 96.0s] The speaker explains the deployment check.",
            [{"text": "The speaker explains the deployment check.", "source_text": "The speaker explains the deployment check.", "start": 75, "end": 96, "score": 0.9}],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What does he say about deployment?", use_llm=False)
        self.assertRegex(response["sources"][0]["timestamp"], r"^\d{2}:\d{2} — \d{2}:\d{2}$")

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    def test_timeline_question_stays_time_relevant(self, _mock_summary_support, mock_context):
        mock_context.return_value = (
            "[240.0s - 255.0s] The video shifts into deployment planning. [255.0s - 270.0s] The speaker explains final validation and launch checks.",
            [
                {"text": "The video shifts into deployment planning.", "source_text": "The video shifts into deployment planning.", "start": 240, "end": 255, "score": 0.91},
                {"text": "The speaker explains final validation and launch checks.", "source_text": "The speaker explains final validation and launch checks.", "start": 255, "end": 270, "score": 0.89},
            ],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What happens around minute 4?", use_llm=False)
        self.assertIn("Around 04:00", response["answer"])
        self.assertTrue(response["sources"])
        self.assertTrue(response["sources"][0]["timestamp"].startswith("04:"))

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    def test_timeline_question_around_minute_8_uses_correct_time_region(self, _mock_summary_support, mock_context):
        mock_context.return_value = (
            "[405.0s - 430.0s] The earlier section discusses setup and context. "
            "[478.0s - 505.0s] Around this point they discuss career highlights and personal questions. "
            "[505.0s - 528.0s] The conversation continues with closing reflections.",
            [
                {"text": "The earlier section discusses setup and context.", "source_text": "The earlier section discusses setup and context.", "start": 405, "end": 430, "score": 0.96},
                {"text": "Around this point they discuss career highlights and personal questions.", "source_text": "Around this point they discuss career highlights and personal questions.", "start": 478, "end": 505, "score": 0.9},
                {"text": "The conversation continues with closing reflections.", "source_text": "The conversation continues with closing reflections.", "start": 505, "end": 528, "score": 0.88},
            ],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("Explain what they are discussing around minute 8.", use_llm=False)
        self.assertIn("Around 07:58", response["answer"])
        self.assertTrue(any(source["timestamp"].startswith("07:") or source["timestamp"].startswith("08:") for source in response["sources"]))

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    def test_factual_question_with_clear_evidence_does_not_abstain(self, _mock_summary_support, mock_context):
        mock_context.return_value = (
            "[125.0s - 173.0s] Christopher Nolan explains that he prefers practical effects because they feel more real and grounded than CGI.",
            [
                {"text": "Christopher Nolan explains that he prefers practical effects because they feel more real and grounded than CGI.", "source_text": "Christopher Nolan explains that he prefers practical effects because they feel more real and grounded than CGI.", "start": 125, "end": 173, "score": 0.95},
            ],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("Why does Christopher Nolan prefer practical effects over CGI?", use_llm=False)
        lowered = response["answer"].lower()
        self.assertNotIn("does not contain enough information", lowered)
        self.assertNotIn("not clearly stated", lowered)
        self.assertIn("practical effects", lowered)
        self.assertNotIn("they feel more real and grounded than cgi", lowered)
        self.assertTrue(response["sources"])

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    def test_cgi_question_removes_lead_in_noise_and_rewrites_semantically(self, _mock_summary_support, mock_context):
        mock_context.return_value = (
            "[125.0s - 173.0s] made out with a few times, why Christopher Nolan doesn't use CGI? I shoot on film because it's the highest quality and something real on camera gives you a better result.",
            [
                {"text": "made out with a few times, why Christopher Nolan doesn't use CGI? I shoot on film because it's the highest quality and something real on camera gives you a better result.", "source_text": "made out with a few times, why Christopher Nolan doesn't use CGI? I shoot on film because it's the highest quality and something real on camera gives you a better result.", "start": 125, "end": 173, "score": 0.95},
            ],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("Why does Christopher Nolan prefer practical effects over CGI?", use_llm=False)
        lowered = response["answer"].lower()
        self.assertIn("christopher nolan explains", lowered)
        self.assertIn("better visual results than cgi", lowered)
        self.assertNotIn("made out with a few times", lowered)
        self.assertNotIn("why christopher nolan", lowered)

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    def test_iron_man_question_with_clear_evidence_does_not_abstain(self, _mock_summary_support, mock_context):
        mock_context.return_value = (
            "[250.0s - 303.0s] Robert Downey Jr. says Iron Man changed his career and remains one of his most meaningful roles.",
            [
                {"text": "Robert Downey Jr. says Iron Man changed his career and remains one of his most meaningful roles.", "source_text": "Robert Downey Jr. says Iron Man changed his career and remains one of his most meaningful roles.", "start": 250, "end": 303, "score": 0.94},
            ],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What does Robert Downey Jr say about Iron Man?", use_llm=False)
        lowered = response["answer"].lower()
        self.assertNotIn("does not contain enough information", lowered)
        self.assertIn("iron man", lowered)
        self.assertIn("major turning point", lowered)
        self.assertNotIn("says iron man changed his career", lowered)
        self.assertTrue(response["sources"])

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    def test_batman_question_uses_relevant_answer_span(self, _mock_summary_support, mock_context):
        mock_context.return_value = (
            "[90.0s - 110.0s] They briefly discuss accents and general film talk. "
            "[300.0s - 328.0s] Christopher Nolan says Batman remains one of the defining parts of his career.",
            [
                {"text": "They briefly discuss accents and general film talk.", "source_text": "They briefly discuss accents and general film talk.", "start": 90, "end": 110, "score": 0.95},
                {"text": "Christopher Nolan says Batman remains one of the defining parts of his career.", "source_text": "Christopher Nolan says Batman remains one of the defining parts of his career.", "start": 300, "end": 328, "score": 0.9},
            ],
        )
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("Do they talk about Batman in this interview?", use_llm=False)
        lowered = response["answer"].lower()
        self.assertIn("batman", lowered)
        self.assertNotIn("accent", lowered)
        self.assertTrue(any("05:" in source["timestamp"] for source in response["sources"]))

    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    @patch.object(VideoRAGEngine, "get_relevant_context")
    def test_beginning_question_uses_timeline_window(self, mock_context, _mock_summary_support):
        mock_context.return_value = ("", [])
        transcript = Transcript.objects.get(video=self.video)
        transcript.json_data = {
            "segments": [
                {"id": 0, "start": 0, "end": 20, "text": "The interview opens with autocomplete questions and introductions."},
                {"id": 1, "start": 20, "end": 40, "text": "Christopher Nolan begins discussing Oppenheimer and filmmaking."},
                {"id": 2, "start": 200, "end": 220, "text": "The middle section shifts to Marvel and Iron Man."},
                {"id": 3, "start": 500, "end": 520, "text": "The ending moves into closing questions."},
            ]
        }
        transcript.save(update_fields=["json_data"])
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What is discussed at the beginning of the interview?", use_llm=False)
        self.assertIn("beginning of the video", response["answer"].lower())
        self.assertTrue(response["sources"])
        self.assertTrue(response["sources"][0]["timestamp"].startswith("00:"))

    @patch.object(ChatbotEngine, "_load_structured_summary_support", return_value=({}, []))
    @patch.object(VideoRAGEngine, "get_relevant_context")
    def test_end_question_uses_timeline_window(self, mock_context, _mock_summary_support):
        mock_context.return_value = ("", [])
        transcript = Transcript.objects.get(video=self.video)
        transcript.json_data = {
            "segments": [
                {"id": 0, "start": 0, "end": 20, "text": "The interview opens with introductions."},
                {"id": 1, "start": 20, "end": 40, "text": "It moves into filmmaking and Oppenheimer."},
                {"id": 2, "start": 480, "end": 500, "text": "Near the end, the discussion turns to personal questions and closing remarks."},
                {"id": 3, "start": 500, "end": 520, "text": "The video ends with final jokes and sign-off."},
            ]
        }
        transcript.save(update_fields=["json_data"])
        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What happens near the end of the video?", use_llm=False)
        self.assertIn("near the end of the video", response["answer"].lower())
        self.assertTrue(response["sources"])
        self.assertTrue(response["sources"][0]["timestamp"].startswith("08:"))
        self.assertNotIn("sign-off.", response["answer"])

    @patch.object(VideoRAGEngine, "search")
    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support")
    def test_summary_question_uses_distributed_sources(self, mock_summary_support, mock_context, mock_search):
        mock_summary_support.return_value = (
            {
                "tldr": "This interview features Robert Downey Jr. and Christopher Nolan discussing filmmaking, Oppenheimer, and personal topics.",
                "key_points": [
                    "Robert Downey Jr. reflects on Iron Man and hobbies.",
                    "Christopher Nolan talks about practical effects.",
                    "The interview mixes film talk with personal questions.",
                ],
                "chapters": [
                    {"title": "Interview Introduction", "timestamp": "00:10"},
                    {"title": "Nolan on Practical Effects", "timestamp": "04:10"},
                    {"title": "Career and Personal Questions", "timestamp": "09:20"},
                ],
            },
            [
                {"text": "Interview Introduction: Robert Downey Jr. and Christopher Nolan open the interview.", "source_text": "Interview Introduction", "start": 10, "end": 40, "score": 0.9},
                {"text": "Nolan on Practical Effects: Nolan explains why he prefers practical effects.", "source_text": "Nolan on Practical Effects", "start": 250, "end": 290, "score": 0.88},
                {"text": "Career and Personal Questions: The interview ends on career and personal questions.", "source_text": "Career and Personal Questions", "start": 560, "end": 600, "score": 0.86},
            ],
        )
        mock_context.return_value = ("", [])
        mock_search.return_value = [
            {"text": "Robert Downey Jr. and Christopher Nolan open the interview.", "source_text": "Robert Downey Jr. and Christopher Nolan open the interview.", "start_time": 12, "end_time": 42, "score": 0.92, "rank_score": 0.92},
            {"text": "Nolan explains his preference for practical effects over CGI.", "source_text": "Nolan explains his preference for practical effects over CGI.", "start_time": 255, "end_time": 285, "score": 0.89, "rank_score": 0.89},
            {"text": "The closing section covers careers, hobbies, and personal anecdotes.", "source_text": "The closing section covers careers, hobbies, and personal anecdotes.", "start_time": 565, "end_time": 595, "score": 0.87, "rank_score": 0.87},
        ]

        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What is this video about?", use_llm=False)
        self.assertGreaterEqual(len(response["sources"]), 3)
        self.assertNotEqual(response["sources"][0]["timestamp"], response["sources"][1]["timestamp"])
        self.assertNotIn('"', response["answer"])
        self.assertNotIn("what do", response["answer"].lower())

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support")
    def test_takeaways_question_returns_natural_paragraph_and_clean_bullets(self, mock_summary_support, mock_context):
        mock_context.return_value = (
            "[10.0s - 40.0s] Christopher Nolan explains why he prefers practical effects over CGI. "
            "[40.0s - 75.0s] Robert Downey Jr. discusses Marvel, Iron Man, and personal hobbies.",
            [
                {"text": "Christopher Nolan explains why he prefers practical effects over CGI.", "source_text": "Christopher Nolan explains why he prefers practical effects over CGI.", "start": 10, "end": 40, "score": 0.93},
                {"text": "Robert Downey Jr. discusses Marvel, Iron Man, and personal hobbies.", "source_text": "Robert Downey Jr. discusses Marvel, Iron Man, and personal hobbies.", "start": 40, "end": 75, "score": 0.9},
            ],
        )
        mock_summary_support.return_value = (
            {
                "tldr": "This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, Marvel, and personal experiences.",
                "key_points": [
                    "Christopher Nolan explains why he prefers practical effects over CGI.",
                    "Robert Downey Jr. discusses Marvel, Iron Man, and personal hobbies.",
                    "The interview explores filmmaking, Oppenheimer, and career highlights.",
                ],
                "chapters": [
                    {"title": "Nolan on Practical Effects", "timestamp": "00:10"},
                    {"title": "Downey on Iron Man", "timestamp": "00:40"},
                    {"title": "Career and Personal Questions", "timestamp": "01:10"},
                ],
            },
            [],
        )

        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("Can you summarize the key takeaways?", use_llm=False)
        self.assertIn("Answer:", response["answer"])
        self.assertIn("Key Points:", response["answer"])
        self.assertIn("Christopher Nolan", response["answer"])
        self.assertNotIn("Interview Introduction", response["answer"])
        self.assertRegex(response["answer"], r"• .+\.")

    @patch.object(VideoRAGEngine, "search")
    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support")
    def test_broad_interview_topics_question_returns_semantic_summary_not_quote(
        self,
        mock_summary_support,
        mock_context,
        mock_search,
    ):
        mock_summary_support.return_value = (
            {
                "tldr": "This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, Marvel, and personal experiences.",
                "key_points": [
                    "Christopher Nolan explains why he prefers practical effects over CGI.",
                    "Robert Downey Jr. discusses Marvel, Iron Man, and personal hobbies.",
                    "The interview covers filmmaking, Oppenheimer, and career highlights.",
                ],
                "chapters": [
                    {"title": "Nolan on Practical Effects", "timestamp": "02:05"},
                    {"title": "Downey on Iron Man", "timestamp": "04:10"},
                    {"title": "Career and Personal Questions", "timestamp": "08:45"},
                ],
            },
            [
                {"text": "Nolan on Practical Effects: Christopher Nolan explains why he prefers practical effects over CGI.", "source_text": "Nolan on Practical Effects", "start": 125, "end": 173, "score": 0.92},
                {"text": "Downey on Iron Man: Robert Downey Jr. reflects on Marvel and Iron Man.", "source_text": "Downey on Iron Man", "start": 250, "end": 303, "score": 0.9},
                {"text": "Career and Personal Questions: The interview also covers hobbies and career highlights.", "source_text": "Career and Personal Questions", "start": 525, "end": 570, "score": 0.87},
            ],
        )
        mock_context.return_value = ("", [])
        mock_search.return_value = [
            {"text": "Christopher Nolan explains why he prefers practical effects over CGI.", "source_text": "Christopher Nolan explains why he prefers practical effects over CGI.", "start_time": 125, "end_time": 173, "score": 0.93, "rank_score": 0.93},
            {"text": "Robert Downey Jr. reflects on Marvel, Iron Man, and personal interests.", "source_text": "Robert Downey Jr. reflects on Marvel, Iron Man, and personal interests.", "start_time": 250, "end_time": 303, "score": 0.91, "rank_score": 0.91},
            {"text": "The later section covers career highlights and personal experiences.", "source_text": "The later section covers career highlights and personal experiences.", "start_time": 525, "end_time": 570, "score": 0.88, "rank_score": 0.88},
        ]

        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What do Christopher Nolan and Robert Downey Jr talk about?", use_llm=False)
        lowered = response["answer"].lower()
        self.assertIn("christopher nolan", lowered)
        self.assertIn("robert downey jr", lowered)
        self.assertIn("filmmaking", lowered)
        self.assertNotIn("\"", response["answer"])
        self.assertNotIn("what do", lowered)
        self.assertGreaterEqual(len(response["sources"]), 3)

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support")
    def test_key_points_differ_by_question_type(self, mock_summary_support, mock_context):
        mock_summary_support.return_value = (
            {
                "tldr": "This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking, Oppenheimer, and Marvel.",
                "key_points": [
                    "Christopher Nolan explains practical effects and filmmaking choices.",
                    "Robert Downey Jr. discusses Marvel and Iron Man.",
                    "The interview covers career highlights and personal experiences.",
                ],
                "chapters": [
                    {"title": "Nolan on Practical Effects", "timestamp": "02:05"},
                    {"title": "Downey on Iron Man", "timestamp": "04:10"},
                ],
            },
            [],
        )

        def _ctx(question, top_k=None):
            if "iron man" in question.lower():
                return (
                    "[250.0s - 303.0s] Robert Downey Jr. says Iron Man changed his career and remains one of his most meaningful roles.",
                    [
                        {"text": "Robert Downey Jr. says Iron Man changed his career and remains one of his most meaningful roles.", "source_text": "Robert Downey Jr. says Iron Man changed his career and remains one of his most meaningful roles.", "start": 250, "end": 303, "score": 0.94},
                    ],
                )
            return (
                "[125.0s - 173.0s] Christopher Nolan explains why he prefers practical effects over CGI. [250.0s - 303.0s] Robert Downey Jr. discusses Iron Man and Marvel.",
                [
                    {"text": "Christopher Nolan explains why he prefers practical effects over CGI.", "source_text": "Christopher Nolan explains why he prefers practical effects over CGI.", "start": 125, "end": 173, "score": 0.93},
                    {"text": "Robert Downey Jr. discusses Iron Man and Marvel.", "source_text": "Robert Downey Jr. discusses Iron Man and Marvel.", "start": 250, "end": 303, "score": 0.9},
                ],
            )

        mock_context.side_effect = _ctx
        engine = ChatbotEngine(str(self.video.id))
        summary_response = engine.ask("What is this video about?", use_llm=False)
        factual_response = engine.ask("What does Robert Downey Jr say about Iron Man?", use_llm=False)
        self.assertNotEqual(summary_response["answer"], factual_response["answer"])
        self.assertIn("Iron Man", factual_response["answer"])
        self.assertIn("•", factual_response["answer"])

    @patch.object(VideoRAGEngine, "get_relevant_context")
    @patch.object(ChatbotEngine, "_load_structured_summary_support")
    def test_malformed_chat_key_points_are_filtered_or_regenerated(self, mock_summary_support, mock_context):
        mock_context.return_value = (
            "[12.0s - 36.0s] Christopher Nolan explains practical effects and Oppenheimer. "
            "[36.0s - 60.0s] Robert Downey Jr. discusses Marvel, Iron Man, and personal hobbies.",
            [
                {"text": "Christopher Nolan explains practical effects and Oppenheimer.", "source_text": "Christopher Nolan explains practical effects and Oppenheimer.", "start": 12, "end": 36, "score": 0.92},
                {"text": "Robert Downey Jr. discusses Marvel, Iron Man, and personal hobbies.", "source_text": "Robert Downey Jr. discusses Marvel, Iron Man, and personal hobbies.", "start": 36, "end": 60, "score": 0.9},
            ],
        )
        mock_summary_support.return_value = (
            {
                "tldr": "This interview features Christopher Nolan and Robert Downey Jr. discussing filmmaking and Marvel.",
                "key_points": [
                    "how did Robert Downey Jr",
                    "I probably should have taken my blood pressure medication",
                    "why Christopher Nolan",
                ],
                "chapters": [
                    {"title": "Nolan on Practical Effects", "timestamp": "00:12"},
                    {"title": "Downey on Iron Man", "timestamp": "00:36"},
                ],
            },
            [],
        )

        engine = ChatbotEngine(str(self.video.id))
        response = engine.ask("What topics are discussed?", use_llm=False)
        lowered = response["answer"].lower()
        self.assertNotIn("how did robert downey jr", lowered)
        self.assertNotIn("blood pressure medication", lowered)
        self.assertNotIn("why christopher nolan", lowered)
        self.assertIn("christopher nolan", lowered)

    def test_format_answer_response_dedupes_similar_key_points(self):
        engine = ChatbotEngine(str(self.video.id))
        formatted = engine._format_answer_response(  # pylint: disable=protected-access
            "Christopher Nolan explains his filmmaking preferences.",
            [
                "Christopher Nolan explains his filmmaking preferences.",
                "Christopher Nolan explains his filmmaking preferences",
                "Robert Downey Jr. discusses Marvel and Iron Man.",
                "Robert Downey Jr discusses Marvel and Iron Man",
            ],
        )
        self.assertEqual(formatted.count("• "), 2)

    def test_factual_answer_avoids_entity_duplication(self):
        engine = ChatbotEngine(str(self.video.id))
        answer = engine._answer_factual_from_segments(  # pylint: disable=protected-access
            "What does Robert Downey Jr say about Iron Man?",
            [
                {
                    "text": "Robert Downey Jr: Robert Downey Jr says Iron Man came from a strong screen test and the opportunity that followed.",
                    "source_text": "Robert Downey Jr: Robert Downey Jr says Iron Man came from a strong screen test and the opportunity that followed.",
                    "start": 250,
                    "end": 280,
                    "score": 0.95,
                    "speaker": "Robert Downey Jr.",
                }
            ],
        )
        self.assertNotIn("Robert Downey Jr. says that Robert Downey Jr.", answer)
        self.assertIn("Robert Downey Jr.", answer)

    def test_speaker_attribution_prefixes_grounded_answer(self):
        engine = ChatbotEngine(str(self.video.id))
        answer = engine._answer_factual_from_segments(  # pylint: disable=protected-access
            "Why does Christopher Nolan prefer practical effects over CGI?",
            [
                {
                    "text": "Christopher Nolan: I shoot on film because something real on camera gives you a better result than CGI.",
                    "source_text": "Christopher Nolan: I shoot on film because something real on camera gives you a better result than CGI.",
                    "start": 125,
                    "end": 150,
                    "score": 0.95,
                    "speaker": "Christopher Nolan",
                }
            ],
        )
        self.assertTrue(answer.startswith("Christopher Nolan explains"))

    def test_source_cards_use_clean_semantic_preview(self):
        engine = ChatbotEngine(str(self.video.id))
        cards = engine._build_source_cards(  # pylint: disable=protected-access
            [
                {
                    "text": "Christopher Nolan: Made out with a few times. I shoot on film because something real on camera gives you a better result than CGI.",
                    "source_text": "Christopher Nolan: Made out with a few times. I shoot on film because something real on camera gives you a better result than CGI.",
                    "start": 93,
                    "end": 139,
                    "score": 0.92,
                    "speaker": "Christopher Nolan",
                }
            ]
        )
        self.assertEqual(len(cards), 1)
        self.assertIn("Christopher Nolan discusses", cards[0]["text"])
        self.assertNotIn("Made out with a few times", cards[0]["text"])

    def test_source_cards_prefer_human_readable_chunk_label_when_available(self):
        engine = ChatbotEngine(str(self.video.id))
        cards = engine._build_source_cards(  # pylint: disable=protected-access
            [
                {
                    "text": "Christopher Nolan explains why he prefers practical effects over CGI.",
                    "source_text": "Christopher Nolan explains why he prefers practical effects over CGI.",
                    "source_label": "Christopher Nolan on practical effects",
                    "start": 125,
                    "end": 173,
                    "score": 0.92,
                    "speaker": "Christopher Nolan",
                }
            ],
            intent="summary",
        )
        self.assertEqual(cards[0]["text"], "Christopher Nolan on practical effects.")

    def test_lightweight_answer_qa_rejects_low_information_filler(self):
        engine = ChatbotEngine(str(self.video.id))
        reason = engine._validate_answer_quality(  # pylint: disable=protected-access
            "What does he say about CGI?",
            "factual",
            "Answer:\nIt covers the main themes that come up.\n\nKey Points:\n• It covers the main themes that come up.",
            [
                {
                    "text": "Christopher Nolan explains why he prefers practical effects over CGI.",
                    "source_text": "Christopher Nolan explains why he prefers practical effects over CGI.",
                    "start": 125,
                    "end": 173,
                    "score": 0.93,
                    "speaker": "Christopher Nolan",
                }
            ],
            "",
        )
        self.assertEqual(reason, "lightweight_answer_qa")

    def test_semantic_chunking_splits_on_topic_shift(self):
        engine = VideoRAGEngine(str(self.video.id))
        texts, metadatas = engine._build_overlapping_chunks(  # pylint: disable=protected-access
            [
                {"text": "Christopher Nolan explains why he prefers practical effects and shooting on film for realism.", "source_text": "Christopher Nolan explains why he prefers practical effects and shooting on film for realism.", "start": 0, "end": 18, "segment_id": 0, "speaker": "Christopher Nolan"},
                {"text": "He says real imagery on camera creates a stronger result than CGI.", "source_text": "He says real imagery on camera creates a stronger result than CGI.", "start": 18, "end": 36, "segment_id": 1, "speaker": "Christopher Nolan"},
                {"text": "Robert Downey Jr then talks about Iron Man, Marvel, and how the role changed his career.", "source_text": "Robert Downey Jr then talks about Iron Man, Marvel, and how the role changed his career.", "start": 52, "end": 74, "segment_id": 2, "speaker": "Robert Downey Jr."},
                {"text": "He reflects on the screen test, Jon Favreau, and the opportunity that followed.", "source_text": "He reflects on the screen test, Jon Favreau, and the opportunity that followed.", "start": 74, "end": 96, "segment_id": 3, "speaker": "Robert Downey Jr."},
            ]
        )
        self.assertGreaterEqual(len(texts), 2)
        self.assertTrue(any(meta.get("source_label") for meta in metadatas))


class ChatVoiceReplyPersistenceTests(TestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.video = Video.objects.create(title="Voice Demo", status="completed")
        self.session = ChatSession.objects.create(video_id=self.video.id, title="Voice session")
        Transcript.objects.create(
            video=self.video,
            language="en",
            transcript_language="en",
            canonical_language="en",
            full_text="Christopher Nolan explains practical effects.",
            transcript_original_text="Christopher Nolan explains practical effects.",
            transcript_canonical_text="Christopher Nolan explains practical effects.",
            json_data={
                "segments": [
                    {
                        "id": 0,
                        "start": 90,
                        "end": 120,
                        "text": "Christopher Nolan: I prefer practical effects because real imagery on camera gives a better result than CGI.",
                        "speaker": "Christopher Nolan",
                    }
                ]
            },
            word_timestamps=[],
        )

    def test_build_voice_narration_uses_natural_structure_and_speaker(self):
        narration = _build_voice_narration(
            "Answer:\nChristopher Nolan explains that practical effects feel more authentic on camera.\n\nKey Points:\n• Practical effects create more believable visuals.\n• The interview discusses CGI and film.",
            [{"speaker": "Christopher Nolan"}],
        )
        self.assertTrue(narration.startswith("Here's what Christopher Nolan says in the interview."))
        self.assertNotIn("Answer:", narration)
        self.assertNotIn("Key Points:", narration)
        self.assertIn("Key takeaway:", narration)

    @patch("chatbot.views.translate_text")
    @patch("chatbot.views.detect_text_language")
    @patch("chatbot.views.ChatbotEngine")
    @patch("videos.tts_utils.text_to_speech")
    def test_chat_view_persists_audio_url_and_voice_narration(
        self,
        mock_tts,
        mock_engine_cls,
        mock_detect_lang,
        mock_translate,
    ):
        mock_detect_lang.return_value = ("en", 0.99, "latin", "script")
        mock_translate.side_effect = lambda text, **kwargs: text
        mock_tts.return_value = "tts/test_reply.mp3"

        engine = mock_engine_cls.return_value
        engine.initialize.return_value = True
        engine.ask.return_value = {
            "answer": "Answer:\nChristopher Nolan explains that he prefers practical effects because real elements on camera feel more authentic than CGI.\n\nKey Points:\n• Practical effects create more believable visuals.\n• The discussion focuses on filmmaking choices.",
            "sources": [
                {
                    "text": "Christopher Nolan discusses practical effects and filming choices.",
                    "timestamp": "01:30 — 02:00",
                    "speaker": "Christopher Nolan",
                    "relevance": 0.95,
                }
            ],
        }

        request = self.factory.post(
            "/api/v1/chatbot/chat/",
            {
                "video_id": str(self.video.id),
                "message": "Why does Christopher Nolan prefer practical effects over CGI?",
                "generate_tts": True,
            },
            format="json",
        )
        response = ChatbotView.as_view()(request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["audio_url"], "/media/tts/test_reply.mp3")

        bot_msg = ChatMessage.objects.filter(sender="bot").latest("created_at")
        self.assertEqual(bot_msg.audio_url, "/media/tts/test_reply.mp3")
        self.assertTrue(bot_msg.voice_narration.startswith("Here's what Christopher Nolan says in the interview."))
        self.assertNotIn("Answer:", bot_msg.voice_narration)

    def test_chat_message_serializer_exposes_audio_fields(self):
        message = ChatMessage.objects.create(
            session=self.session,
            sender="bot",
            message="Answer:\nA clean answer.",
            audio_url="/media/tts/test_reply.mp3",
            voice_narration="Here's what the video says. A clean answer.",
        )
        data = ChatMessageSerializer(message).data
        self.assertEqual(data["audio_url"], "/media/tts/test_reply.mp3")
        self.assertIn("Here's what the video says.", data["voice_narration"])


class ChatIsolationRegressionTests(TestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.video_a = Video.objects.create(title="Video A", status="completed")
        self.video_b = Video.objects.create(title="Video B", status="completed")
        Transcript.objects.create(
            video=self.video_a,
            language="en",
            transcript_language="en",
            canonical_language="en",
            full_text="Transcript for video A.",
            transcript_original_text="Transcript for video A.",
            transcript_canonical_text="Transcript for video A.",
            json_data={"segments": [{"id": 0, "start": 0, "end": 5, "text": "Transcript for video A."}]},
            word_timestamps=[],
        )
        Transcript.objects.create(
            video=self.video_b,
            language="ml",
            transcript_language="ml",
            canonical_language="en",
            full_text="Transcript for video B.",
            transcript_original_text="Transcript for video B.",
            transcript_canonical_text="Transcript for video B.",
            json_data={"segments": [{"id": 0, "start": 0, "end": 5, "text": "Transcript for video B."}]},
            word_timestamps=[],
        )

    @patch("chatbot.views.translate_text", side_effect=lambda text, **kwargs: text)
    @patch("chatbot.views.detect_text_language", return_value=("en", 0.99, "latin", "script"))
    @patch("chatbot.views.ChatbotEngine")
    def test_chat_session_is_not_reused_across_videos(self, mock_engine_cls, _mock_detect, _mock_translate):
        stale_session = ChatSession.objects.create(video_id=self.video_a.id, title="Stale session")
        engine = mock_engine_cls.return_value
        engine.initialize.return_value = True
        engine.ask.return_value = {
            "answer": "This answer belongs to video B.",
            "sources": [{"text": "Transcript for video B.", "timestamp": "0.0s - 5.0s", "relevance": 1.0}],
        }

        request = self.factory.post(
            "/api/v1/chatbot/chat/",
            {
                "video_id": str(self.video_b.id),
                "session_id": str(stale_session.id),
                "message": "What is this video about?",
            },
            format="json",
        )
        response = ChatbotView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.data["session_id"], str(stale_session.id))
        new_session = ChatSession.objects.get(id=response.data["session_id"])
        self.assertEqual(str(new_session.video_id), str(self.video_b.id))

    def test_video_delete_removes_only_its_chatbot_artifacts(self):
        ChatSession.objects.create(video_id=self.video_a.id, title="Session A")
        ChatSession.objects.create(video_id=self.video_b.id, title="Session B")
        VideoIndex.objects.create(video_id=self.video_a.id, is_indexed=True, num_documents=1)
        VideoIndex.objects.create(video_id=self.video_b.id, is_indexed=True, num_documents=1)

        dir_a = Path(settings.BASE_DIR) / "vector_indices" / str(self.video_a.id)
        dir_b = Path(settings.BASE_DIR) / "vector_indices" / str(self.video_b.id)
        dir_a.mkdir(parents=True, exist_ok=True)
        dir_b.mkdir(parents=True, exist_ok=True)
        (dir_a / "data.json").write_text("{}", encoding="utf-8")
        (dir_b / "data.json").write_text("{}", encoding="utf-8")

        self.video_a.delete()

        self.assertFalse(ChatSession.objects.filter(video_id=self.video_a.id).exists())
        self.assertFalse(VideoIndex.objects.filter(video_id=self.video_a.id).exists())
        self.assertFalse(dir_a.exists())

        self.assertTrue(ChatSession.objects.filter(video_id=self.video_b.id).exists())
        self.assertTrue(VideoIndex.objects.filter(video_id=self.video_b.id).exists())
        self.assertTrue(dir_b.exists())
