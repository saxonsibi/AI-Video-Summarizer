"""
Serializers for chatbot app
"""

from rest_framework import serializers
from .models import ChatSession, ChatMessage, VideoIndex
from videos.translation import build_english_view_source_hash


def _chat_sources_from_message(message_obj):
    raw = getattr(message_obj, 'referenced_segments', None)
    if isinstance(raw, dict):
        return list(raw.get('sources') or [])
    if isinstance(raw, list):
        return raw
    return []


def _chat_english_view_cache(message_obj):
    raw = getattr(message_obj, 'referenced_segments', None)
    if not isinstance(raw, dict):
        return {}
    cache = raw.get('_english_view_cache', {})
    if not isinstance(cache, dict):
        return {}
    source_hash = build_english_view_source_hash(
        'chat',
        {
            'answer_text': str(getattr(message_obj, 'message', '') or '').strip(),
            'answer_language': str(getattr(message_obj, 'output_language', '') or '').strip().lower(),
            'sources': _chat_sources_from_message(message_obj),
        },
    )
    if str(cache.get('english_view_source_hash', '') or '') != source_hash:
        return {
            'english_view_answer': '',
            'english_view_available': False,
            'chatbot_english_view_available': False,
            'chatbot_translation_state': 'stale',
            'chatbot_translation_warning': '',
            'chatbot_translation_blocked_reason': 'stale_cached_translation',
        }
    payload = cache.get('payload', {}) if isinstance(cache.get('payload', {}), dict) else {}
    return {
        'english_view_answer': str(payload.get('english_view_text', '') or ''),
        'english_view_available': bool(payload.get('english_view_available', False)),
        'chatbot_english_view_available': bool(payload.get('english_view_available', False)),
        'chatbot_translation_state': str(payload.get('translation_state', '') or ''),
        'chatbot_translation_warning': str(payload.get('translation_warning', '') or ''),
        'chatbot_translation_blocked_reason': str(payload.get('translation_blocked_reason', '') or ''),
    }


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for ChatSession model."""
    
    class Meta:
        model = ChatSession
        fields = ['id', 'video_id', 'title', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for ChatMessage model."""
    referenced_segments = serializers.SerializerMethodField()
    english_view_answer = serializers.SerializerMethodField()
    english_view_available = serializers.SerializerMethodField()
    chatbot_english_view_available = serializers.SerializerMethodField()
    chatbot_translation_state = serializers.SerializerMethodField()
    chatbot_translation_warning = serializers.SerializerMethodField()
    chatbot_translation_blocked_reason = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatMessage
        fields = [
            'id', 'session', 'sender', 'message',
            'user_language', 'output_language', 'retrieval_language',
            'referenced_segments', 'audio_url', 'voice_narration',
            'english_view_answer', 'english_view_available', 'chatbot_english_view_available',
            'chatbot_translation_state', 'chatbot_translation_warning',
            'chatbot_translation_blocked_reason', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']

    def get_referenced_segments(self, obj):
        return _chat_sources_from_message(obj)

    def get_english_view_answer(self, obj):
        return _chat_english_view_cache(obj).get('english_view_answer', '')

    def get_english_view_available(self, obj):
        return bool(_chat_english_view_cache(obj).get('english_view_available', False))

    def get_chatbot_english_view_available(self, obj):
        return bool(_chat_english_view_cache(obj).get('chatbot_english_view_available', False))

    def get_chatbot_translation_state(self, obj):
        return str(_chat_english_view_cache(obj).get('chatbot_translation_state', '') or '')

    def get_chatbot_translation_warning(self, obj):
        return str(_chat_english_view_cache(obj).get('chatbot_translation_warning', '') or '')

    def get_chatbot_translation_blocked_reason(self, obj):
        return str(_chat_english_view_cache(obj).get('chatbot_translation_blocked_reason', '') or '')


class ChatMessageCreateSerializer(serializers.Serializer):
    """Serializer for creating chat messages."""
    
    session_id = serializers.UUIDField(required=False, allow_null=True)
    message = serializers.CharField(max_length=5000)
    video_id = serializers.UUIDField(required=True)
    strict_mode = serializers.BooleanField(required=False, default=False)
    response_language = serializers.CharField(required=False, allow_blank=True, default='auto', max_length=16)
    output_language = serializers.CharField(required=False, allow_blank=True, default='auto', max_length=16)
    context_timestamp = serializers.FloatField(required=False, allow_null=True)
    context_window_seconds = serializers.FloatField(required=False, allow_null=True, default=None)
    # English View: Request English translation for chatbot responses
    english_view = serializers.BooleanField(required=False, default=False)
    
    def validate_message(self, value):
        if not value.strip():
            raise serializers.ValidationError("Message cannot be empty")
        return value.strip()


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat response."""
    
    answer = serializers.CharField()
    english_view_answer = serializers.CharField(required=False, allow_blank=True, default="")
    english_view_available = serializers.BooleanField(required=False, default=False)
    chatbot_answer_language = serializers.CharField(required=False, allow_blank=True, default="")
    chatbot_english_view_available = serializers.BooleanField(required=False, default=False)
    chatbot_translation_state = serializers.CharField(required=False, allow_blank=True, default="")
    chatbot_translation_warning = serializers.CharField(required=False, allow_blank=True, default="")
    chatbot_translation_blocked_reason = serializers.CharField(required=False, allow_blank=True, default="")
    sources = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        default=list
    )
    session_id = serializers.UUIDField()
    error = serializers.CharField(required=False, allow_null=True)


class SuggestedQuestionsSerializer(serializers.Serializer):
    """Serializer for suggested questions."""
    
    questions = serializers.ListField(
        child=serializers.CharField(max_length=500)
    )


class VideoIndexSerializer(serializers.ModelSerializer):
    """Serializer for VideoIndex model."""
    
    class Meta:
        model = VideoIndex
        fields = ['video_id', 'index_type', 'embedding_model', 'is_indexed', 
                 'index_created_at', 'last_updated', 'num_documents', 'dimension']
        read_only_fields = ['video_id', 'index_type', 'embedding_model', 
                           'index_created_at', 'last_updated', 'num_documents', 'dimension']
