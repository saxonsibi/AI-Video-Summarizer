"""
Serializers for chatbot app
"""

from rest_framework import serializers
from .models import ChatSession, ChatMessage, VideoIndex


class ChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for ChatSession model."""
    
    class Meta:
        model = ChatSession
        fields = ['id', 'video_id', 'title', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for ChatMessage model."""
    
    class Meta:
        model = ChatMessage
        fields = ['id', 'session', 'sender', 'message', 'referenced_segments', 'created_at']
        read_only_fields = ['id', 'created_at']


class ChatMessageCreateSerializer(serializers.Serializer):
    """Serializer for creating chat messages."""
    
    session_id = serializers.UUIDField(required=False, allow_null=True)
    message = serializers.CharField(max_length=5000)
    video_id = serializers.UUIDField(required=True)
    
    def validate_message(self, value):
        if not value.strip():
            raise serializers.ValidationError("Message cannot be empty")
        return value.strip()


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat response."""
    
    answer = serializers.CharField()
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
