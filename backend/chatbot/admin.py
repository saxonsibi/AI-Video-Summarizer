"""
Django Admin configuration for chatbot app
"""

from django.contrib import admin
from .models import ChatSession, ChatMessage, VideoIndex


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['video_id', 'title', 'created_at', 'updated_at']
    list_filter = ['created_at']
    search_fields = ['title', 'video_id']


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session', 'sender', 'message_preview', 'created_at']
    list_filter = ['sender', 'created_at']
    search_fields = ['message']
    
    def message_preview(self, obj):
        return obj.message[:100] + '...' if len(obj.message) > 100 else obj.message
    message_preview.short_description = 'Message'


@admin.register(VideoIndex)
class VideoIndexAdmin(admin.ModelAdmin):
    list_display = ['video_id', 'index_type', 'is_indexed', 'num_documents', 'last_updated']
    list_filter = ['index_type', 'is_indexed']
