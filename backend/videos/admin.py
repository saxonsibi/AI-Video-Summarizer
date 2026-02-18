"""
Django Admin configuration for videos app
"""

from django.contrib import admin
from .models import Video, Transcript, Summary, HighlightSegment, ShortVideo, ProcessingTask


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ['title', 'status', 'duration', 'file_size', 'created_at']
    list_filter = ['status', 'created_at', 'file_format']
    search_fields = ['title', 'description']
    readonly_fields = ['id', 'created_at', 'updated_at', 'processed_at']
    ordering = ['-created_at']


@admin.register(Transcript)
class TranscriptAdmin(admin.ModelAdmin):
    list_display = ['video', 'language', 'word_count', 'created_at']
    list_filter = ['language', 'created_at']
    search_fields = ['full_text']
    readonly_fields = ['id', 'created_at']
    
    def word_count(self, obj):
        return obj.get_word_count()
    word_count.short_description = 'Words'


@admin.register(Summary)
class SummaryAdmin(admin.ModelAdmin):
    list_display = ['video', 'summary_type', 'title', 'model_used', 'created_at']
    list_filter = ['summary_type', 'model_used', 'created_at']
    search_fields = ['title', 'content']
    readonly_fields = ['id', 'created_at']


@admin.register(HighlightSegment)
class HighlightSegmentAdmin(admin.ModelAdmin):
    list_display = ['video', 'start_time', 'end_time', 'duration', 'importance_score', 'used_in_short']
    list_filter = ['used_in_short', 'created_at']
    search_fields = ['transcript_snippet', 'reason']
    readonly_fields = ['id', 'created_at']
    
    def duration(self, obj):
        return f"{obj.end_time - obj.start_time:.2f}s"
    duration.short_description = 'Duration'


@admin.register(ShortVideo)
class ShortVideoAdmin(admin.ModelAdmin):
    list_display = ['video', 'duration', 'style', 'status', 'created_at']
    list_filter = ['style', 'status', 'created_at']
    readonly_fields = ['id', 'created_at']


@admin.register(ProcessingTask)
class ProcessingTaskAdmin(admin.ModelAdmin):
    list_display = ['task_type', 'video', 'status', 'progress', 'created_at']
    list_filter = ['task_type', 'status', 'created_at']
    search_fields = ['task_id', 'message', 'error']
    readonly_fields = ['id', 'created_at', 'started_at', 'completed_at']
    ordering = ['-created_at']
