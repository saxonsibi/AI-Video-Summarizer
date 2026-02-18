"""
Serializers for video processing API
"""

import os
from rest_framework import serializers
from .models import Video, Transcript, Summary, HighlightSegment, ShortVideo, ProcessingTask


class VideoUploadSerializer(serializers.Serializer):
    """Serializer for video upload."""
    title = serializers.CharField(max_length=255)
    description = serializers.CharField(required=False, allow_blank=True, default='')
    file = serializers.FileField()
    
    def validate_file(self, value):
        """Validate uploaded file is a video."""
        allowed_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv']
        ext = os.path.splitext(value.name)[1].lower()
        if ext not in allowed_extensions:
            raise serializers.ValidationError(
                f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
            )
        return value


class VideoSerializer(serializers.ModelSerializer):
    """Serializer for Video model."""
    
    filename = serializers.CharField(read_only=True)
    transcripts_count = serializers.SerializerMethodField()
    summaries_count = serializers.SerializerMethodField()
    shorts_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Video
        fields = [
            'id', 'title', 'description', 'original_file', 'filename',
            'duration', 'file_size', 'file_format', 'status',
            'processing_progress', 'error_message', 'created_at',
            'updated_at', 'processed_at', 'transcripts_count',
            'summaries_count', 'shorts_count'
        ]
        read_only_fields = ['id', 'filename', 'duration', 'file_size', 
                           'file_format', 'status', 'processing_progress',
                           'created_at', 'updated_at', 'processed_at']
    
    def get_transcripts_count(self, obj):
        return obj.transcripts.count()
    
    def get_summaries_count(self, obj):
        return obj.summaries.count()
    
    def get_shorts_count(self, obj):
        return obj.short_videos.count()


class TranscriptSerializer(serializers.ModelSerializer):
    """Serializer for Transcript model."""
    
    word_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Transcript
        fields = [
            'id', 'video', 'language', 'full_text', 'json_data',
            'word_timestamps', 'word_count', 'created_at'
        ]
        read_only_fields = ['id', 'video', 'created_at']
    
    def get_word_count(self, obj):
        return obj.get_word_count()


class SummarySerializer(serializers.ModelSerializer):
    """Serializer for Summary model."""
    
    class Meta:
        model = Summary
        fields = [
            'id', 'video', 'summary_type', 'title', 'content',
            'key_topics', 'model_used', 'generation_time', 'created_at'
        ]
        read_only_fields = ['id', 'video', 'model_used', 'generation_time', 'created_at']


class SummaryGenerateSerializer(serializers.Serializer):
    """Serializer for generating summary request."""
    
    summary_type = serializers.ChoiceField(
        choices=['full', 'bullet', 'short', 'timestamps'],
        default='full'
    )
    max_length = serializers.IntegerField(required=False, min_value=50, max_value=1000)
    min_length = serializers.IntegerField(required=False, min_value=20, max_value=500)


class HighlightSegmentSerializer(serializers.ModelSerializer):
    """Serializer for HighlightSegment model."""
    
    duration = serializers.FloatField(read_only=True)
    
    class Meta:
        model = HighlightSegment
        fields = [
            'id', 'video', 'start_time', 'end_time', 'duration',
            'importance_score', 'reason', 'transcript_snippet',
            'used_in_short', 'created_at'
        ]
        read_only_fields = ['id', 'video', 'created_at']


class ShortVideoSerializer(serializers.ModelSerializer):
    """Serializer for ShortVideo model."""
    
    filename = serializers.CharField(read_only=True)
    
    class Meta:
        model = ShortVideo
        fields = [
            'id', 'video', 'file', 'filename', 'duration',
            'thumbnail', 'style', 'include_music', 'music_track',
            'caption_style', 'font_size', 'status', 'created_at'
        ]
        read_only_fields = ['id', 'video', 'created_at']


class ShortVideoGenerateSerializer(serializers.Serializer):
    """Serializer for generating short video request."""
    
    max_duration = serializers.FloatField(default=60.0, required=False, help_text='Maximum short video duration in seconds')
    style = serializers.CharField(default='default', max_length=100, required=False)
    include_music = serializers.BooleanField(default=False, required=False)
    caption_style = serializers.CharField(default='default', max_length=100, required=False)
    font_size = serializers.IntegerField(default=24, required=False)


class ProcessingTaskSerializer(serializers.ModelSerializer):
    """Serializer for ProcessingTask model."""
    
    class Meta:
        model = ProcessingTask
        fields = [
            'id', 'task_type', 'task_id', 'video', 'status', 'progress',
            'message', 'error', 'created_at', 'started_at', 'completed_at'
        ]
        read_only_fields = ['id', 'task_type', 'task_id', 'video']


class VideoDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for video with nested relationships."""
    
    transcripts = TranscriptSerializer(many=True, read_only=True)
    summaries = SummarySerializer(many=True, read_only=True)
    highlight_segments = HighlightSegmentSerializer(many=True, read_only=True)
    short_videos = ShortVideoSerializer(many=True, read_only=True)
    tasks = ProcessingTaskSerializer(many=True, read_only=True)
    
    class Meta:
        model = Video
        fields = [
            'id', 'title', 'description', 'original_file', 'filename',
            'duration', 'file_size', 'file_format', 'status',
            'processing_progress', 'error_message', 'created_at',
            'updated_at', 'processed_at', 'transcripts', 'summaries',
            'highlight_segments', 'short_videos', 'tasks'
        ]
