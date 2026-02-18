"""
Video models for AI Video Summarizer
"""

import os
import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone


def video_upload_path(instance, filename):
    """Generate upload path for video files."""
    ext = os.path.splitext(filename)[1]
    return os.path.join('videos', 'original', f'{uuid.uuid4()}{ext}')


def audio_upload_path(instance, filename):
    """Generate upload path for audio files."""
    ext = os.path.splitext(filename)[1]
    return os.path.join('audio', f'{uuid.uuid4()}{ext}')


def short_video_path(instance, filename):
    """Generate upload path for short video files."""
    ext = os.path.splitext(filename)[1]
    return os.path.join('shorts', f'{uuid.uuid4()}{ext}')


class Video(models.Model):
    """Main video model for storing uploaded videos and their metadata."""
    
    PROCESSING_STATUS = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('transcribing', 'Transcribing'),
        ('summarizing', 'Summarizing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, default='')
    original_file = models.FileField(upload_to=video_upload_path, max_length=500)
    duration = models.FloatField(null=True, blank=True, help_text='Duration in seconds')
    file_size = models.BigIntegerField(default=0, help_text='File size in bytes')
    file_format = models.CharField(max_length=10, blank=True)
    
    # Processing status
    status = models.CharField(max_length=20, choices=PROCESSING_STATUS, default='pending')
    processing_progress = models.IntegerField(default=0, help_text='Progress percentage')
    error_message = models.TextField(blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    # User (optional for now, can be extended for authentication)
    user_id = models.CharField(max_length=100, blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        db_table = 'videos'
    
    def __str__(self):
        return f"{self.title} ({self.status})"
    
    @property
    def filename(self):
        return os.path.basename(self.original_file.name)
    
    def delete(self, *args, **kwargs):
        """Delete files when model is deleted."""
        # Delete associated files
        if self.original_file:
            try:
                self.original_file.delete(save=False)
            except Exception:
                pass
        super().delete(*args, **kwargs)


class Transcript(models.Model):
    """Store video transcript with timestamps."""
    
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='transcripts')
    language = models.CharField(max_length=10, default='en')
    full_text = models.TextField()
    json_data = models.JSONField(help_text='Detailed transcript with timestamps')
    
    # Word-level timestamps for precise video clipping
    word_timestamps = models.JSONField(blank=True, null=True, help_text='Word-level timestamps')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'transcripts'
    
    def __str__(self):
        return f"Transcript for {self.video.title} ({self.language})"
    
    def get_word_count(self):
        """Get word count of transcript."""
        return len(self.full_text.split())


class Summary(models.Model):
    """Store video summaries at different levels of detail."""
    
    SUMMARY_TYPE = [
        ('full', 'Full Summary'),
        ('bullet', 'Bullet Points'),
        ('short', 'Short Script (30-60 sec)'),
        ('timestamps', 'Timestamp Summary'),
    ]
    
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='summaries')
    summary_type = models.CharField(max_length=20, choices=SUMMARY_TYPE)
    title = models.CharField(max_length=255, blank=True, default='')
    content = models.TextField()
    key_topics = models.JSONField(blank=True, null=True, help_text='List of key topics')
    
    # AI model metadata
    model_used = models.CharField(max_length=100, default='facebook/bart-large-cnn')
    generation_time = models.FloatField(default=0.0, help_text='Time taken to generate in seconds')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'summaries'
        unique_together = ['video', 'summary_type']
    
    def __str__(self):
        return f"{self.summary_type} summary for {self.video.title}"


class HighlightSegment(models.Model):
    """Store detected highlight segments for short video generation."""
    
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='highlight_segments')
    
    # Segment timestamps
    start_time = models.FloatField(help_text='Start time in seconds')
    end_time = models.FloatField(help_text='End time in seconds')
    
    # Segment metadata
    importance_score = models.FloatField(default=0.0, help_text='AI-calculated importance score')
    reason = models.CharField(max_length=255, blank=True, help_text='Why this segment was selected')
    transcript_snippet = models.TextField(blank=True, help_text='Transcript text in this segment')
    
    # Usage tracking
    used_in_short = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-importance_score']
        db_table = 'highlight_segments'
    
    def __str__(self):
        return f"Highlight {self.start_time:.1f}s - {self.end_time:.1f}s ({self.importance_score:.2f})"
    
    @property
    def duration(self):
        return self.end_time - self.start_time


class ShortVideo(models.Model):
    """Store generated short videos."""
    
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='short_videos')
    
    # Short video file
    file = models.FileField(upload_to=short_video_path, max_length=500)
    duration = models.FloatField(help_text='Duration in seconds')
    
    # Thumbnail
    thumbnail = models.ImageField(upload_to='thumbnails/', blank=True, null=True)
    
    # Generation settings
    style = models.CharField(max_length=100, default='default', help_text='Style template used')
    include_music = models.BooleanField(default=False)
    music_track = models.CharField(max_length=255, blank=True, null=True)
    
    # Caption settings
    caption_style = models.CharField(max_length=100, default='default')
    font_size = models.IntegerField(default=24)
    
    # Processing status
    status = models.CharField(max_length=20, default='completed')
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        db_table = 'short_videos'
    
    def __str__(self):
        return f"Short video for {self.video.title} ({self.duration:.1f}s)"
    
    @property
    def filename(self):
        return os.path.basename(self.file.name)


class ProcessingTask(models.Model):
    """Track background processing tasks for video operations."""
    
    TASK_TYPE = [
        ('transcription', 'Transcription'),
        ('summarization', 'Summarization'),
        ('highlight_detection', 'Highlight Detection'),
        ('short_generation', 'Short Video Generation'),
        ('thumbnail_generation', 'Thumbnail Generation'),
    ]
    
    TASK_STATUS = [
        ('pending', 'Pending'),
        ('started', 'Started'),
        ('progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task_type = models.CharField(max_length=50, choices=TASK_TYPE)
    task_id = models.CharField(max_length=255, unique=True, help_text='Celery task ID')
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='tasks')
    
    status = models.CharField(max_length=20, choices=TASK_STATUS, default='pending')
    progress = models.IntegerField(default=0)
    message = models.CharField(max_length=255, blank=True)
    
    # Error tracking
    error = models.TextField(blank=True, null=True)
    traceback = models.TextField(blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        db_table = 'processing_tasks'
    
    def __str__(self):
        return f"{self.task_type} - {self.status}"
    
    def mark_started(self):
        """Mark task as started."""
        self.status = 'started'
        self.started_at = timezone.now()
        self.save(update_fields=['status', 'started_at'])
    
    def mark_completed(self):
        """Mark task as completed."""
        self.status = 'completed'
        self.progress = 100
        self.completed_at = timezone.now()
        self.save(update_fields=['status', 'progress', 'completed_at'])
    
    def mark_failed(self, error, traceback=None):
        """Mark task as failed."""
        self.status = 'failed'
        self.error = str(error)
        self.traceback = traceback
        self.save(update_fields=['status', 'error', 'traceback'])
