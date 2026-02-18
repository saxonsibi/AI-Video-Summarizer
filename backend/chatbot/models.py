"""
Models for chatbot app
"""

import uuid
from django.db import models


class ChatSession(models.Model):
    """Store chat sessions for videos."""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video_id = models.UUIDField(help_text='Reference to video')
    
    # Session metadata
    title = models.CharField(max_length=255, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
        db_table = 'chat_sessions'
    
    def __str__(self):
        return f"Chat session for video {self.video_id}"


class ChatMessage(models.Model):
    """Store chat messages within sessions."""
    
    SENDER_CHOICES = [
        ('user', 'User'),
        ('bot', 'Bot'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    
    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    message = models.TextField()
    
    # Optional: reference to transcript segments used
    referenced_segments = models.JSONField(blank=True, null=True, help_text='Transcript segments referenced in answer')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
        db_table = 'chat_messages'
    
    def __str__(self):
        return f"{self.sender}: {self.message[:50]}..."


class VideoIndex(models.Model):
    """Store vector index metadata for videos."""
    
    video_id = models.UUIDField(primary_key=True, help_text='Reference to video')
    
    # Index metadata
    index_type = models.CharField(max_length=50, default='faiss', help_text='Type of vector index')
    embedding_model = models.CharField(max_length=100, default='all-MiniLM-L6-v2')
    
    # Index status
    is_indexed = models.BooleanField(default=False)
    index_created_at = models.DateTimeField(null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)
    
    # Stats
    num_documents = models.IntegerField(default=0)
    dimension = models.IntegerField(default=384)
    
    class Meta:
        db_table = 'video_indices'
    
    def __str__(self):
        return f"Index for video {self.video_id}"
