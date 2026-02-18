"""
API Views for video processing - Synchronous version (no Celery)
"""

import os
import logging
from django.conf import settings
from django.shortcuts import get_object_or_404
from rest_framework import viewsets, status, views
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.db import transaction

from .models import Video, Transcript, Summary, HighlightSegment, ShortVideo, ProcessingTask
from .serializers import (
    VideoSerializer, VideoUploadSerializer, VideoDetailSerializer,
    TranscriptSerializer, SummarySerializer, SummaryGenerateSerializer,
    HighlightSegmentSerializer, ShortVideoSerializer, ShortVideoGenerateSerializer,
    ProcessingTaskSerializer
)
from .utils import (
    extract_audio, transcribe_video, summarize_text, 
    detect_highlights, create_short_video, get_video_duration
)

logger = logging.getLogger(__name__)


class TranscriptViewSet(viewsets.ModelViewSet):
    """ViewSet for managing transcripts with edit capability."""
    serializer_class = TranscriptSerializer
    
    def get_queryset(self):
        return Transcript.objects.all()
    
    def update(self, request, *args, **kwargs):
        """Update transcript text (for manual corrections)."""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        return Response(serializer.data)


def process_video_sync(video_id):
    """Process video synchronously - transcription + summarization + highlights"""
    try:
        video = Video.objects.get(id=video_id)
        
        # Update status
        video.status = 'processing'
        video.processing_progress = 10
        video.save(update_fields=['status', 'processing_progress', 'updated_at'])
        
        # Step 1: Extract audio
        audio_path = extract_audio(video.original_file.path)
        video.processing_progress = 30
        video.save(update_fields=['processing_progress', 'updated_at'])
        
        # Step 2: Transcribe
        transcript_data = transcribe_video(audio_path)
        
        # Save transcript
        Transcript.objects.create(
            video=video,
            language=transcript_data.get('language', 'en'),
            full_text=transcript_data.get('text', ''),
            json_data=transcript_data,
            word_timestamps=transcript_data.get('word_timestamps', [])
        )
        
        video.processing_progress = 70
        video.save(update_fields=['processing_progress', 'updated_at'])
        
        # Step 3: Generate summary
        summary_text = summarize_text(transcript_data.get('text', ''))
        Summary.objects.create(
            video=video,
            summary_type='full',
            title='Full Summary',
            content=summary_text
        )
        
        # Step 4: Try to detect highlights (may fail for short videos)
        try:
            transcript_obj = video.transcripts.first()
            if transcript_obj:
                highlights = detect_highlights(transcript_obj)
                for highlight in highlights:
                    HighlightSegment.objects.create(
                        video=video,
                        start_time=highlight.get('start', 0),
                        end_time=highlight.get('end', 0),
                        importance_score=highlight.get('score', 0.5),
                        transcript_snippet=highlight.get('text', '')
                    )
        except Exception as e:
            logger.warning(f"Highlight detection failed: {str(e)}")
        
        # Complete
        video.status = 'completed'
        video.processing_progress = 100
        video.save(update_fields=['status', 'processing_progress', 'updated_at'])
        
        logger.info(f"Video processing completed for {video_id}")
        
    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        try:
            video = Video.objects.get(id=video_id)
            video.status = 'failed'
            video.error_message = str(e)
            video.save(update_fields=['status', 'error_message', 'updated_at'])
        except:
            pass


class VideoUploadView(views.APIView):
    """Handle video uploads."""
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        """Upload a new video for processing."""
        serializer = VideoUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            video_file = serializer.validated_data['file']
            title = serializer.validated_data.get('title', video_file.name)
            description = serializer.validated_data.get('description', '')
            
            # Create video instance
            video = Video.objects.create(
                title=title,
                description=description,
                original_file=video_file,
                file_size=video_file.size,
                file_format=os.path.splitext(video_file.name)[1].lower()[1:],
                status='pending'
            )
            
            # Process synchronously
            try:
                process_video_sync(str(video.id))
            except Exception as e:
                logger.warning(f"Processing failed: {str(e)}")
            
            response_serializer = VideoSerializer(video)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Video upload failed: {str(e)}")
            return Response(
                {'error': 'Video upload failed', 'detail': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TranscriptViewSet(viewsets.ModelViewSet):
    """ViewSet for managing transcripts."""
    serializer_class = TranscriptSerializer
    
    def get_queryset(self):
        return Transcript.objects.all()
    
    def update(self, request, *args, **kwargs):
        """Update transcript text (for manual corrections)."""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        return Response(serializer.data)


class VideoViewSet(viewsets.ModelViewSet):
    """ViewSet for Video CRUD operations."""
    queryset = Video.objects.all()
    serializer_class = VideoSerializer
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return VideoDetailSerializer
        return VideoSerializer
    
    def get_queryset(self):
        """Filter videos by user if provided."""
        queryset = Video.objects.all()
        return queryset.prefetch_related(
            'transcripts', 'summaries', 'highlight_segments', 'short_videos'
        )
    
    def perform_destroy(self, instance):
        """Delete video and associated files."""
        instance.delete()
    
    @action(detail=True, methods=['get'])
    def transcripts(self, request, pk=None):
        """Get all transcripts for a video."""
        video = self.get_object()
        transcripts = video.transcripts.all()
        serializer = TranscriptSerializer(transcripts, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['put', 'patch'])
    def update_transcript(self, request, pk=None):
        """Update transcript text (for manual corrections)."""
        video = self.get_object()
        transcript = video.transcripts.first()
        
        if not transcript:
            return Response(
                {'error': 'No transcript found for this video'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Update the full_text field
        full_text = request.data.get('full_text')
        if full_text:
            transcript.full_text = full_text
            transcript.save()
        
        serializer = TranscriptSerializer(transcript)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def generate_transcript(self, request, pk=None):
        """Generate transcript for a video."""
        video = self.get_object()
        
        if video.status in ['processing', 'transcribing']:
            return Response(
                {'error': 'Video is already being processed'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Process synchronously
        try:
            process_video_sync(str(video.id))
            video.refresh_from_db()
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        serializer = VideoSerializer(video)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def summaries(self, request, pk=None):
        """Get all summaries for a video."""
        video = self.get_object()
        summaries = video.summaries.all()
        serializer = SummarySerializer(summaries, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def generate_summary(self, request, pk=None):
        """Generate summary for a video."""
        video = self.get_object()
        serializer = SummaryGenerateSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        summary_type = serializer.validated_data.get('summary_type', 'full')
        max_length = serializer.validated_data.get('max_length')
        min_length = serializer.validated_data.get('min_length')
        
        # Delete existing summary and regenerate
        video.summaries.filter(summary_type=summary_type).delete()
        
        # Generate summary synchronously
        try:
            transcript = video.transcripts.first()
            if not transcript:
                return Response(
                    {'error': 'No transcript found. Generate transcript first.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            summary_text = summarize_text(transcript.full_text, summary_type=summary_type, max_length=max_length, min_length=min_length)
            summary = Summary.objects.create(
                video=video,
                summary_type=summary_type,
                title=summary_text.get('title', f'{summary_type.capitalize()} Summary'),
                content=summary_text['content'],  # Save just the plain text
                model_used=summary_text.get('model_used', 'facebook/bart-large-cnn'),
                generation_time=summary_text.get('generation_time', 0)
            )
            
            # Also detect highlights if none exist
            if not video.highlight_segments.exists():
                try:
                    highlights = detect_highlights(transcript)
                    for highlight in highlights:
                        HighlightSegment.objects.create(
                            video=video,
                            start_time=highlight.get('start', 0),
                            end_time=highlight.get('end', 0),
                            importance_score=highlight.get('score', 0.5),
                            transcript_snippet=highlight.get('text', '')
                        )
                    logger.info(f"Created {len(highlights)} highlight segments for video {video.id}")
                except Exception as e:
                    logger.warning(f"Highlight detection failed: {str(e)}")
            
            serializer = SummarySerializer(summary)
            return Response(serializer.data)
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def highlights(self, request, pk=None):
        """Get highlight segments for a video."""
        video = self.get_object()
        segments = video.highlight_segments.all()
        serializer = HighlightSegmentSerializer(segments, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def shorts(self, request, pk=None):
        """Get generated short videos for a video."""
        video = self.get_object()
        shorts = video.short_videos.all()
        serializer = ShortVideoSerializer(shorts, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def generate_short(self, request, pk=None):
        """Generate a short video from a video."""
        video = self.get_object()
        logger.error(f"Generate short request data: {request.data}")
        serializer = ShortVideoGenerateSerializer(data=request.data)
        
        if not serializer.is_valid():
            logger.error(f"Short video serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        logger.error(f"Serializer is valid, validated_data: {serializer.validated_data}")
        
        # Check if highlights exist, if not try to create from transcript
        if not video.highlight_segments.exists():
            # Try to create highlights from transcript
            transcript = video.transcripts.first()
            if transcript and (transcript.json_data or transcript.full_text):
                # Create highlights from transcript segments
                try:
                    from videos.utils import fallback_highlight_detection
                    highlights = fallback_highlight_detection(transcript)
                    
                    if not highlights:
                        # If fallback also returns empty, create basic segments from video duration
                        if video.duration:
                            segment_duration = min(10, video.duration / 5)  # 5 segments of 10 seconds each
                            for i in range(5):
                                start = i * segment_duration
                                end = min((i + 1) * segment_duration, video.duration)
                                highlights.append({
                                    'start_time': start,
                                    'end_time': end,
                                    'importance_score': 0.5,
                                    'reason': 'Auto-generated segment',
                                    'transcript_snippet': f'Segment {i+1}'
                                })
                    
                    # Save highlights to database
                    for hl in highlights:
                        HighlightSegment.objects.create(
                            video=video,
                            start_time=hl['start_time'],
                            end_time=hl['end_time'],
                            transcript_snippet=hl.get('transcript_snippet', ''),
                            importance_score=hl.get('importance_score', 0.5),
                            reason=hl.get('reason', 'Auto-generated from transcript')
                        )
                    logger.info(f"Auto-generated {len(highlights)} highlight segments for video {video.id}")
                except Exception as e:
                    logger.error(f"Failed to auto-generate highlights: {e}")
                    return Response(
                        {'error': f'No highlights found and auto-generation failed: {str(e)}'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            elif not transcript:
                logger.error(f"No highlight segments found for video {video.id}")
                return Response(
                    {'error': 'No highlights found. Please upload and process a video with transcript first.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            else:
                logger.error(f"No highlight segments or transcript data for video {video.id}")
                return Response(
                    {'error': 'No highlights found and no transcript data available. Please process the video first.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # Generate short synchronously
        try:
            # Get highlight segments
            segments = video.highlight_segments.all()
            if not segments:
                return Response(
                    {'error': 'No highlights found. Transcript processing needed first.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Prepare segments data
            max_duration = serializer.validated_data.get('max_duration', 60)
            segments_data = []
            total_duration = 0
            
            for seg in segments:
                seg_duration = seg.end_time - seg.start_time
                if total_duration + seg_duration <= max_duration:
                    segments_data.append({
                        'start_time': seg.start_time,
                        'end_time': seg.end_time,
                        'transcript_snippet': seg.transcript_snippet or ''
                    })
                    total_duration += seg_duration
            
            if not segments_data:
                return Response(
                    {'error': 'No segments fit within max_duration. Try increasing the duration.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get video path
            video_path = video.original_file.path
            
            # Create short video
            short_video_path = create_short_video(
                video_path,
                segments_data,
                style=serializer.validated_data.get('style', 'default'),
                caption_style=serializer.validated_data.get('caption_style', 'default'),
                font_size=24
            )
            
            # Save short video
            from django.core.files import File
            with open(short_video_path, 'rb') as f:
                short = ShortVideo.objects.create(
                    video=video,
                    file=File(f, name=f'short_{video.id}.mp4'),
                    duration=sum(s['end_time'] - s['start_time'] for s in segments_data),
                    style=serializer.validated_data.get('style', 'default'),
                    include_music=serializer.validated_data.get('include_music', False)
                )
            
            serializer = ShortVideoSerializer(short)
            return Response(serializer.data)
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def generate_audio_summary(self, request, pk=None):
        """
        Generate an audio summary (podcast-style) for a video.
        Returns an audio file that can be downloaded.
        """
        video = self.get_object()
        
        try:
            # Get transcript
            transcript = video.transcripts.first()
            if not transcript:
                return Response(
                    {'error': 'No transcript found. Generate transcript first.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get full text
            full_text = transcript.full_text
            if not full_text:
                return Response(
                    {'error': 'Transcript is empty.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Generate audio summary using TTS
            from .tts_utils import generate_podcast_summary
            audio_path = generate_podcast_summary(full_text)
            
            return Response({
                'audio_url': f"/media/{audio_path}",
                'message': 'Audio summary generated successfully'
            })
            
        except Exception as e:
            logger.error(f"Audio summary generation failed: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def tasks(self, request, pk=None):
        """Get processing tasks for a video."""
        video = self.get_object()
        tasks = video.tasks.all()
        serializer = ProcessingTaskSerializer(tasks, many=True)
        return Response(serializer.data)
