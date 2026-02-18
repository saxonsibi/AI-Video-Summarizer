"""
Celery tasks for video processing
"""

import os
import logging
import subprocess
from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .models import Video, Transcript, Summary, HighlightSegment, ShortVideo, ProcessingTask
from .utils import (
    extract_audio, transcribe_video, summarize_text, 
    detect_highlights, create_short_video, get_video_duration
)

logger = logging.getLogger(__name__)


def process_video_transcription_sync(video_id):
    """
    Synchronous version of video transcription for direct calls.
    """
    try:
        # Get video instance
        video = Video.objects.get(id=video_id)
        
        # Update status
        video.status = 'processing'
        video.processing_progress = 10
        video.save(update_fields=['status', 'processing_progress', 'updated_at'])
        
        # Step 1: Extract audio
        audio_path = extract_audio(video.original_file.path)
        
        # Step 2: Transcribe
        video.processing_progress = 30
        video.save(update_fields=['processing_progress', 'updated_at'])
        
        transcript = transcribe_video(audio_path)
        
        # Save transcript
        transcript_obj = Transcript.objects.create(
            video=video,
            language=transcript.get('language', 'en'),
            full_text=transcript.get('text', ''),
            json_data=transcript,
            word_timestamps=transcript.get('word_timestamps', [])
        )
        
        # Update progress
        video.processing_progress = 70
        video.save(update_fields=['processing_progress', 'updated_at'])
        
        # Generate summary
        summary_text = summarize_text(transcript.get('text', ''))
        Summary.objects.create(
            video=video,
            summary_type='full',
            title='Full Summary',
            content=summary_text
        )
        
        # Detect highlights
        try:
            highlights = detect_highlights(transcript_obj)
            for highlight in highlights:
                HighlightSegment.objects.create(
                    video=video,
                    start_time=highlight['start'],
                    end_time=highlight['end'],
                    importance_score=highlight.get('score', 0.5),
                    transcript_snippet=highlight.get('text', '')
                )
        except Exception as e:
            logger.warning(f"Highlight detection failed: {str(e)}")
        
        # Complete
        video.status = 'completed'
        video.processing_progress = 100
        video.processed_at = timezone.now()
        video.save(update_fields=['status', 'processing_progress', 'processed_at', 'updated_at'])
        
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


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_video_transcription(self, video_id):
    """
    Main task for processing video transcription.
    Steps:
    1. Extract audio from video
    2. Transcribe audio using Whisper
    3. Update video status
    """
    try:
        # Get video instance
        video = Video.objects.get(id=video_id)
        
        # Update status
        video.status = 'processing'
        video.processing_progress = 10
        video.save(update_fields=['status', 'processing_progress', 'updated_at'])
        
        # Update task status
        task = ProcessingTask.objects.filter(task_id=self.request.id).first()
        if task:
            task.mark_started()
            task.progress = 10
            task.message = 'Extracting audio from video'
            task.save(update_fields=['status', 'progress', 'message'])
        
        # Step 1: Extract audio
        audio_path = extract_audio(video.original_file.path)
        
        video.processing_progress = 30
        video.save(update_fields=['processing_progress', 'updated_at'])
        
        if task:
            task.progress = 30
            task.message = 'Transcribing audio'
            task.save(update_fields=['progress', 'message'])
        
        # Step 2: Transcribe audio
        transcript_data = transcribe_video(audio_path)
        
        video.processing_progress = 80
        video.save(update_fields=['processing_progress', 'updated_at'])
        
        if task:
            task.progress = 80
            task.message = 'Saving transcript'
            task.save(update_fields=['progress', 'message'])
        
        # Step 3: Save transcript to database
        with transaction.atomic():
            Transcript.objects.create(
                video=video,
                language=transcript_data.get('language', 'en'),
                full_text=transcript_data['text'],
                json_data=transcript_data['segments'],
                word_timestamps=transcript_data.get('word_timestamps')
            )
        
        # Update video status
        video.status = 'completed'
        video.processing_progress = 100
        video.processed_at = timezone.now()
        video.save(update_fields=['status', 'processing_progress', 'processed_at', 'updated_at'])
        
        if task:
            task.mark_completed()
        
        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        logger.info(f"Transcription completed for video {video_id}")
        return {'status': 'completed', 'video_id': str(video_id)}
        
    except Video.DoesNotExist:
        logger.error(f"Video {video_id} not found")
        return {'status': 'error', 'message': 'Video not found'}
    
    except Exception as e:
        logger.error(f"Transcription failed for video {video_id}: {str(e)}")
        
        # Update video status
        try:
            video = Video.objects.get(id=video_id)
            video.status = 'failed'
            video.error_message = str(e)
            video.save(update_fields=['status', 'error_message', 'updated_at'])
            
            # Update task status
            if task:
                task.mark_failed(str(e))
                
        except Exception:
            pass
        
        # Retry if retries available
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {'status': 'error', 'message': str(e)}


@shared_task(bind=True, max_retries=2, default_retry_delay=120)
def generate_summary(self, video_id, summary_type, max_length=None, min_length=None):
    """
    Generate summary for a video transcript.
    
    summary_type: 'full', 'bullet', 'short', 'timestamps'
    """
    try:
        video = Video.objects.get(id=video_id)
        
        # Get latest transcript
        transcript = video.transcripts.order_by('-created_at').first()
        if not transcript:
            raise ValueError("No transcript found for this video")
        
        # Update task status
        task = ProcessingTask.objects.filter(task_id=self.request.id).first()
        if task:
            task.mark_started()
            task.message = f'Generating {summary_type} summary'
            task.save(update_fields=['status', 'message'])
        
        # Generate summary
        summary_result = summarize_text(
            transcript.full_text,
            summary_type=summary_type,
            max_length=max_length,
            min_length=min_length
        )
        
        # Save summary
        with transaction.atomic():
            summary = Summary.objects.create(
                video=video,
                summary_type=summary_type,
                title=summary_result.get('title', ''),
                content=summary_result['content'],
                key_topics=summary_result.get('key_topics'),
                model_used=summary_result.get('model_used', 'facebook/bart-large-cnn'),
                generation_time=summary_result.get('generation_time', 0)
            )
        
        if task:
            task.mark_completed()
        
        logger.info(f"Summary generated for video {video_id}, type: {summary_type}")
        return {
            'status': 'completed', 
            'video_id': str(video_id), 
            'summary_id': str(summary.id)
        }
        
    except Video.DoesNotExist:
        logger.error(f"Video {video_id} not found")
        return {'status': 'error', 'message': 'Video not found'}
    
    except Exception as e:
        logger.error(f"Summary generation failed for video {video_id}: {str(e)}")
        
        try:
            if task:
                task.mark_failed(str(e))
        except Exception:
            pass
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {'status': 'error', 'message': str(e)}


@shared_task(bind=True, max_retries=2, default_retry_delay=120)
def detect_video_highlights(self, video_id):
    """
    Detect highlight segments in video using transcript analysis.
    """
    try:
        video = Video.objects.get(id=video_id)
        
        # Get transcript
        transcript = video.transcripts.order_by('-created_at').first()
        if not transcript:
            raise ValueError("No transcript found for this video")
        
        # Update task status
        task = ProcessingTask.objects.filter(task_id=self.request.id).first()
        if task:
            task.mark_started()
            task.message = 'Detecting highlight segments'
            task.save(update_fields=['status', 'message'])
        
        # Detect highlights
        highlights = detect_highlights(transcript)
        
        # Save highlights
        with transaction.atomic():
            for highlight in highlights:
                HighlightSegment.objects.create(
                    video=video,
                    start_time=highlight['start_time'],
                    end_time=highlight['end_time'],
                    importance_score=highlight['importance_score'],
                    reason=highlight['reason'],
                    transcript_snippet=highlight.get('transcript_snippet', '')
                )
        
        if task:
            task.mark_completed()
        
        logger.info(f"Highlights detected for video {video_id}")
        return {
            'status': 'completed',
            'video_id': str(video_id),
            'highlights_count': len(highlights)
        }
        
    except Exception as e:
        logger.error(f"Highlight detection failed for video {video_id}: {str(e)}")
        
        try:
            if task:
                task.mark_failed(str(e))
        except Exception:
            pass
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {'status': 'error', 'message': str(e)}


@shared_task(bind=True, max_retries=2, default_retry_delay=180)
def generate_short_video(
    self, video_id, max_duration=60.0, style='default',
    include_music=False, caption_style='default', font_size=24
):
    """
    Generate a short video from highlights.
    """
    try:
        video = Video.objects.get(id=video_id)
        
        # Get highlights sorted by importance
        highlights = video.highlight_segments.filter(
            used_in_short=False
        ).order_by('-importance_score')[:10]
        
        if not highlights.exists():
            raise ValueError("No highlights found for this video")
        
        # Update task status
        task = ProcessingTask.objects.filter(task_id=self.request.id).first()
        if task:
            task.mark_started()
            task.message = 'Creating short video'
            task.save(update_fields=['status', 'message'])
        
        # Calculate total duration and select segments
        selected_segments = []
        total_duration = 0
        
        for highlight in highlights:
            segment_duration = highlight.end_time - highlight.start_time
            if total_duration + segment_duration <= max_duration:
                selected_segments.append(highlight)
                total_duration += segment_duration
        
        if not selected_segments:
            raise ValueError("No segments fit within max_duration")
        
        # Update task progress
        if task:
            task.progress = 30
            task.message = 'Processing video segments'
            task.save(update_fields=['progress', 'message'])
        
        # Create short video
        short_video_path = create_short_video(
            video.original_file.path,
            selected_segments,
            style=style,
            caption_style=caption_style,
            font_size=font_size
        )
        
        if task:
            task.progress = 80
            task.message = 'Saving short video'
            task.save(update_fields=['progress', 'message'])
        
        # Save short video record
        from django.core.files import File
        
        with open(short_video_path, 'rb') as f:
            short_video = ShortVideo.objects.create(
                video=video,
                file=File(f, name=os.path.basename(short_video_path)),
                duration=total_duration,
                style=style,
                include_music=include_music,
                caption_style=caption_style,
                font_size=font_size,
                status='completed'
            )
        
        # Mark highlights as used
        for highlight in selected_segments:
            highlight.used_in_short = True
            highlight.save(update_fields=['used_in_short'])
        
        if task:
            task.mark_completed()
        
        logger.info(f"Short video generated for video {video_id}")
        return {
            'status': 'completed',
            'video_id': str(video_id),
            'short_video_id': str(short_video.id),
            'duration': total_duration
        }
        
    except Exception as e:
        logger.error(f"Short video generation failed for video {video_id}: {str(e)}")
        
        try:
            if task:
                task.mark_failed(str(e))
        except Exception:
            pass
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {'status': 'error', 'message': str(e)}


@shared_task
def cleanup_old_files():
    """
    Cleanup task to remove old files and database records.
    Run daily via Celery beat.
    """
    from datetime import timedelta
    from django.core.files.storage import default_storage
    
    # Delete files older than 7 days
    cutoff_date = timezone.now() - timedelta(days=7)
    
    old_videos = Video.objects.filter(created_at__lt=cutoff_date)
    
    for video in old_videos:
        try:
            # Delete files
            if video.original_file:
                video.original_file.delete(save=False)
            
            # Delete related files
            for transcript in video.transcripts.all():
                # Cleanup if there are file references
                pass
            
            for short in video.short_videos.all():
                if short.file:
                    short.file.delete(save=False)
                if short.thumbnail:
                    short.thumbnail.delete(save=False)
            
            logger.info(f"Cleaned up video {video.id}")
        except Exception as e:
            logger.error(f"Cleanup failed for video {video.id}: {str(e)}")
    
    # Delete old task records
    ProcessingTask.objects.filter(
        created_at__lt=cutoff_date,
        status='completed'
    ).delete()
    
    return {'status': 'completed', 'cleaned_videos': old_videos.count()}
