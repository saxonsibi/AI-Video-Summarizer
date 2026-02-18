"""
Django signals for videos app
"""

from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
import logging

logger = logging.getLogger(__name__)


@receiver(post_save, sender='videos.Video')
def video_saved(sender, instance, created, **kwargs):
    """Handle video model save."""
    if created:
        logger.info(f"New video created: {instance.id} - {instance.title}")
    else:
        logger.info(f"Video updated: {instance.id} - Status: {instance.status}")


@receiver(post_delete, sender='videos.Video')
def video_deleted(sender, instance, **kwargs):
    """Handle video model delete and cleanup files."""
    logger.info(f"Video deleted: {instance.id} - {instance.title}")
    
    # File cleanup is handled in model's delete method


@receiver(post_save, sender='videos.Transcript')
def transcript_saved(sender, instance, created, **kwargs):
    """Handle transcript save."""
    if created:
        logger.info(f"Transcript created for video: {instance.video.id}")
        word_count = instance.get_word_count()
        logger.info(f"Transcript word count: {word_count}")


@receiver(post_save, sender='videos.Summary')
def summary_saved(sender, instance, created, **kwargs):
    """Handle summary save."""
    if created:
        logger.info(f"Summary created: {instance.summary_type} for video {instance.video.id}")


@receiver(post_save, sender='videos.ShortVideo')
def short_video_saved(sender, instance, created, **kwargs):
    """Handle short video save."""
    if created:
        logger.info(f"Short video created: {instance.id} for video {instance.video.id}")
