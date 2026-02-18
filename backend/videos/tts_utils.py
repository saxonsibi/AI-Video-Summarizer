"""
Text-to-Speech and Audio Summary Utilities
"""
import os
import logging
import uuid
from gtts import gTTS
from django.conf import settings

logger = logging.getLogger(__name__)


def text_to_speech(text: str, lang: str = 'en', slow: bool = False) -> str:
    """
    Convert text to speech using Google TTS.
    
    Args:
        text: Text to convert to speech
        lang: Language code (default: 'en')
        slow: Whether to use slow speech (default: False)
    
    Returns:
        Path to the generated audio file
    """
    try:
        # Create media directory if it doesn't exist
        media_dir = os.path.join(settings.MEDIA_ROOT, 'tts')
        os.makedirs(media_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(media_dir, filename)
        
        # Generate speech
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(filepath)
        
        # Return relative URL
        relative_path = os.path.join('tts', filename)
        logger.info(f"TTS generated: {relative_path}")
        
        return relative_path
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise


def generate_audio_summary(text: str, duration_target: int = 60) -> str:
    """
    Generate an audio summary from text.
    Creates a shorter audio version focused on key points.
    
    Args:
        text: Full transcript or summary text
        duration_target: Target duration in seconds (default: 60)
    
    Returns:
        Path to the generated audio file
    """
    try:
        # Clean and prepare text for audio
        # Remove timestamps and formatting
        import re
        clean_text = re.sub(r'\[[\d:]+\]', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = clean_text.strip()
        
        # If text is too long, create a summary first
        if len(clean_text) > 1000:
            # Use first few sentences for audio summary
            sentences = clean_text.split('.')
            summary_text = []
            current_length = 0
            
            for sentence in sentences:
                if sentence.strip():
                    summary_text.append(sentence.strip())
                    current_length += len(sentence)
                    # Stop at around 500 characters for reasonable audio length
                    if current_length > 500:
                        break
            
            clean_text = '. '.join(summary_text)
            if not clean_text.endswith('.'):
                clean_text += '.'
        
        # Generate speech
        audio_path = text_to_speech(clean_text, slow=False)
        
        return audio_path
        
    except Exception as e:
        logger.error(f"Audio summary generation failed: {e}")
        raise


def generate_podcast_summary(full_text: str, duration_target: int = 60) -> str:
    """
    Generate a podcast-style audio summary.
    Creates an engaging audio summary that sounds more natural.
    
    Args:
        full_text: Full transcript text
        duration_target: Target duration in seconds (default: 60)
    
    Returns:
        Path to the generated audio file
    """
    try:
        from .utils import summarize_text
        
        # Get a short summary for the podcast
        short_result = summarize_text(full_text, summary_type='short')
        summary_content = short_result.get('content', '')
        
        # Create podcast-style introduction
        podcast_text = f"Here's a summary of the video. {summary_content}"
        
        # Generate the audio
        audio_path = text_to_speech(podcast_text, slow=False)
        
        logger.info(f"Podcast summary generated: {audio_path}")
        
        return audio_path
        
    except Exception as e:
        logger.error(f"Podcast summary generation failed: {e}")
        raise
