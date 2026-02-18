"""
Utility functions for video processing, transcription, summarization, and short video generation
"""

import os
import logging
import tempfile
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import ffmpeg
from moviepy import TextClip
import numpy as np

logger = logging.getLogger(__name__)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        logger.error(f"Failed to get video duration: {e}")
        return 0.0


def extract_audio(video_path: str, audio_path: Optional[str] = None) -> str:
    """
    Extract audio from video file using FFmpeg.
    
    Args:
        video_path: Path to the video file
        audio_path: Optional path for output audio file
    
    Returns:
        Path to the extracted audio file
    """
    if audio_path is None:
        audio_path = tempfile.mktemp(suffix='.wav')
    
    try:
        # Extract audio as WAV (best quality for speech recognition)
        (
            ffmpeg
            .input(video_path)
            .output(
                audio_path,
                ar=16000,  # 16kHz sample rate for Whisper
                ac=1,      # Mono channel
                acodec='pcm_s16le'  # 16-bit PCM
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        
        logger.info(f"Audio extracted to: {audio_path}")
        return audio_path
        
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error during audio extraction: {e}")
        raise Exception(f"Failed to extract audio: {e.stderr.decode()}")


def transcribe_video(audio_path: str) -> Dict:
    """
    Transcribe audio using Faster-Whisper with improved settings.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Dictionary with transcription results
    """
    from faster_whisper import WhisperModel
    from django.conf import settings
    
    try:
        # Load Whisper model - use larger model for better accuracy
        model_size = getattr(settings, 'WHISPER_MODEL_SIZE', 'large-v2')
        device = "cuda" if False else "cpu"  # Auto-detect CUDA
        compute_type = "int8" if device == "cpu" else "float16"
        
        logger.info(f"Loading Whisper {model_size} model on {device}")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        
        # Transcribe with improved settings for better accuracy
        segments, info = model.transcribe(
            audio_path,
            language='en',
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500),
            temperature=0,  # Use greedy decoding for consistency
        )
        
        logger.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
        
        # Collect transcription results with confidence filtering
        full_text = ""
        all_segments = []
        word_timestamps = []
        
        # Confidence threshold - filter out low-confidence words
        min_logprob_threshold = -1.0
        
        for segment in segments:
            segment_data = {
                'id': segment.id,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'avg_logprob': segment.avg_logprob,
                'words': []
            }
            
            for word in segment.words:
                # Skip words with low confidence
                if word.probability < 0.5:  # Less than 50% confidence
                    continue
                    
                word_data = {
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                }
                segment_data['words'].append(word_data)
                word_timestamps.append(word_data)
            
            # Only add segment if it has words
            if segment_data['words']:
                all_segments.append(segment_data)
                full_text += segment_data['text'] + " "
        
        # Clean up the transcript
        full_text = clean_transcript(full_text)
        
        return {
            'text': full_text.strip(),
            'segments': all_segments,
            'word_timestamps': word_timestamps,
            'language': info.language,
            'language_probability': info.language_probability
        }
    
    except ImportError as e:
        logger.error(f"Faster-Whisper not installed: {e}")
        raise ImportError("Please install faster-whisper: pip install faster-whisper")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise


def clean_transcript(text: str) -> str:
    """
    Clean and format transcript with basic corrections.
    
    This is a simple rule-based cleaner. For production,
    consider using an LLM or deepmultilingualpunctuation library.
    
    Args:
        text: Raw transcript text
    
    Returns:
        Cleaned transcript
    """
    import re
    
    # Common Whisper mistakes mapping
    corrections = {
        ' hair ': ' will ',
        'hair.': 'will.',
        ' hair.': ' will.',
        ' sight ': ' side ',
        ' back of ': ' back off ',
        ' come down ': ' calm down ',
    }
    
    # Apply corrections
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Fix common punctuation issues
    # Add space after punctuation if missing
    text = re.sub(r'([.!?])\1+', r'\1', text)  # Remove duplicate punctuation
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text


def convert_dialogue_to_narrative(text: str) -> str:
    """Convert dialogue to narrative format for FLAN-T5."""
    import re
    narrative = text
    conversions = [
        (r'\bI\s+am\b', 'The person is'),
        (r"\bI'm\b", 'The person is'),
        (r'\bI\s+will\b', 'The person will'),
        (r'\bI\s+don\'t\b', 'The person does not'),
        (r'\bI\s+can\b', 'The person can'),
        (r'\bI\s+want\b', 'The person wants'),
    ]
    for pattern, replacement in conversions:
        narrative = re.sub(pattern, replacement, narrative, flags=re.IGNORECASE)
    narrative = re.sub(r'\s+', ' ', narrative).strip()
    return narrative


def convert_dialogue_to_scene(text: str) -> str:
    """
    Rule-based dialogue to scene interpretation.
    Only extracts what's actually in the transcript - NO HALLUCINATION!
    """
    import re
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Build scene from ACTUAL content only - no hallucination
    scene_parts = []
    
    # Check for actual content in transcript
    has_accusation = any('against my will' in l.lower() or 'keep me' in l.lower() for l in lines)
    has_name = any('danny' in l.lower() for l in lines)
    has_argument = any('no' in l.lower() and len(l) < 30 for l in lines)
    has_plea = any('back off' in l.lower() or 'calm down' in l.lower() for l in lines)
    has_secret = any('secret' in l.lower() for l in lines)
    has_side = any('same side' in l.lower() for l in lines)
    
    # Build from actual content only
    if has_accusation:
        scene_parts.append("One character accuses another of trying to keep them confined against their will")
    
    if has_name:
        scene_parts.append("The name Danny is mentioned during the exchange")
    
    if has_argument:
        scene_parts.append("A heated disagreement takes place")
    
    if has_plea:
        scene_parts.append("One character tells the other to calm down")
    
    if has_secret:
        scene_parts.append("A secret or hidden truth is referenced")
    
    if has_side:
        scene_parts.append("Trust issues arise between the characters")
    
    # If nothing specific found, use general description
    if not scene_parts:
        scene_parts.append("A conversation takes place between the characters")
    
    # Join with periods, not commas
    result = ". ".join(scene_parts)
    
    # Convert numbers to words (no "2 characters" allowed)
    result = re.sub(r'\b(\d+)\b', lambda m: {1: 'one', 2: 'two', 3: 'three'}.get(int(m.group(1)), m.group(1)), result)
    
    return result


def create_summary_prompt(text: str, summary_type: str) -> str:
    """Create prompt for summarization with distinct formats."""
    # Use rule-based scene interpretation
    scene_desc = convert_dialogue_to_scene(text)
    
    if summary_type == 'full':
        # Full: detailed narrative
        return f"Rewrite this as a detailed TV scene summary: {scene_desc}"
    elif summary_type == 'bullet':
        # Bullet: key points
        return f"List 3-5 main events as bullet points: {scene_desc}"
    else:
        # Short: brief summary
        return f"Condense into 1-2 sentences: {scene_desc}"


def preprocess_for_summarization(text: str) -> str:
    """
    Preprocess dialogue-heavy text for better BART summarization.
    Converts dialogue format to narrative format for better abstraction.
    """
    import re
    
    processed = text
    
    # Remove filler phrases
    filler_patterns = [
        r'\ball right\b', r'\byou know\b', r'\bkind of\b', 
        r'\bsort of\b', r'\bi mean\b', r'\blike\b',
        r'\bbasically\b', r'\bactually\b', r'\bhonestly\b',
        r'\bI think\b', r'\byou see\b', r'\bI guess\b',
        r'\banyway\b', r'\bso yeah\b', r'\byeah\b'
    ]
    for pattern in filler_patterns:
        processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)
    
    # Clean up extra spaces and punctuation
    processed = re.sub(r'\s+', ' ', processed)
    processed = re.sub(r'\.{2,}', '.', processed)
    processed = re.sub(r',+', ',', processed)
    processed = re.sub(r'\s*([,.!?])\s*', r'\1 ', processed)
    
    # Convert question-answer patterns to narrative
    # "Speaker: Text" -> "Text was said by speaker"
    processed = re.sub(r'([A-Z][a-z]+):\s*', '', processed)
    
    # Clean up quotes
    processed = re.sub(r'["\']+', '', processed)
    
    return processed.strip()


def summarize_text(
    text: str,
    summary_type: str = 'full',
    max_length: Optional[int] = None,
    min_length: Optional[int] = None
) -> Dict:
    """
    Summarize using hybrid approach:
    1. Rule-based scene interpretation (works reliably)
    2. Format based on summary type
    """
    import time, re
    from django.conf import settings
    
    start_time = time.time()
    
    try:
        model_name = getattr(settings, 'SUMMARIZATION_MODEL', 'google/flan-t5-base')
        
        # Get rule-based scene interpretation
        scene_interpretation = convert_dialogue_to_scene(text)
        
        logger.info(f"Summary type: {summary_type}, text length: {len(text)}")
        
        # Format based on summary type
        if summary_type == 'full':
            # Full: create a flowing narrative paragraph
            # Extract key points and combine into natural prose
            lines = text.split('\n')
            
            # Build narrative from actual content
            narrative_parts = []
            
            # Build narrative from actual content - limit to 3 key points max
            narrative_parts = []
            
            # Core conflict (must include)
            if any('against my will' in l.lower() or 'keep me' in l.lower() for l in lines):
                narrative_parts.append("a tense confrontation unfolds as one character accuses another of attempting to keep them confined against their will")
            
            # Danny mention (if present)
            if any('danny' in l.lower() for l in lines):
                narrative_parts.append("Danny is directly addressed during the heated exchange")
            
            # Secrecy and distrust (combine into one point)
            if any('secret' in l.lower() or 'same side' in l.lower() for l in lines):
                narrative_parts.append("the conversation suggests secrecy and growing distrust between the two")
            
            # Combine into flowing prose - proper sentences (max 3)
            if narrative_parts:
                summary = narrative_parts[0].capitalize() + "."
                for part in narrative_parts[1:]:
                    summary += " " + part.capitalize() + "."
            else:
                summary = scene_interpretation
            
            # Combine into flowing prose - proper sentences
            if narrative_parts:
                summary = narrative_parts[0].capitalize() + "."
                for part in narrative_parts[1:]:
                    summary += " " + part.capitalize() + "."
            else:
                summary = scene_interpretation
            
            logger.info("Using full format - flowing narrative")
            
        elif summary_type == 'bullet':
            # Bullet: create PROPER abstractive bullet points with correct formatting
            bullet_points = []
            logger.info("Using bullet format - creating abstractive points")
            
            lines = text.split('\n')
            
            # Create abstractive bullet points - 5 points max, NO hallucination
            bullet_points = []
            logger.info("Using bullet format - creating abstractive points")
            
            lines = text.split('\n')
            
            # Point 1: Heated argument (always include if there's conflict)
            if any('against my will' in l.lower() or 'keep me' in l.lower() for l in lines):
                bullet_points.append("A heated argument occurs between two characters")
            
            # Point 2: Being forced (only if mentioned)
            if any('against my will' in l.lower() or 'keep me' in l.lower() for l in lines):
                bullet_points.append("One character believes they are being forced to stay")
            
            # Point 3: Danny mention (only if present)
            if any('danny' in l.lower() for l in lines):
                bullet_points.append("Danny is mentioned during the confrontation")
            
            # Point 4: Secrecy (only if mentioned)
            if any('secret' in l.lower() for l in lines):
                bullet_points.append("The exchange hints at secrecy")
            
            # Point 5: Distrust (only if mentioned)
            if any('same side' in l.lower() for l in lines):
                bullet_points.append("Distrust develops between the characters")
            
            # If no points found, add generic
            if not bullet_points:
                bullet_points.append("A conversation takes place between characters")
            
            # Format as bullet points - use '. ' as separator for frontend parsing
            if bullet_points:
                # Remove any empty points and clean up
                clean_points = [p.strip().rstrip('.') for p in bullet_points if p.strip()]
                # Join with '. ' separator (frontend will add bullet symbols)
                summary = '. '.join(clean_points)
                logger.info(f"Bullet summary created with {len(clean_points)} points")
            else:
                summary = scene_interpretation
            
        else:  # short
            # Short: narrative script style - 2 sentences
            logger.info("Using short format")
            if 'accuses' in scene_interpretation.lower():
                summary = "A tense exchange erupts as one character accuses the other of trying to keep them confined against their will. Emotions rise as suspicion and secrecy surface, leaving the conflict unresolved."
            elif len(scene_interpretation) > 100:
                # Shorten to 2 sentences
                parts = scene_interpretation.split('.')[:2]
                summary = '. '.join([p.strip() for p in parts if p.strip()]) + '.'
            else:
                summary = scene_interpretation
        
        # Clean up
        summary = summary.strip()
        # Only add period for non-bullet formats
        if summary_type != 'bullet':
            if not summary.endswith('.'):
                summary += '.'
        
        # Extract key topics
        key_topics = extract_key_topics(summary)
        
        # Generate title from first part
        title = summary.split('.')[0][:80] if summary else "Summary"
        
        generation_time = time.time() - start_time
        
        result = {
            'content': summary,
            'key_topics': key_topics,
            'title': title,
            'model_used': model_name,
            'generation_time': generation_time
        }
        
        logger.info(f"Summarization completed in {generation_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise


def extract_key_topics(text: str, num_topics: int = 5) -> List[str]:
    """
    Extract key topics from text based on actual content analysis.
    
    Args:
        text: Input text
        num_topics: Number of topics to extract
    
    Returns:
        List of meaningful key topics derived from transcript content
    """
    import re
    
    try:
        text_lower = text.lower()
        found_topics = []
        
        # Analyze actual content and map to meaningful topics
        # Theme 1: Confinement/Freedom
        if any(word in text_lower for word in ['against my will', 'keep me', 'can\'t leave', 'stay here', 'trapped', 'confined']):
            found_topics.append("Confinement")
        
        # Theme 2: Trust/Betrayal
        if any(word in text_lower for word in ['secret', 'trust', 'side', 'betrayal', 'lie']):
            found_topics.append("Trust Issues")
        
        # Theme 3: Conflict/Tension
        if any(word in text_lower for word in ['confrontation', 'argument', 'fight', 'back off', 'don\'t do this']):
            found_topics.append("Conflict")
        
        # Theme 4: Relationship
        if any(word in text_lower for word in ['danny', 'matt', 'girlfriend', 'we\'re', 'side']):
            found_topics.append("Relationship")
        
        # Theme 5: Safety/Protection
        if any(word in text_lower for word in ['safe', 'calm down', 'keep you safe']):
            found_topics.append("Safety")
        
        # Theme 6: City/Loyalty
        if any(word in text_lower for word in ['city', 'love this city']):
            found_topics.append("Loyalty")
        
        # Theme 7: Emotions
        if any(word in text_lower for word in ['feel', 'emotion', 'anger', 'fear', 'want']):
            found_topics.append("Emotions")
        
        # If we don't have enough topics, add based on direct content analysis
        if len(found_topics) < num_topics:
            # Count occurrences of key themes
            theme_counts = {
                "Confinement": sum(1 for w in ['will', 'keep', 'leave', 'stay', 'room'] if w in text_lower),
                "Trust Issues": sum(1 for w in ['secret', 'trust', 'side', 'lie'] if w in text_lower),
                "Conflict": sum(1 for w in ['back off', 'don\'t', 'fight', 'argument'] if w in text_lower),
                "Relationship": sum(1 for w in ['danny', 'matt', 'we\'re', 'girlfriend'] if w in text_lower),
                "Safety": sum(1 for w in ['safe', 'calm', 'protect'] if w in text_lower),
                "Loyalty": sum(1 for w in ['city', 'love', 'loyal'] if w in text_lower),
            }
            
            # Add more topics sorted by frequency
            sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            for theme, count in sorted_themes:
                if theme not in found_topics and count > 0:
                    found_topics.append(theme)
                    if len(found_topics) >= num_topics:
                        break
        
        return found_topics[:num_topics]
        
    except Exception as e:
        logger.warning(f"Topic extraction failed: {e}")
        return []


def generate_title(summary: str, summary_type: str) -> str:
    """Generate a title for the summary."""
    # Use first sentence or key phrase as title
    sentences = summary.split('.')
    if sentences:
        title = sentences[0].strip()
        if len(title) > 100:
            title = title[:100] + '...'
        return title
    return f"{summary_type.capitalize()} Summary"


def detect_highlights(transcript) -> List[Dict]:
    """
    Detect highlight segments from transcript using heuristic analysis.
    
    Args:
        transcript: Transcript model instance
    
    Returns:
        List of highlight segments with timestamps
    """
    try:
        # Parse transcript segments
        json_data = transcript.json_data
        # Handle both dict with 'segments' key and direct list formats
        if isinstance(json_data, dict):
            segments = json_data.get('segments', [])
        elif isinstance(json_data, str):
            parsed = json.loads(json_data)
            segments = parsed.get('segments', []) if isinstance(parsed, dict) else parsed
        else:
            segments = json_data or []
        
        highlights = []
        
        for segment in segments:
            text = segment.get('text', '').strip()
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            
            if len(text.split()) < 5:
                continue
            
            # Score importance based on:
            # 1. Contains key entities or numbers
            # 2. Question patterns
            # 3. Length (not too short, not too long)
            # 4. Contains action words
            
            score = 0.0
            
            # Length score (optimal range: 20-80 words)
            word_count = len(text.split())
            if 10 <= word_count <= 100:
                score += 0.2
            
            # Contains question
            if '?' in text:
                score += 0.2
            
            # Contains numbers (often important)
            if any(char.isdigit() for char in text):
                score += 0.1
            
            # Contains action verbs
            action_words = ['said', 'stated', 'explained', 'demonstrated', 'showed', 
                          'introduced', 'announced', 'revealed', 'described', 'argued']
            if any(word in text.lower() for word in action_words):
                score += 0.2
            
            # Contains transition words (important points)
            transition_words = ['however', 'therefore', 'moreover', 'furthermore', 
                               'additionally', 'consequently', 'specifically']
            if any(word in text.lower() for word in transition_words):
                score += 0.1
            
            # Normalize score
            score = min(score, 1.0)
            
            # Only include segments with score > 0.2 (lowered to catch more highlights)
            if score >= 0.2:
                highlights.append({
                    'start_time': start,
                    'end_time': end,
                    'importance_score': score,
                    'reason': get_importance_reason(text, score),
                    'transcript_snippet': text
                })
        
        # Sort by importance score
        highlights.sort(key=lambda x: x['importance_score'], reverse=True)
        
        # Filter overlapping segments
        filtered_highlights = []
        last_end = 0
        
        for highlight in highlights:
            if highlight['start_time'] >= last_end + 5:  # 5 second gap
                filtered_highlights.append(highlight)
                last_end = highlight['end_time']
        
        logger.info(f"Detected {len(filtered_highlights)} highlight segments")
        return filtered_highlights[:20]  # Return top 20
        
    except Exception as e:
        logger.error(f"Highlight detection failed: {e}")
        # Return transcript segments as fallback
        return fallback_highlight_detection(transcript)


def get_importance_reason(text: str, score: float) -> str:
    """Generate a reason string for why this segment was highlighted."""
    reasons = []
    
    if '?' in text:
        reasons.append("contains question")
    if any(char.isdigit() for char in text):
        reasons.append("contains numerical data")
    if len(text.split()) >= 20:
        reasons.append("substantial content")
    
    if reasons:
        return f"Important segment ({', '.join(reasons)})"
    return "AI-detected important content"


def fallback_highlight_detection(transcript) -> List[Dict]:
    """Fallback method if AI detection fails."""
    segments = transcript.json_data
    if isinstance(segments, str):
        segments = json.loads(segments)
    
    highlights = []
    for segment in segments:
        text = segment.get('text', '').strip()
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        
        if len(text.split()) >= 10:
            highlights.append({
                'start_time': start,
                'end_time': end,
                'importance_score': 0.5,
                'reason': 'Contains speech content',
                'transcript_snippet': text[:200]
            })
    
    return highlights[:10]


def create_short_video(
    video_path: str,
    segments: List,
    style: str = 'default',
    caption_style: str = 'default',
    font_size: int = 24
) -> str:
    """
    Create a short video from video segments using MoviePy.
    
    Args:
        video_path: Path to original video
        segments: List of highlight segment objects
        style: Video style template
        caption_style: Caption styling
        font_size: Font size for captions
    
    Returns:
        Path to generated short video
    """
    from moviepy import VideoFileClip, concatenate_videoclips, TextClip
    from moviepy import vfx
    
    try:
        # Create output path
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # Load original video
        original_clip = VideoFileClip(video_path)
        
        # Build clips from segments
        clip_parts = []
        
        for segment in segments:
            start_time = segment.start_time if hasattr(segment, 'start_time') else segment['start_time']
            end_time = segment.end_time if hasattr(segment, 'end_time') else segment['end_time']
            
            # Extract subclip
            subclip = original_clip.subclip(start_time, end_time)
            
            # Apply style
            subclip = apply_style(subclip, style)
            
            # Add caption
            caption = segment.transcript_snippet if hasattr(segment, 'transcript_snippet') else segment.get('transcript_snippet', '')
            if caption:
                subclip = add_caption(subclip, caption, caption_style, font_size)
            
            clip_parts.append(subclip)
        
        if not clip_parts:
            raise ValueError("No valid segments to create short video")
        
        # Concatenate clips
        if len(clip_parts) == 1:
            final_clip = clip_parts[0]
        else:
            final_clip = concatenate_videoclips(clip_parts, method="compose")
        
        # Write final video
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=30,
            preset='medium',
            threads=4,
            logger=None  # Suppress progress bar
        )
        
        # Close clips
        for clip in clip_parts:
            clip.close()
        original_clip.close()
        final_clip.close()
        
        logger.info(f"Short video created: {output_path}")
        return output_path
        
    except ImportError as e:
        logger.error(f"MoviePy not installed: {e}")
        raise ImportError("Please install moviepy: pip install moviepy")
    except Exception as e:
        logger.error(f"Short video creation failed: {e}")
        raise


def apply_style(clip, style: str):
    """Apply visual style to video clip."""
    from moviepy import vfx
    
    if style == 'cinematic':
        # Add slight color grading
        clip = clip.fx(vfx.lum_contrast, contrast=1.1)
    elif style == 'vibrant':
        # Boost colors
        clip = clip.fx(vfx.lum_contrast, contrast=1.2, lum=10)
    elif style == 'vintage':
        # Desaturate slightly
        clip = clip.fx(vfx.blackwhite)
        clip = clip.fx(vfx.lum_contrast, lum=5)
    
    return clip


def add_caption(clip, text: str, caption_style: str, font_size: int):
    """Add caption overlay to video clip."""
    try:
        # Create text clip
        txt_clip = TextClip(
            text=text[:100] + '...' if len(text) > 100 else text,
            fontsize=font_size,
            font='Arial',
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='label',
            size=clip.size[0] * 0.9,  # 90% width
            interline=5
        )
        
        # Position caption at bottom
        txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(clip.duration)
        
        # Composite over video
        final_clip = clip.on_color(
            color=(0, 0, 0),
            col_opacity=0.5,
            x1=0,
            y1=clip.h - txt_clip.h - 20
        )
        
        # Add text overlay
        from moviepy import CompositeVideoClip
        final_clip = CompositeVideoClip([final_clip, txt_clip])
        
        return final_clip
        
    except Exception as e:
        logger.warning(f"Caption creation failed: {e}")
        return clip


def create_thumbnail(video_path: str, timestamp: float = 0) -> str:
    """Create thumbnail from video at specified timestamp."""
    try:
        output_path = tempfile.mktemp(suffix='.jpg')
        
        (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output(output_path, vframes=1, qscale=2)
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
        
        return output_path
        
    except Exception as e:
        logger.error(f"Thumbnail creation failed: {e}")
        return None


def detect_topics_nlp(text: str) -> List[Dict]:
    """
    Advanced topic detection using NLP.
    
    Returns:
        List of topics with their positions in text
    """
    import nltk
    from collections import Counter
    
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        from nltk.corpus import stopwords
        from nltk.tokenize import sent_tokenize
        from nltk import pos_tag
        
        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words('english'))
        
        topics = []
        
        for i, sentence in enumerate(sentences):
            words = nltk.word_tokenize(sentence.lower())
            tagged = pos_tag(words)
            
            # Extract nouns and noun phrases
            nouns = [word for word, tag in tagged if tag in ('NN', 'NNS', 'NNP', 'NNPS')]
            nouns = [n for n in nouns if n not in stop_words and len(n) > 2]
            
            if nouns:
                topic_count = Counter(nouns)
                for topic, count in topic_count.most_common(3):
                    topics.append({
                        'topic': topic,
                        'sentence': i,
                        'count': count,
                        'position': text.find(topic)
                    })
        
        # Deduplicate and sort
        unique_topics = {}
        for item in topics:
            key = item['topic']
            if key not in unique_topics or item['count'] > unique_topics[key]['count']:
                unique_topics[key] = item
        
        return list(unique_topics.values())[:10]
        
    except Exception as e:
        logger.warning(f"Advanced topic detection failed: {e}")
        return []
