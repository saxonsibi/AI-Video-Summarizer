#!/usr/bin/env python
"""
Test script to verify transcription works correctly.
Run this from the backend directory:
    python test_transcription.py
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'videoiq.settings')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
django.setup()

from videos.utils import transcribe_video
import logging

logging.basicConfig(level=logging.INFO)

def test_transcription():
    """Test transcription on a sample video."""
    # Find a video file
    media_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media', 'videos', 'original')
    
    if not os.path.exists(media_dir):
        print(f"Media directory not found: {media_dir}")
        return
    
    # Find first video
    videos = [f for f in os.listdir(media_dir) if f.endswith('.mp4')]
    if not videos:
        print("No videos found in media/videos/original")
        return
    
    video_path = os.path.join(media_dir, videos[0])
    print(f"Testing with video: {video_path}")
    
    # Extract audio first
    from videos.utils import extract_audio
    audio_path = extract_audio(video_path)
    print(f"Audio extracted to: {audio_path}")
    
    # Now transcribe
    print("Starting transcription...")
    result = transcribe_video(audio_path)
    
    print(f"\n=== RESULTS ===")
    print(f"Language: {result.get('language')}")
    print(f"Text length: {len(result.get('text', ''))}")
    print(f"Number of segments: {len(result.get('segments', []))}")
    print(f"\nFull transcript:\n{result.get('text', '')}")
    
    # Check if we got the full duration
    if result.get('segments'):
        last_segment = result['segments'][-1]
        print(f"\nLast segment ends at: {last_segment.get('end', 0)} seconds")

if __name__ == '__main__':
    test_transcription()
