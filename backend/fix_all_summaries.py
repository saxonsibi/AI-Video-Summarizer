import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'videoiq.settings')
django.setup()

from videos.models import Transcript, Summary, Video
from videos.utils import summarize_text
import json

def fix_all_transcripts_and_summaries():
    """Fix all transcripts and regenerate summaries for all videos with transcripts."""
    
    # Get all videos that have transcripts
    videos = Video.objects.filter(transcripts__isnull=False).distinct()
    
    print(f"Found {videos.count()} videos with transcripts")
    
    for video in videos:
        print(f"\n{'='*50}")
        print(f"Processing: {video.title} (ID: {video.id})")
        
        transcript = video.transcripts.first()
        if not transcript:
            print("  No transcript found, skipping...")
            continue
        
        # Get the original transcript text
        original_text = transcript.full_text
        if not original_text:
            print("  Empty transcript, skipping...")
            continue
            
        print(f"  Original transcript length: {len(original_text)} chars")
        
        # Apply corrections to transcript
        corrected_text = original_text
        
        # Common corrections
        corrections = {
            'idea bring together': 'idea to bring together',
            'farn': 'fun',
            'move ': 'lose ',
            'Hair ': 'Will ',
        }
        
        for wrong, correct in corrections.items():
            if wrong.lower() in corrected_text.lower():
                import re
                corrected_text = re.sub(re.escape(wrong), correct, corrected_text, flags=re.IGNORECASE)
                print(f"  Fixed: '{wrong}' -> '{correct}'")
        
        # Update transcript if changed
        if corrected_text != original_text:
            transcript.full_text = corrected_text
            transcript.save()
            print(f"  Transcript updated!")
        else:
            print(f"  No corrections needed")
        
        # Delete old summaries
        old_count = Summary.objects.filter(video=video).count()
        if old_count > 0:
            print(f"  Deleting {old_count} old summaries...")
            Summary.objects.filter(video=video).delete()
        
        # Generate new summaries
        print(f"  Generating new summaries...")
        
        try:
            # Full Summary
            full_result = summarize_text(corrected_text, summary_type='full')
            Summary.objects.create(
                video=video,
                summary_type='full',
                title='Full Summary',
                content=json.dumps(full_result)
            )
            print(f"  [OK] Full summary generated")
            
            # Bullet Summary
            bullet_result = summarize_text(corrected_text, summary_type='bullet')
            Summary.objects.create(
                video=video,
                summary_type='bullet',
                title='Bullet Points',
                content=json.dumps(bullet_result)
            )
            print(f"  [OK] Bullet summary generated")
            
            # Short Summary
            short_result = summarize_text(corrected_text, summary_type='short')
            Summary.objects.create(
                video=video,
                summary_type='short',
                title='Short Script (30-60 sec)',
                content=json.dumps(short_result)
            )
            print(f"  [OK] Short summary generated")
            
        except Exception as e:
            print(f"  [X] Error generating summaries: {e}")
    
    print(f"\n{'='*50}")
    print("Done! All summaries have been regenerated.")

if __name__ == "__main__":
    fix_all_transcripts_and_summaries()
