import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_video_summarizer.settings')
django.setup()

from videos.models import Transcript, Summary, Video
from videos.utils import summarize_text
import json

# Get the video and transcript
video = Video.objects.get(id='5ab2d437-aa87-4064-bbec-87139f170254')
transcript = video.transcripts.first()

# Set the corrected full text (script format with proper punctuation and line breaks)
corrected_text = """You're going to try and keep me here against my will.

That's what you're saying, Danny.

But no one loves this city more than I do.

But you can't leave this room.

If you want to do this—

No, I don't want to.

But if I have to...

So you had to keep your girlfriend's secret?

No, we're not.

Danny, we're all on the same side here.

It doesn't feel like it.

Don't do this.

You guys seriously need to back off.

You just stay here.

We can keep you safe, all right?

You just need to calm down.

That's the problem, Matt."""

print("Updating transcript...")
transcript.full_text = corrected_text
transcript.save()

# Delete old summaries
print("Deleting old summaries...")
Summary.objects.filter(video=video).delete()

# Generate new summaries
print("Generating Full Summary...")
full_result = summarize_text(corrected_text, summary_type='full')
Summary.objects.create(
    video=video,
    summary_type='full',
    title='Full Summary',
    content=json.dumps(full_result)
)
print(f"Full: {full_result.get('title', 'N/A')[:50]}...")

print("Generating Bullet Summary...")
bullet_result = summarize_text(corrected_text, summary_type='bullet')
Summary.objects.create(
    video=video,
    summary_type='bullet',
    title='Bullet Points',
    content=json.dumps(bullet_result)
)
print(f"Bullet: {bullet_result.get('title', 'N/A')[:50]}...")

print("Generating Short Summary...")
short_result = summarize_text(corrected_text, summary_type='short')
Summary.objects.create(
    video=video,
    summary_type='short',
    title='Short Script (30-60 sec)',
    content=json.dumps(short_result)
)
print(f"Short: {short_result.get('title', 'N/A')[:50]}...")

print("\n✅ Done! Generated 3 new summaries from corrected transcript.")
