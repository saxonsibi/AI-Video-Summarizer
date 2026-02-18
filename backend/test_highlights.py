#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_video_summarizer.settings')
django.setup()

from videos.models import Video
from videos.utils import detect_highlights

v = Video.objects.get(id='5ab2d437-aa87-4064-bbec-87139f170254')
t = v.transcripts.first()

print('Transcript ID:', t.id)
print('json_data keys:', t.json_data.keys())

jd = t.json_data
segments = jd.get('segments', [])
print('Segments count:', len(segments))
if segments:
    print('First segment keys:', segments[0].keys())
    print('First segment:', segments[0])

# Now test detect_highlights
print('\nTesting detect_highlights...')
highlights = detect_highlights(t)
print('Detected highlights:', len(highlights))
if highlights:
    print('First highlight:', highlights[0])
