#!/usr/bin/env python
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_video_summarizer.settings')
django.setup()

from videos.models import Video

v = Video.objects.get(id='5ab2d437-aa87-4064-bbec-87139f170254')
t = v.transcripts.first()

print('Full text length:', len(t.full_text))
print('Full text preview:', t.full_text[:1000])
