"""
ASGI config for AI Video Summarizer project.
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_video_summarizer.settings')

application = get_asgi_application()
