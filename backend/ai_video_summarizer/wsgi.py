"""
WSGI config for AI Video Summarizer project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_video_summarizer.settings')

application = get_wsgi_application()
