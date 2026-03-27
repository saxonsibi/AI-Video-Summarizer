"""
ASGI config for VideoIQ AI Video Intelligence System project.
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'videoiq.settings')

application = get_asgi_application()
