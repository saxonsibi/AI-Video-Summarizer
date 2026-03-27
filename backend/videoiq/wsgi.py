"""
WSGI config for VideoIQ AI Video Intelligence System project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'videoiq.settings')

application = get_wsgi_application()
