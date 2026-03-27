import os
import sys
import threading

from django.apps import AppConfig
from django.conf import settings


class VideosConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'videos'
    verbose_name = 'Video Processing'
    
    def ready(self):
        # Import signals when app is ready
        import videos.signals  # noqa

        if not getattr(settings, 'ASR_MALAYALAM_WARMUP', False):
            return

        command = next((arg for arg in sys.argv[1:] if not arg.startswith('-')), '')
        if command in {'test', 'migrate', 'makemigrations', 'collectstatic', 'shell', 'dbshell'}:
            return

        if settings.DEBUG and os.environ.get('RUN_MAIN') not in {'true', '1'}:
            return

        def _prewarm():
            try:
                from . import utils as videos_utils
                videos_utils.prewarm_malayalam_asr()
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Malayalam ASR prewarm skipped: %s", exc)

        threading.Thread(target=_prewarm, daemon=True, name='malayalam-asr-prewarm').start()
