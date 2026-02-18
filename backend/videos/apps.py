from django.apps import AppConfig


class VideosConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'videos'
    verbose_name = 'Video Processing'
    
    def ready(self):
        # Import signals when app is ready
        import videos.signals  # noqa
