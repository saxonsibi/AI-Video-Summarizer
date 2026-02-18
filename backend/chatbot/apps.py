from django.apps import AppConfig


class ChatbotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chatbot'
    verbose_name = 'Video Chatbot'
    
    def ready(self):
        # Initialize chatbot components
        pass
