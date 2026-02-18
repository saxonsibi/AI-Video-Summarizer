"""
URL configuration for chatbot app
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ChatSessionViewSet, ChatbotView, VideoIndexViewSet

router = DefaultRouter()
router.register(r'sessions', ChatSessionViewSet, basename='chat-session')
router.register(r'indices', VideoIndexViewSet, basename='video-index')

urlpatterns = [
    path('chat/', ChatbotView.as_view(), name='chatbot'),
    path('', include(router.urls)),
]
