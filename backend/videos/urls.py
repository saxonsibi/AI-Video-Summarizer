"""
URL configuration for videos app - Simplified version
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import VideoUploadView, VideoViewSet, TranscriptViewSet

router = DefaultRouter()
router.register(r'transcripts', TranscriptViewSet, basename='transcript')
router.register(r'', VideoViewSet, basename='video')

urlpatterns = [
    path('upload/', VideoUploadView.as_view(), name='video-upload'),
    path('', include(router.urls)),
]
