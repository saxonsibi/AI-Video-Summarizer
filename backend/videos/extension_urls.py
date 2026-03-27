"""
Compatibility API routes for the Chrome extension.

These endpoints provide a stable contract for extension clients while
internally reusing existing videos/chatbot pipelines.
"""

from django.urls import path
from .extension_views import (
    ExtensionSummarizeView,
    ExtensionStatusView,
    ExtensionResultView,
    ExtensionChatView,
)


urlpatterns = [
    path("summarize", ExtensionSummarizeView.as_view(), name="extension-summarize"),
    path("status", ExtensionStatusView.as_view(), name="extension-status"),
    path("result", ExtensionResultView.as_view(), name="extension-result"),
    path("chat", ExtensionChatView.as_view(), name="extension-chat"),
]

