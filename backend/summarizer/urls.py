"""
URL configuration for summarizer app
"""

from django.urls import path
from .views import SummarizeTextView, health_check

urlpatterns = [
    path('summarize/', SummarizeTextView.as_view(), name='summarize'),
    path('health/', health_check, name='summarizer-health'),
]
