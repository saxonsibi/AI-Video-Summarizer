"""
URL configuration for AI Video Summarizer project.
"""

from django.http import HttpResponseRedirect
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_yasg.views import get_schema_view
from drf_yasg import openapi


def root_redirect(request):
    """Redirect root URL to frontend or API docs."""
    frontend_url = "http://localhost:5173"
    return HttpResponseRedirect(frontend_url)

# API Documentation
schema_view = get_schema_view(
    openapi.Info(
        title="AI Video Summarizer API",
        default_version='v1',
        description="API for AI-powered video summarization, chatbot, and short video generation",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@example.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
)

urlpatterns = [
    # Redirect root to frontend
    path('', root_redirect, name='root'),
    
    # Admin
    path('admin/', admin.site.urls),
    
    # API Documentation
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    
    # App URLs with API v1 prefix
    path('api/v1/videos/', include('videos.urls')),
    path('api/v1/chatbot/', include('chatbot.urls')),
    path('api/v1/summarizer/', include('summarizer.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
