"""
API Views for text summarization
"""

import time
import logging
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view

logger = logging.getLogger(__name__)


class SummarizeTextView(APIView):
    """
    API endpoint for summarizing text.
    Can be used standalone or called from video processing.
    """
    
    def post(self, request):
        """Summarize the provided text."""
        text = request.data.get('text', '')
        summary_type = request.data.get('type', 'full')
        max_length = request.data.get('max_length')
        min_length = request.data.get('min_length')
        
        if not text:
            return Response(
                {'error': 'Text is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            from videos.utils import summarize_text
            
            result = summarize_text(
                text=text,
                summary_type=summary_type,
                max_length=max_length,
                min_length=min_length
            )
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return Response(
                {'error': 'Summarization failed', 'detail': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@api_view(['GET'])
def health_check(request):
    """Health check endpoint for summarizer service."""
    return Response({
        'status': 'healthy',
        'service': 'summarizer',
        'models': {
            'summarization': 'facebook/bart-large-cnn',
            'extraction': 'all-MiniLM-L6-v2'
        }
    })
