"""
Custom exception handlers for videos app
"""

import logging
from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Base exception for video processing errors."""
    def __init__(self, message, details=None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class TranscriptionError(VideoProcessingError):
    """Exception raised when transcription fails."""
    pass


class SummarizationError(VideoProcessingError):
    """Exception raised when summarization fails."""
    pass


class ShortVideoGenerationError(VideoProcessingError):
    """Exception raised when short video generation fails."""
    pass


class InvalidVideoFormatError(VideoProcessingError):
    """Exception raised when video format is invalid."""
    pass


def custom_exception_handler(exc, context):
    """
    Custom exception handler for DRF.
    """
    # Call REST framework's default exception handler first
    response = exception_handler(exc, context)
    
    if response is not None:
        response.data['status_code'] = response.status_code
        return response
    
    # Handle custom exceptions
    if isinstance(exc, VideoProcessingError):
        logger.error(f"Video processing error: {exc.message}", exc_info=True)
        return Response(
            {
                'error': exc.message,
                'details': exc.details,
                'status_code': status.HTTP_500_INTERNAL_SERVER_ERROR
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    # Handle unexpected exceptions
    logger.exception(f"Unexpected error: {exc}")
    return Response(
        {
            'error': 'An unexpected error occurred',
            'detail': str(exc) if str(exc) else 'Unknown error',
            'status_code': status.HTTP_500_INTERNAL_SERVER_ERROR
        },
        status=status.HTTP_500_INTERNAL_SERVER_ERROR
    )
