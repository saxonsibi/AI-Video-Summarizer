"""
API Views for chatbot app
"""

import logging
import uuid
from django.db import transaction
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ChatSession, ChatMessage, VideoIndex
from .serializers import (
    ChatSessionSerializer, ChatMessageSerializer, ChatMessageCreateSerializer,
    ChatResponseSerializer, SuggestedQuestionsSerializer, VideoIndexSerializer
)
from .rag_engine import ChatbotEngine

logger = logging.getLogger(__name__)


class ChatSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for ChatSession CRUD operations."""
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    
    def get_queryset(self):
        """Filter by video_id if provided."""
        queryset = super().get_queryset()
        video_id = self.request.query_params.get('video_id')
        if video_id:
            queryset = queryset.filter(video_id=video_id)
        return queryset
    
    @action(detail=True, methods=['get'])
    def messages(self, request, pk=None):
        """Get all messages in a session."""
        session = self.get_object()
        messages = session.messages.all()
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['delete'])
    def clear(self, request, pk=None):
        """Clear all messages in a session."""
        session = self.get_object()
        session.messages.all().delete()
        return Response({'status': 'cleared'})


class ChatbotView(APIView):
    """
    Main chatbot API endpoint for asking questions about videos.
    """
    
    def post(self, request):
        """Handle chatbot question."""
        serializer = ChatMessageCreateSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        video_id = serializer.validated_data['video_id']
        message = serializer.validated_data['message']
        session_id = serializer.validated_data.get('session_id')
        
        try:
            # Get or create session
            if session_id:
                session = ChatSession.objects.filter(id=session_id).first()
                if not session:
                    session = ChatSession.objects.create(
                        id=session_id,
                        video_id=video_id,
                        title=f"Chat about video {video_id}"
                    )
            else:
                session = ChatSession.objects.create(
                    video_id=video_id,
                    title=f"Chat about video {video_id}"
                )
            
            # Save user message
            user_msg = ChatMessage.objects.create(
                session=session,
                sender='user',
                message=message
            )
            
            # Get transcript and initialize chatbot
            from videos.models import Video, Transcript
            
            try:
                video = Video.objects.get(id=video_id)
                transcript = Transcript.objects.filter(video=video).order_by('-created_at').first()
                
                if not transcript:
                    return Response(
                        {
                            'error': 'No transcript available for this video',
                            'session_id': str(session.id)
                        },
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                # Initialize chatbot engine
                chatbot = ChatbotEngine(str(video_id))
                
                # Build index if needed
                index_exists = chatbot.initialize()
                if not index_exists:
                    # Build index from transcript - use full_text field
                    if transcript.full_text:
                        # Split full text into segments
                        text = transcript.full_text
                        if isinstance(text, str):
                            # Split by sentences or newlines
                            import re
                            # Split by sentence-ending punctuation
                            segments = re.split(r'(?<=[.!?])\s+', text)
                            segments = [s.strip() for s in segments if s.strip() and len(s.strip()) > 10]
                            
                            # Create segment dictionaries
                            transcript_segments = []
                            for i, seg_text in enumerate(segments):
                                transcript_segments.append({
                                    'text': seg_text,
                                    'start': i * 5,  # Approximate 5 seconds per segment
                                    'end': (i + 1) * 5,
                                    'id': i
                                })
                            
                            chatbot.build_from_transcript(transcript_segments)
                        else:
                            # Use json_data if full_text is not a string
                            chatbot.build_from_transcript(transcript.json_data)
                    elif transcript.word_timestamps:
                        # Use word-level timestamps for better indexing
                        chatbot.build_from_transcript(transcript.word_timestamps)
                    else:
                        # Fallback to segment-level
                        chatbot.build_from_transcript(transcript.json_data)
                
                # Get answer
                result = chatbot.ask(message)
                
                # Save bot message with sources
                bot_msg = ChatMessage.objects.create(
                    session=session,
                    sender='bot',
                    message=result['answer'],
                    referenced_segments=result.get('sources', [])
                )
                
                response_data = {
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'session_id': str(session.id)
                }
                
                # Generate TTS for the answer if requested
                generate_tts = request.data.get('generate_tts', False)
                if generate_tts:
                    try:
                        from videos.tts_utils import text_to_speech
                        tts_path = text_to_speech(result['answer'])
                        response_data['audio_url'] = f"/media/{tts_path}"
                    except Exception as e:
                        logger.warning(f"TTS generation failed: {e}")
                
                if result.get('error'):
                    response_data['error'] = result['error']
                
                return Response(response_data)
                
            except Video.DoesNotExist:
                return Response(
                    {'error': 'Video not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
        except Exception as e:
            logger.error(f"Chatbot error: {e}")
            return Response(
                {
                    'error': 'Failed to process question',
                    'detail': str(e),
                    'session_id': str(session.id) if 'session' in locals() else None
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def get(self, request):
        """Get suggested questions for a video."""
        video_id = request.query_params.get('video_id')
        
        if not video_id:
            return Response(
                {'error': 'video_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        chatbot = ChatbotEngine(str(video_id))
        suggested_questions = chatbot.get_suggested_questions()
        
        return Response({'questions': suggested_questions})


class VideoIndexViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for viewing video indices (read-only)."""
    queryset = VideoIndex.objects.all()
    serializer_class = VideoIndexSerializer
    
    def get_queryset(self):
        """Filter by video_id if provided."""
        queryset = super().get_queryset()
        video_id = self.request.query_params.get('video_id')
        if video_id:
            queryset = queryset.filter(video_id=video_id)
        return queryset
    
    @action(detail=False, methods=['post'])
    def build(self, request):
        """Build index for a video."""
        video_id = request.data.get('video_id')
        
        if not video_id:
            return Response(
                {'error': 'video_id is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            from videos.models import Video, Transcript
            
            video = Video.objects.get(id=video_id)
            transcript = Transcript.objects.filter(video=video).order_by('-created_at').first()
            
            if not transcript:
                return Response(
                    {'error': 'No transcript available'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Initialize and build index
            chatbot = ChatbotEngine(str(video_id))
            
            segments = transcript.json_data
            if transcript.word_timestamps:
                success = chatbot.build_from_transcript(transcript.word_timestamps)
            else:
                success = chatbot.build_from_transcript(segments)
            
            if success:
                # Update VideoIndex record
                VideoIndex.objects.update_or_create(
                    video_id=video_id,
                    defaults={
                        'is_indexed': True,
                        'index_created_at': timezone.now(),
                        'num_documents': len(chatbot.rag_engine.documents)
                    }
                )
                return Response({'status': 'index built successfully'})
            else:
                return Response(
                    {'error': 'Failed to build index'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
        except Video.DoesNotExist:
            return Response(
                {'error': 'Video not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Index build error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
