# PROJECT DOCUMENTATION

## 1. Cover Page
- Project Title: AI Video Summarizer
- Intern's Name: [Your Name]
- Internship Period: [Start Date] to [End Date]
- Mentor/Supervisor Name: [Mentor Name]
- Date of Submission: [Submission Date]

## 2. Executive Summary / Abstract
AI Video Summarizer is an AI-powered system that processes uploaded videos and generates meaningful outputs including transcripts, multiple summary formats, chatbot-based question answering, and highlight-based short clips. The platform combines speech recognition, NLP summarization, retrieval-augmented generation, and video editing pipelines into a single workflow.

The project aims to reduce manual effort in consuming long-form video content and improve accessibility by providing concise, searchable, and reusable outputs. It is useful for content creators, learners, and teams who need fast understanding and repurposing of video material.

## 3. Objective(s)
- To build an end-to-end AI video processing system that accepts uploaded videos and generates transcripts automatically.
- To generate multiple summary formats (full, bullet, short) from transcribed content.
- To implement a RAG-based chatbot that answers user questions grounded in video transcript segments.
- To generate highlight-based short videos in vertical format for quick sharing.
- To provide optional text-to-speech audio summaries for improved accessibility.

## 4. Problem Statement
Long-form videos are time-consuming to review, and extracting key points manually is inefficient. Users also struggle to search spoken content and repurpose videos into short clips quickly. This project solves that by automating transcription, summarization, question answering, and short-video generation in one integrated platform.

## 5. Scope of Work
- Will cover:
  - Video upload and metadata handling
  - Audio extraction and speech-to-text transcription
  - NLP-based summarization (full, bullet, short)
  - RAG chatbot with vector retrieval over transcripts
  - Highlight detection and short video generation
  - Audio summary generation via text-to-speech
  - Frontend dashboard for interaction
- Will not cover:
  - Large-scale distributed production deployment
  - Multi-tenant authentication/authorization at enterprise scale
  - Advanced model training from scratch

## 6. Literature Review / Background (Optional for interns)
Existing tools typically focus on one part of the workflow (transcription only, summarization only, or clipping only). Modern approaches combine ASR models (e.g., Whisper family), transformer-based summarization, vector search (FAISS), and LLM-assisted QA (RAG). This project follows that integrated approach to provide complete video understanding and repurposing.

## 7. Features (Proposed System)
- Video upload and processing pipeline
- Automatic transcription using Faster-Whisper
- AI summaries: full, bullet points, and short script
- RAG chatbot with contextual answers from transcript
- Suggested questions for quick interaction
- Short video generation from detected highlights
- Transcript editing and summary regeneration
- Audio summary generation (TTS)
- Processing status tracking and task monitoring

## 8. Methodology / Workflow
Step-by-step workflow:
1. User uploads video through frontend.
2. Backend stores file and extracts audio.
3. Speech-to-text generates transcript with timestamps.
4. Transcript is stored and used for summary generation.
5. Transcript segments are embedded and indexed in FAISS.
6. Chatbot retrieves relevant chunks and generates answers.
7. Highlights are detected from transcript segments.
8. Short video is generated from selected highlight segments.
9. Optional audio summary is generated via TTS.

Simple flow diagram:
Video Upload -> Audio Extraction -> Transcription -> Summary Generation -> RAG Indexing -> Chatbot Q&A

Video Upload -> Transcript Analysis -> Highlight Detection -> Short Video Generation

## 9. Dataset
- Primary data source: User-uploaded videos.
- Derived data:
  - Extracted audio tracks
  - Transcript text and timestamped segments
  - Generated summaries and key topics
  - Highlight segments for short generation
- Preprocessing:
  - Audio normalization via FFmpeg
  - Transcript cleaning and text normalization
  - Segment filtering for embedding and retrieval

## 10. Tech Stack
- Backend: Python, Django, Django REST Framework
- Async Processing: Celery, Redis
- AI/ML: Faster-Whisper, Transformers, Sentence-Transformers, FAISS, LangChain, Groq/Ollama integration
- Video/Audio: FFmpeg, MoviePy, gTTS
- Database: SQLite (development), MySQL-ready configuration
- Frontend: React, Vite, Tailwind CSS, Framer Motion, Axios

## 11. Task Division (Team of 2)
- Intern 1:
  - Backend API development
  - Video/transcription/summarization pipeline
  - Database models and task orchestration
- Intern 2:
  - Frontend UI/UX implementation
  - Chatbot interface and API integration
  - Progress tracking and content visualization

## 12. Timeline / Milestones
- Week 1: Requirement analysis, project setup, base backend/frontend scaffolding.
- Week 2: Video upload, audio extraction, transcription pipeline.
- Week 3: Summary generation modules and transcript management.
- Week 4: RAG indexing and chatbot integration.
- Week 5: Highlight detection and short video generation.
- Week 6: TTS/audio summary, UI polish, testing, and documentation.

## 13. Expected Deliverables
- Complete source code (backend + frontend)
- Configured API endpoints with documentation
- Working AI pipeline for transcription and summarization
- Functional RAG chatbot for video Q&A
- Short video generator from highlights
- Final project report and demo walkthrough

## 14. Risks & Challenges
- ASR quality variation across audio quality/accents.
- High processing time for long videos on CPU.
- Dependency and environment setup complexity (FFmpeg, Redis, model downloads).
- RAG retrieval quality may degrade with noisy transcripts.
- Video clipping/rendering failures for edge cases.

Mitigation plan:
- Use transcript cleanup and confidence filtering.
- Support configurable model sizes for resource constraints.
- Add retries, task tracking, and fallback logic.
- Validate input formats and handle exceptions robustly.

## 15. References (if any)
- OpenAI Whisper / Faster-Whisper documentation
- Hugging Face Transformers documentation
- FAISS documentation
- LangChain documentation
- Django and Django REST Framework documentation
- MoviePy and FFmpeg documentation
