# AI Video Summarizer

An AI-powered application for video summarization, chatbot interaction, and short video generation.

## 🚀 Features

- **Video Upload**: Upload videos for processing
- **Automatic Transcription**: Extract audio and transcribe using Faster-Whisper
- **AI Summarization**: Generate full, bullet-point, and short summaries using BART
- **RAG Chatbot**: Ask questions about video content in natural language
- **Short Video Generator**: Create 9:16 short videos from highlights
- **Progress Tracking**: Real-time progress updates

## 🛠️ Tech Stack

### Backend
- **Django + Django Rest Framework**: API backend
- **Celery**: Background task processing
- **Redis**: Task queue broker
- **Faster-Whisper**: Speech-to-text
- **BART**: Text summarization
- **FAISS**: Vector similarity search
- **LangChain**: RAG pipeline
- **MoviePy**: Video editing

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool
- **Tailwind CSS**: Styling
- **React Router**: Navigation
- **Axios**: HTTP client
- **React Player**: Video playback

## 📁 Project Structure

```
AI Video Summarizer/
├── backend/
│   ├── ai_video_summarizer/   # Django project config
│   ├── videos/                  # Video processing app
│   │   ├── models.py           # Database models
│   │   ├── views.py            # API views
│   │   ├── tasks.py            # Celery tasks
│   │   └── utils.py            # Utility functions
│   ├── chatbot/                 # RAG chatbot app
│   │   ├── rag_engine.py       # RAG implementation
│   │   └── views.py            # Chatbot API
│   └── summarizer/             # Text summarization app
├── frontend/
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── pages/             # Page components
│   │   └── services/           # API services
│   └── package.json
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- FFmpeg (for video processing)
- Redis (for Celery)

### Backend Setup

1. Create virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Run migrations:
```bash
python manage.py migrate
```

5. Start Redis (required for Celery):
```bash
redis-server
```

6. Start Celery worker:
```bash
celery -A ai_video_summarizer worker -l info
```

7. Start Django server:
```bash
python manage.py runserver
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start development server:
```bash
npm run dev
```

3. Open http://localhost:5173

## 📡 API Endpoints

### Videos
- `POST /api/videos/upload/` - Upload video
- `GET /api/videos/` - List videos
- `GET /api/videos/{id}/` - Get video details
- `POST /api/videos/{id}/generate_transcript/` - Generate transcript
- `POST /api/videos/{id}/generate_summary/` - Generate summary
- `POST /api/videos/{id}/generate_short/` - Generate short video

### Chatbot
- `POST /api/chatbot/chat/` - Send message
- `GET /api/chatbot/chat/?video_id=xxx` - Get suggested questions

## 🎯 Usage

1. **Upload a video** from the homepage
2. **Wait for processing** - transcription and summarization happen automatically
3. **View summaries** in the Summaries tab
4. **Ask questions** in the Chatbot tab
5. **Generate shorts** in the Generate Short tab

## 🔧 Configuration

### Whisper Model Size
Choose model size based on your hardware:
- `tiny` - Fastest, lowest accuracy
- `small` - Good balance
- `medium` - Better accuracy, slower
- `large` - Best accuracy, slowest

### Celery Workers
Scale workers for faster processing:
```bash
celery -A ai_video_summarizer worker -l info -c 4
```

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request
