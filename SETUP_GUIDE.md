# AI Video Summarizer - Setup Guide

## Current Status
Your AI Video Summarizer is **fully built** with all core features:
- ✅ Video upload & processing
- ✅ Speech-to-text (Faster-Whisper)
- ✅ Text summarization (BART/T5)
- ✅ Chatbot with RAG (FAISS + LangChain)
- ✅ Short video generation (MoviePy)
- ✅ Voice chatbot (TTS)
- ✅ Professional UI with animations

## Enable Intelligent Chatbot Responses

The chatbot currently uses extractive answers (copying transcript). To get **intelligent abstractive answers** using AI, you have two free options:

---

### Option 1: Groq API (Recommended - Works Instantly) ⭐

Groq provides **free API keys** with generous limits. No downloads needed!

1. **Get a free API key:**
   - Go to: https://console.groq.com/keys
   - Click "Create API Key"
   - Copy the key (starts with `gsk_`)

2. **Set the environment variable:**
   ```powershell
   # In PowerShell (run as Administrator):
   setx GROQ_API_KEY "gsk_your_key_here"
   ```
   
   Or add to your system:
   - Search "Environment Variables" → Edit → New → Add `GROQ_API_KEY`

3. **Restart Django server** (Terminal 2):
   - Stop with `Ctrl+C`
   - Run: `cd backend && python manage.py runserver`

That's it! The chatbot will now use Mixtral-8x7b (free, fast AI) for intelligent responses.

---

### Option 2: Ollama (Local, Slower Download)

If you have good internet, you can run AI locally:

1. **Download a model** (needs 2-4GB download):
   ```bash
   ollama pull phi3    # 2.2GB - faster
   # OR
   ollama pull mistral # 4.4GB - better quality
   ```

2. **Start Ollama server:**
   ```bash
   ollama serve
   ```

3. **Keep Ollama running** while using the app

---

## Testing the Chatbot

After setting up (either option), test with:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chatbot/chat/ `
  -H "Content-Type: application/json" `
  -d '{"video_id": "YOUR_VIDEO_ID", "message": "What is this video about?"}'
```

Or use the UI:
1. Open http://localhost:5173
2. Upload a video and wait for processing
3. Ask questions in the chatbot

---

## Features Quick Reference

| Feature | How to Use |
|---------|------------|
| Upload Video | Drag & drop on home page |
| View Transcript | Click "Transcript" tab |
| Get Summary | Click "Summary" tab (3 formats) |
| Chat with Video | Type in chatbot (needs Groq/Ollama for AI answers) |
| Voice Chat | Click microphone icon in chatbot |
| Generate Short | Click "Generate Short" button |
| Download Audio | Click download icon in summary |

---

## Troubleshooting

**Chatbot returns extractive answers?**
→ Install Groq API key (see Option 1 above)

**Video processing slow?**
→ Smaller videos process faster

**TTS not working?**
→ Check internet connection (uses Google TTS)
