# Ollama Setup for AI Video Summarizer Chatbot

## Installation Steps

### 1. Install Ollama
- Download from: https://ollama.com/download/windows
- Run the installer

### 2. Pull a Model (after installation)
Open a new terminal and run:
```bash
ollama pull mistral
```
This downloads the Mistral model (~4GB).

### 3. Start Ollama Server
```bash
ollama serve
```
Keep this running in the background while using the chatbot.

### 4. Test the Chatbot
Once Ollama is running, test with:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chatbot/chat/ ^
  -H "Content-Type: application/json" ^
  -d "{\"video_id\": \"5ab2d437-aa87-4064-bbec-87139f170254\", \"message\": \"What is this video about?\"}"
```

## What Happens
- The chatbot will use Mistral (powerful open-source model)
- Instead of copying transcript dialogue, it will provide intelligent abstractive answers
- Questions like "What is this video about?" will get analytical summaries
- Questions like "What did he say?" will get relevant quotes

## Troubleshooting
- If Ollama is not running, the chatbot falls back to extractive answering
- Make sure `ollama serve` is running before testing the chatbot
- Default URL: http://localhost:11434
