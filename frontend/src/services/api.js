import axios from 'axios'

const API_BASE_URL = '/api/v1'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // 10 minutes for video uploads with processing
  headers: {
    'Content-Type': 'application/json',
  },
})

// Video API
export const videoAPI = {
  // Upload video
  upload: async (formData, onProgress) => {
    return api.post('/videos/upload/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        if (onProgress) onProgress(percent)
      },
    })
  },

  // Get all videos
  getAll: async (params = {}) => {
    return api.get('/videos/', { params })
  },

  // Get video by ID
  getById: async (id) => {
    return api.get(`/videos/${id}/`)
  },

  // Delete video
  delete: async (id) => {
    return api.delete(`/videos/${id}/`)
  },

  // Generate transcript
  generateTranscript: async (id) => {
    return api.post(`/videos/${id}/generate_transcript/`)
  },

  // Get transcripts
  getTranscripts: async (id) => {
    return api.get(`/videos/${id}/transcripts/`)
  },

  // Update transcript (edit/correct)
  updateTranscript: async (id, data) => {
    return api.patch(`/videos/${id}/update_transcript/`, data)
  },

  // Generate summary
  generateSummary: async (id, data) => {
    return api.post(`/videos/${id}/generate_summary/`, data)
  },

  // Get summaries
  getSummaries: async (id) => {
    return api.get(`/videos/${id}/summaries/`)
  },

  // Get highlights
  getHighlights: async (id) => {
    return api.get(`/videos/${id}/highlights/`)
  },

  // Generate short video
  generateShort: async (id, data) => {
    return api.post(`/videos/${id}/generate_short/`, data)
  },

  // Get short videos
  getShorts: async (id) => {
    return api.get(`/videos/${id}/shorts/`)
  },

  // Generate audio summary
  generateAudioSummary: async (id) => {
    return api.post(`/videos/${id}/generate_audio_summary/`)
  },

  // Get processing tasks
  getTasks: async (id) => {
    return api.get(`/videos/${id}/tasks/`)
  },
}

// Chatbot API
export const chatbotAPI = {
  // Send message
  sendMessage: async (data) => {
    return api.post('/chatbot/chat/', data)
  },

  // Get suggested questions
  getSuggestedQuestions: async (videoId) => {
    return api.get('/chatbot/chat/', { params: { video_id: videoId } })
  },

  // Get chat sessions
  getSessions: async (params = {}) => {
    return api.get('/chatbot/sessions/', { params })
  },

  // Get session messages
  getSessionMessages: async (sessionId) => {
    return api.get(`/chatbot/sessions/${sessionId}/messages/`)
  },
}

// Summarizer API
export const summarizerAPI = {
  // Summarize text
  summarize: async (data) => {
    return api.post('/summarizer/summarize/', data)
  },

  // Health check
  healthCheck: async () => {
    return api.get('/summarizer/health/')
  },
}

export default api
