import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChatBubbleLeftRightIcon, PaperAirplaneIcon, SparklesIcon, SpeakerWaveIcon } from '@heroicons/react/24/outline'
import { SpeakerWaveIcon as SpeakerWaveFilledIcon, PauseIcon } from '@heroicons/react/24/solid'
import { chatbotAPI } from '../services/api'

function ChatBot({ videoId }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [suggestedQuestions, setSuggestedQuestions] = useState([])
  const [playingAudio, setPlayingAudio] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    loadSuggestedQuestions()
  }, [videoId])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const loadSuggestedQuestions = async () => {
    try {
      const response = await chatbotAPI.getSuggestedQuestions(videoId)
      setSuggestedQuestions(response.data.questions || [])
    } catch (error) {
      console.error('Failed to load suggested questions:', error)
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const playAudio = (audioUrl) => {
    // If same audio is playing, toggle pause/play
    if (playingAudio && playingAudio.src === audioUrl) {
      if (playingAudio.paused) {
        playingAudio.play()
        setIsPlaying(true)
      } else {
        playingAudio.pause()
        setIsPlaying(false)
      }
      return
    }
    
    // Stop any existing audio
    if (playingAudio) {
      playingAudio.pause()
      playingAudio = null
    }
    
    if (audioUrl) {
      const audio = new Audio(audioUrl)
      audio.onended = () => {
        setPlayingAudio(null)
        setIsPlaying(false)
      }
      audio.onerror = () => {
        setPlayingAudio(null)
        setIsPlaying(false)
      }
      audio.play()
      setPlayingAudio(audio)
      setIsPlaying(true)
    }
  }

  // Auto-play audio when bot response arrives
  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1]
      if (lastMessage.role === 'bot' && lastMessage.audioUrl && !loading) {
        playAudio(lastMessage.audioUrl)
      }
    }
  }, [messages, loading])

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setLoading(true)

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])

    try {
      const response = await chatbotAPI.sendMessage({
        video_id: videoId,
        message: userMessage,
        session_id: null,
        generate_tts: true
      })

      // Add bot response
      setMessages(prev => [
        ...prev,
        { 
          role: 'bot', 
          content: response.data.answer,
          sources: response.data.sources || [],
          audioUrl: response.data.audio_url || null
        }
      ])
    } catch (error) {
      setMessages(prev => [
        ...prev,
        { 
          role: 'bot', 
          content: 'Sorry, I failed to process your question. Please try again.',
          error: true
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleSuggestedQuestion = (question) => {
    setInput(question)
  }

  return (
    <motion.div 
      className="glass-card flex flex-col h-[500px]"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      {/* Header */}
      <div className="flex items-center space-x-3 p-4 border-b border-white/10">
        <motion.div 
          className="p-2 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-xl"
          whileHover={{ scale: 1.1 }}
        >
          <ChatBubbleLeftRightIcon className="w-6 h-6 text-indigo-400" />
        </motion.div>
        <div>
          <h3 className="text-lg font-semibold text-white">Video Chatbot</h3>
          <p className="text-xs text-white/50">Ask questions about this video</p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
        {messages.length === 0 && (
          <motion.div 
            className="flex flex-col items-center justify-center h-full text-center py-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <motion.div
              className="w-16 h-16 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mb-4"
              animate={{ 
                y: [0, -10, 0],
                scale: [1, 1.05, 1]
              }}
              transition={{ duration: 3, repeat: Infinity }}
            >
              <SparklesIcon className="w-8 h-8 text-indigo-400" />
            </motion.div>
            <p className="text-white/60 mb-2">Ask me anything about this video!</p>
            <p className="text-xs text-white/40">I can summarize, answer questions, or explain topics.</p>
          </motion.div>
        )}

        {messages.map((message, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={message.role === 'user' ? 'flex justify-end' : 'flex justify-start'}
          >
            <div className={message.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-bot'}>
              <div className="flex justify-between items-start gap-2">
                <p className="whitespace-pre-wrap flex-1">{message.content}</p>
                {message.audioUrl && message.role === 'bot' && (
                  <button
                    onClick={() => playAudio(message.audioUrl)}
                    className="p-1 hover:bg-white/10 rounded-full transition-colors flex-shrink-0"
                    title={isPlaying && playingAudio?.src === message.audioUrl ? 'Pause' : 'Play'}
                  >
                    {isPlaying && playingAudio?.src === message.audioUrl ? (
                      <PauseIcon className="w-4 h-4 text-indigo-400" />
                    ) : (
                      <SpeakerWaveFilledIcon className="w-4 h-4 text-white/40 hover:text-white/60" />
                    )}
                  </button>
                )}
              </div>
              
              {/* Sources (for bot messages) */}
              {message.sources && message.sources.length > 0 && (
                <div className="source-reference">
                  <p className="text-xs text-white/40 mb-2">Sources:</p>
                  {message.sources.map((source, i) => (
                    <motion.div 
                      key={i}
                      className="source-item mb-1"
                      whileHover={{ x: 5 }}
                    >
                      <span className="text-white/60">{source.timestamp}</span>
                      <span className="text-white/30 mx-2">•</span>
                      <span className="text-white/50">{source.text?.substring(0, 80)}...</span>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        ))}

        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="chat-bubble-bot">
              <div className="typing-indicator">
                <div className="typing-dot" />
                <div className="typing-dot" />
                <div className="typing-dot" />
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested questions */}
      {messages.length < 2 && suggestedQuestions.length > 0 && (
        <div className="p-4 border-t border-white/10">
          <p className="text-xs text-white/40 mb-3">Suggested questions:</p>
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions.slice(0, 3).map((question, index) => (
              <motion.button
                key={index}
                onClick={() => handleSuggestedQuestion(question)}
                className="suggestion-chip"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                {question}
              </motion.button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-white/10">
        <div className="flex items-center space-x-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your question..."
            className="input-field flex-1"
            disabled={loading}
          />
          <motion.button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="p-3 glow-button flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
            whileHover={!input.trim() || loading ? {} : { scale: 1.05 }}
            whileTap={!input.trim() || loading ? {} : { scale: 0.95 }}
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </motion.button>
        </div>
      </div>
    </motion.div>
  )
}

export default ChatBot
