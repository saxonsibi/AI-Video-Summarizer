import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LinkIcon, XMarkIcon, CheckCircleIcon, ArrowRightIcon } from '@heroicons/react/24/outline'
import { videoAPI } from '../services/api'
import { LANGUAGE_OPTIONS } from '../constants/languages'

function getYouTubeId(rawUrl) {
  if (!rawUrl) return null
  const patterns = [
    /youtube\.com\/watch\?v=([\w-]+)/,
    /youtu\.be\/([\w-]+)/,
    /youtube\.com\/embed\/([\w-]+)/,
    /youtube\.com\/v\/([\w-]+)/,
    /youtube\.com\/shorts\/([\w-]+)/,
  ]

  for (const pattern of patterns) {
    const match = rawUrl.match(pattern)
    if (match?.[1]) return match[1]
  }

  return null
}

function UploadYouTubeURL({ onUploadComplete }) {
  const [url, setUrl] = useState('')
  const [title, setTitle] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(false)
  const [transcriptionLanguage, setTranscriptionLanguage] = useState('auto')
  const [outputLanguage, setOutputLanguage] = useState('auto')

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!url.trim()) {
      setError('Please enter a YouTube URL')
      return
    }

    // Basic validation
    const youtubePatterns = [
      /youtube\.com\/watch\?v=/,
      /youtu\.be\//,
      /youtube\.com\/embed\//,
      /youtube\.com\/v\//,
      /youtube\.com\/shorts\//
    ]
    
    const isValidYouTube = youtubePatterns.some(pattern => pattern.test(url))
    
    if (!isValidYouTube) {
      setError('Please enter a valid YouTube URL')
      return
    }

    setLoading(true)
    setError(null)

    try {
      await videoAPI.uploadYouTubeURL(url.trim(), title.trim(), {
        transcription_language: 'auto',
        output_language: 'auto',
      })
      setLoading(false)
      setSuccess(true)
      
      setTimeout(() => {
        setUrl('')
        setTitle('')
        setSuccess(false)
        if (onUploadComplete) {
          onUploadComplete()
        }
      }, 2000)
      
    } catch (err) {
      setLoading(false)
      setError(err.response?.data?.error || err.response?.data?.detail || 'Failed to process YouTube URL. Please try again.')
    }
  }

  const clearInput = () => {
    setUrl('')
    setTitle('')
    setError(null)
  }

  const youtubeId = getYouTubeId(url.trim())

  return (
    <motion.div
      className="max-w-2xl mx-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="glass-card p-8">
        <motion.h2 
          className="text-2xl font-bold text-white mb-2 flex items-center space-x-3"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <LinkIcon className="w-8 h-8 text-red-400" />
          <span>Add YouTube Video</span>
        </motion.h2>

        <motion.p 
          className="text-white/60 mb-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          Paste a YouTube URL to analyze the video, generate transcripts, and chat with AI
        </motion.p>

        {/* Success state */}
        <AnimatePresence>
          {success && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center py-8"
            >
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200 }}
                className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-emerald-500/20 mb-4"
              >
                <CheckCircleIcon className="w-12 h-12 text-emerald-400" />
              </motion.div>
              <p className="text-xl text-white font-semibold">Video Queued!</p>
              <p className="text-white/60 mt-2">Processing will begin shortly</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Form */}
        {!success && (
          <motion.form
            onSubmit={handleSubmit}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-4"
          >
            {/* YouTube URL Input */}
            <div>
              <label className="block text-sm font-medium text-white/70 mb-2">
                YouTube URL <span className="text-red-400">*</span>
              </label>
              <div className="relative">
                <input
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/30 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                  disabled={loading}
                />
                {url && (
                  <button
                    type="button"
                    onClick={clearInput}
                    className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-lg text-white/40 hover:text-white hover:bg-white/10 transition-colors"
                  >
                    <XMarkIcon className="w-5 h-5" />
                  </button>
                )}
              </div>
              <p className="text-xs text-white/40 mt-1">
                Supports: youtube.com/watch, youtube.com/shorts, youtu.be, youtube.com/embed
              </p>
            </div>

            {/* Optional Title Input */}
            <div>
              <label className="block text-sm font-medium text-white/70 mb-2">
                Custom Title <span className="text-white/30">(optional)</span>
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Give your video a custom title"
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/30 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                disabled={loading}
              />
            </div>

            {youtubeId && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-2xl border border-white/10 bg-black/30 overflow-hidden"
              >
                <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
                  <h3 className="text-sm font-medium text-white">Preview</h3>
                  <a
                    href={url.trim()}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-red-300 hover:text-red-200"
                  >
                    Open on YouTube
                  </a>
                </div>
                <div className="aspect-video bg-black">
                  <iframe
                    src={`https://www.youtube.com/embed/${youtubeId}`}
                    title="YouTube preview"
                    className="w-full h-full"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                </div>
              </motion.div>
            )}

            {/* Submit Button */}
            <motion.button
              type="submit"
              disabled={loading || !url.trim()}
              className="glow-button w-full flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              whileHover={{ scale: loading ? 1 : 1.02 }}
              whileTap={{ scale: loading ? 1 : 0.98 }}
            >
              {loading ? (
                <>
                  <motion.div 
                    className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                  />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <ArrowRightIcon className="w-5 h-5" />
                  <span>Add YouTube Video</span>
                </>
              )}
            </motion.button>

            {/* Info box */}
            <div className="mt-4 p-4 bg-indigo-500/10 border border-indigo-500/30 rounded-xl">
              <p className="text-sm text-indigo-300">
                <strong>What happens next?</strong>
              </p>
              <ul className="text-sm text-white/60 mt-2 space-y-1">
                <li>• We'll download the audio from the video</li>
                <li>• Generate accurate transcripts</li>
                <li>• Create AI-powered summaries</li>
                <li>• Enable chat with your video content</li>
              </ul>
            </div>
          </motion.form>
        )}

        {/* Error message */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl"
            >
              <p className="text-red-400">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

export default UploadYouTubeURL
