import { useState } from 'react'
import { motion } from 'framer-motion'
import { SpeakerWaveIcon, ArrowDownTrayIcon } from '@heroicons/react/24/outline'
import { videoAPI } from '../services/api'

function AudioSummaryButton({ videoId }) {
  const [loading, setLoading] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)
  const [message, setMessage] = useState('')

  const generateAudio = async () => {
    if (loading) return
    
    setLoading(true)
    setMessage('')
    
    try {
      const response = await videoAPI.generateAudioSummary(videoId)
      setAudioUrl(response.data.audio_url)
      setMessage('Audio summary generated!')
    } catch (error) {
      console.error('Failed to generate audio:', error)
      setMessage('Failed to generate audio')
    } finally {
      setLoading(false)
    }
  }

  const downloadAudio = () => {
    if (!audioUrl) return
    
    const link = document.createElement('a')
    link.href = audioUrl
    link.download = 'video-summary.mp3'
    link.click()
  }

  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center space-x-3 mb-4">
        <motion.div
          className="p-2 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-xl"
          whileHover={{ scale: 1.1 }}
        >
          <SpeakerWaveIcon className="w-6 h-6 text-indigo-400" />
        </motion.div>
        <div>
          <h3 className="text-lg font-semibold text-white">Audio Summary</h3>
          <p className="text-xs text-white/50">Get a podcast-style audio summary</p>
        </div>
      </div>

      <p className="text-white/60 text-sm mb-4">
        Generate an audio version of the video summary that you can listen to on the go. 
        It's like a mini-podcast of the video!
      </p>

      {message && (
        <p className={`text-sm mb-4 ${message.includes('Failed') ? 'text-red-400' : 'text-green-400'}`}>
          {message}
        </p>
      )}

      <div className="flex gap-3">
        <motion.button
          onClick={generateAudio}
          disabled={loading}
          className="flex-1 bg-gradient-to-r from-indigo-500 to-purple-500 text-white py-2 px-4 rounded-xl font-medium disabled:opacity-50"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {loading ? 'Generating...' : 'Generate Audio'}
        </motion.button>

        {audioUrl && (
          <motion.button
            onClick={downloadAudio}
            className="flex items-center space-x-2 bg-white/10 hover:bg-white/20 text-white py-2 px-4 rounded-xl"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <ArrowDownTrayIcon className="w-5 h-5" />
            <span>Download</span>
          </motion.button>
        )}
      </div>

      {audioUrl && (
        <div className="mt-4">
          <audio controls className="w-full" src={audioUrl}>
            Your browser does not support audio.
          </audio>
        </div>
      )}
    </motion.div>
  )
}

export default AudioSummaryButton
