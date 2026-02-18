import { useState } from 'react'
import { motion } from 'framer-motion'
import { ScissorsIcon, ArrowDownTrayIcon, CheckCircleIcon, ArrowPathIcon } from '@heroicons/react/24/outline'
import { videoAPI } from '../services/api'

function GenerateShortButton({ videoId, onGenerationComplete }) {
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [shortVideo, setShortVideo] = useState(null)
  const [error, setError] = useState(null)
  const [settings, setSettings] = useState({
    max_duration: 60,
    style: 'default',
    include_music: false,
    caption_style: 'default'
  })

  const handleGenerate = async () => {
    setLoading(true)
    setProgress(0)
    setError(null)

    try {
      // Flatten settings for API
      const payload = {
        max_duration: settings.max_duration,
        style: settings.style,
        include_music: settings.include_music,
        caption_style: settings.caption_style
      }
      
      const response = await videoAPI.generateShort(videoId, payload)
      
      // Check if response has task_id (async) or direct result (sync)
      if (response.data.task_id) {
        await pollTaskStatus(response.data.task_id)
      } else if (response.data.id) {
        // Direct result - short video was created
        setShortVideo(response.data)
        if (onGenerationComplete) {
          onGenerationComplete()
        }
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate short video')
    } finally {
      setLoading(false)
    }
  }

  const pollTaskStatus = async (taskId) => {
    const maxAttempts = 60
    let attempts = 0

    while (attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      try {
        const response = await api.get(`/videos/tasks/by_task_id/?task_id=${taskId}`)
        const task = response.data
        
        setProgress(task.progress || 0)
        
        if (task.status === 'completed') {
          const shortsResponse = await videoAPI.getShorts(videoId)
          if (shortsResponse.data.length > 0) {
            setShortVideo(shortsResponse.data[0])
          }
          if (onGenerationComplete) {
            onGenerationComplete()
          }
          return
        }
        
        if (task.status === 'failed') {
          throw new Error(task.error || 'Task failed')
        }
      } catch (err) {
        console.error('Polling error:', err)
      }
      
      attempts++
    }
    
    throw new Error('Timeout waiting for task completion')
  }

  const handleDownload = () => {
    if (shortVideo) {
      window.open(shortVideo.file, '_blank')
    }
  }

  return (
    <motion.div 
      className="glass-card p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <motion.div 
            className="p-3 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl"
            whileHover={{ scale: 1.05 }}
          >
            <ScissorsIcon className="w-6 h-6 text-purple-400" />
          </motion.div>
          <div>
            <h3 className="text-lg font-semibold text-white">Generate Short Video</h3>
            <p className="text-sm text-white/50">Create a 9:16 short from highlights</p>
          </div>
        </div>
      </div>

      {/* Settings */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm text-white/60 mb-2">Max Duration</label>
          <select
            value={settings.max_duration}
            onChange={(e) => setSettings({ ...settings, max_duration: Number(e.target.value) })}
            className="setting-select w-full"
            disabled={loading}
          >
            <option value={30}>30 seconds</option>
            <option value={60}>60 seconds</option>
            <option value={90}>90 seconds</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm text-white/60 mb-2">Style</label>
          <select
            value={settings.style}
            onChange={(e) => setSettings({ ...settings, style: e.target.value })}
            className="setting-select w-full"
            disabled={loading}
          >
            <option value="default">Default</option>
            <option value="cinematic">Cinematic</option>
            <option value="vibrant">Vibrant</option>
            <option value="vintage">Vintage</option>
          </select>
        </div>
      </div>

      {/* Progress */}
      {loading && (
        <motion.div 
          className="mb-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-white/50">Generating...</span>
            <span className="text-white/70">{progress}%</span>
          </div>
          <div className="progress-bar">
            <motion.div 
              className="progress-bar-fill"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        </motion.div>
      )}

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 p-3 bg-red-500/10 border border-red-500/30 rounded-xl"
        >
          <p className="text-red-400 text-sm">{error}</p>
        </motion.div>
      )}

      {/* Actions */}
      <div className="flex space-x-3">
        {!shortVideo ? (
          <motion.button
            onClick={handleGenerate}
            disabled={loading}
            className="glow-button flex-1 disabled:opacity-50"
            whileHover={!loading ? { scale: 1.02 } : {}}
            whileTap={!loading ? { scale: 0.98 } : {}}
          >
            {loading ? (
              <>
                <motion.div 
                  className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full mr-2"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                />
                Generating...
              </>
            ) : (
              'Generate Short'
            )}
          </motion.button>
        ) : (
          <>
            <motion.button
              onClick={handleDownload}
              className="glow-button flex-1 flex items-center justify-center space-x-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <ArrowDownTrayIcon className="w-5 h-5" />
              <span>Download Short</span>
            </motion.button>
            <div className="flex items-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="p-2 bg-emerald-500/20 rounded-full"
              >
                <CheckCircleIcon className="w-6 h-6 text-emerald-400" />
              </motion.div>
            </div>
          </>
        )}
      </div>

      {/* Short video preview */}
      {shortVideo && (
        <motion.div 
          className="mt-4 p-3 bg-white/5 rounded-xl"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <p className="text-sm text-white/50">
            Duration: <span className="text-white">{shortVideo.duration?.toFixed(1)}s</span>
          </p>
        </motion.div>
      )}
    </motion.div>
  )
}

import api from '../services/api'

export default GenerateShortButton
