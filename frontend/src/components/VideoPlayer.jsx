import { useState } from 'react'
import { motion } from 'framer-motion'
import ReactPlayer from 'react-player'
import { PlayPauseIcon, SpeakerWaveIcon, SpeakerXMarkIcon } from '@heroicons/react/24/solid'

function VideoPlayer({ url, title, onTimeChange }) {
  const [playing, setPlaying] = useState(false)
  const [muted, setMuted] = useState(false)
  const [volume, setVolume] = useState(0.8)
  const [played, setPlayed] = useState(0)
  const [duration, setDuration] = useState(0)

  const handleProgress = (state) => {
    setPlayed(state.played)
    if (onTimeChange) {
      onTimeChange(state.playedSeconds)
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <motion.div 
      className="video-player-container"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="relative bg-black rounded-2xl overflow-hidden shadow-2xl">
        <ReactPlayer
          url={url}
          width="100%"
          height="100%"
          playing={playing}
          muted={muted}
          volume={volume}
          onProgress={handleProgress}
          onDuration={setDuration}
          controls={false}
          style={{ position: 'absolute', top: 0, left: 0 }}
        />

        {/* Custom controls overlay */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-6">
          {/* Progress bar */}
          <motion.div 
            className="h-1.5 bg-white/20 rounded-full cursor-pointer mb-4 overflow-hidden"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect()
              const pos = (e.clientX - rect.left) / rect.width
            }}
            whileHover={{ height: '6px' }}
          >
            <motion.div 
              className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full"
              style={{ width: `${played * 100}%` }}
              layoutId="progress"
            />
          </motion.div>

          {/* Controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {/* Play/Pause */}
              <motion.button
                onClick={() => setPlaying(!playing)}
                className="p-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                {playing ? (
                  <PlayPauseIcon className="w-6 h-6 text-white" />
                ) : (
                  <PlayPauseIcon className="w-6 h-6 text-white" />
                )}
              </motion.button>

              {/* Volume */}
              <motion.button
                onClick={() => setMuted(!muted)}
                className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                {muted || volume === 0 ? (
                  <SpeakerXMarkIcon className="w-5 h-5 text-white" />
                ) : (
                  <SpeakerWaveIcon className="w-5 h-5 text-white" />
                )}
              </motion.button>

              {/* Time */}
              <span className="text-sm text-white/80 font-medium">
                {formatTime(played * duration)} / {formatTime(duration)}
              </span>
            </div>

            {/* Title */}
            <div className="text-sm text-white/80 truncate max-w-xs">
              {title}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default VideoPlayer
