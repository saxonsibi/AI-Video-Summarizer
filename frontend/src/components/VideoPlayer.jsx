import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import ReactPlayer from 'react-player'
import { ArrowsPointingOutIcon } from '@heroicons/react/24/outline'
import { PauseIcon, PlayIcon, SpeakerWaveIcon, SpeakerXMarkIcon } from '@heroicons/react/24/solid'

function parseTimestampToSeconds(value) {
  if (value == null) return 0
  if (typeof value === 'number') return value
  const parts = String(value)
    .split(':')
    .map((part) => Number(part))
    .filter((part) => !Number.isNaN(part))
  if (parts.length === 3) return (parts[0] * 3600) + (parts[1] * 60) + parts[2]
  if (parts.length === 2) return (parts[0] * 60) + parts[1]
  if (parts.length === 1) return parts[0]
  return 0
}

function VideoPlayer({
  url,
  title,
  onTimeChange,
  seekToSeconds = null,
  highlights = [],
  fit = 'contain',
  showHighlights = true,
}) {
  const playerRef = useRef(null)
  const wrapperRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [muted, setMuted] = useState(false)
  const [volume, setVolume] = useState(0.8)
  const [played, setPlayed] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    if (seekToSeconds == null || Number.isNaN(seekToSeconds)) return
    if (!playerRef.current || typeof playerRef.current.seekTo !== 'function') return
    try {
      playerRef.current.seekTo(seekToSeconds, 'seconds')
      setPlaying(true)
    } catch (error) {
      console.warn('Failed to seek player:', error)
    }
  }, [seekToSeconds])

  const handleProgress = (state) => {
    setPlayed(state.played)
    if (onTimeChange) {
      onTimeChange(state.playedSeconds)
    }
  }

  const handlePlay = () => {
    setPlaying(true)
  }

  const handlePause = () => {
    setPlaying(false)
  }

  const handleEnded = () => {
    setPlaying(false)
    setPlayed(1)
    if (onTimeChange) {
      onTimeChange(duration || 0)
    }
  }

  const formatTime = (seconds) => {
    if (!seconds || Number.isNaN(seconds)) return '0:00'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleProgressSeek = (event) => {
    if (!duration || !playerRef.current || typeof playerRef.current.seekTo !== 'function') return
    const rect = event.currentTarget.getBoundingClientRect()
    const pos = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
    playerRef.current.seekTo(pos, 'fraction')
    setPlayed(pos)
    setPlaying(true)
  }

  const jumpTo = (seconds) => {
    if (!playerRef.current || typeof playerRef.current.seekTo !== 'function') return
    playerRef.current.seekTo(seconds, 'seconds')
    setPlaying(true)
  }

  const handleTogglePlayback = () => {
    if (!playerRef.current) return

    if (played >= 0.999 && typeof playerRef.current.seekTo === 'function') {
      playerRef.current.seekTo(0, 'seconds')
      setPlayed(0)
      if (onTimeChange) {
        onTimeChange(0)
      }
      setPlaying(true)
      return
    }

    setPlaying((prev) => !prev)
  }

  const handleFullscreen = async () => {
    const element = wrapperRef.current
    if (!element) return

    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen()
        return
      }

      if (element.requestFullscreen) {
        await element.requestFullscreen()
      }
    } catch (error) {
      console.warn('Failed to toggle fullscreen:', error)
    }
  }

  const normalizedHighlights = highlights
    .map((highlight) => ({
      ...highlight,
      seconds: parseTimestampToSeconds(highlight.timestamp),
    }))
    .filter((highlight) => highlight.seconds >= 0)

  const shouldShowHighlights = showHighlights && duration >= 10 && normalizedHighlights.length > 0
  const visibleHighlights = shouldShowHighlights
    ? normalizedHighlights.slice(0, duration < 30 ? 3 : 6)
    : []
  const hasFooter = Boolean(title || visibleHighlights.length > 0)

  return (
    <motion.div
      ref={wrapperRef}
      className="video-player-container"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className={`relative bg-black overflow-hidden shadow-2xl aspect-video ${hasFooter ? 'rounded-t-2xl' : 'rounded-2xl'}`}>
        <ReactPlayer
          ref={playerRef}
          url={url}
          width="100%"
          height="100%"
          playing={playing}
          muted={muted}
          volume={volume}
          onProgress={handleProgress}
          onPlay={handlePlay}
          onPause={handlePause}
          onEnded={handleEnded}
          onDuration={setDuration}
          controls={false}
          style={{ position: 'absolute', top: 0, left: 0 }}
          config={{
            file: {
              attributes: {
                preload: 'metadata',
                playsInline: true,
                style: {
                  objectFit: fit,
                  backgroundColor: '#000000',
                },
              },
            },
            youtube: {
              playerVars: {
                rel: 0,
                modestbranding: 1,
              },
            },
          }}
        />

        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-6">
          <motion.div
            className="relative h-1.5 bg-white/20 rounded-full cursor-pointer mb-4 overflow-visible"
            onClick={handleProgressSeek}
            whileHover={{ height: '6px' }}
          >
            <motion.div
              className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full"
              style={{ width: `${played * 100}%` }}
              layoutId="progress"
            />
            {duration > 0 && normalizedHighlights.map((highlight, index) => {
              const left = `${Math.max(0, Math.min(100, (highlight.seconds / duration) * 100))}%`
              return (
                <button
                  key={`${highlight.timestamp}-${index}`}
                  type="button"
                  title={`${highlight.timestamp || '0:00'} - ${highlight.title}`}
                  aria-label={`Jump to ${highlight.title} at ${highlight.timestamp}`}
                  className="absolute top-1/2 w-3 h-3 -translate-y-1/2 -translate-x-1/2 rounded-full border border-white/70 bg-fuchsia-400 shadow-[0_0_18px_rgba(217,70,239,0.65)] hover:scale-110 transition-transform"
                  style={{ left }}
                  onClick={(event) => {
                    event.stopPropagation()
                    jumpTo(highlight.seconds)
                  }}
                />
              )
            })}
          </motion.div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <motion.button
                type="button"
                onClick={handleTogglePlayback}
                className="p-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                title={playing ? 'Pause video' : 'Play video'}
              >
                {playing ? (
                  <PauseIcon className="w-6 h-6 text-white" />
                ) : (
                  <PlayIcon className="w-6 h-6 text-white" />
                )}
              </motion.button>

              <motion.button
                onClick={() => setMuted((prev) => !prev)}
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

              <span className="text-sm text-white/80 font-medium">
                {formatTime(played * duration)} / {formatTime(duration)}
              </span>
            </div>

            <div className="flex items-center gap-3">
              <motion.button
                type="button"
                onClick={handleFullscreen}
                className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
                whileHover={{ scale: 1.08 }}
                whileTap={{ scale: 0.92 }}
                title="Fullscreen"
              >
                <ArrowsPointingOutIcon className="w-5 h-5 text-white" />
              </motion.button>
            </div>
          </div>
        </div>
      </div>

      {hasFooter && (
        <div className="bg-slate-950/80 border border-white/10 border-t-0 rounded-b-2xl px-4 py-3">
          {title && (
            <div className="text-sm text-white/80 truncate mb-2">
              {title}
            </div>
          )}

          {visibleHighlights.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {visibleHighlights.map((highlight, index) => (
                <button
                  key={`chip-${highlight.timestamp}-${index}`}
                  type="button"
                  onClick={() => jumpTo(highlight.seconds)}
                  className="px-3 py-1.5 rounded-full bg-white/10 hover:bg-fuchsia-500/20 border border-white/10 hover:border-fuchsia-400/40 text-left transition-colors"
                >
                  <span className="text-[11px] font-mono text-fuchsia-300 mr-2">{highlight.timestamp || '0:00'}</span>
                  <span className="text-xs text-white/80">{highlight.title}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </motion.div>
  )
}

export default VideoPlayer
