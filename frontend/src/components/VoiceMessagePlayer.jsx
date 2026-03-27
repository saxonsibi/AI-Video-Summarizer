import { useEffect, useMemo, useRef, useState } from 'react'
import { PlayIcon, PauseIcon } from '@heroicons/react/24/solid'

function formatDuration(seconds) {
  const safe = Math.max(0, Math.floor(Number(seconds) || 0))
  const minutes = Math.floor(safe / 60)
  const remaining = safe % 60
  return `${String(minutes).padStart(2, '0')}:${String(remaining).padStart(2, '0')}`
}

function VoiceMessagePlayer({
  audioUrl,
  activeAudioUrl,
  onSetActiveAudioUrl,
  autoPlay = false,
  onBubbleClick = null,
}) {
  const audioRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [hasAutoPlayed, setHasAutoPlayed] = useState(false)
  const [isScrubbing, setIsScrubbing] = useState(false)
  const [scrubTime, setScrubTime] = useState(0)
  const isActive = activeAudioUrl === audioUrl

  const waveformBars = useMemo(() => [10, 18, 14, 22, 12, 20, 16, 24, 13, 18, 15], [])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return undefined

    const handleLoaded = () => setDuration(audio.duration || 0)
    const handleTimeUpdate = () => {
      if (isScrubbing) return
      setCurrentTime(audio.currentTime || 0)
    }
    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleEnded = () => {
      setIsPlaying(false)
      setCurrentTime(audio.duration || 0)
      if (onSetActiveAudioUrl) onSetActiveAudioUrl(null)
    }

    audio.addEventListener('loadedmetadata', handleLoaded)
    audio.addEventListener('timeupdate', handleTimeUpdate)
    audio.addEventListener('play', handlePlay)
    audio.addEventListener('pause', handlePause)
    audio.addEventListener('ended', handleEnded)

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoaded)
      audio.removeEventListener('timeupdate', handleTimeUpdate)
      audio.removeEventListener('play', handlePlay)
      audio.removeEventListener('pause', handlePause)
      audio.removeEventListener('ended', handleEnded)
    }
  }, [isScrubbing, onSetActiveAudioUrl])

  useEffect(() => {
    if (!isPlaying || isScrubbing) return undefined
    const audio = audioRef.current
    if (!audio) return undefined

    const timer = window.setInterval(() => {
      setCurrentTime(audio.currentTime || 0)
    }, 100)

    return () => window.clearInterval(timer)
  }, [isPlaying, isScrubbing])

  useEffect(() => {
    setHasAutoPlayed(false)
  }, [audioUrl])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio || !audioUrl) return

    const attemptAutoPlay = async () => {
      if (!autoPlay || hasAutoPlayed) return
      if (onSetActiveAudioUrl) onSetActiveAudioUrl(audioUrl)
      try {
        await audio.play()
        setHasAutoPlayed(true)
      } catch (_error) {
        // If the browser delays readiness, wait for canplay and try once more.
      }
    }

    const handleCanPlay = () => {
      if (!autoPlay || hasAutoPlayed) return
      if (onSetActiveAudioUrl) onSetActiveAudioUrl(audioUrl)
      audio.play().then(() => {
        setHasAutoPlayed(true)
      }).catch(() => {})
    }

    attemptAutoPlay()
    audio.addEventListener('canplay', handleCanPlay)
    return () => {
      audio.removeEventListener('canplay', handleCanPlay)
    }
  }, [audioUrl, autoPlay, hasAutoPlayed, onSetActiveAudioUrl])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio || isActive) return
    if (!audio.paused) audio.pause()
  }, [isActive])

  const handleToggle = async () => {
    const audio = audioRef.current
    if (!audio) return

    if (isActive && !audio.paused) {
      audio.pause()
      return
    }

    if (onSetActiveAudioUrl) onSetActiveAudioUrl(audioUrl)
    try {
      await audio.play()
    } catch (_error) {
      setIsPlaying(false)
    }
  }

  const handleSeek = (event) => {
    const audio = audioRef.current
    if (!audio || !duration) return
    const rect = event.currentTarget.getBoundingClientRect()
    const percent = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1)
    audio.currentTime = duration * percent
    setCurrentTime(audio.currentTime)
  }

  const handleScrubChange = (event) => {
    const nextTime = Number(event.target.value || 0)
    setIsScrubbing(true)
    setScrubTime(nextTime)
    setCurrentTime(nextTime)
  }

  const commitScrub = () => {
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = scrubTime
    setCurrentTime(scrubTime)
    setIsScrubbing(false)
  }

  const displayedTime = isScrubbing ? scrubTime : currentTime
  const progressPercent = duration ? Math.min((displayedTime / duration) * 100, 100) : 0

  return (
    <div
      className="mt-4 max-w-sm rounded-[20px] border border-white/10 bg-white/[0.045] px-4 py-3 shadow-[0_12px_30px_rgba(15,23,42,0.35)]"
      onClick={() => {
        if (onBubbleClick) onBubbleClick()
      }}
    >
      <audio ref={audioRef} src={audioUrl} preload="metadata" autoPlay={autoPlay} />
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleToggle}
          className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-emerald-400/90 to-cyan-400/80 text-slate-950 transition-transform hover:scale-[1.03]"
          title={isPlaying && isActive ? 'Pause voice message' : 'Play voice message'}
        >
          {isPlaying && isActive ? <PauseIcon className="h-5 w-5" /> : <PlayIcon className="h-5 w-5 pl-0.5" />}
        </button>

        <div className="min-w-0 flex-1">
          <div className="group relative block w-full rounded-full py-2">
            <div className="absolute inset-x-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-white/10" />
            <div
              className="absolute left-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-gradient-to-r from-emerald-300 to-cyan-300"
              style={{ width: `${progressPercent}%` }}
            />
            <button
              type="button"
              onClick={handleSeek}
              className="absolute inset-0 z-10 rounded-full"
              title="Seek audio"
            />
            <input
              type="range"
              min="0"
              max={Math.max(duration, 0)}
              step="0.1"
              value={displayedTime}
              onChange={handleScrubChange}
              onMouseUp={commitScrub}
              onTouchEnd={commitScrub}
              onKeyUp={commitScrub}
              className="absolute inset-0 z-20 h-full w-full cursor-pointer opacity-0"
              aria-label="Scrub voice reply"
            />
            <div className="relative flex items-end justify-between gap-1">
              {waveformBars.map((height, index) => {
                const filled = ((index + 1) / waveformBars.length) * 100 <= progressPercent + 4
                return (
                  <span
                    key={`${audioUrl}-${index}`}
                    className={`w-full rounded-full transition-colors ${filled ? 'bg-emerald-300/95' : 'bg-white/25 group-hover:bg-white/35'}`}
                    style={{ height: `${height}px` }}
                  />
                )
              })}
            </div>
          </div>

          <div className="mt-2 flex items-center justify-between text-[11px] font-medium tracking-wide text-white/50">
            <span>Voice reply</span>
            <span>{formatDuration(displayedTime)} / {formatDuration(duration || displayedTime)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default VoiceMessagePlayer
