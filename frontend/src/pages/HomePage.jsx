import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  VideoCameraIcon, 
  DocumentTextIcon, 
  ChatBubbleLeftRightIcon,
  SparklesIcon,
  ClockIcon,
  TrashIcon,
  ArrowRightIcon,
  CloudArrowUpIcon,
  PlayIcon,
  LinkIcon,
  ComputerDesktopIcon
} from '@heroicons/react/24/outline'
import { videoAPI } from '../services/api'
import UploadVideo from '../components/UploadVideo'
import UploadYouTubeURL from '../components/UploadYouTubeURL'
import ScreenRecorder from '../components/ScreenRecorder'

// Custom AI Video Logo SVG
function AIVideoLogo({ className = "w-12 h-12" }) {
  return (
    <svg 
      width="48" 
      height="48" 
      viewBox="0 0 32 32" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <rect 
        x="2" 
        y="2" 
        width="28" 
        height="28" 
        rx="8" 
        fill="url(#heroLogoGradient)"
      />
      <rect 
        x="2" 
        y="2" 
        width="28" 
        height="28" 
        rx="8" 
        fill="url(#heroLogoGradient)"
        filter="url(#heroGlow)"
        opacity="0.4"
      />
      <path 
        d="M12 9L22 16L12 23V9Z" 
        fill="white" 
        stroke="white" 
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <circle cx="24" cy="8" r="1.5" fill="#f472b6" />
      <circle cx="26" cy="10" r="1" fill="#a78bfa" opacity="0.8" />
      <circle cx="8" cy="24" r="1" fill="#f472b6" opacity="0.6" />
      <path 
        d="M23 6L24 4L25 6" 
        stroke="#f472b6" 
        strokeWidth="1" 
        strokeLinecap="round"
        opacity="0.8"
      />
      <defs>
        <linearGradient id="heroLogoGradient" x1="2" y1="2" x2="30" y2="30" gradientUnits="userSpaceOnUse">
          <stop stopColor="#8b5cf6" />
          <stop offset="0.5" stopColor="#a855f7" />
          <stop offset="1" stopColor="#ec4899" />
        </linearGradient>
        <filter id="heroGlow" x="-4" y="-4" width="40" height="40">
          <feGaussianBlur stdDeviation="3" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
    </svg>
  )
}

function HomePage() {
  const location = useLocation()
  const isPanelMode = new URLSearchParams(location.search).get('layout') === 'panel'
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(true)
  const [showUpload, setShowUpload] = useState(false)
  const [uploadType, setUploadType] = useState('file') // 'file', 'youtube', 'screen'
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })

  useEffect(() => {
    if (isPanelMode) return
    loadVideos()
    
    // Track mouse for parallax effect
    const handleMouseMove = (e) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth - 0.5) * 20,
        y: (e.clientY / window.innerHeight - 0.5) * 20
      })
    }
    
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [isPanelMode])

  useEffect(() => {
    if (!isPanelMode) return
    loadVideos()
  }, [isPanelMode])

  // Auto-refresh video list when there are pending videos
  useEffect(() => {
    const hasPendingVideos = videos.some(v => v.status === 'pending' || v.status === 'processing')
    
    if (!hasPendingVideos || loading) return
    
    const interval = setInterval(() => {
      loadVideos()
    }, 3000) // Refresh every 3 seconds
    
    return () => clearInterval(interval)
  }, [videos, loading])

  const loadVideos = async () => {
    try {
      console.log("Loading videos...")
      const response = await videoAPI.getAll()
      console.log("Response:", response)
      console.log("Response data:", response.data)
      
      // Handle DRF paginated response { count, results } or plain array
      const data = response.data
      if (data && data.results && Array.isArray(data.results)) {
        setVideos(data.results)
        console.log("Setting videos from results:", data.results.length)
      } else if (Array.isArray(data)) {
        setVideos(data)
        console.log("Setting videos from array:", data.length)
      } else {
        console.log("No videos found, data:", data)
        setVideos([])
      }
    } catch (error) {
      console.error('Failed to load videos:', error)
      console.error('Error response:', error.response)
      setVideos([])
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this video?')) return
    
    try {
      await videoAPI.delete(id)
      setVideos(videos.filter(v => v.id !== id))
    } catch (error) {
      console.error('Failed to delete video:', error)
    }
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    })
  }

  const formatDuration = (seconds) => {
    if (!seconds) return '-'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getYouTubeId = (url) => {
    if (!url) return null
    const patterns = [
      /youtube\.com\/watch\?v=([\w-]+)/,
      /youtu\.be\/([\w-]+)/,
      /youtube\.com\/embed\/([\w-]+)/,
      /youtube\.com\/v\/([\w-]+)/,
      /youtube\.com\/shorts\/([\w-]+)/
    ]
    for (const p of patterns) {
      const match = url.match(p)
      if (match && match[1]) return match[1]
    }
    return null
  }

  const getStatusBadge = (status, progress = 0) => {
    switch (status) {
      case 'completed':
        return <span className="status-badge status-badge-success">✅ Completed</span>
      case 'transcribing':
        return (
          <span className="status-badge bg-blue-500/20 text-blue-400">
            🎤 Transcribing ({progress}%)
          </span>
        )
      case 'summarizing':
        return (
          <span className="status-badge bg-purple-500/20 text-purple-400">
            📝 Summarizing ({progress}%)
          </span>
        )
      case 'processing':
        return (
          <span className="status-badge status-badge-processing">
            ⚡ Processing ({progress}%)
          </span>
        )
      case 'failed':
        return <span className="status-badge status-badge-failed">❌ Failed</span>
      case 'pending':
        return <span className="status-badge bg-amber-500/20 text-amber-400">⏳ Queued</span>
      default:
        return <span className="status-badge bg-white/10 text-white/60">⏳ Queued</span>
    }
  }

  return (
    <div className={isPanelMode ? 'panel-home' : 'relative'}>
      {/* Hero Section */}
      <motion.section 
        className={`hero-gradient relative overflow-hidden ${isPanelMode ? 'rounded-3xl mb-6' : ''}`}
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        {/* Animated background elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <motion.div 
            className="absolute top-20 left-1/4 w-72 h-72 bg-indigo-500/20 rounded-full blur-3xl"
            animate={{
              x: [0, 30, 0],
              y: [0, -30, 0],
            }}
            transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div 
            className="absolute top-40 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl"
            animate={{
              x: [0, -40, 0],
              y: [0, 40, 0],
            }}
            transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.div 
            className="absolute bottom-20 left-1/3 w-80 h-80 bg-pink-500/15 rounded-full blur-3xl"
            animate={{
              x: [0, 50, 0],
              y: [0, -20, 0],
            }}
            transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
          />
        </div>

        <div className={`relative z-10 container mx-auto px-4 ${isPanelMode ? 'py-10' : 'py-24'}`}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex justify-center mb-6"
          >
            <div className="relative">
              <motion.div 
                className="absolute inset-0 bg-purple-500/60 blur-2xl opacity-60 rounded-3xl"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 3, repeat: Infinity }}
              />
              <div className={`relative ${isPanelMode ? 'w-16 h-16' : 'w-20 h-20'} p-1 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-sm`}>
                <AIVideoLogo className="w-full h-full" />
              </div>
            </div>
          </motion.div>

          <motion.h1 
            className={`${isPanelMode ? 'text-4xl leading-tight' : 'text-6xl md:text-7xl'} font-bold text-center mb-6`}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <span className="text-white">Transform Your Videos with</span>
            <br />
            <span className="gradient-text">AI Magic</span>
          </motion.h1>

          <motion.p 
            className={`${isPanelMode ? 'text-base max-w-md mb-8' : 'text-xl max-w-2xl mb-10'} text-white/60 text-center mx-auto`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            Upload files, paste YouTube links, or capture your screen to generate
            readable transcripts, structured summaries, and source-grounded answers with timestamps.
          </motion.p>

          <motion.div 
            className="flex flex-wrap justify-center gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <motion.button
              onClick={() => setShowUpload(true)}
              className={`glow-button flex items-center space-x-2 ${isPanelMode ? 'px-6 py-3 text-base' : 'px-8 py-4 text-lg'}`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
            >
              <CloudArrowUpIcon className="w-6 h-6" />
              <span>Upload Video</span>
            </motion.button>
            {isPanelMode ? (
              <motion.button
                onClick={() => {
                  const videosSection = document.getElementById('videos-section')
                  videosSection?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                }}
                className="glass-card px-6 py-3 text-base font-semibold text-white flex items-center space-x-2"
                whileHover={{ scale: 1.05, bgColor: 'rgba(255,255,255,0.15)' }}
                whileTap={{ scale: 0.98 }}
              >
                <span>View Videos</span>
                <ArrowRightIcon className="w-5 h-5" />
              </motion.button>
            ) : (
              <Link to="#features">
                <motion.button
                  className="glass-card px-8 py-4 text-lg font-semibold text-white flex items-center space-x-2"
                  whileHover={{ scale: 1.05, bgColor: 'rgba(255,255,255,0.15)' }}
                  whileTap={{ scale: 0.98 }}
                >
                  <span>Explore Features</span>
                  <ArrowRightIcon className="w-5 h-5" />
                </motion.button>
              </Link>
            )}
          </motion.div>
        </div>

        {/* Scroll indicator */}
        {!isPanelMode && (
          <motion.div 
            className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <div className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center pt-2">
              <motion.div 
                className="w-1.5 h-1.5 bg-white/60 rounded-full"
                animate={{ y: [0, 12, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
            </div>
          </motion.div>
        )}
      </motion.section>

      {/* Upload Section */}
      {showUpload && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className={isPanelMode ? 'mb-6' : 'container mx-auto px-4 mb-16'}
        >
          {/* Upload Type Tabs */}
          <div className={`flex justify-center ${isPanelMode ? 'mb-4' : 'mb-8'}`}>
            <div className="glass-card p-1 rounded-xl flex space-x-1">
              <motion.button
                onClick={() => setUploadType('file')}
                className={`px-6 py-3 rounded-lg flex items-center space-x-2 transition-colors ${
                  uploadType === 'file' 
                    ? 'bg-indigo-500/20 text-indigo-300' 
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <CloudArrowUpIcon className="w-5 h-5" />
                <span>Upload File</span>
              </motion.button>
              <motion.button
                onClick={() => setUploadType('youtube')}
                className={`px-6 py-3 rounded-lg flex items-center space-x-2 transition-colors ${
                  uploadType === 'youtube' 
                    ? 'bg-red-500/20 text-red-300' 
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <LinkIcon className="w-5 h-5" />
                <span>YouTube URL</span>
              </motion.button>
              <motion.button
                onClick={() => setUploadType('screen')}
                className={`px-6 py-3 rounded-lg flex items-center space-x-2 transition-colors ${
                  uploadType === 'screen' 
                    ? 'bg-purple-500/20 text-purple-300' 
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <ComputerDesktopIcon className="w-5 h-5" />
                <span>Screen Record</span>
              </motion.button>
            </div>
          </div>

          {/* Upload Component */}
          {uploadType === 'file' && (
            <UploadVideo onUploadComplete={() => {
              loadVideos()
              setShowUpload(false)
            }} />
          )}
          {uploadType === 'youtube' && (
            <UploadYouTubeURL onUploadComplete={() => {
              loadVideos()
              setShowUpload(false)
            }} />
          )}
          {uploadType === 'screen' && (
            <ScreenRecorder onVideoRecorded={() => {
              loadVideos()
              setShowUpload(false)
            }} />
          )}
        </motion.div>
      )}

      {/* Features Section */}
      {!isPanelMode && (
      <section id="features" className="py-20">
        <div className="container mx-auto px-4">
          <motion.h2 
            className="text-4xl font-bold text-center text-white mb-4"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            Powerful <span className="gradient-text">Features</span>
          </motion.h2>
          
          <motion.p 
            className="text-white/60 text-center mb-12 max-w-xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
          >
            Built around the features the product actually supports today, not placeholder AI copy.
          </motion.p>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                icon: DocumentTextIcon,
                title: 'Structured Summary Dashboard',
                desc: 'Every video is organized into TLDR, key points, action items, and chapter jumps instead of one long summary block.',
              },
              {
                icon: ChatBubbleLeftRightIcon,
                title: 'Grounded Video Chatbot',
                desc: 'Ask questions about the full video or a specific moment and get concise answers with clickable timestamp sources.',
              },
              {
                icon: SparklesIcon,
                title: 'Readable Transcript Output',
                desc: 'Playback-synced transcript view with clean formatting, search, active highlights, and timestamp-based navigation.',
              },
              {
                icon: PlayIcon,
                title: 'AI Highlight Timeline',
                desc: 'Chapters and smart moments appear directly on the player so users can jump through long videos quickly.',
              },
              {
                icon: ComputerDesktopIcon,
                title: 'Multiple Input Workflows',
                desc: 'Process local uploads, YouTube URLs, and screen recordings inside the same workflow without changing tools.',
              },
              {
                icon: ClockIcon,
                title: 'Progressive Processing UX',
                desc: 'Users see upload, transcript, summary, and indexing progress while the system builds results in the background.',
              },
            ].map((feature, index) => (
              <motion.div
                key={index}
                className="feature-card"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ y: -5 }}
              >
                <div className="feature-icon">
                  <feature.icon className="w-8 h-8 text-indigo-400" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-white/60 text-sm">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      )}

      {/* Videos Section */}
      <section id="videos-section" className={isPanelMode ? 'py-2' : 'py-16'}>
        <div className={isPanelMode ? '' : 'container mx-auto px-4'}>
          <motion.div 
            className="flex flex-wrap items-center justify-between mb-8 gap-4"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <div>
              <h2 className="text-3xl font-bold text-white">Your Videos</h2>
              <p className="text-white/60">{videos.length} video{videos.length !== 1 ? 's' : ''}</p>
            </div>
            <button
              onClick={() => setShowUpload((prev) => !prev)}
              className={`${
                isPanelMode ? 'glass-card px-4 py-2.5 text-sm' : 'glow-button'
              } flex items-center space-x-2`}
            >
              <CloudArrowUpIcon className="w-5 h-5" />
              <span>{showUpload ? 'Hide Upload' : 'Add Video'}</span>
            </button>
          </motion.div>

          {loading ? (
            <div className={isPanelMode ? 'grid grid-cols-1 gap-4' : 'grid md:grid-cols-2 lg:grid-cols-3 gap-6'}>
              {[1, 2, 3].map((i) => (
                <div key={i} className="glass-card p-4">
                  <div className="skeleton aspect-video rounded-xl mb-4" />
                  <div className="skeleton h-6 w-3/4 rounded mb-2" />
                  <div className="skeleton h-4 w-1/2 rounded" />
                </div>
              ))}
            </div>
          ) : videos.length === 0 ? (
            <motion.div 
              className={`glass-card text-center ${isPanelMode ? 'py-8' : 'py-16'}`}
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
            >
              <motion.div
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <VideoCameraIcon className="w-24 h-24 mx-auto text-indigo-400/30 mb-6" />
              </motion.div>
              <h3 className="text-3xl font-bold gradient-text mb-3">No videos yet</h3>
              <p className="text-white/50 text-lg mb-8">Upload your first video and watch the AI work its magic</p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button
                  onClick={() => setShowUpload(true)}
                  className="glow-button inline-flex items-center justify-center space-x-2 px-8 py-3"
                >
                  <CloudArrowUpIcon className="w-5 h-5" />
                  <span>Upload Video</span>
                </button>
              </div>
            </motion.div>
          ) : (
            <div className={isPanelMode ? 'grid grid-cols-1 gap-4' : 'grid md:grid-cols-2 lg:grid-cols-3 gap-6'}>
              {videos.map((video, index) => (
                <motion.div
                  key={video.id}
                  className="video-grid-item group"
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Link to={isPanelMode ? `/video/${video.id}?layout=panel` : `/video/${video.id}`} className="block">
                    <div className="aspect-video bg-white/5 relative overflow-hidden">
                    {video.original_file ? (
                      <>
                        <video
                          src={video.original_file}
                          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                          <div className="w-16 h-16 rounded-full bg-white/20 backdrop-blur-lg flex items-center justify-center">
                            <PlayIcon className="w-8 h-8 text-white ml-1" />
                          </div>
                        </div>
                      </>
                    ) : video.youtube_url ? (
                      <>
                        <img
                          src={`https://img.youtube.com/vi/${getYouTubeId(video.youtube_url)}/hqdefault.jpg`}
                          alt={video.title}
                          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                          <div className="w-16 h-16 rounded-full bg-white/20 backdrop-blur-lg flex items-center justify-center">
                            <PlayIcon className="w-8 h-8 text-white ml-1" />
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <VideoCameraIcon className="w-16 h-16 text-white/20" />
                      </div>
                    )}
                    </div>
                    
                    <div className="p-4">
                      <h3 className="font-semibold text-white truncate mb-2 group-hover:text-indigo-300 transition-colors">
                        {video.title}
                      </h3>
                      
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center space-x-3 text-white/50">
                          <span className="flex items-center space-x-1">
                            <ClockIcon className="w-4 h-4" />
                            <span>{formatDuration(video.duration)}</span>
                          </span>
                          <span>{formatDate(video.created_at)}</span>
                        </div>
                        {getStatusBadge(video.status, video.processing_progress)}
                      </div>
                      
                      <div className="mt-3 pt-3 border-t border-white/10 flex items-center justify-between text-xs text-white/40">
                        <span>
                          {video.transcripts_count || 0} transcript{video.transcripts_count !== 1 ? 's' : ''}
                          {video.summaries_count > 0 && ` • ${video.summaries_count} summary`}
                        </span>
                        <button
                          onClick={(e) => {
                            e.preventDefault()
                            handleDelete(video.id)
                          }}
                          className="delete-button"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </Link>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* CTA Section */}
      {!isPanelMode && (
      <section className="py-20">
        <div className="container mx-auto px-4">
          <motion.div 
            className="glass-card relative overflow-hidden"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            {/* Background gradient */}
            <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10" />
            
            <div className="relative z-10 text-center py-12 px-8">
              <div className="flex justify-center mb-4">
                <AIVideoLogo className="w-16 h-16" />
              </div>
              <h2 className="text-3xl font-bold text-white mb-4">
                Ready to Transform Your Videos?
              </h2>
              <p className="text-white/60 max-w-xl mx-auto mb-8">
                Start uploading videos and experience the power of AI-driven 
                summarization, analysis, and content creation.
              </p>
              <button
                onClick={() => setShowUpload(true)}
                className="glow-button inline-flex items-center space-x-2"
              >
                <CloudArrowUpIcon className="w-5 h-5" />
                <span>Get Started Free</span>
              </button>
            </div>
          </motion.div>
        </div>
      </section>
      )}
    </div>
  )
}

export default HomePage
