import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  VideoCameraIcon, 
  DocumentTextIcon, 
  ChatBubbleLeftRightIcon,
  ScissorsIcon,
  SparklesIcon,
  ClockIcon,
  TrashIcon,
  ArrowRightIcon,
  CloudArrowUpIcon,
  PlayIcon
} from '@heroicons/react/24/outline'
import { videoAPI } from '../services/api'
import UploadVideo from '../components/UploadVideo'

function HomePage() {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(true)
  const [showUpload, setShowUpload] = useState(false)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })

  useEffect(() => {
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
  }, [])

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

  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed':
        return <span className="status-badge status-badge-success">Completed</span>
      case 'processing':
        return <span className="status-badge status-badge-processing">Processing</span>
      case 'failed':
        return <span className="status-badge status-badge-failed">Failed</span>
      default:
        return <span className="status-badge bg-white/10 text-white/60">Pending</span>
    }
  }

  return (
    <div className="relative">
      {/* Hero Section */}
      <motion.section 
        className="hero-gradient relative overflow-hidden"
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

        <div className="relative z-10 container mx-auto px-4 py-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex justify-center mb-6"
          >
            <div className="relative">
              <motion.div 
                className="absolute inset-0 bg-indigo-500 blur-2xl opacity-50 rounded-full"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 3, repeat: Infinity }}
              />
              <div className="relative p-4 glass-card rounded-2xl">
                <SparklesIcon className="w-12 h-12 gradient-text" />
              </div>
            </div>
          </motion.div>

          <motion.h1 
            className="text-6xl md:text-7xl font-bold text-center mb-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <span className="text-white">Transform Your Videos with</span>
            <br />
            <span className="gradient-text">AI Magic</span>
          </motion.h1>

          <motion.p 
            className="text-xl text-white/60 text-center max-w-2xl mx-auto mb-10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            Upload any video and get instant AI-powered summaries, 
            ask questions in natural language, and generate engaging 
            short clips automatically.
          </motion.p>

          <motion.div 
            className="flex flex-wrap justify-center gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <motion.button
              onClick={() => setShowUpload(true)}
              className="glow-button flex items-center space-x-2 px-8 py-4 text-lg"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.98 }}
            >
              <CloudArrowUpIcon className="w-6 h-6" />
              <span>Upload Video</span>
            </motion.button>
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
          </motion.div>
        </div>

        {/* Scroll indicator */}
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
      </motion.section>

      {/* Upload Section */}
      {showUpload && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="container mx-auto px-4 mb-16"
        >
          <UploadVideo onUploadComplete={() => {
            loadVideos()
            setShowUpload(false)
          }} />
        </motion.div>
      )}

      {/* Features Section */}
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
            Everything you need to analyze, understand, and repurpose your video content
          </motion.p>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { icon: DocumentTextIcon, title: 'Auto Summaries', desc: 'Get detailed, bullet-point, and short summaries powered by BART AI' },
              { icon: ChatBubbleLeftRightIcon, title: 'Smart Chatbot', desc: 'Ask questions about your video content in natural language' },
              { icon: ScissorsIcon, title: 'Short Generator', desc: 'Automatically create 9:16 short videos from highlights' },
              { icon: SparklesIcon, title: 'AI Transcription', desc: 'Accurate speech-to-text with Faster-Whisper technology' }
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

      {/* Videos Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
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
              onClick={() => setShowUpload(true)}
              className="glow-button flex items-center space-x-2"
            >
              <CloudArrowUpIcon className="w-5 h-5" />
              <span>Add Video</span>
            </button>
          </motion.div>

          {loading ? (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
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
              className="glass-card text-center py-16"
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
            >
              <VideoCameraIcon className="w-20 h-20 mx-auto text-white/20 mb-4" />
              <h3 className="text-2xl font-semibold text-white mb-2">No videos yet</h3>
              <p className="text-white/60 mb-6">Upload your first video to get started</p>
              <button
                onClick={() => setShowUpload(true)}
                className="glow-button inline-flex items-center space-x-2"
              >
                <CloudArrowUpIcon className="w-5 h-5" />
                <span>Upload Video</span>
              </button>
            </motion.div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {videos.map((video, index) => (
                <motion.div
                  key={video.id}
                  className="video-grid-item group"
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.05 }}
                >
                  <Link to={`/video/${video.id}`} className="block">
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
                        {getStatusBadge(video.status)}
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
              <SparklesIcon className="w-16 h-16 gradient-text mx-auto mb-4" />
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
    </div>
  )
}

export default HomePage
