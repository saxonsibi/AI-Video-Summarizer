import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  ArrowLeftIcon,
  DocumentTextIcon,
  ChatBubbleLeftRightIcon,
  ScissorsIcon,
  ArrowPathIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline'
import { videoAPI } from '../services/api'
import VideoPlayer from '../components/VideoPlayer'
import ChatBot from '../components/ChatBot'
import GenerateShortButton from '../components/GenerateShortButton'
import ProgressTracker from '../components/ProgressTracker'
import SummaryCard from '../components/SummaryCard'
import TranscriptEditor from '../components/TranscriptEditor'

function VideoDetailPage() {
  const { videoId } = useParams()
  const [video, setVideo] = useState(null)
  const [summaries, setSummaries] = useState([])
  const [shorts, setShorts] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('player')
  const [generatingTranscript, setGeneratingTranscript] = useState(false)
  const [editingTranscript, setEditingTranscript] = useState(false)
  const [activeSummary, setActiveSummary] = useState('full')

  useEffect(() => {
    loadVideo()
  }, [videoId])

  const loadVideo = async () => {
    try {
      const response = await videoAPI.getById(videoId)
      setVideo(response.data)
      setSummaries(response.data.summaries || [])
      setShorts(response.data.short_videos || [])
    } catch (error) {
      console.error('Failed to load video:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateTranscript = async () => {
    setGeneratingTranscript(true)
    try {
      await videoAPI.generateTranscript(videoId)
      const maxAttempts = 120
      let attempts = 0
      
      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000))
        const response = await videoAPI.getById(videoId)
        setVideo(response.data)
        
        if (response.data.status === 'completed' || response.data.status === 'failed') {
          break
        }
        attempts++
      }
      
      if (response.data.status === 'completed') {
        loadVideo()
      }
    } catch (error) {
      console.error('Failed to generate transcript:', error)
    } finally {
      setGeneratingTranscript(false)
    }
  }

  const handleGenerateSummary = async (summaryType) => {
    try {
      await videoAPI.generateSummary(videoId, { summary_type: summaryType })
      await new Promise(resolve => setTimeout(resolve, 5000))
      loadVideo()
    } catch (error) {
      console.error('Failed to generate summary:', error)
    }
  }

  const handleShortComplete = () => {
    loadVideo()
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <motion.div 
          className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
      </div>
    )
  }

  if (!video) {
    return (
      <div className="text-center py-20">
        <p className="text-white/60">Video not found</p>
        <Link to="/" className="glow-button inline-block mt-4">
          Back to Home
        </Link>
      </div>
    )
  }

  const hasTranscript = video.transcripts && video.transcripts.length > 0
  const videoUrl = video.original_file ? `http://localhost:8000${video.original_file}` : null

  return (
    <div>
      {/* Header */}
      <motion.div 
        className="flex items-center space-x-4 mb-6"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
      >
        <Link 
          to="/" 
          className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
        >
          <ArrowLeftIcon className="w-6 h-6 text-white/60" />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-white">{video.title}</h1>
          <p className="text-white/50 text-sm">{video.description || 'No description'}</p>
        </div>
      </motion.div>

      {/* Tabs */}
      <motion.div 
        className="flex space-x-1 bg-white/5 p-1 rounded-xl mb-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        {[
          { id: 'player', icon: <span className="text-lg">▶</span>, label: 'Player' },
          { id: 'summaries', icon: <DocumentTextIcon className="w-5 h-5" />, label: 'Summaries' },
          { id: 'chatbot', icon: <ChatBubbleLeftRightIcon className="w-5 h-5" />, label: 'Chatbot', disabled: !hasTranscript },
          { id: 'shorts', icon: <ScissorsIcon className="w-5 h-5" />, label: 'Generate Short', disabled: video.status !== 'completed' }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => !tab.disabled && setActiveTab(tab.id)}
            disabled={tab.disabled}
            className={`tab-button ${activeTab === tab.id ? 'tab-button-active' : ''} ${tab.disabled ? 'opacity-40 cursor-not-allowed' : ''}`}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </motion.div>

      {/* Content */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            {activeTab === 'player' && (
              <div className="space-y-6">
                {videoUrl ? (
                  <VideoPlayer url={videoUrl} title={video.title} />
                ) : (
                  <div className="glass-card text-center py-20">
                    <p className="text-white/50">Video file not available</p>
                  </div>
                )}

                {/* Transcript section */}
                <div className="glass-card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">Transcript</h3>
                    {hasTranscript ? (
                      <div className="flex items-center gap-2">
                        <span className="status-badge status-badge-success flex items-center space-x-1">
                          <CheckCircleIcon className="w-4 h-4" />
                          <span>Ready</span>
                        </span>
                        <button
                          onClick={() => setEditingTranscript(true)}
                          className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg flex items-center gap-2 transition-colors"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                          </svg>
                          Edit
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={handleGenerateTranscript}
                        disabled={generatingTranscript || video.status !== 'completed'}
                        className="glow-button text-sm flex items-center space-x-2 disabled:opacity-50"
                      >
                        {generatingTranscript ? (
                          <>
                            <motion.div 
                              className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full"
                              animate={{ rotate: 360 }}
                              transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                            />
                            <span>Generating...</span>
                          </>
                        ) : (
                          'Generate Transcript'
                        )}
                      </button>
                    )}
                  </div>

                  {editingTranscript ? (
                    <TranscriptEditor
                      videoId={videoId}
                      transcript={video.transcripts[0]}
                      onSave={(text) => {
                        video.transcripts[0].full_text = text
                        setEditingTranscript(false)
                      }}
                      onCancel={() => setEditingTranscript(false)}
                    />
                  ) : (
                    hasTranscript && video.transcripts[0] && (
                      <div className="bg-white/5 rounded-xl p-4 max-h-96 overflow-y-auto custom-scrollbar">
                        <p className="text-white/70 whitespace-pre-wrap">
                          {video.transcripts[0].full_text}
                        </p>
                      </div>
                    )
                  )}

                  {!hasTranscript && video.status !== 'completed' && (
                    <p className="text-white/50">
                      Transcript will be available after video processing is complete.
                    </p>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'summaries' && (
              <div className="space-y-6">
                {/* Generate summaries */}
                <div className="glass-card p-6">
                  <h3 className="text-lg font-semibold text-white mb-4">Generate Summary</h3>
                  <div className="grid grid-cols-3 gap-4">
                    {['full', 'bullet', 'short'].map((type) => (
                      <button
                        key={type}
                        onClick={() => handleGenerateSummary(type)}
                        disabled={!hasTranscript}
                        className="glow-button text-sm disabled:opacity-50"
                      >
                        {type.charAt(0).toUpperCase() + type.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Summary cards - Tab based display */}
                {summaries.length > 0 ? (
                  <div className="space-y-4">
                    {/* Tab buttons */}
                    <div className="flex gap-2 p-1 bg-slate-800/50 rounded-xl">
                      {['full', 'bullet', 'short'].map((type) => {
                        const summary = summaries.find(s => s.summary_type === type)
                        if (!summary) return null
                        return (
                          <button
                            key={type}
                            onClick={() => setActiveSummary(type)}
                            className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-all ${
                              activeSummary === type
                                ? 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white'
                                : 'text-slate-400 hover:text-white hover:bg-white/5'
                            }`}
                          >
                            {type === 'full' ? 'Full Summary' : type === 'bullet' ? 'Bullet Points' : 'Short Script'}
                          </button>
                        )
                      })}
                    </div>
                    
                    {/* Active summary card */}
                    {(() => {
                      const activeSummaryData = summaries.find(s => s.summary_type === activeSummary)
                      return activeSummaryData ? (
                        <SummaryCard
                          key={activeSummaryData.id}
                          video={video}
                          summary={activeSummaryData}
                          onUpdate={loadVideo}
                        />
                      ) : null
                    })()}
                  </div>
                ) : (
                  <div className="glass-card text-center py-12">
                    <DocumentTextIcon className="w-12 h-12 mx-auto text-white/20 mb-4" />
                    <p className="text-white/50">No summaries yet</p>
                    <p className="text-sm text-white/30 mt-2">
                      Generate a summary to see it here
                    </p>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'chatbot' && (
              hasTranscript ? (
                <ChatBot videoId={videoId} />
              ) : (
                <div className="glass-card text-center py-20">
                  <ChatBubbleLeftRightIcon className="w-16 h-16 mx-auto text-white/20 mb-4" />
                  <p className="text-white/50">
                    Transcript is required to use the chatbot
                  </p>
                  <p className="text-sm text-white/30 mt-2">
                    Generate a transcript first
                  </p>
                </div>
              )
            )}

            {activeTab === 'shorts' && (
              <div className="space-y-6">
                {video.status === 'completed' ? (
                  <GenerateShortButton
                    videoId={videoId}
                    onGenerationComplete={handleShortComplete}
                  />
                ) : (
                  <div className="glass-card text-center py-20">
                    <ScissorsIcon className="w-16 h-16 mx-auto text-white/20 mb-4" />
                    <p className="text-white/50">
                      Video processing must be complete
                    </p>
                    <p className="text-sm text-white/30 mt-2">
                      Current status: {video.status}
                    </p>
                  </div>
                )}

                {/* Generated shorts */}
                {shorts.length > 0 && (
                  <div className="glass-card p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">
                      Generated Shorts ({shorts.length})
                    </h3>
                    <div className="space-y-3">
                      {shorts.map((short) => (
                        <div
                          key={short.id}
                          className="flex items-center justify-between p-3 bg-white/5 rounded-xl hover:bg-white/10 transition-colors"
                        >
                          <div className="flex items-center space-x-3">
                            <div className="p-2 bg-purple-500/20 rounded-lg">
                              <ScissorsIcon className="w-5 h-5 text-purple-400" />
                            </div>
                            <div>
                              <p className="text-white">{short.duration?.toFixed(1)}s</p>
                              <p className="text-xs text-white/40">
                                {new Date(short.created_at).toLocaleDateString()}
                              </p>
                            </div>
                          </div>
                          <a
                            href={short.file}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="glow-button text-sm"
                          >
                            Download
                          </a>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </div>

        {/* Sidebar */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <ProgressTracker video={video} />

          {/* Video info */}
          <div className="glass-card mt-6 p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Video Info</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-white/50">Duration</span>
                <span className="text-white">
                  {video.duration ? `${(video.duration / 60).toFixed(1)} min` : '-'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/50">Format</span>
                <span className="text-white uppercase">{video.file_format || '-'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/50">Size</span>
                <span className="text-white">
                  {(video.file_size / (1024 * 1024)).toFixed(1)} MB
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/50">Uploaded</span>
                <span className="text-white">
                  {new Date(video.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default VideoDetailPage
