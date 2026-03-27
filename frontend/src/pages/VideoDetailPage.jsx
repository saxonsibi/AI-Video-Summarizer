import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  ArrowLeftIcon,
  ChatBubbleLeftRightIcon,
  CheckCircleIcon,
  PlayCircleIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  QueueListIcon,
} from '@heroicons/react/24/outline'
import { videoAPI } from '../services/api'
import VideoPlayer from '../components/VideoPlayer'
import ChatBot from '../components/ChatBot'
import ProgressTracker from '../components/ProgressTracker'
import TranscriptEditor from '../components/TranscriptEditor'
import { SummarySkeleton, TranscriptSkeleton } from '../components/Skeleton'
import SummaryDashboard from '../components/SummaryDashboard'
import { LANGUAGE_OPTIONS } from '../constants/languages'
import { useToast } from '../context/ToastContext'

const EMPTY_STRUCTURED_SUMMARY = {
  tldr: '',
  key_points: [],
  action_items: [],
  chapters: [],
}

function normalizeLegacySummaryContent(summary) {
  const raw = summary?.content
  if (!raw) return {}
  if (typeof raw === 'object') return raw
  try {
    return JSON.parse(raw)
  } catch {
    return { content: String(raw) }
  }
}

function splitLegacyBulletText(value) {
  return String(value || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^[\u2022*-]+\s*/, '').trim())
    .filter(Boolean)
}

function extractLegacyActionItems(...values) {
  const actionPatterns = [
    /\b(?:should|need to|must|remember to|make sure|try to|focus on|practice|review|prepare|follow up|check|use)\b/i,
  ]

  const seen = new Set()
  const items = []

  for (const value of values) {
    const candidates = Array.isArray(value)
      ? value
      : splitLegacyBulletText(value).concat(
          String(value || '')
            .split(/(?<=[.!?])\s+/)
            .map((line) => line.trim())
            .filter(Boolean),
        )

    for (const raw of candidates) {
      const normalized = String(raw || '').replace(/^[\u2022*-]+\s*/, '').trim()
      if (!normalized) continue
      if (!actionPatterns.some((pattern) => pattern.test(normalized))) continue
      const dedupeKey = normalized.toLowerCase()
      if (seen.has(dedupeKey)) continue
      seen.add(dedupeKey)
      items.push(normalized.replace(/[.]+$/, ''))
      if (items.length >= 6) return items
    }
  }

  return items
}

function buildSummaryDashboardFallback(summaries) {
  const items = Array.isArray(summaries) ? summaries : []
  const shortSummary = items.find((item) => item?.summary_type === 'short')
  const bulletSummary = items.find((item) => item?.summary_type === 'bullet')
  const timestampSummary = items.find((item) => item?.summary_type === 'timestamps')

  const shortContent = normalizeLegacySummaryContent(shortSummary)
  const bulletContent = normalizeLegacySummaryContent(bulletSummary)
  const timestampContent = normalizeLegacySummaryContent(timestampSummary)

  const tldr = String(
    shortContent?.content
      || shortContent?.summary_text
      || shortSummary?.content
      || ''
  ).trim()

  const keyPoints = (
    bulletContent?.key_points
    || bulletContent?.key_topics
    || splitLegacyBulletText(bulletContent?.content || bulletContent?.summary_text || bulletSummary?.content)
  )

  const actionItems = (
    bulletContent?.action_items
    || shortContent?.action_items
    || timestampContent?.action_items
    || extractLegacyActionItems(
      bulletContent?.content || bulletContent?.summary_text || bulletSummary?.content,
      shortContent?.content || shortContent?.summary_text || shortSummary?.content,
      timestampContent?.content || timestampContent?.summary_text || timestampSummary?.content,
      keyPoints,
    )
  )

  const chapters = Array.isArray(timestampContent?.chapters)
    ? timestampContent.chapters
    : splitLegacyBulletText(timestampContent?.content || timestampContent?.summary_text || timestampSummary?.content).map((line, index) => {
        const match = line.match(/(\d{1,2}:\d{2}(?::\d{2})?)\s*[-—:]?\s*(.*)/)
        return {
          timestamp: match?.[1] || '00:00',
          title: (match?.[2] || line || `Chapter ${index + 1}`).trim(),
        }
      })

  return {
    tldr,
    key_points: Array.isArray(keyPoints) ? keyPoints.filter(Boolean) : [],
    action_items: Array.isArray(actionItems) ? actionItems.filter(Boolean) : [],
    chapters: Array.isArray(chapters) ? chapters.filter((item) => item?.title || item?.timestamp) : [],
  }
}

function VideoDetailPage() {
  const { videoId } = useParams()
  const location = useLocation()
  const isPanelMode = new URLSearchParams(location.search).get('layout') === 'panel'

  const toast = useToast()
  const [video, setVideo] = useState(null)
  const [summaries, setSummaries] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('player')
  const [generatingTranscript, setGeneratingTranscript] = useState(false)
  const [editingTranscript, setEditingTranscript] = useState(false)
  const [transcriptQuery, setTranscriptQuery] = useState('')
  const [currentPlayerTime, setCurrentPlayerTime] = useState(0)
  const [seekToSeconds, setSeekToSeconds] = useState(null)
  const [momentContext, setMomentContext] = useState(null)
  const [activeTranscriptSegmentKey, setActiveTranscriptSegmentKey] = useState(null)
  const transcriptContainerRef = useRef(null)
  const transcriptRowRefs = useRef(new Map())
  const previousStatusRef = useRef(null)
  const previousTranscriptCountRef = useRef(0)
  const previousSummaryCountRef = useRef(0)
  const didPrimeStatusRef = useRef(false)

  useEffect(() => {
    loadVideo()
  }, [videoId])

  useEffect(() => {
    if (!video || ['completed', 'failed'].includes(video.status)) return undefined
    const intervalId = setInterval(() => {
      loadVideo(true)
    }, 4000)
    return () => clearInterval(intervalId)
  }, [video?.id, video?.status])

  useEffect(() => {
    if (!video) return

    const transcriptCount = Array.isArray(video.transcripts) ? video.transcripts.length : 0
    const summaryCount = Array.isArray(video.summaries) ? video.summaries.length : 0

    if (!didPrimeStatusRef.current) {
      previousStatusRef.current = video.status
      previousTranscriptCountRef.current = transcriptCount
      previousSummaryCountRef.current = summaryCount
      didPrimeStatusRef.current = true
      return
    }

    if (previousStatusRef.current !== video.status) {
      const statusMessages = {
        pending: 'Video queued for processing.',
        uploaded: 'Upload accepted. Preparing the job.',
        processing: 'Upload complete. Extracting audio now.',
        extracting_audio: 'Extracting audio now.',
        transcribing: 'Transcript generation started.',
        cleaning_transcript: 'Cleaning and validating the transcript.',
        transcript_ready: 'Transcript ready.',
        summarizing_quick: 'Building the quick summary first.',
        summarizing_final: 'Refining the final structured summary.',
        summarizing: 'Building summaries.',
        indexing_chat: 'Preparing the chatbot index.',
        completed: 'Video intelligence is ready.',
        failed: video.error_message || 'Processing failed. Check the status panel for details.',
      }

      if (video.status === 'failed') {
        toast.error(statusMessages[video.status])
      } else if (video.status === 'completed') {
        toast.success(statusMessages[video.status])
      } else if (statusMessages[video.status]) {
        toast.info(statusMessages[video.status])
      }
    }

    if (transcriptCount > previousTranscriptCountRef.current) {
      toast.success('Transcript generated successfully.')
    }

    if (summaryCount > previousSummaryCountRef.current) {
      toast.success('Summary sections updated.')
    }

    previousStatusRef.current = video.status
    previousTranscriptCountRef.current = transcriptCount
    previousSummaryCountRef.current = summaryCount
  }, [toast, video])

  const loadVideo = async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const response = await videoAPI.getById(videoId)
      setVideo(response.data)
      setSummaries(response.data.summaries || [])
    } catch (error) {
      console.error('Failed to load video:', error)
    } finally {
      if (!silent) setLoading(false)
    }
  }

  const handleGenerateTranscript = async () => {
    setGeneratingTranscript(true)
    try {
      toast.info('Starting transcript generation.')
      await videoAPI.generateTranscript(videoId, {
        transcription_language: 'auto',
        output_language: 'auto',
      })
      await loadVideo()
    } catch (error) {
      console.error('Failed to generate transcript:', error)
      toast.error('Transcript generation failed to start.')
    } finally {
      setGeneratingTranscript(false)
    }
  }

  const handleGenerateSummary = async (summaryType) => {
    try {
      toast.info(`Generating ${summaryType} summary.`)
      await videoAPI.generateSummary(videoId, {
        summary_type: summaryType,
        output_language: 'auto',
      })
      await loadVideo(true)
      await loadVideo()
    } catch (error) {
      console.error('Failed to generate summary:', error)
      toast.error(`Failed to generate ${summaryType} summary.`)
    }
  }

  const buildDisplayTranscript = (text) => {
    if (!text) return ''
    let out = text
    out = out.replace(/\s+/g, ' ').trim()
    out = out.replace(/([.!?])([A-Za-z])/g, '$1 $2')
    out = out.replace(/\s{2,}/g, ' ').trim()

    const sentences = out
      .split(/(?<=[.!?])\s+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => s.charAt(0).toUpperCase() + s.slice(1))

    const rebuilt = sentences.join(' ')
    const finalSents = rebuilt.split(/(?<=[.!?])\s+/).filter(Boolean)
    const paragraphs = []
    for (let i = 0; i < finalSents.length; i += 4) {
      paragraphs.push(finalSents.slice(i, i + 4).join(' '))
    }
    return paragraphs.join('\n\n').trim()
  }

  const latestTranscript = useMemo(
    () => (video?.transcripts && video.transcripts.length > 0 ? video.transcripts[0] : null),
    [video],
  )

  const transcriptSegments = useMemo(() => {
    if (!latestTranscript?.json_data) return []
    if (Array.isArray(latestTranscript.json_data)) return latestTranscript.json_data
    if (Array.isArray(latestTranscript.json_data.segments)) return latestTranscript.json_data.segments
    return []
  }, [latestTranscript])

  const filteredSegments = useMemo(() => {
    if (!transcriptQuery.trim()) return transcriptSegments
    const q = transcriptQuery.toLowerCase()
    return transcriptSegments.filter((seg) => String(seg.text || '').toLowerCase().includes(q))
  }, [transcriptQuery, transcriptSegments])

  useEffect(() => {
    const activeSegment = filteredSegments.find((segment) => {
      const start = Number(segment.start ?? segment.start_time ?? 0)
      const end = Number(segment.end ?? segment.end_time ?? start + 1)
      return currentPlayerTime >= start && currentPlayerTime <= end
    })

    if (!activeSegment) {
      setActiveTranscriptSegmentKey((prev) => (prev == null ? prev : null))
      return
    }

    const start = Number(activeSegment.start ?? activeSegment.start_time ?? 0)
    const end = Number(activeSegment.end ?? activeSegment.end_time ?? start + 1)
    const nextKey = `${start}-${end}`
    setActiveTranscriptSegmentKey((prev) => (prev === nextKey ? prev : nextKey))
  }, [currentPlayerTime, filteredSegments])

  useEffect(() => {
    if (!activeTranscriptSegmentKey) return
    const container = transcriptContainerRef.current
    const row = transcriptRowRefs.current.get(activeTranscriptSegmentKey)
    if (!container || !row) return

    const padding = 12
    const visibleTop = container.scrollTop
    const visibleBottom = visibleTop + container.clientHeight
    const rowTop = row.offsetTop
    const rowBottom = rowTop + row.offsetHeight

    if (rowTop < visibleTop + padding) {
      container.scrollTop = Math.max(0, rowTop - padding)
      return
    }

    if (rowBottom > visibleBottom - padding) {
      container.scrollTop = Math.max(0, rowBottom - container.clientHeight + padding)
    }

  }, [activeTranscriptSegmentKey])

  const hasTranscript = Boolean(latestTranscript)
  const structuredSummary = video?.structured_summary || EMPTY_STRUCTURED_SUMMARY
  const fallbackSummary = useMemo(() => buildSummaryDashboardFallback(video?.summaries || []), [video?.summaries])
  const effectiveSummary = useMemo(() => {
    const hasStructuredContent = Boolean(
      structuredSummary.tldr
      || structuredSummary.key_points?.length
      || structuredSummary.action_items?.length
      || structuredSummary.chapters?.length,
    )
    return hasStructuredContent ? structuredSummary : fallbackSummary
  }, [structuredSummary, fallbackSummary])
  const metadata = video?.processing_metadata || {}
  const hasStructuredSummaryContent = Boolean(
    effectiveSummary.tldr
      || effectiveSummary.key_points?.length
      || effectiveSummary.action_items?.length
      || effectiveSummary.chapters?.length,
  )
  const showSidebar = activeTab === 'player'
  const transcriptStageStatuses = ['uploaded', 'processing', 'extracting_audio', 'transcribing', 'cleaning_transcript']
  const summaryStageStatuses = ['transcript_ready', 'summarizing_quick', 'summarizing_final', 'summarizing']
  const showTranscriptSkeleton = !hasTranscript && transcriptStageStatuses.includes(video?.status)
  const showSummarySkeleton = !hasStructuredSummaryContent && summaryStageStatuses.includes(video?.status) && hasTranscript
  const chatPreparing = hasTranscript && ['transcript_ready', 'summarizing_quick', 'summarizing_final', 'summarizing', 'indexing_chat'].includes(video?.status)

  const playbackUrl = useMemo(() => {
    if (video?.original_file) {
      return video.original_file.startsWith('http')
        ? video.original_file
        : `${window.location.origin}${video.original_file}`
    }
    if (video?.youtube_url) return video.youtube_url
    return null
  }, [video])

  const formatTime = (seconds) => {
    const s = Math.max(0, Math.floor(Number(seconds) || 0))
    const hours = Math.floor(s / 3600)
    const mins = Math.floor((s % 3600) / 60)
    const secs = s % 60
    if (hours > 0) return `${hours}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
    return `${mins}:${String(secs).padStart(2, '0')}`
  }

  const parseTimestampToSeconds = (value) => {
    if (!value) return 0
    if (typeof value === 'number') return value
    const parts = String(value)
      .split(':')
      .map((p) => Number(p))
      .filter((n) => !Number.isNaN(n))
    if (parts.length === 3) return (parts[0] * 3600) + (parts[1] * 60) + parts[2]
    if (parts.length === 2) return (parts[0] * 60) + parts[1]
    if (parts.length === 1) return parts[0]
    return 0
  }

  const jumpTo = (seconds) => {
    setSeekToSeconds(Number(seconds) || 0)
    setActiveTab('player')
  }

  const openMomentChat = (segment) => {
    const start = Number(segment.start ?? segment.start_time ?? 0)
    const end = Number(segment.end ?? segment.end_time ?? start)
    const text = String(segment.text || '').trim()
    setMomentContext({
      timestamp: start,
      label: `${formatTime(start)} — ${formatTime(end)}`,
      excerpt: buildDisplayTranscript(text),
    })
    setActiveTab('chatbot')
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

  return (
    <div className={isPanelMode ? 'panel-detail' : ''}>
      <motion.div
        className={`flex items-center space-x-4 ${isPanelMode ? 'mb-4' : 'mb-6'}`}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
      >
        <Link
          to={isPanelMode ? '/?layout=panel' : '/'}
          className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
        >
          <ArrowLeftIcon className="w-6 h-6 text-white/60" />
        </Link>
        <div className="flex-1">
          <h1 className={`${isPanelMode ? 'text-lg' : 'text-2xl'} font-bold text-white`}>{video.title}</h1>
          {!isPanelMode && <p className="text-white/50 text-sm">{video.description || 'No description'}</p>}
        </div>
      </motion.div>

      <motion.div
        className={`flex space-x-1 bg-white/5 p-1 rounded-xl ${isPanelMode ? 'mb-4' : 'mb-6'}`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        {[
          { id: 'player', icon: <PlayCircleIcon className="w-5 h-5" />, label: 'Player' },
          { id: 'summaries', icon: <DocumentTextIcon className="w-5 h-5" />, label: 'Summaries' },
          { id: 'chatbot', icon: <ChatBubbleLeftRightIcon className="w-5 h-5" />, label: 'Chatbot', disabled: !hasTranscript },
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

      <div className={`grid ${isPanelMode || !showSidebar ? 'grid-cols-1 gap-4' : 'lg:grid-cols-3 gap-6'}`}>
        <div className={isPanelMode || !showSidebar ? '' : 'lg:col-span-2'}>
          {activeTab === 'player' && (
            <div className="space-y-6">
              <div className={`glass-card ${isPanelMode ? 'p-4' : 'p-6'}`}>
              <div className="flex items-center gap-3 mb-4">
                <PlayCircleIcon className="w-6 h-6 text-indigo-300" />
                <h2 className="text-xl font-semibold text-white">Player</h2>
              </div>

              {playbackUrl ? (
                <VideoPlayer
                  url={playbackUrl}
                  title={video.title}
                  seekToSeconds={seekToSeconds}
                  onTimeChange={setCurrentPlayerTime}
                  highlights={effectiveSummary.chapters || []}
                />
              ) : (
                <div className="glass-card text-center py-20">
                  <p className="text-white/50">Video file not available</p>
                </div>
              )}
            </div>

            <div className={`glass-card ${isPanelMode ? 'p-4' : 'p-6'}`}>
              <div className="flex items-center gap-3 mb-4">
                <QueueListIcon className="w-6 h-6 text-indigo-300" />
                <h2 className="text-xl font-semibold text-white">Transcript</h2>
              </div>

              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Transcript Viewer</h3>
                {hasTranscript ? (
                  <div className="flex items-center gap-2">
                    <span className="status-badge status-badge-success flex items-center space-x-1">
                      <CheckCircleIcon className="w-4 h-4" />
                      <span>Ready</span>
                    </span>
                    <button
                      onClick={() => setEditingTranscript(true)}
                      className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition-colors"
                    >
                      Edit
                    </button>
                  </div>
                ) : (
                  <span className="text-sm text-white/50">Transcript not generated yet</span>
                )}
              </div>

              <div className="mb-4">
                <div className="flex items-end">
                  <button
                    onClick={handleGenerateTranscript}
                    disabled={generatingTranscript || ['processing', 'extracting_audio', 'transcribing', 'cleaning_transcript', 'transcript_ready', 'summarizing_quick', 'summarizing_final', 'summarizing', 'indexing_chat'].includes(video.status)}
                    className="w-full px-3 py-2 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 text-sm disabled:opacity-50"
                  >
                    {generatingTranscript ? 'Generating...' : hasTranscript ? 'Regenerate Transcript' : 'Generate Transcript'}
                  </button>
                </div>
              </div>

              {editingTranscript ? (
                <TranscriptEditor
                  videoId={videoId}
                  transcript={latestTranscript}
                  onSave={(text) => {
                    if (latestTranscript) {
                      latestTranscript.full_text = text
                    }
                    setEditingTranscript(false)
                  }}
                  onCancel={() => setEditingTranscript(false)}
                />
              ) : (
                showTranscriptSkeleton ? (
                  <div className="bg-white/5 rounded-xl p-4">
                    <TranscriptSkeleton />
                  </div>
                ) : (
                  hasTranscript && (
                  <div className="space-y-4">
                    <div className="relative">
                      <MagnifyingGlassIcon className="w-4 h-4 text-white/40 absolute left-3 top-2.5" />
                      <input
                        value={transcriptQuery}
                        onChange={(e) => setTranscriptQuery(e.target.value)}
                        placeholder="Search transcript..."
                        className="w-full pl-9 pr-3 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder-white/40"
                      />
                    </div>

                    {filteredSegments.length > 0 ? (
                      <div
                        ref={transcriptContainerRef}
                        className={`bg-white/5 rounded-xl overflow-y-auto custom-scrollbar ${isPanelMode ? 'max-h-72' : 'max-h-[32rem]'}`}
                      >
                        {filteredSegments.map((segment, index) => {
                          const start = Number(segment.start ?? segment.start_time ?? 0)
                          const end = Number(segment.end ?? segment.end_time ?? start + 1)
                          const text = String(segment.text || '').trim()
                          const segmentKey = `${start}-${end}`
                          const active = activeTranscriptSegmentKey === segmentKey
                          return (
                            <div
                              key={`${start}-${end}-${index}`}
                              ref={(element) => {
                                if (element) {
                                  transcriptRowRefs.current.set(segmentKey, element)
                                } else {
                                  transcriptRowRefs.current.delete(segmentKey)
                                }
                              }}
                              style={{ scrollMarginBlock: '96px' }}
                              className={`w-full text-left px-4 py-3 border-b border-white/5 hover:bg-white/10 transition-colors ${
                                active ? 'bg-indigo-500/15 ring-1 ring-indigo-400/30' : ''
                              }`}
                            >
                              <div className="flex items-center justify-between gap-3 mb-2">
                                <button
                                  type="button"
                                  onClick={() => jumpTo(start)}
                                  className="text-xs text-indigo-300 font-mono hover:text-indigo-200"
                                >
                                  {formatTime(start)}
                                </button>
                                <button
                                  type="button"
                                  onClick={() => openMomentChat({ ...segment, start })}
                                  className="px-2.5 py-1 rounded-full bg-fuchsia-500/10 border border-fuchsia-400/20 text-[11px] text-fuchsia-200 hover:bg-fuchsia-500/20 transition-colors"
                                >
                                  Ask AI about this moment
                                </button>
                              </div>
                              <div className="text-sm text-white/80 whitespace-pre-wrap">
                                {buildDisplayTranscript(text)}
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    ) : (
                      <div className={`bg-white/5 rounded-xl p-4 overflow-y-auto custom-scrollbar ${isPanelMode ? 'max-h-64' : 'max-h-96'}`}>
                        <p className="text-white/70 whitespace-pre-wrap">
                          {buildDisplayTranscript(
                            latestTranscript?.full_text || ''
                          )}
                        </p>
                      </div>
                    )}
                  </div>
                  )
                )
              )}
            </div>
            </div>
          )}

          {activeTab === 'summaries' && (
            <div className={`detail-workspace-panel ${isPanelMode ? 'p-4' : 'p-6'}`}>
              <div className="flex items-center gap-3 mb-4">
                <DocumentTextIcon className="w-6 h-6 text-indigo-300" />
                <h2 className="text-xl font-semibold text-white">Summaries</h2>
              </div>

              {showSummarySkeleton ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <SummarySkeleton />
                  <SummarySkeleton />
                  <SummarySkeleton />
                  <SummarySkeleton />
                </div>
              ) : (
                <SummaryDashboard
                  summary={effectiveSummary}
                  onJumpToTimestamp={(timestamp) => jumpTo(parseTimestampToSeconds(timestamp))}
                />
              )}
            </div>
          )}

          {activeTab === 'chatbot' && (
            <div className={`detail-workspace-panel ${isPanelMode ? 'p-4' : 'p-6'}`}>
              <div className="flex items-center gap-3 mb-4">
                <ChatBubbleLeftRightIcon className="w-6 h-6 text-indigo-300" />
                <h2 className="text-xl font-semibold text-white">Chatbot</h2>
              </div>

              {hasTranscript && !chatPreparing ? (
                <ChatBot
                  videoId={videoId}
                  momentContext={momentContext}
                  onClearMomentContext={() => setMomentContext(null)}
                  onJumpToTimestamp={jumpTo}
                />
              ) : hasTranscript ? (
                <div className="glass-card text-center py-20">
                  <ChatBubbleLeftRightIcon className="w-16 h-16 mx-auto text-white/20 mb-4" />
                  <p className="text-white/70">Preparing chat...</p>
                  <p className="text-sm text-white/30 mt-2">The retrieval index is still being built from the transcript.</p>
                </div>
              ) : (
                <div className="glass-card text-center py-20">
                  <ChatBubbleLeftRightIcon className="w-16 h-16 mx-auto text-white/20 mb-4" />
                  <p className="text-white/50">Transcript is required to use the chatbot</p>
                  <p className="text-sm text-white/30 mt-2">Generate a transcript first</p>
                </div>
              )}
            </div>
          )}
        </div>

        {showSidebar && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <ProgressTracker video={video} />

            <div className={`glass-card ${isPanelMode ? 'mt-4 p-4' : 'mt-6 p-6'}`}>
              <h3 className="text-lg font-semibold text-white mb-4">Processing Metadata</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-white/50">Language</span>
                  <span className="text-white uppercase">{metadata.language || '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">ASR Engine</span>
                  <span className="text-white">{metadata.asr_engine || '-'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Quality Score</span>
                  <span className="text-white">
                    {typeof metadata.transcript_quality_score === 'number'
                      ? metadata.transcript_quality_score.toFixed(2)
                      : '-'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Processing Time</span>
                  <span className="text-white">
                    {typeof metadata.processing_time_seconds === 'number'
                      ? `${metadata.processing_time_seconds.toFixed(1)}s`
                      : '-'}
                  </span>
                </div>
              </div>
            </div>

            <div className={`glass-card ${isPanelMode ? 'mt-4 p-4' : 'mt-6 p-6'}`}>
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
                    {video.file_size ? `${(video.file_size / (1024 * 1024)).toFixed(1)} MB` : '-'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Uploaded</span>
                  <span className="text-white">{new Date(video.created_at).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default VideoDetailPage
