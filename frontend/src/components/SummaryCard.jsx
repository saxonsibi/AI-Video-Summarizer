import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { DocumentTextIcon, ArrowPathIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline'
import { videoAPI } from '../services/api'

function SummaryCard({ video, summary, onUpdate }) {
  const [loading, setLoading] = useState(false)
  const [isExpanded, setIsExpanded] = useState(true)
  const [parsedContent, setParsedContent] = useState(null)

  // Parse JSON content from backend
  useEffect(() => {
    try {
      if (typeof summary.content === 'string') {
        const parsed = JSON.parse(summary.content)
        setParsedContent(parsed)
      } else {
        setParsedContent(summary.content)
      }
    } catch (e) {
      setParsedContent({ content: summary.content, title: '', key_topics: [] })
    }
  }, [summary.content])

  const summaryTypeLabels = {
    full: 'Full Summary',
    bullet: 'Bullet Points',
    short: 'Short Script',
    timestamps: 'Timestamp Summary'
  }

  const handleRegenerate = async () => {
    setLoading(true)
    try {
      await videoAPI.generateSummary(video.id, { summary_type: summary.summary_type })
      if (onUpdate) onUpdate()
    } catch (error) {
      console.error('Failed to regenerate summary:', error)
    } finally {
      setLoading(false)
    }
  }

  const getContentText = () => {
    if (!parsedContent) return ''
    return parsedContent.content || parsedContent.summary_text || ''
  }

  const getKeyTopics = () => {
    if (!parsedContent) return []
    return parsedContent.key_topics || []
  }

  return (
    <motion.div
      className="summary-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -5 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-xl">
            <DocumentTextIcon className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <h4 className="font-semibold text-white">{summaryTypeLabels[summary.summary_type]}</h4>
            <p className="text-xs text-white/40">
              Generated in {(summary.generation_time || 0).toFixed(1)}s
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <motion.button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-2 rounded-lg bg-white/5 text-white/40 hover:text-white hover:bg-white/10 transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            {isExpanded ? (
              <ChevronUpIcon className="w-5 h-5" />
            ) : (
              <ChevronDownIcon className="w-5 h-5" />
            )}
          </motion.button>
          <motion.button
            onClick={handleRegenerate}
            disabled={loading}
            className="p-2 rounded-lg bg-white/5 text-white/40 hover:text-white hover:bg-white/10 transition-colors"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <ArrowPathIcon className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
          </motion.button>
        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="pt-4 space-y-2">
              {summary.summary_type === 'bullet' ? (
                <ul className="space-y-2">
                  {getContentText().split('. ').filter(line => line.trim()).map((line, i) => (
                    <li key={i} className="ml-4 text-white/70 flex items-start gap-2">
                      <span className="text-indigo-400 mt-1">•</span>
                      <span>{line.trim()}{line.trim().endsWith('.') ? '' : '.'}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-white/70 whitespace-pre-wrap">
                  {getContentText()}
                </div>
              )}
            </div>

            {/* Key topics */}
            {getKeyTopics().length > 0 && (
              <div className="mt-4 pt-4 border-t border-white/10">
                <p className="text-xs text-white/40 mb-2">Key Topics:</p>
                <div className="flex flex-wrap gap-2">
                  {getKeyTopics().map((topic, index) => (
                    <motion.span
                      key={index}
                      className="px-3 py-1 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/20 rounded-full text-xs text-indigo-300"
                      whileHover={{ scale: 1.05 }}
                    >
                      {topic}
                    </motion.span>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default SummaryCard
