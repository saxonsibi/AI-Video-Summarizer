import { motion } from 'framer-motion'
import { ArrowPathIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline'

function ProgressTracker({ video }) {
  const statusColors = {
    pending: 'bg-slate-500',
    uploaded: 'bg-slate-500',
    processing: 'bg-amber-500',
    extracting_audio: 'bg-amber-500',
    transcribing: 'bg-blue-500',
    cleaning_transcript: 'bg-cyan-500',
    transcript_ready: 'bg-cyan-500',
    summarizing_quick: 'bg-purple-500',
    summarizing_final: 'bg-purple-500',
    summarizing: 'bg-purple-500',
    indexing_chat: 'bg-fuchsia-500',
    completed: 'bg-emerald-500',
    failed: 'bg-red-500',
  }

  const statusLabels = {
    pending: 'Pending',
    uploaded: 'Uploaded',
    processing: 'Processing',
    extracting_audio: 'Extracting Audio',
    transcribing: 'Transcribing',
    cleaning_transcript: 'Cleaning Transcript',
    transcript_ready: 'Transcript Ready',
    summarizing_quick: 'Quick Summary',
    summarizing_final: 'Final Summary',
    summarizing: 'Summarizing',
    indexing_chat: 'Preparing Chat',
    completed: 'Completed',
    failed: 'Failed',
  }

  const hasTranscript = Array.isArray(video?.transcripts) && video.transcripts.length > 0
  const hasSummaries = Array.isArray(video?.summaries) && video.summaries.length > 0

  const steps = [
    { key: 'uploading', label: 'Uploading Video', icon: '1' },
    { key: 'extracting', label: 'Extracting Audio', icon: '2' },
    { key: 'transcript', label: 'Generating Transcript', icon: '3' },
    { key: 'summary', label: 'Creating Summary', icon: '4' },
    { key: 'index', label: 'Building Chatbot Index', icon: '5' },
  ]

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon className="w-5 h-5 text-emerald-400" />
      case 'failed':
        return <XCircleIcon className="w-5 h-5 text-red-400" />
      default:
        return <ArrowPathIcon className="w-5 h-5 text-amber-400 animate-spin" />
    }
  }

  const getStepStatus = (stepKey) => {
    if (video.status === 'failed') return 'pending'
    if (video.status === 'completed') return 'completed'

    if (stepKey === 'uploading') return 'completed'

    if (stepKey === 'extracting') {
      if (['uploaded', 'processing', 'extracting_audio'].includes(video.status) && !hasTranscript) return 'active'
      return hasTranscript || ['transcribing', 'cleaning_transcript', 'transcript_ready', 'summarizing_quick', 'summarizing_final', 'indexing_chat'].includes(video.status) ? 'completed' : 'pending'
    }

    if (stepKey === 'transcript') {
      if (['transcribing', 'cleaning_transcript'].includes(video.status)) return 'active'
      return hasTranscript || ['transcript_ready', 'summarizing_quick', 'summarizing_final', 'indexing_chat'].includes(video.status) ? 'completed' : 'pending'
    }

    if (stepKey === 'summary') {
      if (['summarizing_quick', 'summarizing_final', 'summarizing'].includes(video.status) && !hasSummaries) return 'active'
      return hasSummaries || ['indexing_chat', 'completed'].includes(video.status) ? 'completed' : 'pending'
    }

    if (stepKey === 'index') {
      if (video.status === 'indexing_chat') return 'active'
      return video.status === 'completed' ? 'completed' : 'pending'
    }

    return 'pending'
  }

  const formatDate = (dateString) => {
    if (!dateString) return '-'
    return new Date(dateString).toLocaleString()
  }

  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white">Processing Status</h3>
        <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${statusColors[video.status]} bg-opacity-20`}>
          {getStatusIcon(video.status)}
          <span
            className={`text-sm font-medium ${
              video.status === 'completed'
                ? 'text-emerald-400'
                : video.status === 'failed'
                  ? 'text-red-400'
                  : video.status === 'pending'
                    ? 'text-slate-400'
                    : 'text-amber-400'
            }`}
          >
            {statusLabels[video.status]}
          </span>
        </div>
      </div>

      {video.status !== 'completed' && video.status !== 'failed' && (
        <div className="mb-6">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-white/50">Progress</span>
            <span className="text-white/70">{video.processing_progress ?? 0}%</span>
          </div>
          <div className="progress-bar">
            <motion.div
              className="progress-bar-fill"
              initial={{ width: 0 }}
              animate={{ width: `${video.processing_progress ?? 0}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>
      )}

      <div className="space-y-3">
        {steps.map((step, index) => {
          const stepStatus = getStepStatus(step.key)
          return (
            <motion.div
              key={step.key}
              className={`flex items-center space-x-3 p-3 rounded-xl transition-all duration-300 ${
                stepStatus === 'active'
                  ? 'bg-indigo-500/10 border border-indigo-500/30'
                  : stepStatus === 'completed'
                    ? 'bg-emerald-500/5'
                    : 'bg-white/5'
              }`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.08 }}
            >
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold ${
                  stepStatus === 'completed'
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : stepStatus === 'active'
                      ? 'bg-indigo-500/20 text-indigo-300'
                      : 'bg-white/10 text-white/40'
                }`}
              >
                {step.icon}
              </div>
              <span
                className={`flex-1 ${
                  stepStatus === 'completed'
                    ? 'text-white/70'
                    : stepStatus === 'active'
                      ? 'text-white'
                      : 'text-white/30'
                }`}
              >
                {step.label}
              </span>
              {stepStatus === 'completed' && <CheckCircleIcon className="w-5 h-5 text-emerald-400" />}
              {stepStatus === 'active' && (
                <motion.div
                  className="w-5 h-5 border-2 border-indigo-400 border-t-transparent rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                />
              )}
            </motion.div>
          )
        })}
      </div>

      {video.error_message && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl"
        >
          <p className="text-red-400 text-sm">{video.error_message}</p>
        </motion.div>
      )}

      <div className="mt-6 pt-4 border-t border-white/10 grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-white/40 mb-1">Created</p>
          <p className="text-white/70">{formatDate(video.created_at)}</p>
        </div>
        <div>
          <p className="text-white/40 mb-1">Processed</p>
          <p className="text-white/70">{formatDate(video.processed_at)}</p>
        </div>
      </div>
    </motion.div>
  )
}

export default ProgressTracker
