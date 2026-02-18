import { motion } from 'framer-motion'
import { CheckCircleIcon, XCircleIcon, ArrowPathIcon } from '@heroicons/react/24/outline'

function ProgressTracker({ video }) {
  const statusColors = {
    pending: 'bg-slate-500',
    processing: 'bg-amber-500',
    transcribing: 'bg-blue-500',
    summarizing: 'bg-purple-500',
    completed: 'bg-emerald-500',
    failed: 'bg-red-500'
  }

  const statusLabels = {
    pending: 'Pending',
    processing: 'Processing',
    transcribing: 'Transcribing',
    summarizing: 'Summarizing',
    completed: 'Completed',
    failed: 'Failed'
  }

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

  const formatDate = (dateString) => {
    if (!dateString) return '-'
    return new Date(dateString).toLocaleString()
  }

  const steps = [
    { key: 'upload', label: 'Upload Complete', icon: '📤' },
    { key: 'transcribe', label: 'Transcribing Speech', icon: '🎙️' },
    { key: 'summarize', label: 'Generating Summaries', icon: '📝' },
    { key: 'complete', label: 'Ready to Use', icon: '✨' }
  ]

  const getStepStatus = (stepKey) => {
    if (video.status === 'completed') return 'completed'
    if (video.status === 'pending') return stepKey === 'upload' ? 'active' : 'pending'
    if (stepKey === 'upload') return 'completed'
    if (stepKey === 'transcribe' && video.status === 'transcribing') return 'active'
    if (stepKey === 'summarize' && video.status === 'summarizing') return 'active'
    if (stepKey === 'complete' && video.status === 'completed') return 'completed'
    if (['transcribing', 'summarizing', 'processing'].includes(video.status)) return 'pending'
    return 'pending'
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
          <span className={`text-sm font-medium ${
            video.status === 'completed' ? 'text-emerald-400' :
            video.status === 'failed' ? 'text-red-400' :
            video.status === 'pending' ? 'text-slate-400' :
            'text-amber-400'
          }`}>
            {statusLabels[video.status]}
          </span>
        </div>
      </div>

      {/* Progress bar */}
      {video.status !== 'completed' && video.status !== 'failed' && (
        <div className="mb-6">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-white/50">Progress</span>
            <span className="text-white/70">{video.processing_progress}%</span>
          </div>
          <div className="progress-bar">
            <motion.div 
              className="progress-bar-fill"
              initial={{ width: 0 }}
              animate={{ width: `${video.processing_progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>
      )}

      {/* Status steps */}
      <div className="space-y-3">
        {steps.map((step, index) => {
          const status = getStepStatus(step.key)
          return (
            <motion.div
              key={step.key}
              className={`flex items-center space-x-3 p-3 rounded-xl transition-all duration-300 ${
                status === 'active' ? 'bg-indigo-500/10 border border-indigo-500/30' :
                status === 'completed' ? 'bg-emerald-500/5' :
                'bg-white/5'
              }`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className={`${
                status === 'completed' ? 'text-emerald-400' :
                status === 'active' ? 'text-indigo-400' :
                'text-white/30'
              }`}>
                {step.icon}
              </div>
              <span className={`flex-1 ${
                status === 'completed' ? 'text-white/70' :
                status === 'active' ? 'text-white' :
                'text-white/30'
              }`}>
                {step.label}
              </span>
              {status === 'completed' && (
                <CheckCircleIcon className="w-5 h-5 text-emerald-400" />
              )}
              {status === 'active' && (
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

      {/* Error message */}
      {video.error_message && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl"
        >
          <p className="text-red-400 text-sm">{video.error_message}</p>
        </motion.div>
      )}

      {/* Timestamps */}
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
