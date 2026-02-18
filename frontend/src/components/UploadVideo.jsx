import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CloudArrowUpIcon, DocumentIcon, XMarkIcon, CheckCircleIcon } from '@heroicons/react/24/outline'
import { videoAPI } from '../services/api'

function UploadVideo({ onUploadComplete }) {
  const [dragOver, setDragOver] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(false)
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => {
    setDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }

  const handleFileSelect = (file) => {
    const allowedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm']
    if (!allowedTypes.includes(file.type)) {
      setError('Invalid file type. Please upload a video file (MP4, MOV, AVI, MKV, WebM)')
      return
    }
    setSelectedFile(file)
    setError(null)
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setUploading(true)
    setProgress(0)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('title', selectedFile.name.replace(/\.[^/.]+$/, ''))

      await videoAPI.upload(formData, (percent) => {
        setProgress(percent)
      })

      setUploading(false)
      setProgress(100)
      setSuccess(true)
      
      setTimeout(() => {
        setSelectedFile(null)
        setSuccess(false)
        if (onUploadComplete) {
          onUploadComplete()
        }
      }, 2000)
      
    } catch (err) {
      setUploading(false)
      setError(err.response?.data?.error || 'Upload failed. Please try again.')
    }
  }

  const removeFile = () => {
    setSelectedFile(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <motion.div
      className="max-w-2xl mx-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="glass-card p-8">
        <motion.h2 
          className="text-2xl font-bold text-white mb-2 flex items-center space-x-3"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <CloudArrowUpIcon className="w-8 h-8 text-indigo-400" />
          <span>Upload Video</span>
        </motion.h2>

        <motion.p 
          className="text-white/60 mb-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          Drag and drop your video or click to browse
        </motion.p>

        {/* Drop zone */}
        <motion.div
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          whileHover={{ scale: dragOver ? 1 : 1.01 }}
          whileTap={{ scale: 0.99 }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => e.target.files[0] && handleFileSelect(e.target.files[0])}
            disabled={uploading}
          />

          {!uploading && !success && (
            <>
              <motion.div
                className="mb-4"
                animate={{ 
                  y: [0, -10, 0],
                  scale: [1, 1.1, 1]
                }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <CloudArrowUpIcon className="w-16 h-16 mx-auto text-indigo-400/60" />
              </motion.div>
              <p className="text-lg text-white/80 mb-2">
                Drop your video here
              </p>
              <p className="text-sm text-white/40">
                or click to browse
              </p>
              <p className="text-xs text-white/30 mt-4">
                Supports: MP4, MOV, AVI, MKV, WebM (max 500MB)
              </p>
            </>
          )}

          {/* Success state */}
          {success && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center"
            >
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200 }}
                className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-emerald-500/20 mb-4"
              >
                <CheckCircleIcon className="w-12 h-12 text-emerald-400" />
              </motion.div>
              <p className="text-xl text-white font-semibold">Upload Complete!</p>
            </motion.div>
          )}

          {/* Upload progress */}
          {uploading && (
            <motion.div 
              className="text-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <motion.div 
                className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full mx-auto mb-4"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              />
              <p className="text-lg text-white/80 mb-4">Uploading...</p>
              <div className="max-w-xs mx-auto">
                <div className="progress-bar mb-2">
                  <motion.div 
                    className="progress-bar-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <p className="text-sm text-white/50">{progress}%</p>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Selected file */}
        <AnimatePresence>
          {selectedFile && !uploading && !success && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-4 p-4 bg-white/5 border border-white/10 rounded-xl flex items-center justify-between"
            >
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-indigo-500/20 rounded-lg">
                  <DocumentIcon className="w-6 h-6 text-indigo-400" />
                </div>
                <div>
                  <p className="text-white font-medium truncate max-w-xs">{selectedFile.name}</p>
                  <p className="text-sm text-white/50">{formatFileSize(selectedFile.size)}</p>
                </div>
              </div>
              <motion.button
                onClick={removeFile}
                className="p-2 rounded-lg bg-white/5 text-white/40 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <XMarkIcon className="w-5 h-5" />
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Upload button */}
        {selectedFile && !uploading && !success && (
          <motion.button
            onClick={handleUpload}
            className="glow-button w-full mt-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Start Upload
          </motion.button>
        )}

        {/* Error message */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl"
            >
              <p className="text-red-400">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

export default UploadVideo
