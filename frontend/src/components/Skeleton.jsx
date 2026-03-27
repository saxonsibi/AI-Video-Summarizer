import { motion } from 'framer-motion'

export function Skeleton({ className = '', variant = 'rect' }) {
  const baseStyles = 'bg-white/5'
  const variants = {
    rect: 'rounded-lg',
    circle: 'rounded-full',
    text: 'rounded h-4',
  }
  
  return (
    <motion.div
      className={`${baseStyles} ${variants[variant]} ${className}`}
      animate={{ opacity: [0.3, 0.6, 0.3] }}
      transition={{ duration: 1.5, repeat: Infinity }}
    />
  )
}

export function VideoCardSkeleton() {
  return (
    <div className="glass-card p-4 space-y-4">
      <div className="flex gap-4">
        <Skeleton className="w-40 h-24" />
        <div className="flex-1 space-y-3">
          <Skeleton className="w-3/4 h-5" />
          <Skeleton className="w-1/2 h-4" />
          <div className="flex gap-2">
            <Skeleton className="w-16 h-6 rounded-full" />
            <Skeleton className="w-16 h-6 rounded-full" />
          </div>
        </div>
      </div>
    </div>
  )
}

export function TranscriptSkeleton() {
  return (
    <div className="space-y-3">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="flex gap-3">
          <Skeleton className="w-12 h-4 mt-1" />
          <div className="flex-1 space-y-2">
            <Skeleton className="w-full h-4" />
            <Skeleton className="w-2/3 h-4" />
          </div>
        </div>
      ))}
    </div>
  )
}

export function SummarySkeleton() {
  return (
    <div className="glass-card p-6 space-y-4">
      <Skeleton className="w-1/3 h-6" />
      <div className="space-y-2">
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="w-full h-4" />
        ))}
      </div>
    </div>
  )
}
