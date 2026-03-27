import { Link } from 'react-router-dom'
import { VideoCameraIcon } from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'

// Custom AI Video Logo Component
function AIVideoLogo() {
  return (
    <svg 
      width="32" 
      height="32" 
      viewBox="0 0 32 32" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
      className="w-full h-full"
    >
      {/* Outer soft glow */}
      <rect
        x="1.5"
        y="1.5"
        width="29"
        height="29"
        rx="9"
        fill="url(#logoGradient)"
        filter="url(#glow)"
        opacity="0.55"
      />

      {/* Main rounded square */}
      <rect 
        x="3" 
        y="3" 
        width="26" 
        height="26" 
        rx="8" 
        fill="url(#logoGradient)"
      />

      {/* Inner highlight */}
      <rect
        x="4"
        y="4"
        width="24"
        height="24"
        rx="7"
        fill="url(#logoHighlight)"
        opacity="0.35"
      />

      {/* Play triangle */}
      <path 
        d="M12.2 9.4L22.2 16L12.2 22.6V9.4Z" 
        fill="white" 
        stroke="white" 
        strokeWidth="1.3"
        strokeLinejoin="round"
      />

      {/* AI sparkle dots */}
      <circle cx="23.8" cy="8.2" r="1.3" fill="#f9a8d4" />
      <circle cx="25.6" cy="10" r="0.9" fill="#c4b5fd" opacity="0.9" />
      <circle cx="8.4" cy="23.6" r="0.9" fill="#f9a8d4" opacity="0.7" />
      
      {/* Gradient definitions */}
      <defs>
        <linearGradient id="logoGradient" x1="2" y1="2" x2="30" y2="30" gradientUnits="userSpaceOnUse">
          <stop stopColor="#7c3aed" />
          <stop offset="0.5" stopColor="#a855f7" />
          <stop offset="1" stopColor="#f472b6" />
        </linearGradient>
        <linearGradient id="logoHighlight" x1="4" y1="4" x2="28" y2="28" gradientUnits="userSpaceOnUse">
          <stop stopColor="#ffffff" stopOpacity="0.5" />
          <stop offset="1" stopColor="#ffffff" stopOpacity="0" />
        </linearGradient>
        <filter id="glow" x="-4" y="-4" width="40" height="40">
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

function Header({ compact = false }) {
  return (
    <motion.header 
      className={`header-glass ${compact ? 'header-compact' : ''}`}
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 100 }}
    >
      <div className={`${compact ? 'px-3 py-3' : 'container mx-auto px-4 py-4'}`}>
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-3 group">
            <motion.div 
              className={`relative ${compact ? 'w-8 h-8' : 'w-10 h-10'}`}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              <AIVideoLogo />
            </motion.div>
            <span className={`${compact ? 'text-base' : 'text-xl'} font-bold gradient-text group-hover:scale-105 transition-transform`}>
              VideoIQ
            </span>
          </Link>
          
          {!compact && (
          <nav className="hidden md:flex items-center space-x-2">
            <Link 
              to="/" 
              className="px-4 py-2 rounded-lg text-white/60 hover:text-white hover:bg-white/5 transition-all duration-300"
            >
              Home
            </Link>
            <Link 
              to="/#features" 
              className="px-4 py-2 rounded-lg text-white/60 hover:text-white hover:bg-white/5 transition-all duration-300"
            >
              Features
            </Link>
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="p-2 rounded-lg text-white/40 hover:text-white hover:bg-white/5 transition-all duration-300"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
            </a>
          </nav>
          )}
          
          <motion.div 
            className="flex items-center space-x-3"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <Link 
              to={compact ? '/?layout=panel' : '/'}
              className={`${compact ? 'flex' : 'hidden sm:flex'} items-center space-x-2 px-3 py-2 glass-card hover:bg-white/10 transition-all duration-300`}
            >
              <VideoCameraIcon className="w-5 h-5 text-indigo-400" />
              <span className="text-white/80">{compact ? 'Videos' : 'My Videos'}</span>
            </Link>
          </motion.div>
        </div>
      </div>
    </motion.header>
  )
}

export default Header
