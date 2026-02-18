import { Link } from 'react-router-dom'
import { SparklesIcon, VideoCameraIcon } from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'

function Header() {
  return (
    <motion.header 
      className="header-glass"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 100 }}
    >
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-3 group">
            <motion.div 
              className="relative"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-purple-600 blur-lg opacity-50 rounded-xl" />
              <div className="relative p-2.5 glass-card rounded-xl">
                <SparklesIcon className="w-6 h-6 gradient-text" />
              </div>
            </motion.div>
            <span className="text-xl font-bold gradient-text group-hover:scale-105 transition-transform">
              AI Video Summarizer
            </span>
          </Link>
          
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
          
          <motion.div 
            className="flex items-center space-x-3"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <Link 
              to="/"
              className="hidden sm:flex items-center space-x-2 px-4 py-2 glass-card hover:bg-white/10 transition-all duration-300"
            >
              <VideoCameraIcon className="w-5 h-5 text-indigo-400" />
              <span className="text-white/80">My Videos</span>
            </Link>
          </motion.div>
        </div>
      </div>
    </motion.header>
  )
}

export default Header
