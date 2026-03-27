import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
import Header from './components/Header'
import HomePage from './pages/HomePage'
import VideoDetailPage from './pages/VideoDetailPage'
import { ToastProvider } from './context/ToastContext'
import { useEffect, useMemo } from 'react'

function AppLayout() {
  const location = useLocation()
  const isPanelMode = useMemo(() => {
    const params = new URLSearchParams(location.search)
    return params.get('layout') === 'panel'
  }, [location.search])

  useEffect(() => {
    document.body.classList.toggle('panel-mode', isPanelMode)
    return () => document.body.classList.remove('panel-mode')
  }, [isPanelMode])

  return (
    <div
      className={isPanelMode ? 'app-root app-root-panel' : 'app-root'}
      style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)' }}
    >
      <Header compact={isPanelMode} />
      <main className={isPanelMode ? 'app-main-panel' : 'app-main-default'}>
        <div className={isPanelMode ? 'panel-shell' : ''}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/video/:videoId" element={<VideoDetailPage />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

function App() {
  return (
    <ToastProvider>
      <Router>
        <AppLayout />
      </Router>
    </ToastProvider>
  )
}

export default App
