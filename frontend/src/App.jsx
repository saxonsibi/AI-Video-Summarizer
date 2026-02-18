import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import HomePage from './pages/HomePage'
import VideoDetailPage from './pages/VideoDetailPage'

function App() {
  return (
    <Router>
      <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)' }}>
        <Header />
        <main style={{ padding: '2rem' }}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/video/:videoId" element={<VideoDetailPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
