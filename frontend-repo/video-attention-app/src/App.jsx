import { useState } from 'react'
import './App.css'
import VideoWithOverlay from './components/VideoWithOverlay.jsx'

const videos = [
  {
    videoSrc: '/videos/video1.mp4',
    videoTitle: 'Video 1',
    buttonLabel: 'Attention prompt 1',
  },
  {
    videoSrc: '/videos/video2.mp4',
    videoTitle: 'Video 2',
    buttonLabel: 'Attention prompt 2',
  },
]

function App() {
  const [activeVideoIndex, setActiveVideoIndex] = useState(0)

  return (
    <main className="viewer">
      <div className="viewer__tabs" role="tablist" aria-label="Video tabs">
        {videos.map((video, index) => {
          const isActive = index === activeVideoIndex

          return (
            <button
              key={video.videoSrc}
              type="button"
              className={`viewer__tab ${isActive ? 'viewer__tab--active' : ''}`}
              aria-pressed={isActive}
              aria-label={`Show video ${index + 1}`}
              onClick={() => setActiveVideoIndex(index)}
            >
              {index + 1}
            </button>
          )
        })}
      </div>

      <section className="viewer__stage" aria-label="Video player">
        {videos.map((video, index) => (
          <VideoWithOverlay
            key={video.videoSrc}
            {...video}
            isActive={index === activeVideoIndex}
          />
        ))}
      </section>
    </main>
  )
}

export default App
