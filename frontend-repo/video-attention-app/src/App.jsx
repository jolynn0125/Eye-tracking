import { useRef, useState } from 'react'
import './App.css'
import VideoWithOverlay from './components/VideoWithOverlay.jsx'
import { completeSession, startSession } from './lib/sessionApi.js'

const videos = [
  {
    videoKey: 'video1',
    videoSrc: '/videos/video1.mp4',
    videoTitle: 'Video 1',
    buttonLabel: 'Attention prompt 1',
  },
  {
    videoKey: 'video2',
    videoSrc: '/videos/video2.mp4',
    videoTitle: 'Video 2',
    buttonLabel: 'Attention prompt 2',
  },
]

function App() {
  const [formState, setFormState] = useState({
    name: '',
    email: '',
  })
  const [session, setSession] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState('')
  const [activeVideoIndex, setActiveVideoIndex] = useState(0)
  const completedVideosRef = useRef({
    video1: false,
    video2: false,
  })

  async function handleSessionSubmit(event) {
    event.preventDefault()

    const name = formState.name.trim()
    const email = formState.email.trim()

    if (!name || !email) {
      setSubmitError('Name and email are required.')
      return
    }

    setIsSubmitting(true)
    setSubmitError('')

    try {
      const createdSession = await startSession({ name, email })

      setSession(createdSession)
      completedVideosRef.current = { video1: false, video2: false }
      setActiveVideoIndex(0)
    } catch (error) {
      setSubmitError(error.message)
    } finally {
      setIsSubmitting(false)
    }
  }

  function handleVideoEnded(videoKey) {
    let shouldCompleteSession = false

    const currentCompletedVideos = completedVideosRef.current

    if (currentCompletedVideos[videoKey]) {
      return
    }

    completedVideosRef.current = {
      ...currentCompletedVideos,
      [videoKey]: true,
    }

    if (completedVideosRef.current.video1 && completedVideosRef.current.video2) {
      shouldCompleteSession = true
    }

    if (shouldCompleteSession && session) {
      completeSession({ sessionId: session.sessionId }).catch(() => {})
    }
  }

  if (!session) {
    return (
      <main className="gate-shell">
        <div className="gate-backdrop" />
        <section className="gate-modal" aria-labelledby="session-modal-title">
          <h1 id="session-modal-title">Start session</h1>
          <p className="gate-modal__copy">
            Enter your name and email before the videos can be viewed.
          </p>

          <form className="gate-form" onSubmit={handleSessionSubmit}>
            <label className="gate-field">
              <span>Name</span>
              <input
                type="text"
                value={formState.name}
                onChange={(event) =>
                  setFormState((currentFormState) => ({
                    ...currentFormState,
                    name: event.target.value,
                  }))
                }
                required
                autoComplete="name"
              />
            </label>

            <label className="gate-field">
              <span>Email</span>
              <input
                type="email"
                value={formState.email}
                onChange={(event) =>
                  setFormState((currentFormState) => ({
                    ...currentFormState,
                    email: event.target.value,
                  }))
                }
                required
                autoComplete="email"
              />
            </label>

            {submitError ? <p className="gate-error">{submitError}</p> : null}

            <button type="submit" className="gate-submit" disabled={isSubmitting}>
              {isSubmitting ? 'Starting...' : 'Enter videos'}
            </button>
          </form>
        </section>
      </main>
    )
  }

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
            session={session}
            onVideoEnded={handleVideoEnded}
          />
        ))}
      </section>
    </main>
  )
}

export default App
