import { useEffect, useRef, useState } from 'react'
import './VideoWithOverlay.css'
import { recordPrompt } from '../lib/sessionApi.js'

const OVERLAY_INTERVAL_MS = 2 * 60 * 1000
const OVERLAY_VISIBLE_MS = 4000

function VideoWithOverlay({
  videoSrc,
  videoTitle,
  buttonLabel,
  videoKey,
  isActive,
  session,
  onVideoEnded = () => {},
}) {
  const [isOverlayVisible, setIsOverlayVisible] = useState(false)
  const videoRef = useRef(null)
  const intervalRef = useRef(null)
  const hideTimeoutRef = useRef(null)

  function showOverlay() {
    setIsOverlayVisible(true)

    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current)
    }

    hideTimeoutRef.current = window.setTimeout(() => {
      setIsOverlayVisible(false)
    }, OVERLAY_VISIBLE_MS)
  }

  async function handleOverlayClick() {
    const currentPlaybackTime = videoRef.current?.currentTime ?? 0

    console.log('Overlay button clicked', {
      videoTitle,
      timestamp: new Date().toISOString(),
      currentPlaybackTimeSeconds: Number(currentPlaybackTime.toFixed(2)),
    })

    if (!session?.sessionId) {
      return
    }

    try {
      await recordPrompt({
        sessionId: session.sessionId,
        name: session.name,
        email: session.email,
        videoKey,
      })
    } catch (error) {
      console.error('Unable to store overlay click', error)
    }
  }

  useEffect(() => {
    const videoElement = videoRef.current

    if (!videoElement) {
      return undefined
    }

    if (isActive) {
      videoElement.play().catch(() => {})
    } else {
      videoElement.pause()
    }

    return undefined
  }, [isActive])

  useEffect(() => {
    intervalRef.current = window.setInterval(showOverlay, OVERLAY_INTERVAL_MS)

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }

      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current)
      }
    }
  }, [])

  return (
    <article className={`video-layer ${isActive ? 'video-layer--active' : ''}`}>
      <div className="video-frame">
        <video
          ref={videoRef}
          className="video-element"
          controls
          preload="metadata"
          onEnded={() => onVideoEnded(videoKey)}
        >
          <source src={videoSrc} type="video/mp4" />
          Your browser does not support the video tag.
        </video>

        {isOverlayVisible ? (
          <div className="video-overlay" aria-hidden="false">
            <button type="button" className="video-overlay__button" onClick={handleOverlayClick}>
              {buttonLabel}
            </button>
          </div>
        ) : null}
      </div>
    </article>
  )
}

export default VideoWithOverlay