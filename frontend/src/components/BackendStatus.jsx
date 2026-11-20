
import { useEffect, useState } from 'react'

export default function BackendStatus() {
  const [status, setStatus] = useState('Checking')

  useEffect(() => {
    let active = true

    fetch('/api/health')
      .then((response) => {
        if (!response.ok) {
          throw new Error('API unavailable')
        }
        return response.json()
      })
      .then((data) => {
        if (active) {
          setStatus(
            data.status === 'healthy' ? 'Connected' : 'Unavailable'
          )
        }
      })
      .catch(() => {
        if (active) {
          setStatus('Disconnected')
        }
      })

    return () => {
      active = false
    }
  }, [])

  return (
    <div className={`backend-status ${status.toLowerCase()}`}>
      <span className="status-dot"></span>
      <span>Backend: {status}</span>
    </div>
  )
}
