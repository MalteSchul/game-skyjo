import { useEffect, useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

type BackendStatus = 'checking' | 'online' | 'offline'

function App() {
  const [status, setStatus] = useState<BackendStatus>('checking')

  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then((res) => setStatus(res.ok ? 'online' : 'offline'))
      .catch(() => setStatus('offline'))
  }, [])

  return (
    <main>
      <h1>Skyjo</h1>
      <p>Backend: {status}</p>
    </main>
  )
}

export default App
