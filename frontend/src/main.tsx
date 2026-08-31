import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import McTreeExplorerPage from './tools/mcts-tree/McTreeExplorerPage.tsx'
import GameReplayPage from './tools/game-replay/GameReplayPage.tsx'

// Path-based routes for developer tools, not a router: nginx.conf and
// Vite's dev server both fall back to index.html for any unmatched path
// (SPA-style), so this is the only wiring a second/third "page" needs here.
const path = window.location.pathname
const page = path.startsWith('/tools/mcts-tree') ? (
  <McTreeExplorerPage />
) : path.startsWith('/tools/game-replay') ? (
  <GameReplayPage />
) : (
  <App />
)

createRoot(document.getElementById('root')!).render(<StrictMode>{page}</StrictMode>)
