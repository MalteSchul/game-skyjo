import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import McTreeExplorerPage from './tools/mcts-tree/McTreeExplorerPage.tsx'

// A path-based route for one developer tool, not a router: nginx.conf and
// Vite's dev server both fall back to index.html for any unmatched path
// (SPA-style), so this is the only wiring a second "page" needs here.
const isToolsRoute = window.location.pathname.startsWith('/tools/mcts-tree')

createRoot(document.getElementById('root')!).render(
  <StrictMode>{isToolsRoute ? <McTreeExplorerPage /> : <App />}</StrictMode>,
)
