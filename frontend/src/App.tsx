import { useEffect, useState } from 'react'
import {
  ApiError,
  API_BASE_URL,
  applyAction,
  createMatch,
  getMatch,
  getMatchHistory,
  gotoMatchHistoryNode,
  startNextRound,
} from './api/matchClient'
import { clearStoredMatchId, loadStoredMatchId, storeMatchId } from './api/matchStorage'
import type { ActionTypeName, MatchHistoryOut, MatchStateOut } from './api/types'
import HistoryPanel from './game/HistoryPanel'
import MatchView from './game/MatchView'
import Scoreboard from './game/Scoreboard'

type BackendStatus = 'checking' | 'online' | 'offline'

function App() {
  const [status, setStatus] = useState<BackendStatus>('checking')
  const [match, setMatch] = useState<MatchStateOut | null>(null)
  const [history, setHistory] = useState<MatchHistoryOut | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  // True while we're trying to reload a match remembered from a previous
  // visit, so we don't flash the "start a new match" form before it resolves.
  const [restoring, setRestoring] = useState(() => loadStoredMatchId() !== null)
  // When set, the next click on a face-down card discards the drawn card and
  // reveals that one instead of placing the drawn card there.
  const [discardMode, setDiscardMode] = useState(false)

  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then((res) => setStatus(res.ok ? 'online' : 'offline'))
      .catch(() => setStatus('offline'))
  }, [])

  useEffect(() => {
    const storedMatchId = loadStoredMatchId()
    if (!storedMatchId) return

    getMatch(storedMatchId)
      .then((restored) => {
        setMatch(restored)
        return refreshHistory(restored.match_id)
      })
      .catch(() => clearStoredMatchId())
      .finally(() => setRestoring(false))
  }, [])

  // History is a nice-to-have sidebar, not core to playing the match — a
  // failed refresh just leaves it stale/empty rather than surfacing an error.
  async function refreshHistory(matchId: string) {
    try {
      setHistory(await getMatchHistory(matchId))
    } catch {
      // ignore
    }
  }

  async function runAction(action: { type: ActionTypeName; position?: number }) {
    if (!match || busy) return
    setBusy(true)
    setError(null)
    try {
      const updated = await applyAction(match.match_id, action)
      setMatch(updated)
      setDiscardMode(false)
      await refreshHistory(updated.match_id)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong.')
    } finally {
      setBusy(false)
    }
  }

  async function handleCreate(playerCount: number, seed: number | undefined, playerNames: string[]) {
    setBusy(true)
    setError(null)
    try {
      const created = await createMatch({ player_count: playerCount, seed, player_names: playerNames })
      setMatch(created)
      storeMatchId(created.match_id)
      await refreshHistory(created.match_id)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong.')
    } finally {
      setBusy(false)
    }
  }

  function handlePlayAgain() {
    clearStoredMatchId()
    setMatch(null)
    setHistory(null)
    setError(null)
  }

  async function handleNextRound() {
    if (!match || busy) return
    setBusy(true)
    setError(null)
    try {
      const updated = await startNextRound(match.match_id)
      setMatch(updated)
      await refreshHistory(updated.match_id)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong.')
    } finally {
      setBusy(false)
    }
  }

  async function handleGotoHistory(nodeId: string) {
    if (!match || busy) return
    setBusy(true)
    setError(null)
    try {
      const updated = await gotoMatchHistoryNode(match.match_id, nodeId)
      setMatch(updated)
      setDiscardMode(false)
      await refreshHistory(updated.match_id)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong.')
    } finally {
      setBusy(false)
    }
  }

  function handleCardClick(playerIndex: number, position: number) {
    if (!match || playerIndex !== match.current_player) return

    if (match.phase === 'initial_flip') {
      runAction({ type: 'flip_initial', position })
      return
    }

    if (match.phase === 'awaiting_placement') {
      runAction({ type: discardMode ? 'discard_and_reveal' : 'place', position })
    }
  }

  return (
    <>
      <header className="app-header">
        <h1>Skyjo</h1>
        {match && <Scoreboard playerNames={match.player_names} scores={match.total_scores} currentPlayer={match.current_player} />}
      </header>

      <main>
        {status === 'offline' ? (
          <p role="alert" className="error-banner">
            Backend unreachable — start it and reload.
          </p>
        ) : restoring ? (
          <p>Restoring your match…</p>
        ) : (
          <div className="app-body">
            {match && (
              <HistoryPanel history={history} playerNames={match.player_names} onGoto={handleGotoHistory} busy={busy} />
            )}
            <MatchView
              match={match}
              error={error}
              busy={busy}
              discardMode={discardMode}
              onCreate={handleCreate}
              onDrawStock={() => runAction({ type: 'draw_stock' })}
              onDrawDiscard={() => runAction({ type: 'draw_discard' })}
              onSetDiscardMode={setDiscardMode}
              onCardClick={handleCardClick}
              onNextRound={handleNextRound}
              onPlayAgain={handlePlayAgain}
            />
          </div>
        )}
      </main>
    </>
  )
}

export default App
