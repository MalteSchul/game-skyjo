import { useEffect, useMemo, useState } from 'react'
import type { ActionTypeName, MatchStateOut } from '../api/types'
import CenterPiles from './CenterPiles'
import KeyboardHelp from './KeyboardHelp'
import NewMatchForm from './NewMatchForm'
import PlayerBoard from './PlayerBoard'

function isTypingTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
}

function legalPositionsFor(match: MatchStateOut, type: ActionTypeName): Set<number> {
  return new Set(
    match.legal_actions.filter((action) => action.type === type && action.position !== null).map((a) => a.position!),
  )
}

interface MatchViewProps {
  match: MatchStateOut | null
  error: string | null
  busy: boolean
  discardMode: boolean
  onCreate: (playerCount: number, seed: number | undefined, playerNames: string[]) => void
  onDrawStock: () => void
  onDrawDiscard: () => void
  onSetDiscardMode: (discardMode: boolean) => void
  onCardClick: (playerIndex: number, position: number) => void
  onNextRound: () => void
  onPlayAgain: () => void
}

function MatchView({
  match,
  error,
  busy,
  discardMode,
  onCreate,
  onDrawStock,
  onDrawDiscard,
  onSetDiscardMode,
  onCardClick,
  onNextRound,
  onPlayAgain,
}: MatchViewProps) {
  const placePositions = useMemo(() => (match ? legalPositionsFor(match, 'place') : new Set<number>()), [match])
  const discardRevealPositions = useMemo(
    () => (match ? legalPositionsFor(match, 'discard_and_reveal') : new Set<number>()),
    [match],
  )
  const flipPositions = useMemo(() => (match ? legalPositionsFor(match, 'flip_initial') : new Set<number>()), [match])
  const canDrawStock = useMemo(() => (match ? match.legal_actions.some((a) => a.type === 'draw_stock') : false), [match])
  const canDrawDiscard = useMemo(
    () => (match ? match.legal_actions.some((a) => a.type === 'draw_discard') : false),
    [match],
  )
  const canToggleMode = match?.phase === 'awaiting_placement' && discardRevealPositions.size > 0

  const [helpOpen, setHelpOpen] = useState(false)

  // Global one-key shortcuts for the actions a turn can take: drawing, the
  // place/discard-reveal mode toggle, and round transitions. Per-card
  // selection is handled separately, by arrow-key roving tabindex inside
  // each PlayerBoard.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (!match || event.ctrlKey || event.metaKey || event.altKey || isTypingTarget(event.target)) return

      if (event.key === '?') {
        event.preventDefault()
        setHelpOpen((open) => !open)
        return
      }
      if (helpOpen) {
        if (event.key === 'Escape') setHelpOpen(false)
        return
      }
      if (busy) return

      switch (event.key.toLowerCase()) {
        case 's':
          if (canDrawStock) onDrawStock()
          break
        case 'd':
          if (canDrawDiscard) onDrawDiscard()
          break
        case 'm':
          if (canToggleMode) onSetDiscardMode(!discardMode)
          break
        case 'n':
          if (match.phase === 'round_over') onNextRound()
          break
        case 'p':
          if (match.phase === 'game_over') onPlayAgain()
          break
        default:
          break
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [
    match,
    busy,
    helpOpen,
    canDrawStock,
    canDrawDiscard,
    canToggleMode,
    discardMode,
    onDrawStock,
    onDrawDiscard,
    onSetDiscardMode,
    onNextRound,
    onPlayAgain,
  ])

  if (!match) {
    return (
      <div className="match-view">
        <NewMatchForm onCreate={onCreate} submitting={busy} />
        {error && <p className="error-banner">{error}</p>}
      </div>
    )
  }

  return (
    <div className="match-view">
      {error && <p className="error-banner">{error}</p>}

      <div className="match-toolbar">
        <button type="button" className="shortcuts-trigger" onClick={() => setHelpOpen(true)}>
          ⌨ Shortcuts <kbd>?</kbd>
        </button>
      </div>
      <KeyboardHelp open={helpOpen} onClose={() => setHelpOpen(false)} />

      <CenterPiles
        stockCount={match.stock_count}
        discardTop={match.discard_top}
        drawnCard={match.drawn_card}
        canDrawStock={canDrawStock}
        canDrawDiscard={canDrawDiscard}
        onDrawStock={onDrawStock}
        onDrawDiscard={onDrawDiscard}
        showModeToggle={match.phase === 'awaiting_placement' && discardRevealPositions.size > 0}
        discardMode={discardMode}
        onSetDiscardMode={onSetDiscardMode}
      />

      {match.phase === 'round_over' && (
        <div className="status-panel">
          <h2>Round over</h2>
          <ul className="round-scores">
            {match.round_scores?.map((score, i) => (
              <li key={i}>
                <span>{match.player_names[i]}</span>
                <span>{score} pts</span>
              </li>
            ))}
          </ul>
          <button type="button" className="btn-primary" onClick={onNextRound} disabled={busy}>
            Start next round
          </button>
        </div>
      )}

      {match.phase === 'game_over' && (
        <div className="status-panel">
          <h2>Game over</h2>
          <p className="winner-line">
            Winner: <strong>{match.player_names[match.total_scores.indexOf(Math.min(...match.total_scores))]}</strong>
          </p>
          <button type="button" className="btn-primary" onClick={onPlayAgain}>
            Play again
          </button>
        </div>
      )}

      <div className="boards">
        {match.boards.map((board, playerIndex) => {
          let clickable = new Set<number>()
          if (playerIndex === match.current_player) {
            if (match.phase === 'initial_flip') clickable = flipPositions
            else if (match.phase === 'awaiting_placement') clickable = discardMode ? discardRevealPositions : placePositions
          }
          return (
            <PlayerBoard
              key={playerIndex}
              board={board}
              name={match.player_names[playerIndex]}
              isCurrentPlayer={playerIndex === match.current_player}
              isFinalTurn={match.finisher !== null}
              clickablePositions={clickable}
              onCardClick={(position) => onCardClick(playerIndex, position)}
            />
          )
        })}
      </div>
    </div>
  )
}

export default MatchView
