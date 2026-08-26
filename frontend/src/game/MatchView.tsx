import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ActionTypeName, MatchStateOut, PlayerTypeName } from '../api/types'
import CenterPiles from './CenterPiles'
import KeyboardHelp from './KeyboardHelp'
import NewMatchForm from './NewMatchForm'
import PlayerBoard, { BOARD_COLUMNS } from './PlayerBoard'
import RestartConfirm from './RestartConfirm'

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
  onCreate: (
    playerCount: number,
    seed: number | undefined,
    playerNames: string[],
    playerTypes: PlayerTypeName[],
    playerMctsModels: (string | null)[],
    playerMctsNumSimulations: (number | null)[],
  ) => void
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
  // While a bot is deciding, nothing is actionable — regardless of what
  // `legal_actions`/`current_player` say, since those describe the bot's
  // seat, not anything a human at the keyboard can do right now.
  const thinking = match?.status === 'thinking'
  const placePositions = useMemo(() => (match ? legalPositionsFor(match, 'place') : new Set<number>()), [match])
  const discardRevealPositions = useMemo(
    () => (match ? legalPositionsFor(match, 'discard_and_reveal') : new Set<number>()),
    [match],
  )
  const flipPositions = useMemo(() => (match ? legalPositionsFor(match, 'flip_initial') : new Set<number>()), [match])
  const canDrawStock = useMemo(
    () => (match && !thinking ? match.legal_actions.some((a) => a.type === 'draw_stock') : false),
    [match, thinking],
  )
  const canDrawDiscard = useMemo(
    () => (match && !thinking ? match.legal_actions.some((a) => a.type === 'draw_discard') : false),
    [match, thinking],
  )
  const canToggleMode = match?.phase === 'awaiting_placement' && !thinking && discardRevealPositions.size > 0

  // The current player's selectable cards, regardless of phase — this is
  // also what arrow-key/roving-focus navigation operates over.
  const activeClickablePositions = useMemo(() => {
    if (!match || thinking) return new Set<number>()
    if (match.phase === 'initial_flip') return flipPositions
    if (match.phase === 'awaiting_placement') return discardMode ? discardRevealPositions : placePositions
    return new Set<number>()
  }, [match, thinking, discardMode, flipPositions, discardRevealPositions, placePositions])
  const activeClickableList = useMemo(
    () => Array.from(activeClickablePositions).sort((a, b) => a - b),
    [activeClickablePositions],
  )

  const [helpOpen, setHelpOpen] = useState(false)
  // Guards the restart shortcut/button while a match is in progress, so a
  // stray "P" press can't silently discard it. Once the game is over there's
  // nothing left to lose, so that case skips this and restarts immediately.
  const [restartConfirmOpen, setRestartConfirmOpen] = useState(false)

  function requestRestart() {
    if (match?.phase === 'game_over') onPlayAgain()
    else setRestartConfirmOpen(true)
  }

  function confirmRestart() {
    setRestartConfirmOpen(false)
    onPlayAgain()
  }

  // Roving-tabindex focus cursor for the current player's board. Lives here
  // (not inside PlayerBoard) so arrow keys can be handled as a global
  // shortcut below — working the instant it's your turn, not only once
  // something inside the grid already happens to have DOM focus.
  const buttonRefs = useRef(new Map<number, HTMLButtonElement>())
  const [rovingPosition, setRovingPosition] = useState<number | null>(null)
  // Position numbers are per-board-local (0-11), not globally unique, so a
  // roving position surviving a turn change could coincidentally still be
  // "valid" on the new current player's board while actually pointing at a
  // different card. Track whose turn the roving position belongs to so a
  // player change always forces a re-focus, never just a position check.
  const rovingPlayerRef = useRef<number | null>(null)

  // Auto-focus the first available card whenever it becomes (or stops
  // being) your turn — including right when the match starts. Without this,
  // arrow keys would need a prior Tab press to "enter" the grid, and
  // disabling the previous target's button also blurs it, so focus would
  // otherwise drop out of the grid after every single action.
  useEffect(() => {
    const samePlayer = rovingPlayerRef.current === (match?.current_player ?? null)
    if (samePlayer && rovingPosition !== null && activeClickablePositions.has(rovingPosition)) return
    rovingPlayerRef.current = match?.current_player ?? null
    const next = activeClickableList[0] ?? null
    setRovingPosition(next)
    if (next !== null) buttonRefs.current.get(next)?.focus()
  }, [match, activeClickableList, activeClickablePositions, rovingPosition])

  const moveFocus = useCallback(
    (dCol: number, dRow: number) => {
      if (!match || activeClickableList.length === 0) return
      const totalCards = match.boards[match.current_player].cards.length
      const rows = Math.ceil(totalCards / BOARD_COLUMNS)
      let position = rovingPosition ?? activeClickableList[0]
      for (let step = 0; step < totalCards; step++) {
        const row = Math.floor(position / BOARD_COLUMNS)
        const col = position % BOARD_COLUMNS
        const nextRow = (row + dRow + rows) % rows
        const nextCol = (col + dCol + BOARD_COLUMNS) % BOARD_COLUMNS
        position = nextRow * BOARD_COLUMNS + nextCol
        if (activeClickablePositions.has(position)) {
          setRovingPosition(position)
          buttonRefs.current.get(position)?.focus()
          return
        }
      }
    },
    [match, activeClickableList, activeClickablePositions, rovingPosition],
  )

  // Global one-key shortcuts for everything a turn can do: drawing (q/w),
  // the place/discard-reveal mode select (also q/w — drawing and placement
  // never overlap in phase), round transitions, and moving the roving-focus
  // cursor between cards.
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
      if (restartConfirmOpen) {
        if (event.key === 'Escape') setRestartConfirmOpen(false)
        else if (event.key === 'Enter' || event.key.toLowerCase() === 'p') confirmRestart()
        return
      }
      if (busy) return

      switch (event.key) {
        case 'ArrowUp':
          event.preventDefault()
          moveFocus(0, -1)
          return
        case 'ArrowDown':
          event.preventDefault()
          moveFocus(0, 1)
          return
        case 'ArrowLeft':
          event.preventDefault()
          moveFocus(-1, 0)
          return
        case 'ArrowRight':
          event.preventDefault()
          moveFocus(1, 0)
          return
        default:
          break
      }

      switch (event.key.toLowerCase()) {
        case 'q':
          if (canDrawStock) onDrawStock()
          else if (canToggleMode) onSetDiscardMode(false)
          break
        case 'w':
          if (canDrawDiscard) onDrawDiscard()
          else if (canToggleMode) onSetDiscardMode(true)
          break
        case 'n':
          if (match.phase === 'round_over') onNextRound()
          break
        case 'p':
          requestRestart()
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
    restartConfirmOpen,
    canDrawStock,
    canDrawDiscard,
    canToggleMode,
    moveFocus,
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
        <button type="button" className="restart-trigger" onClick={requestRestart} disabled={busy}>
          ⟲ Restart <kbd>P</kbd>
        </button>
        <button type="button" className="shortcuts-trigger" onClick={() => setHelpOpen(true)}>
          ⌨ Shortcuts <kbd>?</kbd>
        </button>
      </div>
      <KeyboardHelp open={helpOpen} onClose={() => setHelpOpen(false)} />
      <RestartConfirm open={restartConfirmOpen} onConfirm={confirmRestart} onCancel={() => setRestartConfirmOpen(false)} />

      {thinking && (
        <p className="thinking-banner" role="status">
          {match.player_names[match.thinking_player ?? match.current_player]} is thinking
          {match.thinking_progress != null ? ` (${Math.round(match.thinking_progress * 100)}%)` : '…'}
        </p>
      )}

      <CenterPiles
        stockCount={match.stock_count}
        discardTop={match.discard_top}
        drawnCard={match.drawn_card}
        canDrawStock={canDrawStock}
        canDrawDiscard={canDrawDiscard}
        onDrawStock={onDrawStock}
        onDrawDiscard={onDrawDiscard}
        showModeToggle={canToggleMode}
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
          const isCurrent = playerIndex === match.current_player
          return (
            <PlayerBoard
              key={playerIndex}
              board={board}
              name={match.player_names[playerIndex]}
              isCurrentPlayer={isCurrent}
              isFinalTurn={match.finisher !== null}
              clickablePositions={isCurrent ? activeClickablePositions : new Set<number>()}
              rovingPosition={isCurrent ? rovingPosition : null}
              onCardClick={(position) => onCardClick(playerIndex, position)}
              onCardRef={
                isCurrent
                  ? (position, el) => {
                      if (el) buttonRefs.current.set(position, el)
                      else buttonRefs.current.delete(position)
                    }
                  : undefined
              }
              onCardFocus={isCurrent ? (position) => setRovingPosition(position) : undefined}
            />
          )
        })}
      </div>
    </div>
  )
}

export default MatchView
