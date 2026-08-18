import { useMemo } from 'react'
import type { ActionTypeName, MatchStateOut } from '../api/types'
import CenterPiles from './CenterPiles'
import NewMatchForm from './NewMatchForm'
import PlayerBoard from './PlayerBoard'

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
