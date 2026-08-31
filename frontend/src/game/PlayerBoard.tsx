import type { BoardOut } from '../api/types'
import Card from './Card'

export const BOARD_COLUMNS = 4

/** Sum of this board's currently face-up cards - a face-down card's `value`
 * is always redacted to `null` on the wire (see `CardOut.from_card`), and a
 * cleared column's slots are `null` outright, so summing every non-null
 * value already means exactly "known" cards without checking `face_up`
 * separately. Lets a player track their running visible score without doing
 * the addition themselves. */
function visibleTotal(board: BoardOut): number {
  return board.cards.reduce((sum, card) => sum + (card?.value ?? 0), 0)
}

interface PlayerBoardProps {
  board: BoardOut
  name: string
  isCurrentPlayer: boolean
  isFinalTurn: boolean
  clickablePositions: ReadonlySet<number>
  rovingPosition: number | null
  onCardClick: (position: number) => void
  onCardRef?: (position: number, el: HTMLButtonElement | null) => void
  onCardFocus?: (position: number) => void
}

/**
 * One clickable card is a roving Tab stop (tabIndex 0), the rest are
 * Tab-skipped; MatchView drives which one via `rovingPosition` (so arrow
 * keys work globally, not just once focus already happens to be on this
 * board) and this component reports focus changes back up via
 * `onCardFocus`, so Tab/click-driven focus stays in sync too.
 */
function PlayerBoard({
  board,
  name,
  isCurrentPlayer,
  isFinalTurn,
  clickablePositions,
  rovingPosition,
  onCardClick,
  onCardRef,
  onCardFocus,
}: PlayerBoardProps) {
  const turnLabel = isFinalTurn ? 'final turn' : 'current turn'
  return (
    <section className={`player-board ${isCurrentPlayer ? 'player-board-active' : ''}`}>
      <h3 className="player-board-name">
        {name}
        {' '}
        <span className="visible-total-chip" aria-hidden="true" title="Sum of this player's face-up cards">
          Σ {visibleTotal(board)}
        </span>
        {isCurrentPlayer && (
          <>
            {' '}
            <span className="sr-only">({turnLabel})</span>
            <span className={`turn-chip ${isFinalTurn ? 'turn-chip-final' : ''}`} aria-hidden="true">
              {isFinalTurn ? 'Final turn' : 'Current turn'}
            </span>
          </>
        )}
      </h3>
      <div className="board-grid" style={{ gridTemplateColumns: `repeat(${BOARD_COLUMNS}, auto)` }}>
        {board.cards.map((card, position) => {
          const clickable = clickablePositions.has(position)
          return (
            <Card
              key={position}
              ref={onCardRef ? (el) => onCardRef(position, el) : undefined}
              card={card}
              onClick={clickable ? () => onCardClick(position) : undefined}
              onFocus={clickable && onCardFocus ? () => onCardFocus(position) : undefined}
              tabIndex={position === rovingPosition ? 0 : -1}
            />
          )
        })}
      </div>
    </section>
  )
}

export default PlayerBoard
