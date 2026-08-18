import type { BoardOut } from '../api/types'
import Card from './Card'

const COLUMNS = 4

interface PlayerBoardProps {
  board: BoardOut
  name: string
  isCurrentPlayer: boolean
  isFinalTurn: boolean
  clickablePositions: ReadonlySet<number>
  onCardClick: (position: number) => void
}

function PlayerBoard({ board, name, isCurrentPlayer, isFinalTurn, clickablePositions, onCardClick }: PlayerBoardProps) {
  const turnLabel = isFinalTurn ? 'final turn' : 'current turn'
  return (
    <section className={`player-board ${isCurrentPlayer ? 'player-board-active' : ''}`}>
      <h3 className="player-board-name">
        {name}
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
      <div className="board-grid" style={{ gridTemplateColumns: `repeat(${COLUMNS}, auto)` }}>
        {board.cards.map((card, position) => {
          const clickable = clickablePositions.has(position)
          return (
            <Card
              key={position}
              card={card}
              onClick={clickable ? () => onCardClick(position) : undefined}
            />
          )
        })}
      </div>
    </section>
  )
}

export default PlayerBoard
