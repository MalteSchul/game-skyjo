import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
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

/**
 * Arrow-key navigation across this board's clickable cards, following the
 * standard "roving tabindex" pattern: one clickable card is a normal Tab
 * stop, the rest are Tab-skipped, and the arrow keys move which one that is
 * (wrapping within the 4-column grid, skipping over non-clickable cells).
 */
function PlayerBoard({ board, name, isCurrentPlayer, isFinalTurn, clickablePositions, onCardClick }: PlayerBoardProps) {
  const turnLabel = isFinalTurn ? 'final turn' : 'current turn'
  const buttonRefs = useRef(new Map<number, HTMLButtonElement>())
  const rows = Math.ceil(board.cards.length / COLUMNS)

  const clickableList = useMemo(() => Array.from(clickablePositions).sort((a, b) => a - b), [clickablePositions])
  const [rovingPosition, setRovingPosition] = useState<number | null>(clickableList[0] ?? null)

  // When it becomes (or stops being) this board's turn, the roving target
  // changes. Move real DOM focus along with it — without this, a keyboard
  // player would have to Tab back into the grid after every single action,
  // since disabling the old target's button also blurs it.
  useEffect(() => {
    if (rovingPosition !== null && clickablePositions.has(rovingPosition)) return
    const next = clickableList[0] ?? null
    setRovingPosition(next)
    if (next !== null) buttonRefs.current.get(next)?.focus()
  }, [clickableList, clickablePositions, rovingPosition])

  function moveFocus(from: number, dCol: number, dRow: number) {
    let position = from
    for (let step = 0; step < board.cards.length; step++) {
      const row = Math.floor(position / COLUMNS)
      const col = position % COLUMNS
      const nextRow = (row + dRow + rows) % rows
      const nextCol = (col + dCol + COLUMNS) % COLUMNS
      position = nextRow * COLUMNS + nextCol
      if (clickablePositions.has(position)) {
        setRovingPosition(position)
        buttonRefs.current.get(position)?.focus()
        return
      }
    }
  }

  function handleGridKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (clickableList.length === 0) return
    const current = rovingPosition ?? clickableList[0]
    switch (event.key) {
      case 'ArrowUp':
        event.preventDefault()
        moveFocus(current, 0, -1)
        break
      case 'ArrowDown':
        event.preventDefault()
        moveFocus(current, 0, 1)
        break
      case 'ArrowLeft':
        event.preventDefault()
        moveFocus(current, -1, 0)
        break
      case 'ArrowRight':
        event.preventDefault()
        moveFocus(current, 1, 0)
        break
      default:
        break
    }
  }

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
      <div
        className="board-grid"
        style={{ gridTemplateColumns: `repeat(${COLUMNS}, auto)` }}
        onKeyDown={handleGridKeyDown}
      >
        {board.cards.map((card, position) => {
          const clickable = clickablePositions.has(position)
          return (
            <Card
              key={position}
              ref={(el) => {
                if (el) buttonRefs.current.set(position, el)
                else buttonRefs.current.delete(position)
              }}
              card={card}
              onClick={clickable ? () => onCardClick(position) : undefined}
              tabIndex={position === rovingPosition ? 0 : -1}
            />
          )
        })}
      </div>
    </section>
  )
}

export default PlayerBoard
