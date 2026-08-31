import type { BoardOut, CardOut } from '../../api/types'
import Card from '../../game/Card'

const BOARD_COLUMNS = 4

/** This is the true, un-redacted state (see `types.ts`), so a face-down
 * card's real value is always in the data - `revealHidden` only controls
 * whether *this component* shows it, so the tool can switch between "what
 * really happened" and "what this player could actually see" without
 * needing two copies of the data. */
function displayedCard(card: CardOut | null, revealHidden: boolean): CardOut | null {
  if (card === null || card.face_up || !revealHidden) return card
  return { value: card.value, face_up: true }
}

/** Whether a card is only showing its value because of `revealHidden` - it
 * was actually still face-down in the real game. Drives the striped overlay
 * below, so a forced reveal never looks identical to a genuinely-known card. */
function isForcedReveal(card: CardOut | null, revealHidden: boolean): boolean {
  return card !== null && !card.face_up && revealHidden
}

interface ReplayBoardProps {
  board: BoardOut
  name: string
  isActing: boolean
  revealHidden: boolean
}

/** A read-only rendering of one player's board for the game-replay tool -
 * reuses the live game's own `Card` component (and its `.player-board`/
 * `.board-grid` styling from index.css) so a recorded position looks exactly
 * like it would have in the real app, just without any click handlers. */
export default function ReplayBoard({ board, name, isActing, revealHidden }: ReplayBoardProps) {
  return (
    <section className={`player-board ${isActing ? 'player-board-active' : ''}`}>
      <h3 className="player-board-name">
        {name}
        {isActing && (
          <>
            {' '}
            <span className="turn-chip" aria-hidden="true">
              Acting
            </span>
          </>
        )}
      </h3>
      <div className="board-grid" style={{ gridTemplateColumns: `repeat(${BOARD_COLUMNS}, auto)` }}>
        {board.cards.map((card, position) => (
          <div
            key={position}
            className={isForcedReveal(card, revealHidden) ? 'gr-secret-card' : undefined}
            title={isForcedReveal(card, revealHidden) ? 'Still face-down in the real game - shown only via "reveal hidden cards"' : undefined}
          >
            <Card card={displayedCard(card, revealHidden)} />
          </div>
        ))}
      </div>
    </section>
  )
}
