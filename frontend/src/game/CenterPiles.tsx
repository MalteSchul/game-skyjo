import { cardTone } from './cardTone'

interface CenterPilesProps {
  stockCount: number
  discardTop: number | null
  drawnCard: number | null
  canDrawStock: boolean
  canDrawDiscard: boolean
  onDrawStock: () => void
  onDrawDiscard: () => void
  showModeToggle: boolean
  discardMode: boolean
  onSetDiscardMode: (discardMode: boolean) => void
}

/**
 * Stays mounted for the whole round (initial flip through round/game over) so the
 * piles, drawn card and mode toggle occupy one fixed spot instead of swapping
 * DOM blocks per phase — that swapping was pushing the boards below up and down
 * on every draw/place. The toggle row keeps its layout slot via `visibility`
 * even when inapplicable, rather than unmounting, so its own appearance never
 * shifts anything either.
 */
function CenterPiles({
  stockCount,
  discardTop,
  drawnCard,
  canDrawStock,
  canDrawDiscard,
  onDrawStock,
  onDrawDiscard,
  showModeToggle,
  discardMode,
  onSetDiscardMode,
}: CenterPilesProps) {
  const discardToneClass = discardTop !== null ? cardTone(discardTop) : ''

  return (
    <div className="turn-panel">
      <div className="center-piles">
        <button
          type="button"
          className="pile pile-stock"
          onClick={onDrawStock}
          disabled={!canDrawStock}
          aria-label={`Stock, ${stockCount} cards left`}
        >
          <span className="pile-label" aria-hidden="true">
            Stock
          </span>
          <span className="pile-count" aria-hidden="true">
            {stockCount}
          </span>
        </button>
        <button
          type="button"
          className={`pile pile-discard ${discardToneClass}`}
          onClick={onDrawDiscard}
          disabled={!canDrawDiscard}
          aria-label={discardTop !== null ? `Discard, top card value ${discardTop}` : 'Discard, empty'}
        >
          <span className="pile-label" aria-hidden="true">
            Discard
          </span>
          <span className={`pile-value ${discardTop === null ? 'pile-value-empty' : ''}`} aria-hidden="true">
            {discardTop !== null ? discardTop : '—'}
          </span>
        </button>
        <div
          className={`drawn-card-slot ${drawnCard === null ? 'drawn-card-slot-hidden' : ''}`}
          aria-hidden={drawnCard === null}
        >
          <span className="drawn-card-label">Drawn</span>
          <div className={`card card-face-up card-lg ${drawnCard !== null ? cardTone(drawnCard) : ''}`}>
            {drawnCard ?? ''}
          </div>
        </div>
      </div>
      <div
        className={`mode-toggle-row ${showModeToggle ? '' : 'mode-toggle-row-hidden'}`}
        aria-hidden={!showModeToggle}
      >
        <div className="mode-toggle" role="group" aria-label="Drawn card action">
          <button
            type="button"
            className={`mode-toggle-btn ${discardMode ? '' : 'mode-toggle-btn-active'}`}
            onClick={() => onSetDiscardMode(false)}
            disabled={!showModeToggle}
            tabIndex={showModeToggle ? 0 : -1}
          >
            Place on board
          </button>
          <button
            type="button"
            className={`mode-toggle-btn ${discardMode ? 'mode-toggle-btn-active' : ''}`}
            onClick={() => onSetDiscardMode(true)}
            disabled={!showModeToggle}
            tabIndex={showModeToggle ? 0 : -1}
          >
            Discard &amp; reveal
          </button>
        </div>
      </div>
    </div>
  )
}

export default CenterPiles
