import type { CardOut } from '../api/types'
import { cardTone } from './cardTone'

interface CardProps {
  card: CardOut | null
  onClick?: () => void
}

/** Renders one board slot: a cleared column (empty), a face-down back, or a revealed value. */
function Card({ card, onClick }: CardProps) {
  if (card === null) {
    return <div className="card card-empty" aria-label="cleared slot" />
  }

  const clickable = onClick !== undefined
  const tone = card.face_up && card.value !== null ? cardTone(card.value) : ''
  const className = ['card', card.face_up ? 'card-face-up' : 'card-face-down', tone, clickable ? 'card-clickable' : '']
    .filter(Boolean)
    .join(' ')

  return (
    <button
      type="button"
      className={className}
      onClick={onClick}
      disabled={!clickable}
      aria-label={card.face_up ? `card ${card.value}` : 'face-down card'}
    >
      {card.face_up ? card.value : ''}
    </button>
  )
}

export default Card
