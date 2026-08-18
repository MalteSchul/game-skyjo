/** Maps a revealed card value to one of Skyjo's five colour bands, driving the
 * `tone-*` CSS classes shared by board cards, the drawn-card display, and the
 * discard pile swatch. */
export type CardTone = 'tone-neg' | 'tone-zero' | 'tone-low' | 'tone-mid' | 'tone-high'

export function cardTone(value: number): CardTone {
  if (value <= -1) return 'tone-neg'
  if (value === 0) return 'tone-zero'
  if (value <= 4) return 'tone-low'
  if (value <= 8) return 'tone-mid'
  return 'tone-high'
}
