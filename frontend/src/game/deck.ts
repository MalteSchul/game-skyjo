export type Card = number

// TODO: duplicated in backend/src/skyjo/domain/deck.py — no shared source of truth yet.
// Official Skyjo deck: 150 cards.
const CARD_COUNTS: ReadonlyArray<readonly [Card, number]> = [
  [-2, 5],
  [-1, 10],
  [0, 15],
  ...Array.from({ length: 12 }, (_, i) => [i + 1, 10] as const),
]

export const DECK_SIZE = CARD_COUNTS.reduce((sum, [, count]) => sum + count, 0)

// mulberry32: small deterministic PRNG so a seed produces a reproducible shuffle.
function mulberry32(seed: number): () => number {
  let state = seed
  return () => {
    state |= 0
    state = (state + 0x6d2b79f5) | 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Builds and shuffles the standard 150-card Skyjo deck.
 * Pass a seed for a reproducible shuffle (e.g. in tests); omit it for a random one.
 */
export function buildDeck(seed?: number): Card[] {
  if (seed !== undefined && typeof seed !== 'number') {
    throw new TypeError('buildDeck: seed must be a number if provided')
  }

  const deck: Card[] = []
  for (const [value, count] of CARD_COUNTS) {
    for (let i = 0; i < count; i++) {
      deck.push(value)
    }
  }

  const random = seed === undefined ? Math.random : mulberry32(seed)
  for (let i = deck.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    ;[deck[i], deck[j]] = [deck[j], deck[i]]
  }

  return deck
}
