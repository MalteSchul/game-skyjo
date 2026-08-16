import { describe, expect, it } from 'vitest'
import { buildDeck, DECK_SIZE } from './deck'

function countsByValue(deck: number[]): Map<number, number> {
  const counts = new Map<number, number>()
  for (const card of deck) {
    counts.set(card, (counts.get(card) ?? 0) + 1)
  }
  return counts
}

describe('buildDeck', () => {
  it('happy path: builds the official 150-card deck with the correct composition', () => {
    const deck = buildDeck(42)

    expect(deck).toHaveLength(DECK_SIZE)
    expect(deck).toHaveLength(150)

    const counts = countsByValue(deck)
    expect(counts.get(-2)).toBe(5)
    expect(counts.get(-1)).toBe(10)
    expect(counts.get(0)).toBe(15)
    for (let value = 1; value <= 12; value++) {
      expect(counts.get(value)).toBe(10)
    }
  })

  it('happy path: same seed produces the same shuffle order', () => {
    expect(buildDeck(7)).toEqual(buildDeck(7))
  })

  it('sad path: rejects a non-number seed instead of silently producing a bad deck', () => {
    // @ts-expect-error intentionally passing an invalid seed type
    expect(() => buildDeck('not-a-seed')).toThrow(TypeError)
  })

  it('bad path: seed=0 is a falsy value but must still be treated as a valid seed', () => {
    const deck = buildDeck(0)

    expect(deck).toHaveLength(150)
    expect(countsByValue(deck).get(0)).toBe(15)
    // Deterministic: must match a second call with the same seed.
    expect(deck).toEqual(buildDeck(0))
  })
})
