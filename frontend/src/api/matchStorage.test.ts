import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearStoredMatchId, loadStoredMatchId, storeMatchId } from './matchStorage'

afterEach(() => {
  localStorage.clear()
  vi.restoreAllMocks()
})

describe('matchStorage', () => {
  it('round-trips a stored match id (happy path)', () => {
    storeMatchId('abc123')
    expect(loadStoredMatchId()).toBe('abc123')

    clearStoredMatchId()
    expect(loadStoredMatchId()).toBeNull()
  })

  it('returns null when nothing has been stored yet (sad path)', () => {
    expect(loadStoredMatchId()).toBeNull()
  })

  it('swallows storage errors instead of throwing (bad path)', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('storage disabled')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('storage disabled')
    })

    expect(() => storeMatchId('abc123')).not.toThrow()
    expect(loadStoredMatchId()).toBeNull()
  })
})
