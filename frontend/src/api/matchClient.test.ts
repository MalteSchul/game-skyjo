import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, applyAction, createMatch, getMatch, startNextRound } from './matchClient'
import type { MatchStateOut } from './types'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const MATCH: MatchStateOut = {
  match_id: 'abc123',
  phase: 'initial_flip',
  boards: [{ cards: [] }],
  player_names: ['Player 1', 'Player 2'],
  stock_count: 140,
  discard_top: 3,
  current_player: 0,
  drawn_card: null,
  finisher: null,
  players_awaiting_final_turn: [],
  round_scores: null,
  total_scores: [0, 0],
  target_score: 100,
  legal_actions: [],
}

describe('matchClient', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('creates a match and returns the parsed state (happy path)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(MATCH, 201))
    vi.stubGlobal('fetch', fetchMock)

    const result = await createMatch({ player_count: 2, seed: 42 })

    expect(result).toEqual(MATCH)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('http://localhost:8000/matches')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({ player_count: 2, seed: 42 })
  })

  it('fetches an existing match by id (happy path)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(MATCH)))

    await expect(getMatch('abc123')).resolves.toEqual(MATCH)
  })

  it('raises an ApiError with the backend detail on an illegal action (sad path)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Action(type=<ActionType.DRAW_STOCK...) is not legal" }, 409)),
    )

    await expect(applyAction('abc123', { type: 'draw_stock' })).rejects.toMatchObject({
      name: 'ApiError',
      status: 409,
      message: expect.stringContaining('not legal'),
    })
  })

  it('raises an ApiError when the match does not exist (sad path)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'match not found' }, 404)))

    await expect(startNextRound('missing')).rejects.toBeInstanceOf(ApiError)
  })

  it('raises an ApiError instead of throwing a raw network error when the server is unreachable (bad path)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('fetch failed')))

    await expect(getMatch('abc123')).rejects.toMatchObject({ name: 'ApiError', status: 0 })
  })
})
