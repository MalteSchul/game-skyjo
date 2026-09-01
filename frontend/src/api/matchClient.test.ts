import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  ApiError,
  applyAction,
  createMatch,
  getMatch,
  getMatchHistory,
  getMctsModels,
  gotoMatchHistoryNode,
  startNextRound,
} from './matchClient'
import type { MatchHistoryOut, MatchStateOut } from './types'

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
  player_types: ['human', 'human'],
  stock_count: 140,
  discard_top: 3,
  current_player: 0,
  drawn_card: null,
  finisher: null,
  players_awaiting_final_turn: [],
  round_scores: null,
  total_scores: [0, 0],
  round_history: [],
  target_score: 100,
  legal_actions: [],
  status: 'idle',
  thinking_player: null,
  thinking_progress: null,
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

  it('fetches the list of selectable mcts models (happy path)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(['strong', 'weak'])))

    await expect(getMctsModels()).resolves.toEqual(['strong', 'weak'])
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

  it('fetches a match history tree (happy path)', async () => {
    const history: MatchHistoryOut = {
      head_id: 'n1',
      nodes: [
        { node_id: 'n1', parent_id: null, seq: 0, round_index: 0, actor: null, current_player: 0, phase: 'initial_flip', edge: { kind: 'root', action_type: null, position: null }, has_mcts_tree: false, mcts_visit_share: null, mcts_prior_overridden: null },
      ],
    }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(history)))

    await expect(getMatchHistory('abc123')).resolves.toEqual(history)
  })

  it('jumps to a history node and returns the resulting state (happy path)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(MATCH))
    vi.stubGlobal('fetch', fetchMock)

    const result = await gotoMatchHistoryNode('abc123', 'n1')

    expect(result).toEqual(MATCH)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('http://localhost:8000/matches/abc123/history/n1/goto')
    expect(init.method).toBe('POST')
  })

  it('raises an ApiError when jumping to an unknown history node (sad path)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: "history node 'n9' not found" }, 404)))

    await expect(gotoMatchHistoryNode('abc123', 'n9')).rejects.toBeInstanceOf(ApiError)
  })
})
