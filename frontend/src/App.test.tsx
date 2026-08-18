import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from './api/matchClient'
import { loadStoredMatchId, storeMatchId } from './api/matchStorage'
import type { MatchHistoryOut, MatchStateOut } from './api/types'
import App from './App'

const { createMatch, applyAction, startNextRound, getMatch, getMatchHistory, gotoMatchHistoryNode } = vi.hoisted(() => ({
  createMatch: vi.fn(),
  applyAction: vi.fn(),
  startNextRound: vi.fn(),
  getMatch: vi.fn(),
  getMatchHistory: vi.fn(),
  gotoMatchHistoryNode: vi.fn(),
}))

vi.mock('./api/matchClient', async () => {
  const actual = await vi.importActual<typeof import('./api/matchClient')>('./api/matchClient')
  return { ...actual, createMatch, applyAction, startNextRound, getMatch, getMatchHistory, gotoMatchHistoryNode }
})

const INITIAL_FLIP_MATCH: MatchStateOut = {
  match_id: 'm1',
  phase: 'initial_flip',
  boards: [
    { cards: Array.from({ length: 12 }, () => ({ value: null, face_up: false })) },
    { cards: Array.from({ length: 12 }, () => ({ value: null, face_up: false })) },
  ],
  player_names: ['Player 1', 'Player 2'],
  stock_count: 142,
  discard_top: 6,
  current_player: 0,
  drawn_card: null,
  finisher: null,
  players_awaiting_final_turn: [],
  round_scores: null,
  total_scores: [0, 0],
  target_score: 100,
  legal_actions: Array.from({ length: 12 }, (_, i) => ({ type: 'flip_initial' as const, position: i })),
}

const AWAITING_DRAW_MATCH: MatchStateOut = {
  ...INITIAL_FLIP_MATCH,
  phase: 'awaiting_draw',
  legal_actions: [
    { type: 'draw_stock', position: null },
    { type: 'draw_discard', position: null },
  ],
}

const ROOT_ONLY_HISTORY: MatchHistoryOut = {
  head_id: 'root',
  nodes: [
    {
      node_id: 'root',
      parent_id: null,
      seq: 0,
      round_index: 0,
      actor: null,
      current_player: 0,
      phase: 'initial_flip',
      edge: { kind: 'root', action_type: null, position: null },
    },
  ],
}

beforeEach(() => {
  createMatch.mockReset()
  applyAction.mockReset()
  startNextRound.mockReset()
  getMatch.mockReset()
  getMatchHistory.mockReset()
  gotoMatchHistoryNode.mockReset()
  getMatchHistory.mockResolvedValue(ROOT_ONLY_HISTORY)
  localStorage.clear()
  // The health check in App's useEffect hits the real `fetch`, not matchClient
  // — stub it so tests don't make a real network call or race on its result.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }))
})

afterEach(() => {
  vi.unstubAllGlobals()
  localStorage.clear()
})

describe('App', () => {
  it('creates a match and renders the resulting boards (happy path)', async () => {
    createMatch.mockResolvedValue(INITIAL_FLIP_MATCH)
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(await screen.findByRole('heading', { name: /Player 1/ })).toBeInTheDocument()
    expect(createMatch).toHaveBeenCalledWith({ player_count: 2, seed: undefined, player_names: ['', ''] })
  })

  it('shows the scoreboard in the header once a match starts, not before (happy path)', async () => {
    createMatch.mockResolvedValue(INITIAL_FLIP_MATCH)
    render(<App />)
    const header = screen.getByRole('banner')

    expect(within(header).queryByText('Player 1')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(await within(header).findByText('Player 1')).toBeInTheDocument()
    expect(within(header).getByText('Player 2')).toBeInTheDocument()
  })

  it('flips a card by clicking it during initial_flip and applies the resulting state', async () => {
    createMatch.mockResolvedValue(INITIAL_FLIP_MATCH)
    applyAction.mockResolvedValue({ ...AWAITING_DRAW_MATCH })
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))
    await screen.findByRole('heading', { name: /Player 1/ })

    const faceDownCards = screen.getAllByRole('button', { name: 'face-down card' })
    fireEvent.click(faceDownCards[0])

    expect(applyAction).toHaveBeenCalledWith('m1', { type: 'flip_initial', position: 0 })
    expect(await screen.findByRole('button', { name: 'Stock, 142 cards left' })).toBeInTheDocument()
  })

  it('shows the backend error message when an action is rejected (sad path)', async () => {
    createMatch.mockResolvedValue(INITIAL_FLIP_MATCH)
    applyAction.mockRejectedValue(new ApiError(409, 'that action is not legal right now'))
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))
    await screen.findByRole('heading', { name: /Player 1/ })

    fireEvent.click(screen.getAllByRole('button', { name: 'face-down card' })[0])

    expect(await screen.findByText('that action is not legal right now')).toBeInTheDocument()
  })

  it('shows a generic error message when match creation fails without reaching the server (bad path)', async () => {
    createMatch.mockRejectedValue(new Error('boom'))
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(await screen.findByText('Something went wrong.')).toBeInTheDocument()
  })

  it('remembers the match id and restores it on the next visit (happy path)', async () => {
    createMatch.mockResolvedValue(INITIAL_FLIP_MATCH)
    const { unmount } = render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))
    await screen.findByRole('heading', { name: /Player 1/ })

    expect(loadStoredMatchId()).toBe('m1')
    // Simulate a page refresh: the whole tree unmounts and a fresh App boots.
    unmount()

    getMatch.mockResolvedValue(AWAITING_DRAW_MATCH)
    render(<App />)

    expect(await screen.findByRole('button', { name: 'Stock, 142 cards left' })).toBeInTheDocument()
    expect(getMatch).toHaveBeenCalledWith('m1')
  })

  it('falls back to the new-match form and forgets the id when the remembered match no longer exists (sad path)', async () => {
    storeMatchId('stale-id')
    getMatch.mockRejectedValue(new ApiError(404, "match 'stale-id' not found"))

    render(<App />)

    expect(await screen.findByRole('button', { name: 'Start match' })).toBeInTheDocument()
    expect(loadStoredMatchId()).toBeNull()
  })

  it('forgets the stored match once the player starts a new one after game over (bad path: avoids getting stuck on a finished match)', async () => {
    createMatch.mockResolvedValue(INITIAL_FLIP_MATCH)
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))
    await screen.findByRole('heading', { name: /Player 1/ })
    applyAction.mockResolvedValue({ ...INITIAL_FLIP_MATCH, phase: 'game_over', total_scores: [80, 30] })
    fireEvent.click(screen.getAllByRole('button', { name: 'face-down card' })[0])
    await screen.findByRole('button', { name: 'Play again' })

    fireEvent.click(screen.getByRole('button', { name: 'Play again' }))

    expect(screen.getByRole('button', { name: 'Start match' })).toBeInTheDocument()
    expect(loadStoredMatchId()).toBeNull()
  })

  it('loads and shows the match history sidebar once a match starts (happy path)', async () => {
    createMatch.mockResolvedValue(INITIAL_FLIP_MATCH)
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(await screen.findByText('Game start')).toBeInTheDocument()
    expect(getMatchHistory).toHaveBeenCalledWith('m1')
  })

  it('jumps to a past history node when clicked (happy path)', async () => {
    createMatch.mockResolvedValue(INITIAL_FLIP_MATCH)
    applyAction.mockResolvedValue(AWAITING_DRAW_MATCH)
    gotoMatchHistoryNode.mockResolvedValue(INITIAL_FLIP_MATCH)
    // First fetch (right after creation) has the root as the head; the second
    // (after the flip action) has moved the head to a new leaf, so "Game
    // start" becomes a clickable past entry instead of the disabled head.
    getMatchHistory.mockResolvedValueOnce(ROOT_ONLY_HISTORY).mockResolvedValueOnce({
      head_id: 'flip0',
      nodes: [
        ...ROOT_ONLY_HISTORY.nodes,
        {
          node_id: 'flip0',
          parent_id: 'root',
          seq: 1,
          round_index: 0,
          actor: 0,
          current_player: 1,
          phase: 'awaiting_draw',
          edge: { kind: 'action', action_type: 'flip_initial', position: 0 },
        },
      ],
    })
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))
    await screen.findByRole('heading', { name: /Player 1/ })
    fireEvent.click(screen.getAllByRole('button', { name: 'face-down card' })[0])
    await screen.findByText('Player 1 flipped card 1')

    fireEvent.click(screen.getByRole('button', { name: 'Game start' }))

    expect(gotoMatchHistoryNode).toHaveBeenCalledWith('m1', 'root')
    expect(getMatchHistory).toHaveBeenLastCalledWith('m1')
  })

  it('shows the backend error message when jumping to history fails (sad path)', async () => {
    createMatch.mockResolvedValue(INITIAL_FLIP_MATCH)
    applyAction.mockResolvedValue(AWAITING_DRAW_MATCH)
    gotoMatchHistoryNode.mockRejectedValue(new ApiError(404, 'history node not found'))
    getMatchHistory.mockResolvedValueOnce(ROOT_ONLY_HISTORY).mockResolvedValueOnce({
      head_id: 'flip0',
      nodes: [
        ...ROOT_ONLY_HISTORY.nodes,
        {
          node_id: 'flip0',
          parent_id: 'root',
          seq: 1,
          round_index: 0,
          actor: 0,
          current_player: 1,
          phase: 'awaiting_draw',
          edge: { kind: 'action', action_type: 'flip_initial', position: 0 },
        },
      ],
    })
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))
    await screen.findByRole('heading', { name: /Player 1/ })
    fireEvent.click(screen.getAllByRole('button', { name: 'face-down card' })[0])
    await screen.findByText('Player 1 flipped card 1')

    fireEvent.click(screen.getByRole('button', { name: 'Game start' }))

    expect(await screen.findByText('history node not found')).toBeInTheDocument()
  })

  it('still renders the board when the history fetch fails, leaving the sidebar empty (bad path)', async () => {
    createMatch.mockResolvedValue(INITIAL_FLIP_MATCH)
    getMatchHistory.mockReset()
    getMatchHistory.mockRejectedValue(new Error('boom'))
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(await screen.findByRole('heading', { name: /Player 1/ })).toBeInTheDocument()
    expect(screen.queryByLabelText('Match history')).not.toBeInTheDocument()
  })
})
