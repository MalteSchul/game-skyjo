import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from './api/matchClient'
import type { MatchStateOut } from './api/types'
import App from './App'

const { createMatch, applyAction, startNextRound } = vi.hoisted(() => ({
  createMatch: vi.fn(),
  applyAction: vi.fn(),
  startNextRound: vi.fn(),
}))

vi.mock('./api/matchClient', async () => {
  const actual = await vi.importActual<typeof import('./api/matchClient')>('./api/matchClient')
  return { ...actual, createMatch, applyAction, startNextRound }
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

beforeEach(() => {
  createMatch.mockReset()
  applyAction.mockReset()
  startNextRound.mockReset()
  // The health check in App's useEffect hits the real `fetch`, not matchClient
  // — stub it so tests don't make a real network call or race on its result.
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }))
})

afterEach(() => {
  vi.unstubAllGlobals()
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
})
