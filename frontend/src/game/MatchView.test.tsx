import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { MatchStateOut } from '../api/types'
import MatchView from './MatchView'

function blankBoard() {
  return { cards: Array.from({ length: 12 }, () => ({ value: null, face_up: false })) }
}

const BASE_MATCH: MatchStateOut = {
  match_id: 'm1',
  phase: 'initial_flip',
  boards: [blankBoard(), blankBoard()],
  player_names: ['Ada', 'Grace'],
  stock_count: 140,
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

function noop() {}

describe('MatchView', () => {
  it('renders the new-match form and forwards submissions when there is no match yet (happy path)', () => {
    const onCreate = vi.fn()
    render(
      <MatchView
        match={null}
        error={null}
        busy={false}
        discardMode={false}
        onCreate={onCreate}
        onDrawStock={noop}
        onDrawDiscard={noop}
        onSetDiscardMode={noop}
        onCardClick={noop}
        onNextRound={noop}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(2, undefined, ['', ''])
  })

  it('calls onCardClick with the player index and position for a clickable board card (happy path)', () => {
    const onCardClick = vi.fn()
    render(
      <MatchView
        match={BASE_MATCH}
        error={null}
        busy={false}
        discardMode={false}
        onCreate={noop}
        onDrawStock={noop}
        onDrawDiscard={noop}
        onSetDiscardMode={noop}
        onCardClick={onCardClick}
        onNextRound={noop}
      />,
    )

    fireEvent.click(screen.getAllByRole('button', { name: 'face-down card' })[0])

    expect(onCardClick).toHaveBeenCalledWith(0, 0)
  })

  it('calls onSetDiscardMode when the drawn-card mode toggle is used during awaiting_placement (happy path)', () => {
    const onSetDiscardMode = vi.fn()
    const match: MatchStateOut = {
      ...BASE_MATCH,
      phase: 'awaiting_placement',
      drawn_card: 6,
      legal_actions: [
        { type: 'place', position: 0 },
        { type: 'discard_and_reveal', position: 1 },
      ],
    }
    render(
      <MatchView
        match={match}
        error={null}
        busy={false}
        discardMode={false}
        onCreate={noop}
        onDrawStock={noop}
        onDrawDiscard={noop}
        onSetDiscardMode={onSetDiscardMode}
        onCardClick={noop}
        onNextRound={noop}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Discard & reveal' }))

    expect(onSetDiscardMode).toHaveBeenCalledWith(true)
  })

  it("marks the acting player's board as their final turn once a player finishes (happy path)", () => {
    const match: MatchStateOut = {
      ...BASE_MATCH,
      phase: 'awaiting_draw',
      current_player: 1,
      finisher: 0,
      players_awaiting_final_turn: [1],
      legal_actions: [
        { type: 'draw_stock', position: null },
        { type: 'draw_discard', position: null },
      ],
    }
    render(
      <MatchView
        match={match}
        error={null}
        busy={false}
        discardMode={false}
        onCreate={noop}
        onDrawStock={noop}
        onDrawDiscard={noop}
        onSetDiscardMode={noop}
        onCardClick={noop}
        onNextRound={noop}
      />,
    )

    expect(screen.getByRole('heading', { name: /Grace \(final turn\)/ })).toBeInTheDocument()
    expect(screen.getByText('Final turn')).toBeInTheDocument()
  })

  it('labels the board "current turn" instead of "final turn" during ordinary play (sad path)', () => {
    render(
      <MatchView
        match={BASE_MATCH}
        error={null}
        busy={false}
        discardMode={false}
        onCreate={noop}
        onDrawStock={noop}
        onDrawDiscard={noop}
        onSetDiscardMode={noop}
        onCardClick={noop}
        onNextRound={noop}
      />,
    )

    expect(screen.queryByText('Final turn')).not.toBeInTheDocument()
    expect(screen.getByText('Current turn')).toBeInTheDocument()
  })

  it('shows the error message when the error prop is set (sad path)', () => {
    render(
      <MatchView
        match={BASE_MATCH}
        error="that action is not legal right now"
        busy={false}
        discardMode={false}
        onCreate={noop}
        onDrawStock={noop}
        onDrawDiscard={noop}
        onSetDiscardMode={noop}
        onCardClick={noop}
        onNextRound={noop}
      />,
    )

    expect(screen.getByText('that action is not legal right now')).toBeInTheDocument()
  })

  it('disables "Start next round" while busy and calls onNextRound when clicked (bad path: prevents double-submit)', () => {
    const onNextRound = vi.fn()
    const match: MatchStateOut = { ...BASE_MATCH, phase: 'round_over', round_scores: [4, 9] }
    const { rerender } = render(
      <MatchView
        match={match}
        error={null}
        busy
        discardMode={false}
        onCreate={noop}
        onDrawStock={noop}
        onDrawDiscard={noop}
        onSetDiscardMode={noop}
        onCardClick={noop}
        onNextRound={onNextRound}
      />,
    )

    expect(screen.getByRole('button', { name: 'Start next round' })).toBeDisabled()

    rerender(
      <MatchView
        match={match}
        error={null}
        busy={false}
        discardMode={false}
        onCreate={noop}
        onDrawStock={noop}
        onDrawDiscard={noop}
        onSetDiscardMode={noop}
        onCardClick={noop}
        onNextRound={onNextRound}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Start next round' }))

    expect(onNextRound).toHaveBeenCalledTimes(1)
  })
})
