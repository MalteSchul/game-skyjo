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
  player_types: ['human', 'human'],
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
  status: 'idle',
  thinking_player: null,
  thinking_progress: null,
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
        onPlayAgain={noop}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Start match' }))

    expect(onCreate).toHaveBeenCalledWith(2, undefined, ['', ''], ['human', 'human'])
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
        onPlayAgain={noop}
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
        onPlayAgain={noop}
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
        onPlayAgain={noop}
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
        onPlayAgain={noop}
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
        onPlayAgain={noop}
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
        onPlayAgain={noop}
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
        onPlayAgain={noop}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Start next round' }))

    expect(onNextRound).toHaveBeenCalledTimes(1)
  })

  it('calls onPlayAgain when "Play again" is clicked after game_over (happy path)', () => {
    const onPlayAgain = vi.fn()
    const match: MatchStateOut = { ...BASE_MATCH, phase: 'game_over', total_scores: [42, 17] }
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
        onPlayAgain={onPlayAgain}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Play again' }))

    expect(onPlayAgain).toHaveBeenCalledTimes(1)
  })

  it('the toolbar Restart button asks for confirmation mid-game before calling onPlayAgain (happy path)', () => {
    const onPlayAgain = vi.fn()
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
        onPlayAgain={onPlayAgain}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: /Restart/ }))
    expect(onPlayAgain).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Restart' }))
    expect(onPlayAgain).toHaveBeenCalledTimes(1)
  })

  describe('keyboard shortcuts', () => {
    const DRAW_MATCH: MatchStateOut = {
      ...BASE_MATCH,
      phase: 'awaiting_draw' as MatchStateOut['phase'],
      legal_actions: [
        { type: 'draw_stock', position: null },
        { type: 'draw_discard', position: null },
      ],
    }

    it('draws from the stock and discard pile via "q" and "w" (happy path)', () => {
      const onDrawStock = vi.fn()
      const onDrawDiscard = vi.fn()
      render(
        <MatchView
          match={DRAW_MATCH}
          error={null}
          busy={false}
          discardMode={false}
          onCreate={noop}
          onDrawStock={onDrawStock}
          onDrawDiscard={onDrawDiscard}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={noop}
          onPlayAgain={noop}
        />,
      )

      fireEvent.keyDown(window, { key: 'q' })
      fireEvent.keyDown(window, { key: 'w' })

      expect(onDrawStock).toHaveBeenCalledTimes(1)
      expect(onDrawDiscard).toHaveBeenCalledTimes(1)
    })

    it('selects place/discard-reveal mode with "q"/"w" (happy path)', () => {
      const onSetDiscardMode = vi.fn()
      const placeMatch: MatchStateOut = {
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
          match={placeMatch}
          error={null}
          busy={false}
          discardMode={false}
          onCreate={noop}
          onDrawStock={noop}
          onDrawDiscard={noop}
          onSetDiscardMode={onSetDiscardMode}
          onCardClick={noop}
          onNextRound={noop}
          onPlayAgain={noop}
        />,
      )

      fireEvent.keyDown(window, { key: 'w' })
      fireEvent.keyDown(window, { key: 'q' })

      expect(onSetDiscardMode).toHaveBeenNthCalledWith(1, true)
      expect(onSetDiscardMode).toHaveBeenNthCalledWith(2, false)
    })

    it('starts the next round with "n" and plays again with "p" (happy path)', () => {
      const onNextRound = vi.fn()
      const roundOverMatch: MatchStateOut = { ...BASE_MATCH, phase: 'round_over', round_scores: [4, 9] }
      const { unmount } = render(
        <MatchView
          match={roundOverMatch}
          error={null}
          busy={false}
          discardMode={false}
          onCreate={noop}
          onDrawStock={noop}
          onDrawDiscard={noop}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={onNextRound}
          onPlayAgain={noop}
        />,
      )
      fireEvent.keyDown(window, { key: 'n' })
      expect(onNextRound).toHaveBeenCalledTimes(1)
      unmount()

      const onPlayAgain = vi.fn()
      const gameOverMatch: MatchStateOut = { ...BASE_MATCH, phase: 'game_over', total_scores: [42, 17] }
      render(
        <MatchView
          match={gameOverMatch}
          error={null}
          busy={false}
          discardMode={false}
          onCreate={noop}
          onDrawStock={noop}
          onDrawDiscard={noop}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={noop}
          onPlayAgain={onPlayAgain}
        />,
      )
      fireEvent.keyDown(window, { key: 'p' })
      expect(onPlayAgain).toHaveBeenCalledTimes(1)
    })

    it('confirms before restarting mid-game: "p" opens a confirm dialog, a second "p" restarts (happy path)', () => {
      const onPlayAgain = vi.fn()
      render(
        <MatchView
          match={DRAW_MATCH}
          error={null}
          busy={false}
          discardMode={false}
          onCreate={noop}
          onDrawStock={noop}
          onDrawDiscard={noop}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={noop}
          onPlayAgain={onPlayAgain}
        />,
      )

      fireEvent.keyDown(window, { key: 'p' })
      expect(onPlayAgain).not.toHaveBeenCalled()
      expect(screen.getByRole('dialog', { name: 'Restart the game?' })).toBeInTheDocument()

      fireEvent.keyDown(window, { key: 'p' })
      expect(onPlayAgain).toHaveBeenCalledTimes(1)
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    })

    it('cancels the mid-game restart confirmation with Escape, without restarting (sad path)', () => {
      const onPlayAgain = vi.fn()
      render(
        <MatchView
          match={DRAW_MATCH}
          error={null}
          busy={false}
          discardMode={false}
          onCreate={noop}
          onDrawStock={noop}
          onDrawDiscard={noop}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={noop}
          onPlayAgain={onPlayAgain}
        />,
      )

      fireEvent.keyDown(window, { key: 'p' })
      expect(screen.getByRole('dialog', { name: 'Restart the game?' })).toBeInTheDocument()

      fireEvent.keyDown(window, { key: 'Escape' })
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
      expect(onPlayAgain).not.toHaveBeenCalled()
    })

    it('ignores the restart shortcut while busy, to avoid opening the confirm dialog mid-action (bad path)', () => {
      const onPlayAgain = vi.fn()
      render(
        <MatchView
          match={DRAW_MATCH}
          error={null}
          busy
          discardMode={false}
          onCreate={noop}
          onDrawStock={noop}
          onDrawDiscard={noop}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={noop}
          onPlayAgain={onPlayAgain}
        />,
      )

      fireEvent.keyDown(window, { key: 'p' })
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
      expect(onPlayAgain).not.toHaveBeenCalled()
    })

    it('opens the shortcuts help on "?" and closes it on Escape (happy path)', () => {
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
          onPlayAgain={noop}
        />,
      )

      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
      fireEvent.keyDown(window, { key: '?' })
      expect(screen.getByRole('dialog', { name: 'Keyboard shortcuts' })).toBeInTheDocument()
      fireEvent.keyDown(window, { key: 'Escape' })
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    })

    it('ignores draw/round shortcuts that are not legal right now (sad path)', () => {
      const onDrawStock = vi.fn()
      const onNextRound = vi.fn()
      render(
        <MatchView
          match={BASE_MATCH}
          error={null}
          busy={false}
          discardMode={false}
          onCreate={noop}
          onDrawStock={onDrawStock}
          onDrawDiscard={noop}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={onNextRound}
          onPlayAgain={noop}
        />,
      )

      // BASE_MATCH is in initial_flip: neither drawing nor next-round is legal.
      fireEvent.keyDown(window, { key: 'q' })
      fireEvent.keyDown(window, { key: 'n' })

      expect(onDrawStock).not.toHaveBeenCalled()
      expect(onNextRound).not.toHaveBeenCalled()
    })

    it('ignores shortcuts entirely while busy, to avoid double-submitting an action (bad path)', () => {
      const onDrawStock = vi.fn()
      render(
        <MatchView
          match={DRAW_MATCH}
          error={null}
          busy
          discardMode={false}
          onCreate={noop}
          onDrawStock={onDrawStock}
          onDrawDiscard={noop}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={noop}
          onPlayAgain={noop}
        />,
      )

      fireEvent.keyDown(window, { key: 'q' })

      expect(onDrawStock).not.toHaveBeenCalled()
    })

    it('auto-focuses the first available card as soon as the board renders, with no prior Tab/click (happy path)', () => {
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
          onPlayAgain={noop}
        />,
      )

      const firstCard = screen.getAllByRole('button', { name: 'face-down card' })[0]
      expect(firstCard).toHaveFocus()
      expect(firstCard).toHaveAttribute('tabindex', '0')
    })

    it('moves focus with the arrow keys even when nothing is focused, skipping unavailable cards (happy path)', () => {
      const sparseMatch: MatchStateOut = {
        ...BASE_MATCH,
        legal_actions: [0, 2, 6].map((position) => ({ type: 'flip_initial' as const, position })),
      }
      render(
        <MatchView
          match={sparseMatch}
          error={null}
          busy={false}
          discardMode={false}
          onCreate={noop}
          onDrawStock={noop}
          onDrawDiscard={noop}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={noop}
          onPlayAgain={noop}
        />,
      )

      // Simulate a fresh page load where nothing has focus yet.
      ;(document.activeElement as HTMLElement | null)?.blur()
      expect(document.body).toHaveFocus()

      const cards = screen.getAllByRole('button', { name: 'face-down card' })
      fireEvent.keyDown(window, { key: 'ArrowRight' })
      expect(cards[2]).toHaveFocus()

      fireEvent.keyDown(window, { key: 'ArrowDown' })
      expect(cards[6]).toHaveFocus()
    })

    it("moves the focused player's board along with the turn once the match state advances (happy path)", () => {
      const afterFirstFlip: MatchStateOut = {
        ...BASE_MATCH,
        current_player: 1,
        boards: [
          { cards: [{ value: 5, face_up: true }, ...blankBoard().cards.slice(1)] },
          blankBoard(),
        ],
        legal_actions: Array.from({ length: 12 }, (_, i) => ({ type: 'flip_initial' as const, position: i })),
      }
      const { rerender } = render(
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
          onPlayAgain={noop}
        />,
      )

      rerender(
        <MatchView
          match={afterFirstFlip}
          error={null}
          busy={false}
          discardMode={false}
          onCreate={noop}
          onDrawStock={noop}
          onDrawDiscard={noop}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={noop}
          onPlayAgain={noop}
        />,
      )

      expect((document.activeElement as HTMLElement).closest('.player-board')).toHaveTextContent('Grace')
    })
  })

  describe('bot thinking status', () => {
    it('shows a progress percentage while a bot is thinking (happy path)', () => {
      const match: MatchStateOut = { ...BASE_MATCH, status: 'thinking', thinking_player: 0, thinking_progress: 0.42 }
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
          onPlayAgain={noop}
        />,
      )

      expect(screen.getByRole('status')).toHaveTextContent('Ada is thinking (42%)')
    })

    it('falls back to an indeterminate indicator when no progress has been reported yet (happy path)', () => {
      const match: MatchStateOut = { ...BASE_MATCH, status: 'thinking', thinking_player: 1, thinking_progress: null }
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
          onPlayAgain={noop}
        />,
      )

      expect(screen.getByRole('status')).toHaveTextContent('Grace is thinking…')
    })

    it('shows no thinking indicator while idle (sad path)', () => {
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
          onPlayAgain={noop}
        />,
      )

      expect(screen.queryByRole('status')).not.toBeInTheDocument()
    })

    it('ignores card clicks while a bot is thinking, even on an otherwise-legal position (bad path)', () => {
      const onCardClick = vi.fn()
      const match: MatchStateOut = { ...BASE_MATCH, status: 'thinking', thinking_player: 1, thinking_progress: 0.1 }
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
          onCardClick={onCardClick}
          onNextRound={noop}
          onPlayAgain={noop}
        />,
      )

      const cards = screen.getAllByRole('button', { name: 'face-down card' })
      expect(cards.every((card) => card.hasAttribute('disabled'))).toBe(true)

      fireEvent.click(cards[0])
      expect(onCardClick).not.toHaveBeenCalled()
    })

    it('ignores the draw shortcut while a bot is thinking (bad path)', () => {
      const onDrawStock = vi.fn()
      const match: MatchStateOut = {
        ...BASE_MATCH,
        phase: 'awaiting_draw',
        status: 'thinking',
        thinking_player: 0,
        thinking_progress: 0.1,
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
          onDrawStock={onDrawStock}
          onDrawDiscard={noop}
          onSetDiscardMode={noop}
          onCardClick={noop}
          onNextRound={noop}
          onPlayAgain={noop}
        />,
      )

      fireEvent.keyDown(window, { key: 'q' })

      expect(onDrawStock).not.toHaveBeenCalled()
    })
  })
})
