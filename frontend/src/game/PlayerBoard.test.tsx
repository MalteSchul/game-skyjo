import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { BoardOut } from '../api/types'
import PlayerBoard from './PlayerBoard'

function boardWith(cards: BoardOut['cards']): BoardOut {
  return { cards }
}

describe('PlayerBoard', () => {
  it('invokes onCardClick with the position only for clickable cells (happy path)', () => {
    const onCardClick = vi.fn()
    const board = boardWith([
      { value: null, face_up: false },
      { value: 5, face_up: true },
    ])
    render(
      <PlayerBoard
        board={board}
        name="Player 1"
        isCurrentPlayer
        isFinalTurn={false}
        clickablePositions={new Set([0])}
        onCardClick={onCardClick}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'face-down card' }))
    expect(onCardClick).toHaveBeenCalledWith(0)
  })

  it('renders non-clickable cells as disabled buttons (sad path: click has no effect)', () => {
    const onCardClick = vi.fn()
    const board = boardWith([{ value: 5, face_up: true }])
    render(
      <PlayerBoard
        board={board}
        name="Player 2"
        isCurrentPlayer={false}
        isFinalTurn={false}
        clickablePositions={new Set()}
        onCardClick={onCardClick}
      />,
    )

    const button = screen.getByRole('button', { name: 'card 5' })
    fireEvent.click(button)
    expect(button).toBeDisabled()
    expect(onCardClick).not.toHaveBeenCalled()
  })

  it('renders cleared column slots without a button, even at a "clickable" position (bad path)', () => {
    render(
      <PlayerBoard
        board={boardWith([null])}
        name="Player 1"
        isCurrentPlayer
        isFinalTurn={false}
        clickablePositions={new Set([0])}
        onCardClick={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('labels the board with the given name and current-turn state', () => {
    render(
      <PlayerBoard
        board={boardWith([])}
        name="Grace"
        isCurrentPlayer
        isFinalTurn={false}
        clickablePositions={new Set()}
        onCardClick={vi.fn()}
      />,
    )

    expect(screen.getByRole('heading', { name: /Grace \(current turn\)/ })).toBeInTheDocument()
  })

  it('labels the board as the final turn instead of the current turn when the round is closing out', () => {
    render(
      <PlayerBoard
        board={boardWith([])}
        name="Grace"
        isCurrentPlayer
        isFinalTurn
        clickablePositions={new Set()}
        onCardClick={vi.fn()}
      />,
    )

    expect(screen.getByRole('heading', { name: /Grace \(final turn\)/ })).toBeInTheDocument()
    expect(screen.queryByText('Current turn')).not.toBeInTheDocument()
  })
})
