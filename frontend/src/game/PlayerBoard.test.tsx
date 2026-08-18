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

  function twelveFaceDownCards(): BoardOut['cards'] {
    return Array.from({ length: 12 }, () => ({ value: null, face_up: false }))
  }

  it('gives the first clickable card the roving tabindex, and moves it with the arrow keys, skipping unavailable cards (happy path)', () => {
    const { container } = render(
      <PlayerBoard
        board={boardWith(twelveFaceDownCards())}
        name="Ada"
        isCurrentPlayer
        isFinalTurn={false}
        clickablePositions={new Set([0, 2, 5, 6, 9])}
        onCardClick={vi.fn()}
      />,
    )
    const buttons = screen.getAllByRole('button')
    expect(buttons[0]).toHaveAttribute('tabindex', '0')
    expect(buttons[2]).toHaveAttribute('tabindex', '-1')

    const grid = container.querySelector('.board-grid')!

    // Row 0: [0 x x x] -> ArrowRight skips position 1 (not clickable) to land on 2.
    fireEvent.keyDown(grid, { key: 'ArrowRight' })
    expect(document.activeElement).toBe(buttons[2])
    expect(buttons[2]).toHaveAttribute('tabindex', '0')
    expect(buttons[0]).toHaveAttribute('tabindex', '-1')

    // Column 2, row 1 (position 6) is clickable directly below position 2.
    fireEvent.keyDown(grid, { key: 'ArrowDown' })
    expect(document.activeElement).toBe(buttons[6])

    // Position 5 is directly to the left of 6, and is clickable.
    fireEvent.keyDown(grid, { key: 'ArrowLeft' })
    expect(document.activeElement).toBe(buttons[5])
  })

  it('wraps focus around the grid edges back to a clickable card (happy path)', () => {
    const { container } = render(
      <PlayerBoard
        board={boardWith(twelveFaceDownCards())}
        name="Ada"
        isCurrentPlayer
        isFinalTurn={false}
        clickablePositions={new Set([0, 8])}
        onCardClick={vi.fn()}
      />,
    )
    const buttons = screen.getAllByRole('button')
    const grid = container.querySelector('.board-grid')!

    // Position 0 is the top-left cell (row 0); pressing Up wraps to row 2, same column.
    fireEvent.keyDown(grid, { key: 'ArrowUp' })
    expect(document.activeElement).toBe(buttons[8])
  })

  it('does nothing on arrow keys when the board has no clickable cards (sad path)', () => {
    const { container } = render(
      <PlayerBoard
        board={boardWith(twelveFaceDownCards())}
        name="Ada"
        isCurrentPlayer={false}
        isFinalTurn={false}
        clickablePositions={new Set()}
        onCardClick={vi.fn()}
      />,
    )
    const grid = container.querySelector('.board-grid')!

    expect(() => fireEvent.keyDown(grid, { key: 'ArrowRight' })).not.toThrow()
    for (const button of screen.getAllByRole('button')) {
      expect(button).not.toHaveAttribute('tabindex', '0')
    }
  })
})
