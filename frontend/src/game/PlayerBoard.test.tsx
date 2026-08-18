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
        rovingPosition={0}
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
        rovingPosition={null}
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
        rovingPosition={0}
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
        rovingPosition={null}
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
        rovingPosition={null}
        onCardClick={vi.fn()}
      />,
    )

    expect(screen.getByRole('heading', { name: /Grace \(final turn\)/ })).toBeInTheDocument()
    expect(screen.queryByText('Current turn')).not.toBeInTheDocument()
  })

  it('gives only the card matching rovingPosition a tabindex of 0, and the other clickable cards -1 (happy path)', () => {
    const board = boardWith([
      { value: null, face_up: false },
      { value: null, face_up: false },
      { value: null, face_up: false },
    ])
    render(
      <PlayerBoard
        board={board}
        name="Ada"
        isCurrentPlayer
        isFinalTurn={false}
        clickablePositions={new Set([0, 1, 2])}
        rovingPosition={1}
        onCardClick={vi.fn()}
      />,
    )
    const buttons = screen.getAllByRole('button')

    expect(buttons[0]).toHaveAttribute('tabindex', '-1')
    expect(buttons[1]).toHaveAttribute('tabindex', '0')
    expect(buttons[2]).toHaveAttribute('tabindex', '-1')
  })

  it('reports which position gained focus via onCardFocus, only for clickable cards (happy path)', () => {
    const onCardFocus = vi.fn()
    const board = boardWith([{ value: null, face_up: false }])
    render(
      <PlayerBoard
        board={board}
        name="Ada"
        isCurrentPlayer
        isFinalTurn={false}
        clickablePositions={new Set([0])}
        rovingPosition={0}
        onCardClick={vi.fn()}
        onCardFocus={onCardFocus}
      />,
    )

    screen.getByRole('button').focus()

    expect(onCardFocus).toHaveBeenCalledWith(0)
  })

  it('registers and clears button refs via onCardRef as cards mount and unmount (sad path)', () => {
    const onCardRef = vi.fn()
    const { rerender } = render(
      <PlayerBoard
        board={boardWith([{ value: null, face_up: false }])}
        name="Ada"
        isCurrentPlayer
        isFinalTurn={false}
        clickablePositions={new Set([0])}
        rovingPosition={0}
        onCardClick={vi.fn()}
        onCardRef={onCardRef}
      />,
    )
    expect(onCardRef).toHaveBeenCalledWith(0, expect.any(HTMLButtonElement))

    rerender(
      <PlayerBoard
        board={boardWith([null])}
        name="Ada"
        isCurrentPlayer
        isFinalTurn={false}
        clickablePositions={new Set()}
        rovingPosition={null}
        onCardClick={vi.fn()}
        onCardRef={onCardRef}
      />,
    )
    expect(onCardRef).toHaveBeenCalledWith(0, null)
  })
})
