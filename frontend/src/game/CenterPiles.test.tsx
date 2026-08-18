import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import CenterPiles from './CenterPiles'

function baseProps() {
  return {
    stockCount: 140,
    discardTop: 6 as number | null,
    drawnCard: null as number | null,
    canDrawStock: true,
    canDrawDiscard: true,
    onDrawStock: vi.fn(),
    onDrawDiscard: vi.fn(),
    showModeToggle: false,
    discardMode: false,
    onSetDiscardMode: vi.fn(),
  }
}

describe('CenterPiles', () => {
  it('labels the stock as a count and the discard as a value, distinctly (happy path)', () => {
    render(<CenterPiles {...baseProps()} stockCount={125} discardTop={7} />)

    const stockButton = screen.getByRole('button', { name: 'Stock, 125 cards left' })
    const discardButton = screen.getByRole('button', { name: 'Discard, top card value 7' })
    expect(stockButton).toHaveTextContent('125')
    expect(discardButton).toHaveTextContent('7')
    // The count pill and the value pill are visually distinct classes — one
    // is a plain quantity, the other is coloured like the card it names,
    // via the tone class carried on the button itself.
    expect(stockButton.querySelector('.pile-count')).toBeInTheDocument()
    expect(discardButton.querySelector('.pile-value')).toBeInTheDocument()
    expect(discardButton).toHaveClass('tone-mid')
  })

  it('shows an empty placeholder instead of a value when the discard pile has no top card (sad path)', () => {
    render(<CenterPiles {...baseProps()} discardTop={null} canDrawDiscard={false} />)

    expect(screen.getByRole('button', { name: 'Discard, empty' })).toHaveTextContent('—')
  })

  it('does not call onDrawStock when the stock pile is disabled (bad path)', () => {
    const onDrawStock = vi.fn()
    render(<CenterPiles {...baseProps()} stockCount={0} discardTop={3} canDrawStock={false} onDrawStock={onDrawStock} />)

    fireEvent.click(screen.getByRole('button', { name: 'Stock, 0 cards left' }))

    expect(onDrawStock).not.toHaveBeenCalled()
  })

  it('calls onSetDiscardMode when the mode toggle is shown (happy path)', () => {
    const onSetDiscardMode = vi.fn()
    render(<CenterPiles {...baseProps()} showModeToggle drawnCard={6} onSetDiscardMode={onSetDiscardMode} />)

    fireEvent.click(screen.getByRole('button', { name: 'Discard & reveal' }))

    expect(onSetDiscardMode).toHaveBeenCalledWith(true)
  })

  it('hides the mode toggle from the accessibility tree and interaction when there is nothing to toggle (sad path)', () => {
    render(<CenterPiles {...baseProps()} showModeToggle={false} />)

    // The toggle stays mounted (it reserves its layout space so the piles
    // above it never shift), but aria-hidden + disabled keep it out of the
    // accessibility tree and tab order until there's a drawn card to act on.
    expect(screen.queryByRole('button', { name: 'Discard & reveal' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Place on board' })).not.toBeInTheDocument()
  })

  it('keeps the piles visible (disabled, not unmounted) when neither pile can be drawn from (bad path)', () => {
    render(<CenterPiles {...baseProps()} canDrawStock={false} canDrawDiscard={false} />)

    expect(screen.getByRole('button', { name: 'Stock, 140 cards left' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Discard, top card value 6' })).toBeDisabled()
  })

  it('reveals the drawn card value once a card has been drawn (happy path)', () => {
    render(<CenterPiles {...baseProps()} drawnCard={9} />)

    expect(screen.getByText('9')).toBeInTheDocument()
    expect(screen.getByText('9').closest('.drawn-card-slot')).not.toHaveClass('drawn-card-slot-hidden')
  })

  it('hides the drawn-card slot from the accessibility tree before any card is drawn (sad path)', () => {
    render(<CenterPiles {...baseProps()} drawnCard={null} />)

    // Stays mounted (it reserves the piles row's height so drawing a card
    // never resizes the row), just hidden — same pattern as the toggle above.
    const slot = document.querySelector('.drawn-card-slot')
    expect(slot).toHaveClass('drawn-card-slot-hidden')
    expect(slot).toHaveAttribute('aria-hidden', 'true')
  })
})
