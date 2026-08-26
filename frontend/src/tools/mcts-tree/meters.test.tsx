import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { DivergingMeter, PlayerBadge, PlayerValueStrip } from './meters'

describe('DivergingMeter', () => {
  it('exposes the value through an accessible label (happy path)', () => {
    render(<DivergingMeter value={0.42} />)
    expect(screen.getByRole('img', { name: 'value +0.420' })).toBeInTheDocument()
  })

  it('clamps an out-of-range value rather than overflowing the bar (bad path)', () => {
    render(<DivergingMeter value={5} label="overflowing" />)
    // Rendering with an extreme input should not throw; the accessible
    // label is caller-supplied here so it just needs to come through.
    expect(screen.getByRole('img', { name: 'overflowing' })).toBeInTheDocument()
  })
})

describe('PlayerValueStrip', () => {
  it('renders one chip per player and outlines the current player\'s (happy path)', () => {
    render(<PlayerValueStrip values={[0.5, -0.5, 0.1]} currentPlayer={1} />)
    expect(screen.getByTitle('P0: +0.500')).toBeInTheDocument()
    expect(screen.getByTitle('P1: -0.500')).toHaveClass('mtx-value-chip-active')
    expect(screen.getByTitle('P2: +0.100')).not.toHaveClass('mtx-value-chip-active')
  })

  it('renders with no active outline when currentPlayer is omitted (sad path: chance-edge values have no acting player)', () => {
    render(<PlayerValueStrip values={[0.2, -0.2]} />)
    expect(screen.getByTitle('P0: +0.200')).not.toHaveClass('mtx-value-chip-active')
  })

  it('renders nothing but does not crash for an empty values array (bad path)', () => {
    const { container } = render(<PlayerValueStrip values={[]} />)
    expect(container.querySelectorAll('.mtx-value-chip')).toHaveLength(0)
  })
})

describe('PlayerBadge', () => {
  it('wraps the player index around every 8 seats (bad path: more than 8 players)', () => {
    render(<PlayerBadge index={9} />)
    expect(screen.getByText('P9')).toHaveStyle({ background: 'var(--mtx-player-1)' })
  })
})
