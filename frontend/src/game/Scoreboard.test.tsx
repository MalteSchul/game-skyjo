import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Scoreboard from './Scoreboard'

describe('Scoreboard', () => {
  it('renders each player name next to their score (happy path)', () => {
    render(<Scoreboard playerNames={['Ada', 'Grace']} scores={[4, 9]} currentPlayer={0} />)

    expect(screen.getByText('Ada')).toBeInTheDocument()
    expect(screen.getByText('4')).toBeInTheDocument()
    expect(screen.getByText('Grace')).toBeInTheDocument()
    expect(screen.getByText('9')).toBeInTheDocument()
  })

  it('marks only the current player as active (sad path: others stay unmarked)', () => {
    render(<Scoreboard playerNames={['Ada', 'Grace']} scores={[4, 9]} currentPlayer={1} />)

    expect(screen.getByText('Ada').closest('li')).not.toHaveClass('score-chip-active')
    expect(screen.getByText('Grace').closest('li')).toHaveClass('score-chip-active')
  })

  it('renders nothing but the empty list for a match with no players (bad path)', () => {
    const { container } = render(<Scoreboard playerNames={[]} scores={[]} currentPlayer={0} />)

    expect(container.querySelector('ul.scoreboard')).toBeEmptyDOMElement()
  })
})
