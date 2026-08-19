import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import Scoreboard from './Scoreboard'

describe('Scoreboard', () => {
  it('renders each player name next to their score (happy path)', () => {
    render(
      <Scoreboard
        playerNames={['Ada', 'Grace']}
        playerTypes={['human', 'human']}
        scores={[4, 9]}
        currentPlayer={0}
        status="idle"
        thinkingPlayer={null}
      />,
    )

    expect(screen.getByText('Ada')).toBeInTheDocument()
    expect(screen.getByText('4')).toBeInTheDocument()
    expect(screen.getByText('Grace')).toBeInTheDocument()
    expect(screen.getByText('9')).toBeInTheDocument()
  })

  it('marks only the current player as active (sad path: others stay unmarked)', () => {
    render(
      <Scoreboard
        playerNames={['Ada', 'Grace']}
        playerTypes={['human', 'human']}
        scores={[4, 9]}
        currentPlayer={1}
        status="idle"
        thinkingPlayer={null}
      />,
    )

    expect(screen.getByText('Ada').closest('li')).not.toHaveClass('score-chip-active')
    expect(screen.getByText('Grace').closest('li')).toHaveClass('score-chip-active')
  })

  it('tags a random_bot seat but leaves human seats untagged (sad path)', () => {
    render(
      <Scoreboard
        playerNames={['Ada', 'Grace']}
        playerTypes={['random_bot', 'human']}
        scores={[4, 9]}
        currentPlayer={0}
        status="idle"
        thinkingPlayer={null}
      />,
    )

    expect(screen.getByText('(Bot)')).toBeInTheDocument()
    expect(screen.getByText('Grace').closest('li')).not.toHaveTextContent('(Bot)')
  })

  it('shows the thinking variant only for the bot seat currently deciding (happy path)', () => {
    render(
      <Scoreboard
        playerNames={['Ada', 'Grace']}
        playerTypes={['random_bot', 'random_bot']}
        scores={[4, 9]}
        currentPlayer={0}
        status="thinking"
        thinkingPlayer={0}
      />,
    )

    expect(screen.getByText('(Bot · thinking)')).toBeInTheDocument()
    expect(screen.getByText('Grace').closest('li')).toHaveTextContent('(Bot)')
    expect(screen.getByText('Grace').closest('li')).not.toHaveTextContent('(Bot · thinking)')
  })

  it('renders nothing but the empty list for a match with no players (bad path)', () => {
    const { container } = render(
      <Scoreboard playerNames={[]} playerTypes={[]} scores={[]} currentPlayer={0} status="idle" thinkingPlayer={null} />,
    )

    expect(container.querySelector('ul.scoreboard')).toBeEmptyDOMElement()
  })
})
