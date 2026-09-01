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

  it('tags a thinking_bot seat the same as a random_bot seat (happy path)', () => {
    render(
      <Scoreboard
        playerNames={['Ada', 'Grace']}
        playerTypes={['thinking_bot', 'human']}
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

  it('shows each round\'s raw score per player, marking a doubled round only beside its finisher (happy path)', () => {
    render(
      <Scoreboard
        playerNames={['Ada', 'Grace']}
        playerTypes={['human', 'human']}
        scores={[12, 42]}
        currentPlayer={0}
        status="idle"
        thinkingPlayer={null}
        roundHistory={[{ scores: [12, 21], finisher: 1 }]}
      />,
    )

    const adaChip = screen.getByText('Ada').closest('li')!
    const graceChip = screen.getByText('Grace').closest('li')!
    // Grace finished this round without the sole lowest score (12 < 21), so
    // only her entry gets the doubled marker - Ada's own (lower, unrelated)
    // round score never does, even though it's the one that "won".
    expect(adaChip).toHaveTextContent('12')
    expect(adaChip.querySelector('.score-chip-round-doubled')).not.toBeInTheDocument()
    expect(graceChip).toHaveTextContent('21')
    expect(graceChip.querySelector('.score-chip-round-doubled')).toHaveTextContent('×2')
  })

  it('marks no round as doubled when the finisher had the sole lowest score (sad path)', () => {
    render(
      <Scoreboard
        playerNames={['Ada', 'Grace']}
        playerTypes={['human', 'human']}
        scores={[12, 42]}
        currentPlayer={0}
        status="idle"
        thinkingPlayer={null}
        roundHistory={[{ scores: [12, 21], finisher: 0 }]}
      />,
    )

    expect(document.querySelector('.score-chip-round-doubled')).not.toBeInTheDocument()
  })

  it('renders no round breakdown at all before any round has closed (bad path)', () => {
    render(
      <Scoreboard
        playerNames={['Ada', 'Grace']}
        playerTypes={['human', 'human']}
        scores={[0, 0]}
        currentPlayer={0}
        status="idle"
        thinkingPlayer={null}
        roundHistory={[]}
      />,
    )

    expect(document.querySelector('.score-chip-rounds')).not.toBeInTheDocument()
  })

  it('renders nothing but the empty list for a match with no players (bad path)', () => {
    const { container } = render(
      <Scoreboard playerNames={[]} playerTypes={[]} scores={[]} currentPlayer={0} status="idle" thinkingPlayer={null} />,
    )

    expect(container.querySelector('ul.scoreboard')).toBeEmptyDOMElement()
  })
})
