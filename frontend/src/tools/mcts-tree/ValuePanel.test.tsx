import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ValuePanel from './ValuePanel'
import type { DecisionNode } from './types'

const BASE: DecisionNode = {
  kind: 'decision',
  current_player: 0,
  phase: 'awaiting_draw',
  is_terminal: false,
  visit_count: 5,
  value: null,
  rank_probs: null,
  edges: [],
}

describe('ValuePanel', () => {
  it('shows the predicted value per player when the node has been expanded (happy path)', () => {
    render(<ValuePanel node={{ ...BASE, value: [0.4, -0.4] }} />)
    expect(screen.getByText('Network value · predicted utility per player')).toBeInTheDocument()
    expect(screen.getByText('+0.400')).toBeInTheDocument()
    expect(screen.getByText('-0.400')).toBeInTheDocument()
  })

  it('says the node has not been expanded yet instead of showing empty numbers (sad path)', () => {
    render(<ValuePanel node={{ ...BASE, value: null }} />)
    expect(screen.getByText('Not expanded in this snapshot yet')).toBeInTheDocument()
  })

  it('renders a rank_probs cell per player/rank pair when populated (happy path)', () => {
    render(<ValuePanel node={{ ...BASE, value: [0.4, -0.4], rank_probs: [[0.7, 0.3], [0.3, 0.7]] }} />)
    expect(screen.getByText('rank_probs — P(player i finishes at rank r)')).toBeInTheDocument()
    expect(screen.getAllByText('70%')).toHaveLength(2)
    expect(screen.getAllByText('30%')).toHaveLength(2)
  })

  it('omits the rank_probs table entirely when it was not captured (bad path: uniform stand-in evaluator)', () => {
    render(<ValuePanel node={{ ...BASE, value: [0, 0], rank_probs: null }} />)
    expect(screen.queryByText(/rank_probs/)).not.toBeInTheDocument()
  })
})
