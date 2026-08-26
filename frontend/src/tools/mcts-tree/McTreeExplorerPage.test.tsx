import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import McTreeExplorerPage from './McTreeExplorerPage'
import type { DecisionNode } from './types'

function decisionRoot(overrides: Partial<DecisionNode> = {}): DecisionNode {
  return {
    kind: 'decision',
    current_player: 0,
    phase: 'awaiting_draw',
    is_terminal: false,
    visit_count: 30,
    value: [0.3, -0.3],
    rank_probs: null,
    points_pred: null,
    edges: [
      {
        action: { type: 'DRAW_STOCK', position: null },
        prior: 0.6,
        prior_before_noise: 0.6,
        visit_count: 20,
        mean_value: [0.3, -0.3],
        q: 0.3,
        u: 0.1,
        puct_score: 0.4,
        child: null,
      },
      {
        action: { type: 'DRAW_DISCARD', position: null },
        prior: 0.4,
        prior_before_noise: 0.4,
        visit_count: 10,
        mean_value: [0.1, -0.1],
        q: 0.1,
        u: 0.2,
        puct_score: 0.3,
        child: null,
      },
    ],
    ...overrides,
  }
}

const SNAPSHOTS = {
  '5': decisionRoot({ visit_count: 5, edges: [] }),
  '30': decisionRoot(),
}

// A second fixture where '5' has the *same* two edges as '30', just earlier
// in the search — unlike SNAPSHOTS above (whose '5' is empty, for exercising
// "nothing recorded yet"), this is what a compare-deltas test needs: a
// matching edge to diff against.
const COMPARE_SNAPSHOTS = {
  '5': decisionRoot({
    visit_count: 5,
    edges: [
      { action: { type: 'DRAW_STOCK', position: null }, prior: 0.6, prior_before_noise: 0.6, visit_count: 0, mean_value: [0, 0], q: 0, u: 0.6, puct_score: 0.6, child: null },
      { action: { type: 'DRAW_DISCARD', position: null }, prior: 0.4, prior_before_noise: 0.4, visit_count: 5, mean_value: [0.05, -0.05], q: 0.05, u: 0.1, puct_score: 0.15, child: null },
    ],
  }),
  '30': decisionRoot(),
}

function uploadFile(input: HTMLElement, contents: string, name = 'tree.json') {
  const file = new File([contents], name, { type: 'application/json' })
  fireEvent.change(input, { target: { files: [file] } })
}

afterEach(() => {
  document.body.classList.remove('mcts-tools-page')
})

describe('McTreeExplorerPage', () => {
  it('loads a snapshot file and shows the root stat tiles for the latest snapshot (happy path)', async () => {
    render(<McTreeExplorerPage />)
    uploadFile(screen.getByLabelText('Load tree export file'), JSON.stringify(SNAPSHOTS))

    expect(await screen.findByText('N=30 · 30 visits')).toBeInTheDocument()
    expect(screen.getAllByText('Draw stock').length).toBeGreaterThan(0)
    expect(screen.getByText('Simulations run')).toBeInTheDocument()
  })

  it('shows the JSON parse error inline instead of crashing on malformed input (sad path)', async () => {
    render(<McTreeExplorerPage />)
    uploadFile(screen.getByLabelText('Load tree export file'), '{not valid json')

    expect(await screen.findByRole('alert')).toHaveTextContent("isn't valid JSON")
  })

  it('shows a schema error for valid JSON that contains no tree data (bad path)', async () => {
    render(<McTreeExplorerPage />)
    uploadFile(screen.getByLabelText('Load tree export file'), JSON.stringify({ foo: 'bar' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Couldn\'t find any MCTS tree snapshots')
  })

  it('switches the displayed snapshot when the scrubber moves, resetting its own tree expansion (happy path)', async () => {
    render(<McTreeExplorerPage />)
    uploadFile(screen.getByLabelText('Load tree export file'), JSON.stringify(SNAPSHOTS))
    await screen.findByText('Simulations run')

    const treeShell = () => document.querySelector('.mtx-tree-shell') as HTMLElement
    expect(within(treeShell()).getByText('Draw stock')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Simulation snapshot'), { target: { value: '0' } })

    expect(await screen.findByText('N=5 · 5 visits')).toBeInTheDocument()
    // N=5's root has no edges at all — the tree pane should reflect that,
    // even though the (snapshot-independent) trend chart above it still
    // shows "Draw stock" from the other snapshot's history.
    expect(within(treeShell()).queryByText('Draw stock')).not.toBeInTheDocument()
    expect(within(treeShell()).getByText(/No edges recorded/)).toBeInTheDocument()
  })

  it('annotates visit-count deltas once a comparison snapshot is selected (happy path)', async () => {
    render(<McTreeExplorerPage />)
    uploadFile(screen.getByLabelText('Load tree export file'), JSON.stringify(COMPARE_SNAPSHOTS))
    await screen.findByText('Simulations run')

    fireEvent.change(screen.getByLabelText('Show deltas vs.'), { target: { value: '5' } })

    const delta = await screen.findByText((_, el) => el?.className === 'mtx-delta mtx-delta-up' && el.textContent === '▲20')
    expect(delta).toBeInTheDocument()
  })

  it('opts the shared #root out of the game\'s fixed-width layout while mounted, and restores it on unmount', () => {
    const { unmount } = render(<McTreeExplorerPage />)
    expect(document.body).toHaveClass('mcts-tools-page')
    unmount()
    expect(document.body).not.toHaveClass('mcts-tools-page')
  })
})
