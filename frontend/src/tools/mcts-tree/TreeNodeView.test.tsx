import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import NodeBlock from './TreeNodeView'
import { ViewSettingsContext } from './ViewSettingsContext'
import type { ChanceNode, DecisionEdge, DecisionNode } from './types'

function decisionEdge(overrides: Partial<DecisionEdge> = {}): DecisionEdge {
  return {
    action: { type: 'DRAW_STOCK', position: null },
    prior: 0.5,
    prior_before_noise: 0.5,
    visit_count: 0,
    mean_value: [0, 0],
    q: 0,
    u: 0,
    puct_score: 0,
    child: null,
    ...overrides,
  }
}

const GRANDCHILD: DecisionNode = {
  kind: 'decision',
  current_player: 1,
  phase: 'awaiting_placement',
  is_terminal: false,
  visit_count: 4,
  value: [0.1, -0.1],
  rank_probs: null,
  points_pred: null,
  edges: [],
}

const CHANCE_CHILD: ChanceNode = {
  kind: 'chance',
  edges: [{ card_value: 4, prior: 1, visit_count: 4, mean_value: [0.1, -0.1], child: GRANDCHILD }],
}

const ROOT: DecisionNode = {
  kind: 'decision',
  current_player: 0,
  phase: 'awaiting_draw',
  is_terminal: false,
  visit_count: 12,
  value: [0.2, -0.2],
  rank_probs: null,
  points_pred: null,
  edges: [
    decisionEdge({ action: { type: 'DRAW_STOCK', position: null }, visit_count: 10, child: CHANCE_CHILD }),
    decisionEdge({ action: { type: 'DRAW_DISCARD', position: null }, visit_count: 0, child: null }),
  ],
}

/** Mirrors exactly how `McTreeExplorerPage` drives `NodeBlock`: expand state
 * lives outside the tree, toggled by path id. This is the harness the
 * regression test below exercises — a per-row click must flip the *same*
 * element the expanded-state check reads, which a previous vanilla-JS
 * version of this tool got wrong (toggled a class with no matching CSS
 * rule, so individual rows silently did nothing while "expand all" still
 * worked, because it touched a different element). */
function Harness({
  filterText = '',
  hideZero = false,
  showWinRate = false,
}: {
  filterText?: string
  hideZero?: boolean
  showWinRate?: boolean
}) {
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set())
  return (
    <ViewSettingsContext.Provider value={{ filterText, hideZero, showWinRate }}>
      <NodeBlock
        node={ROOT}
        compareNode={null}
        path={[]}
        expandedPaths={expandedPaths}
        onToggle={(path) =>
          setExpandedPaths((prev) => {
            const id = path.join('|')
            const next = new Set(prev)
            if (next.has(id)) next.delete(id)
            else next.add(id)
            return next
          })
        }
        onFocusTrend={vi.fn()}
      />
    </ViewSettingsContext.Provider>
  )
}

describe('NodeBlock / edge row expand-collapse', () => {
  it('reveals the chance child on the first click and hides it again on the second (happy path — the reported bug)', () => {
    render(<Harness />)
    const row = screen.getByRole('button', { name: /Draw stock/ })
    expect(row).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('✲ chance — unresolved reveal')).not.toBeInTheDocument()

    fireEvent.click(row)
    expect(row).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText('✲ chance — unresolved reveal')).toBeInTheDocument()

    fireEvent.click(row)
    expect(row).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('✲ chance — unresolved reveal')).not.toBeInTheDocument()
  })

  it('expanding one edge does not affect its sibling (happy path)', () => {
    render(<Harness />)
    fireEvent.click(screen.getByRole('button', { name: /Draw stock/ }))

    const discardRow = screen.getByRole('button', { name: /Draw discard/ })
    expect(discardRow).toBeDisabled()
    expect(discardRow).not.toHaveAttribute('aria-expanded')
  })

  it('recurses through a chance node into its decision grandchild when expanded (happy path)', () => {
    render(<Harness />)
    fireEvent.click(screen.getByRole('button', { name: /Draw stock/ }))
    fireEvent.click(screen.getByRole('button', { name: /Card 4/ }))

    expect(screen.getByText('awaiting_placement')).toBeInTheDocument()
  })

  it('gives an unexpandable edge (no child) a disabled row instead of a dead click target (sad path)', () => {
    render(<Harness />)
    const discardRow = screen.getByRole('button', { name: /Draw discard/ })
    expect(discardRow).toBeDisabled()
    fireEvent.click(discardRow)
    expect(screen.queryByText('✲ chance — unresolved reveal')).not.toBeInTheDocument()
  })

  it('dims rows that do not match the action-type filter without removing them (bad path)', () => {
    render(<Harness filterText="DISCARD" />)
    const stockRow = screen.getByRole('button', { name: /Draw stock/ })
    expect(stockRow.closest('.mtx-edge-row')).toHaveClass('mtx-dimmed')
    expect(screen.getByRole('button', { name: /Draw discard/ }).closest('.mtx-edge-row')).not.toHaveClass('mtx-dimmed')
  })

  it('hides zero-visit edges entirely when hideZero is set, without touching visited ones (bad path)', () => {
    render(<Harness hideZero />)
    expect(screen.queryByRole('button', { name: /Draw discard/ })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Draw stock/ })).toBeInTheDocument()
  })
})
