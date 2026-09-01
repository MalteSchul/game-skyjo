import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { HistoryNodeOut, MatchHistoryOut } from '../api/types'
import HistoryPanel from './HistoryPanel'

function node(overrides: Partial<HistoryNodeOut>): HistoryNodeOut {
  return {
    node_id: 'n0',
    parent_id: null,
    seq: 0,
    round_index: 0,
    actor: null,
    current_player: 0,
    phase: 'initial_flip',
    edge: { kind: 'root', action_type: null, position: null },
    has_mcts_tree: false,
    mcts_visit_share: null,
    mcts_prior_overridden: null,
    ...overrides,
  }
}

describe('HistoryPanel', () => {
  it('renders nothing when there is no history yet (bad path)', () => {
    const { container } = render(<HistoryPanel history={null} matchId="m0" playerNames={[]} onGoto={vi.fn()} busy={false} />)

    expect(container).toBeEmptyDOMElement()
  })

  it('renders each node as a labeled, clickable entry and calls onGoto with its id (happy path)', () => {
    const history: MatchHistoryOut = {
      head_id: 'n0',
      nodes: [
        node({ node_id: 'n0' }),
        node({
          node_id: 'n1',
          parent_id: 'n0',
          seq: 1,
          actor: 0,
          edge: { kind: 'action', action_type: 'flip_initial', position: 2 },
        }),
      ],
    }
    const onGoto = vi.fn()
    render(<HistoryPanel history={history} matchId="m0" playerNames={['Ada', 'Grace']} onGoto={onGoto} busy={false} />)

    const panel = screen.getByLabelText('Match history')
    expect(within(panel).getByText('Game start')).toBeInTheDocument()
    const move = within(panel).getByText('Ada flipped r1c3')
    fireEvent.click(move)

    expect(onGoto).toHaveBeenCalledWith('n1')
  })

  it('disables the entry for the current head instead of letting it be re-selected (sad path)', () => {
    const history: MatchHistoryOut = { head_id: 'n0', nodes: [node({ node_id: 'n0' })] }
    const onGoto = vi.fn()
    render(<HistoryPanel history={history} matchId="m0" playerNames={[]} onGoto={onGoto} busy={false} />)

    const button = screen.getByRole('button', { name: 'Game start' })
    fireEvent.click(button)

    expect(button).toBeDisabled()
    expect(onGoto).not.toHaveBeenCalled()
  })

  it('lists entries with the most recent move at the top (happy path: reverse-chronological)', () => {
    const history: MatchHistoryOut = {
      head_id: 'n2',
      nodes: [
        node({ node_id: 'n0' }),
        node({
          node_id: 'n1',
          parent_id: 'n0',
          seq: 1,
          actor: 0,
          edge: { kind: 'action', action_type: 'flip_initial', position: 0 },
        }),
        node({
          node_id: 'n2',
          parent_id: 'n1',
          seq: 2,
          actor: 1,
          edge: { kind: 'action', action_type: 'flip_initial', position: 1 },
        }),
      ],
    }
    render(<HistoryPanel history={history} matchId="m0" playerNames={['Ada', 'Grace']} onGoto={vi.fn()} busy={false} />)

    const labels = screen.getAllByRole('button').map((btn) => btn.textContent)
    expect(labels).toEqual(['Grace flipped r1c2', 'Ada flipped r1c1', 'Game start'])
  })

  it('puts a diverging branch in its own graph lane, not nested under the trunk (happy path)', () => {
    const history: MatchHistoryOut = {
      head_id: 'a',
      nodes: [
        node({ node_id: 'root' }),
        node({
          node_id: 'a',
          parent_id: 'root',
          seq: 1,
          actor: 0,
          edge: { kind: 'action', action_type: 'flip_initial', position: 0 },
        }),
        node({
          node_id: 'b',
          parent_id: 'root',
          seq: 2,
          actor: 0,
          edge: { kind: 'action', action_type: 'flip_initial', position: 1 },
        }),
      ],
    }
    const { container } = render(<HistoryPanel history={history} matchId="m0" playerNames={['Ada']} onGoto={vi.fn()} busy={false} />)

    expect(screen.getByText('Ada flipped r1c1')).toBeInTheDocument()
    expect(screen.getByText('Ada flipped r1c2')).toBeInTheDocument()
    // Two sibling branches off the same root must occupy two distinct graph
    // lanes (columns), which is what makes this a graph rather than a list.
    const dots = container.querySelectorAll('.lane-dot')
    const laneOffsets = new Set(Array.from(dots).map((dot) => (dot as HTMLElement).style.left))
    expect(laneOffsets.size).toBeGreaterThan(1)
  })

  it('reuses a freed lane for a later, unrelated branch instead of growing wider (happy path: compact graph)', () => {
    // root -> a1 -> a2 -> a3 -> a4 (head), with two short side branches off
    // a1 and a2 that each end immediately (leaves). b1's branch merges back
    // in (freeing its lane) before d1's branch needs one, so d1 should reuse
    // exactly the lane b1 vacated rather than allocating a brand new one.
    const history: MatchHistoryOut = {
      head_id: 'a4',
      nodes: [
        node({ node_id: 'root', seq: 0 }),
        node({ node_id: 'a1', parent_id: 'root', seq: 10, actor: 0, edge: { kind: 'action', action_type: 'flip_initial', position: 0 } }),
        node({ node_id: 'b1', parent_id: 'a1', seq: 12, actor: 0, edge: { kind: 'action', action_type: 'flip_initial', position: 1 } }),
        node({ node_id: 'd1', parent_id: 'a1', seq: 15, actor: 0, edge: { kind: 'action', action_type: 'flip_initial', position: 2 } }),
        node({ node_id: 'a2', parent_id: 'a1', seq: 20, actor: 0, edge: { kind: 'action', action_type: 'flip_initial', position: 3 } }),
        node({ node_id: 'c1', parent_id: 'a2', seq: 22, actor: 0, edge: { kind: 'action', action_type: 'draw_stock', position: null } }),
        node({ node_id: 'a3', parent_id: 'a2', seq: 30, actor: 0, edge: { kind: 'action', action_type: 'draw_discard', position: null } }),
        node({ node_id: 'a4', parent_id: 'a3', seq: 40, actor: 0, edge: { kind: 'action', action_type: 'place', position: 5 } }),
      ],
    }
    const { container } = render(<HistoryPanel history={history} matchId="m0" playerNames={['Ada']} onGoto={vi.fn()} busy={false} />)

    function laneLeftOf(text: string): string {
      const button = screen.getByText(text)
      const row = button.closest('li')!
      return (row.querySelector('.lane-dot') as HTMLElement).style.left
    }
    const dots = container.querySelectorAll('.lane-dot')
    const distinctLanes = new Set(Array.from(dots).map((dot) => (dot as HTMLElement).style.left))
    // Trunk + b1's lane + d1/c1 sharing one reused lane = 3 lanes total, not 4.
    expect(distinctLanes.size).toBe(3)
    expect(laneLeftOf('Ada drew from the stock')).toBe(laneLeftOf('Ada flipped r1c3'))
  })

  it('shows only a window of steps around the current head and lists the rest behind "View full history" (happy path)', () => {
    const nodes: HistoryNodeOut[] = [node({ node_id: 'n0', seq: 0 })]
    for (let i = 1; i <= 14; i++) {
      nodes.push(
        node({
          node_id: `n${i}`,
          parent_id: `n${i - 1}`,
          seq: i,
          actor: 0,
          edge: { kind: 'action', action_type: 'flip_initial', position: i % 4 },
        }),
      )
    }
    const history: MatchHistoryOut = { head_id: 'n14', nodes }
    render(<HistoryPanel history={history} matchId="m0" playerNames={['Ada']} onGoto={vi.fn()} busy={false} />)

    const panel = screen.getByLabelText('Match history')
    // Head (n14) ± 5 = rows for n9..n14 => 6 visible entries, 9 hidden below.
    expect(within(panel).getAllByRole('button', { name: /Ada/ })).toHaveLength(6)
    expect(within(panel).getByText('⋯ 9 earlier')).toBeInTheDocument()
    expect(within(panel).queryByText('Game start')).not.toBeInTheDocument()

    fireEvent.click(within(panel).getByRole('button', { name: 'View full history' }))

    const modal = screen.getByRole('dialog', { name: 'Full match history' })
    expect(within(modal).getByText('Game start')).toBeInTheDocument()
    expect(within(modal).getAllByRole('button', { name: /Ada/ })).toHaveLength(14)
  })

  it('portals the full-history popup out of the scrolling sidebar so it is not clipped by it (happy path)', () => {
    const nodes: HistoryNodeOut[] = [node({ node_id: 'n0', seq: 0 })]
    for (let i = 1; i <= 12; i++) {
      nodes.push(node({ node_id: `n${i}`, parent_id: `n${i - 1}`, seq: i, actor: 0, edge: { kind: 'action', action_type: 'draw_stock', position: null } }))
    }
    const history: MatchHistoryOut = { head_id: 'n12', nodes }
    const { container } = render(<HistoryPanel history={history} matchId="m0" playerNames={['Ada']} onGoto={vi.fn()} busy={false} />)

    fireEvent.click(screen.getByRole('button', { name: 'View full history' }))

    // The sidebar (`.history-panel`) has `overflow-y: auto`, which clips any
    // fixed-position descendant to its own small box regardless of the
    // descendant's viewport-relative positioning — so the modal must be a
    // sibling of the sidebar, not nested inside it, or it renders clipped
    // and appears to be behind the rest of the page instead of on top of it.
    const modal = screen.getByRole('dialog', { name: 'Full match history' })
    expect(container.contains(modal)).toBe(false)
    expect(document.body.contains(modal)).toBe(true)
  })

  it('links to the tree viewer only for a node with a search tree, pointed at that node and match (happy path)', () => {
    const history: MatchHistoryOut = {
      head_id: 'n1',
      nodes: [
        node({ node_id: 'n0' }),
        node({
          node_id: 'n1',
          parent_id: 'n0',
          seq: 1,
          actor: 0,
          edge: { kind: 'action', action_type: 'draw_stock', position: null },
          has_mcts_tree: true,
        }),
      ],
    }
    render(<HistoryPanel history={history} matchId="m0" playerNames={['Ada']} onGoto={vi.fn()} busy={false} />)

    const treeLinks = screen.getAllByRole('link', { name: /search tree/ })
    expect(treeLinks).toHaveLength(1)
    expect(treeLinks[0]).toHaveAttribute('href', '/tools/mcts-tree?matchId=m0&nodeId=n1')
  })

  it('shows the search\'s visit share as a percentage badge, colored by how confident it was (happy path)', () => {
    const history: MatchHistoryOut = {
      head_id: 'n1',
      nodes: [
        node({ node_id: 'n0' }),
        node({
          node_id: 'n1',
          parent_id: 'n0',
          seq: 1,
          actor: 0,
          edge: { kind: 'action', action_type: 'draw_stock', position: null },
          has_mcts_tree: true,
          mcts_visit_share: 0.82,
        }),
      ],
    }
    const { container } = render(<HistoryPanel history={history} matchId="m0" playerNames={['Ada']} onGoto={vi.fn()} busy={false} />)

    const badge = screen.getByText('82%')
    expect(badge).toBeInTheDocument()
    expect(badge).toHaveClass('history-visit-share-high')
    expect(container.querySelector('.history-prior-overridden')).not.toBeInTheDocument()
  })

  it('marks a node where search overrode the network\'s raw top prior (happy path)', () => {
    const history: MatchHistoryOut = {
      head_id: 'n1',
      nodes: [
        node({ node_id: 'n0' }),
        node({
          node_id: 'n1',
          parent_id: 'n0',
          seq: 1,
          actor: 0,
          edge: { kind: 'action', action_type: 'draw_stock', position: null },
          has_mcts_tree: true,
          mcts_visit_share: 0.35,
          mcts_prior_overridden: true,
        }),
      ],
    }
    render(<HistoryPanel history={history} matchId="m0" playerNames={['Ada']} onGoto={vi.fn()} busy={false} />)

    expect(screen.getByText('35%')).toHaveClass('history-visit-share-low')
    expect(screen.getByRole('img', { name: /overrode the network/ })).toBeInTheDocument()
  })

  it('omits the visit-share badge and override marker for a node with no recorded search (sad path)', () => {
    const history: MatchHistoryOut = { head_id: 'n0', nodes: [node({ node_id: 'n0' })] }
    const { container } = render(<HistoryPanel history={history} matchId="m0" playerNames={[]} onGoto={vi.fn()} busy={false} />)

    expect(container.querySelector('.history-visit-share')).not.toBeInTheDocument()
    expect(container.querySelector('.history-prior-overridden')).not.toBeInTheDocument()
  })

  it('closes the full-history popup without navigating when the backdrop is clicked (sad path)', () => {
    const nodes: HistoryNodeOut[] = [node({ node_id: 'n0', seq: 0 })]
    for (let i = 1; i <= 12; i++) {
      nodes.push(node({ node_id: `n${i}`, parent_id: `n${i - 1}`, seq: i, actor: 0, edge: { kind: 'action', action_type: 'draw_stock', position: null } }))
    }
    const history: MatchHistoryOut = { head_id: 'n12', nodes }
    const onGoto = vi.fn()
    render(<HistoryPanel history={history} matchId="m0" playerNames={['Ada']} onGoto={onGoto} busy={false} />)

    fireEvent.click(screen.getByRole('button', { name: 'View full history' }))
    expect(screen.getByRole('dialog', { name: 'Full match history' })).toBeInTheDocument()

    fireEvent.click(document.querySelector('.history-modal-backdrop')!)

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(onGoto).not.toHaveBeenCalled()
  })
})
