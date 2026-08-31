import { describe, expect, it } from 'vitest'
import type { ChanceNode, DecisionEdge, DecisionNode } from './types'
import {
  actionLabel,
  bestLinePrefixIds,
  buildTrendSeries,
  collectAllPathIds,
  computeBestLine,
  edgeKey,
  fmtPct,
  fmtSigned,
  getNodeAtPath,
  pathId,
  topEdgeByVisits,
} from './treeUtils'

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

// root (P0) --DRAW_STOCK(10 visits)--> chance --card 4(10)--> leaf (P1, terminal)
//      \--DRAW_DISCARD(2 visits)--> leaf2 (P1, not terminal, no edges)
const TERMINAL_LEAF: DecisionNode = {
  kind: 'decision',
  current_player: 1,
  phase: 'game_over',
  is_terminal: true,
  visit_count: 10,
  value: [1, -1],
  rank_probs: null,
  points_pred: null,
  edges: [],
}

const CHANCE: ChanceNode = {
  kind: 'chance',
  edges: [{ card_value: 4, prior: 1, visit_count: 10, mean_value: [1, -1], child: TERMINAL_LEAF }],
}

const LEAF2: DecisionNode = {
  kind: 'decision',
  current_player: 1,
  phase: 'awaiting_draw',
  is_terminal: false,
  visit_count: 0,
  value: null,
  rank_probs: null,
  points_pred: null,
  edges: [],
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
    decisionEdge({ action: { type: 'DRAW_STOCK', position: null }, visit_count: 10, child: CHANCE }),
    decisionEdge({ action: { type: 'DRAW_DISCARD', position: null }, visit_count: 2, child: LEAF2 }),
  ],
}

describe('actionLabel', () => {
  it('renders known action types in plain language (happy path)', () => {
    expect(actionLabel({ type: 'DRAW_STOCK', position: null })).toBe('Draw stock')
    expect(actionLabel({ type: 'PLACE', position: 5 })).toBe('Place → r2c2')
  })

  it('falls back to "TYPE @ position" for an unrecognized action type (sad path)', () => {
    expect(actionLabel({ type: 'SOME_NEW_ACTION', position: 3 })).toBe('SOME_NEW_ACTION @ 3')
  })

  it('falls back to the bare type when position is null and the type is unknown (bad path)', () => {
    expect(actionLabel({ type: 'SOME_NEW_ACTION', position: null })).toBe('SOME_NEW_ACTION')
  })
})

describe('edgeKey / pathId', () => {
  it('produces stable, distinct keys for decision and chance edges (happy path)', () => {
    expect(edgeKey(ROOT.edges[0], 'decision')).toBe('d:DRAW_STOCK:null')
    expect(edgeKey(CHANCE.edges[0], 'chance')).toBe('c:4')
  })

  it('joins a path into one id without colliding on separators used inside keys (bad path)', () => {
    expect(pathId(['d:DRAW_STOCK:null', 'c:4'])).toBe('d:DRAW_STOCK:null|c:4')
  })
})

describe('getNodeAtPath', () => {
  it('walks decision and chance edges to the node at a path (happy path)', () => {
    expect(getNodeAtPath(ROOT, ['d:DRAW_STOCK:null', 'c:4'])).toBe(TERMINAL_LEAF)
  })

  it('returns the root itself for an empty path (happy path)', () => {
    expect(getNodeAtPath(ROOT, [])).toBe(ROOT)
  })

  it('returns null for a path that does not exist in this tree (sad path)', () => {
    expect(getNodeAtPath(ROOT, ['d:PLACE:0'])).toBeNull()
  })

  it('returns null once the path runs past a leaf with no edges (bad path)', () => {
    expect(getNodeAtPath(ROOT, ['d:DRAW_DISCARD:null', 'd:DRAW_STOCK:null'])).toBeNull()
  })
})

describe('topEdgeByVisits', () => {
  it('picks the edge with the highest visit_count (happy path)', () => {
    expect(topEdgeByVisits(ROOT.edges)).toBe(ROOT.edges[0])
  })

  it('returns null for an empty edge list (bad path)', () => {
    expect(topEdgeByVisits([])).toBeNull()
  })
})

describe('computeBestLine', () => {
  it('follows the most-visited edge through a chance node to the leaf (happy path)', () => {
    expect(computeBestLine(ROOT)).toEqual(['d:DRAW_STOCK:null', 'c:4'])
  })

  it('stops at an edge with a child but zero visits rather than descending into it (sad path)', () => {
    const unvisitedChild: DecisionNode = { ...LEAF2, current_player: 0 }
    const node: DecisionNode = {
      ...ROOT,
      edges: [decisionEdge({ visit_count: 0, child: unvisitedChild })],
    }
    expect(computeBestLine(node)).toEqual([])
  })

  it('stops cleanly at a terminal node with no edges instead of throwing (bad path)', () => {
    expect(computeBestLine(TERMINAL_LEAF)).toEqual([])
  })
})

describe('bestLinePrefixIds / collectAllPathIds', () => {
  it('returns every growing prefix of the best line (happy path)', () => {
    expect(bestLinePrefixIds(['a', 'b', 'c'])).toEqual(['a', 'a|b', 'a|b|c'])
  })

  it('returns an empty list for an empty best line (bad path)', () => {
    expect(bestLinePrefixIds([])).toEqual([])
  })

  it('collects every reachable path in the tree, including through chance nodes (happy path)', () => {
    const ids = collectAllPathIds(ROOT)
    expect(ids).toEqual(
      expect.arrayContaining(['d:DRAW_STOCK:null', 'd:DRAW_STOCK:null|c:4', 'd:DRAW_DISCARD:null']),
    )
    expect(ids).toHaveLength(3)
  })
})

describe('buildTrendSeries', () => {
  const early: DecisionNode = { ...ROOT, edges: [decisionEdge({ action: { type: 'DRAW_STOCK', position: null }, visit_count: 1 })] }
  const snapshots = { '1': early, '12': ROOT }
  const order = ['1', '12']

  it('tracks each action\'s visit share across snapshots at the given path (happy path)', () => {
    const series = buildTrendSeries(snapshots, order, [])
    const stock = series.find((s) => s.key === 'd:DRAW_STOCK:null')
    expect(stock?.points).toEqual([
      { snapshotKey: '1', n: 1, visitCount: 1, share: 1 },
      { snapshotKey: '12', n: 12, visitCount: 10, share: 10 / 12 },
    ])
  })

  it('omits a point for a snapshot where the action had not appeared yet (sad path)', () => {
    const series = buildTrendSeries(snapshots, order, [])
    const discard = series.find((s) => s.key === 'd:DRAW_DISCARD:null')
    expect(discard?.points).toHaveLength(1)
    expect(discard?.points[0].snapshotKey).toBe('12')
  })

  it('returns an empty list when the path points at a node missing from every snapshot (bad path)', () => {
    expect(buildTrendSeries(snapshots, order, ['d:PLACE:0'])).toEqual([])
  })
})

describe('formatting helpers', () => {
  it('formats percentages and signed numbers (happy path)', () => {
    expect(fmtPct(0.625, 1)).toBe('62.5%')
    expect(fmtSigned(0.5)).toBe('+0.500')
    expect(fmtSigned(-0.5)).toBe('-0.500')
  })

  it('signs exactly zero as positive rather than "-0.000" (bad path)', () => {
    expect(fmtSigned(0)).toBe('+0.000')
  })
})
