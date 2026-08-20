import { describe, expect, it } from 'vitest'
import { normalizeSnapshots, sortedSnapshotKeys, TreeParseError } from './treeParse'
import type { DecisionNode } from './types'

const DECISION_NODE: DecisionNode = {
  kind: 'decision',
  current_player: 0,
  phase: 'awaiting_draw',
  is_terminal: false,
  visit_count: 0,
  value: null,
  rank_probs: null,
  edges: [],
}

describe('normalizeSnapshots', () => {
  it('accepts a dict of simulation-count to tree (happy path)', () => {
    const data = { '0': DECISION_NODE, '10': DECISION_NODE }
    expect(normalizeSnapshots(data)).toEqual(data)
  })

  it('wraps a single bare tree (top-level "kind") under key "0" (happy path)', () => {
    expect(normalizeSnapshots(DECISION_NODE)).toEqual({ '0': DECISION_NODE })
  })

  it('drops entries that are not tree-shaped while keeping the valid ones (sad path)', () => {
    const data = { '0': DECISION_NODE, note: 'not a tree', empty: {} }
    expect(normalizeSnapshots(data)).toEqual({ '0': DECISION_NODE })
  })

  it('throws a TreeParseError for valid JSON with no tree-shaped values at all (bad path)', () => {
    expect(() => normalizeSnapshots({ foo: 'bar' })).toThrow(TreeParseError)
  })

  it('throws for primitives, arrays, and null rather than crashing (bad path)', () => {
    expect(() => normalizeSnapshots(null)).toThrow(TreeParseError)
    expect(() => normalizeSnapshots(42)).toThrow(TreeParseError)
    expect(() => normalizeSnapshots([DECISION_NODE])).toThrow(TreeParseError)
  })
})

describe('sortedSnapshotKeys', () => {
  it('sorts numeric keys numerically, not lexically (happy path)', () => {
    expect(sortedSnapshotKeys({ '10': DECISION_NODE, '2': DECISION_NODE, '0': DECISION_NODE })).toEqual(['0', '2', '10'])
  })

  it('falls back to lexical order for non-numeric keys (bad path)', () => {
    expect(sortedSnapshotKeys({ b: DECISION_NODE, a: DECISION_NODE })).toEqual(['a', 'b'])
  })
})
