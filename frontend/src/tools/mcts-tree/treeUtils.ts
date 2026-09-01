import type { ChanceEdge, DecisionAction, DecisionEdge, SnapshotMap, TreeNode } from './types'

export type EdgeKind = 'decision' | 'chance'

const BOARD_COLUMNS = 4

/** 1-indexed "rRcC" for a board position (0-indexed, row-major over
 * BOARD_COLUMNS-wide rows) - the one position-naming scheme used everywhere
 * a position needs a human label (see `game/HistoryPanel.tsx`'s own copy of
 * this, duplicated rather than imported for the same reason `actionLabel`
 * below is duplicated into `tools/game-replay/replayUtils.ts`). */
function rowColLabel(position: number): string {
  return `r${Math.floor(position / BOARD_COLUMNS) + 1}c${(position % BOARD_COLUMNS) + 1}`
}

const ACTION_LABELS: Record<string, (position: number | null) => string> = {
  DRAW_STOCK: () => 'Draw stock',
  DRAW_DISCARD: () => 'Draw discard',
  PLACE: (p) => `Place → ${rowColLabel(p!)}`,
  FLIP_INITIAL: (p) => `Flip ${rowColLabel(p!)}`,
  DISCARD_AND_REVEAL: (p) => `Discard & reveal ${rowColLabel(p!)}`,
}

export function actionLabel(action: DecisionAction): string {
  const fn = ACTION_LABELS[action.type]
  if (fn) return fn(action.position)
  return action.position != null ? `${action.type} @ ${action.position}` : action.type
}

/** Identifies an edge by what it represents (an action, or a revealed card
 * value) rather than by object identity, so the same edge can be found again
 * in a *different* snapshot's tree — the basis for both "compare against"
 * deltas and the move-preference-over-time chart. */
export function edgeKey(edge: DecisionEdge | ChanceEdge, kind: EdgeKind): string {
  return kind === 'chance' ? `c:${(edge as ChanceEdge).card_value}` : `d:${(edge as DecisionEdge).action.type}:${(edge as DecisionEdge).action.position}`
}

export function pathId(path: string[]): string {
  return path.join('|')
}

export function topEdgeByVisits<E extends { visit_count: number }>(edges: E[]): E | null {
  if (edges.length === 0) return null
  return edges.reduce((best, e) => (e.visit_count > best.visit_count ? e : best))
}

/** Walks a node/edge-key path (as produced by `edgeKey`) from `root`,
 * returning the node reached, or `null` if the path doesn't exist in this
 * particular tree (e.g. a branch that was never visited at a lower
 * simulation count). Used to find "the same node" across snapshots. */
export function getNodeAtPath(root: TreeNode, path: string[]): TreeNode | null {
  let node: TreeNode | null = root
  for (const key of path) {
    if (!node) return null
    if (node.kind === 'decision') {
      const edge: DecisionEdge | undefined = node.edges.find((e) => edgeKey(e, 'decision') === key)
      node = edge ? edge.child : null
    } else {
      const edge: ChanceEdge | undefined = node.edges.find((e) => edgeKey(e, 'chance') === key)
      node = edge ? edge.child : null
    }
  }
  return node
}

/** The "principal variation": repeatedly follow the most-visited edge (through
 * chance nodes too), stopping at an unvisited or unexpanded edge. Mirrors
 * `bots.mcts_bot.MctsBot.choose_action`'s "most-visited wins" rule, one level
 * at a time, so it's the answer search actually settled on, not just the
 * deepest exported branch. */
export function computeBestLine(root: TreeNode, cap = 8): string[] {
  const path: string[] = []
  let node: TreeNode | null = root
  for (let i = 0; i < cap; i++) {
    if (!node || node.kind !== 'decision' || node.edges.length === 0) break
    const top: DecisionEdge | null = topEdgeByVisits(node.edges)
    if (!top || top.visit_count === 0 || !top.child) break
    path.push(edgeKey(top, 'decision'))
    node = top.child
    if (node && node.kind === 'chance') {
      const topChance: ChanceEdge | null = topEdgeByVisits(node.edges)
      if (!topChance || topChance.visit_count === 0 || !topChance.child) break
      path.push(edgeKey(topChance, 'chance'))
      node = topChance.child
    }
  }
  return path
}

/** Every path prefix of `bestLine`, as ids — the set of node-paths that
 * should start expanded so the principal variation is visible without
 * clicking anything. */
export function bestLinePrefixIds(bestLine: string[]): string[] {
  const ids: string[] = []
  for (let i = 1; i <= bestLine.length; i++) {
    ids.push(pathId(bestLine.slice(0, i)))
  }
  return ids
}

/** Every reachable path-id in `root`'s subtree, for "expand all" — computed
 * by walking the already-in-memory data tree, not the DOM, so it works
 * without first rendering the branches it's about to expand. */
export function collectAllPathIds(root: TreeNode, prefix: string[] = []): string[] {
  const ids: string[] = []
  if (root.kind === 'decision') {
    for (const edge of root.edges) {
      const next = [...prefix, edgeKey(edge, 'decision')]
      ids.push(pathId(next))
      if (edge.child) ids.push(...collectAllPathIds(edge.child, next))
    }
  } else {
    for (const edge of root.edges) {
      const next = [...prefix, edgeKey(edge, 'chance')]
      ids.push(pathId(next))
      if (edge.child) ids.push(...collectAllPathIds(edge.child, next))
    }
  }
  return ids
}

export interface TrendPoint {
  snapshotKey: string
  n: number
  visitCount: number
  share: number
}

export interface TrendSeries {
  key: string
  label: string
  points: TrendPoint[]
}

/** For the node at `path` (root by default), tracks each of its actions'
 * visit share across every snapshot in `order` — the data behind the
 * "move preference over time" chart. An action missing from an early
 * snapshot (not yet explored) simply has no point there rather than a
 * zero, so the line only draws once the action exists. */
export function buildTrendSeries(snapshots: SnapshotMap, order: string[], path: string[]): TrendSeries[] {
  const perAction = new Map<string, TrendSeries>()
  for (const key of order) {
    const root = snapshots[key]
    if (!root) continue
    const node = getNodeAtPath(root, path)
    if (!node || node.kind !== 'decision' || node.edges.length === 0) continue
    const total = node.edges.reduce((sum, e) => sum + e.visit_count, 0) || 1
    for (const edge of node.edges) {
      const key2 = edgeKey(edge, 'decision')
      let series = perAction.get(key2)
      if (!series) {
        series = { key: key2, label: actionLabel(edge.action), points: [] }
        perAction.set(key2, series)
      }
      series.points.push({ snapshotKey: key, n: Number(key), visitCount: edge.visit_count, share: edge.visit_count / total })
    }
  }
  return [...perAction.values()]
}

export function fmtPct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`
}

export function fmtSigned(x: number, digits = 3): string {
  const s = x.toFixed(digits)
  return x >= 0 ? `+${s}` : s
}

export function fmtNum(x: number, digits = 3): string {
  return x.toFixed(digits)
}
