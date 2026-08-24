import { useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import type { HistoryNodeOut, MatchHistoryOut } from '../api/types'

const LANE_WIDTH = 18
const WINDOW_RADIUS = 5

// Reuses the game's existing card-tone palette so branch colors feel like
// part of the same design system instead of an arbitrary new one.
const LANE_COLORS = [
  'var(--accent)',
  'var(--tone-mid-bg)',
  'var(--tone-low-bg)',
  'var(--tone-high-bg)',
  'var(--tone-zero-bg)',
  'var(--tone-neg-bg)',
]

function laneColor(lane: number): string {
  return LANE_COLORS[lane % LANE_COLORS.length]
}

function describeNode(node: HistoryNodeOut, playerNames: string[]): string {
  const { edge } = node
  if (edge.kind === 'root') return 'Game start'
  if (edge.kind === 'next_round') return `Round ${node.round_index + 1} begins`

  const actorName = (node.actor !== null && playerNames[node.actor]) || `Player ${(node.actor ?? 0) + 1}`
  const slot = edge.position !== null ? edge.position + 1 : null

  switch (edge.action_type) {
    case 'flip_initial':
      return `${actorName} flipped card ${slot}`
    case 'draw_stock':
      return `${actorName} drew from the stock`
    case 'draw_discard':
      return `${actorName} drew from the discard`
    case 'place':
      return `${actorName} placed on card ${slot}`
    case 'discard_and_reveal':
      return `${actorName} discarded & flipped card ${slot}`
    default:
      return `${actorName} played`
  }
}

interface RowGraphInfo {
  node: HistoryNodeOut
  lane: number
  verticalTop: boolean // line continues up into the row above
  verticalBottom: boolean // line continues down into the row below
  passThroughLanes: number[] // other lanes with no node here, just passing through
  elbowLanes: number[] // other lanes that terminate here, merging into `lane`
}

interface GraphLayout {
  rows: RowGraphInfo[] // newest (highest seq) first
  laneCount: number
}

/** Lays the tree out the way `git log --graph` does: sweep newest-to-oldest,
 * keep a pool of lane columns, and free a lane the moment its branch merges
 * back into its parent so a later, unrelated branch can reuse that same
 * column instead of the graph growing wider forever. The line to the current
 * head is preferred as the "primary" continuation at every merge point, so
 * it reads as the straight trunk. */
function buildGraphLayout(nodes: HistoryNodeOut[], headId: string): GraphLayout {
  const byId = new Map(nodes.map((n) => [n.node_id, n]))
  const headAncestors = new Set<string>()
  for (let cur = byId.get(headId); cur; cur = cur.parent_id ? byId.get(cur.parent_id) : undefined) {
    headAncestors.add(cur.node_id)
  }

  const sortedRows = [...nodes].sort((a, b) => b.seq - a.seq)

  const pendingByParent = new Map<string, Array<{ lane: number; childId: string }>>()
  const activeLanes = new Set<number>()
  const freeLanes: number[] = []
  let nextLane = 0

  function allocateLane(): number {
    if (freeLanes.length > 0) {
      freeLanes.sort((a, b) => a - b)
      return freeLanes.shift()!
    }
    return nextLane++
  }

  const rows: RowGraphInfo[] = []
  for (const node of sortedRows) {
    const waiting = pendingByParent.get(node.node_id) ?? []
    pendingByParent.delete(node.node_id)

    let lane: number
    let verticalTop: boolean
    const elbowLanes: number[] = []

    if (waiting.length === 0) {
      lane = allocateLane()
      activeLanes.add(lane)
      verticalTop = false
    } else {
      let primary = waiting.find((w) => headAncestors.has(w.childId))
      if (!primary) {
        primary = waiting.reduce((best, w) => (byId.get(w.childId)!.seq > byId.get(best.childId)!.seq ? w : best))
      }
      lane = primary.lane
      verticalTop = true
      for (const w of waiting) {
        if (w.lane === lane) continue
        elbowLanes.push(w.lane)
        activeLanes.delete(w.lane)
        freeLanes.push(w.lane)
      }
    }

    const passThroughLanes = Array.from(activeLanes).filter((l) => l !== lane)

    const verticalBottom = node.parent_id !== null
    if (verticalBottom) {
      const siblings = pendingByParent.get(node.parent_id!) ?? []
      siblings.push({ lane, childId: node.node_id })
      pendingByParent.set(node.parent_id!, siblings)
    } else {
      activeLanes.delete(lane)
    }

    rows.push({ node, lane, verticalTop, verticalBottom, passThroughLanes, elbowLanes })
  }

  return { rows, laneCount: nextLane }
}

interface GraphRowsProps {
  rows: RowGraphInfo[]
  laneCount: number
  headId: string
  matchId: string
  playerNames: string[]
  onGoto: (nodeId: string) => void
  busy: boolean
}

function GraphRows({ rows, laneCount, headId, matchId, playerNames, onGoto, busy }: GraphRowsProps) {
  const gutterWidth = laneCount * LANE_WIDTH
  return (
    <ol className="history-graph">
      {rows.map((info) => {
        const isHead = info.node.node_id === headId
        return (
          <li key={info.node.node_id} className="history-row">
            <div className="history-gutter" style={{ width: gutterWidth }} aria-hidden="true">
              <div
                className="lane-line"
                style={{
                  left: info.lane * LANE_WIDTH + LANE_WIDTH / 2 - 1,
                  top: info.verticalTop ? 0 : '50%',
                  bottom: info.verticalBottom ? 0 : '50%',
                  background: laneColor(info.lane),
                }}
              />
              {info.passThroughLanes.map((l) => (
                <div
                  key={l}
                  className="lane-line"
                  style={{ left: l * LANE_WIDTH + LANE_WIDTH / 2 - 1, top: 0, bottom: 0, background: laneColor(l) }}
                />
              ))}
              {info.elbowLanes.map((l) => (
                <div key={l}>
                  <div
                    className="lane-line"
                    style={{ left: l * LANE_WIDTH + LANE_WIDTH / 2 - 1, top: 0, bottom: '50%', background: laneColor(l) }}
                  />
                  <div
                    className="lane-elbow"
                    style={{
                      left: Math.min(l, info.lane) * LANE_WIDTH + LANE_WIDTH / 2 - 1,
                      width: Math.abs(l - info.lane) * LANE_WIDTH,
                      background: laneColor(l),
                    }}
                  />
                </div>
              ))}
              <div
                className={`lane-dot ${isHead ? 'lane-dot-active' : ''}`}
                style={{ left: info.lane * LANE_WIDTH + LANE_WIDTH / 2, background: laneColor(info.lane) }}
              />
            </div>
            <button
              type="button"
              className={`history-node ${isHead ? 'history-node-active' : ''}`}
              onClick={() => onGoto(info.node.node_id)}
              disabled={busy || isHead}
            >
              {describeNode(info.node, playerNames)}
            </button>
            {info.node.has_mcts_tree && (
              <a
                className="history-tree-link"
                href={`/tools/mcts-tree?matchId=${matchId}&nodeId=${info.node.node_id}`}
                target="_blank"
                rel="noreferrer"
                title="View the MCTS search tree behind this move"
                aria-label="View the MCTS search tree behind this move"
              >
                🌳
              </a>
            )}
          </li>
        )
      })}
    </ol>
  )
}

interface HistoryPanelProps {
  history: MatchHistoryOut | null
  matchId: string
  playerNames: string[]
  onGoto: (nodeId: string) => void
  busy: boolean
}

function HistoryPanel({ history, matchId, playerNames, onGoto, busy }: HistoryPanelProps) {
  const [showFull, setShowFull] = useState(false)
  const layout = useMemo(
    () => (history ? buildGraphLayout(history.nodes, history.head_id) : null),
    [history],
  )

  useEffect(() => {
    if (!showFull) return
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setShowFull(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [showFull])

  if (!history || !layout) return null

  const headRow = layout.rows.findIndex((r) => r.node.node_id === history.head_id)
  const windowStart = Math.max(0, headRow - WINDOW_RADIUS)
  const windowEnd = Math.min(layout.rows.length - 1, headRow + WINDOW_RADIUS)
  const windowedRows = layout.rows.slice(windowStart, windowEnd + 1)
  const hiddenAbove = windowStart
  const hiddenBelow = layout.rows.length - 1 - windowEnd
  const hasMore = hiddenAbove > 0 || hiddenBelow > 0

  function goto(nodeId: string) {
    onGoto(nodeId)
    setShowFull(false)
  }

  return (
    <aside className="history-panel" aria-label="Match history">
      <h2>History</h2>
      {hiddenAbove > 0 && <p className="history-window-hint">⋯ {hiddenAbove} more recent</p>}
      <GraphRows
        rows={windowedRows}
        laneCount={layout.laneCount}
        headId={history.head_id}
        matchId={matchId}
        playerNames={playerNames}
        onGoto={goto}
        busy={busy}
      />
      {hiddenBelow > 0 && <p className="history-window-hint">⋯ {hiddenBelow} earlier</p>}
      {hasMore && (
        <button type="button" className="history-view-full" onClick={() => setShowFull(true)}>
          View full history
        </button>
      )}
      {showFull &&
        createPortal(
          <div className="history-modal-backdrop" onClick={() => setShowFull(false)}>
            <div
              className="history-modal"
              role="dialog"
              aria-modal="true"
              aria-label="Full match history"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="history-modal-header">
                <h2>Full history</h2>
                <button type="button" className="history-modal-close" onClick={() => setShowFull(false)} aria-label="Close">
                  ×
                </button>
              </div>
              <div className="history-modal-body">
                <GraphRows
                  rows={layout.rows}
                  laneCount={layout.laneCount}
                  headId={history.head_id}
                  matchId={matchId}
                  playerNames={playerNames}
                  onGoto={goto}
                  busy={busy}
                />
              </div>
            </div>
          </div>,
          document.body,
        )}
    </aside>
  )
}

export default HistoryPanel
