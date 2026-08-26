import { useState } from 'react'
import { cardTone } from '../../game/cardTone'
import { PlayerBadge, PlayerValueStrip } from './meters'
import type { ChanceEdge, ChanceNode, DecisionEdge, DecisionNode, TreeNode } from './types'
import { actionLabel, edgeKey, fmtNum, fmtPct, fmtSigned, pathId } from './treeUtils'
import { useViewSettings } from './ViewSettingsContext'
import ValuePanel from './ValuePanel'

export interface NodeBlockProps {
  node: TreeNode
  compareNode: TreeNode | null
  path: string[]
  expandedPaths: Set<string>
  onToggle: (path: string[]) => void
  onFocusTrend: (path: string[]) => void
}

/** Renders one decision or chance node — its header plus the list of edges
 * leading out of it. Recurses through `edgeKey`-matched `compareNode` (for
 * the "compare against" deltas) and a lifted `expandedPaths` set (for
 * expand/collapse) rather than any local/DOM-level toggle state, so there is
 * exactly one source of truth for "is this branch open" — see the git
 * history for the bug this replaced, where a per-row DOM class toggle
 * silently targeted the wrong element. */
export default function NodeBlock(props: NodeBlockProps) {
  if (props.node.kind === 'chance') {
    return (
      <ChanceNodeBlockView
        {...props}
        node={props.node}
        compareNode={props.compareNode && props.compareNode.kind === 'chance' ? props.compareNode : null}
      />
    )
  }
  return (
    <DecisionNodeBlockView
      {...props}
      node={props.node}
      compareNode={props.compareNode && props.compareNode.kind === 'decision' ? props.compareNode : null}
    />
  )
}

function findCompareDecisionEdge(compareNode: DecisionNode | null, key: string): DecisionEdge | null {
  return compareNode?.edges.find((e) => edgeKey(e, 'decision') === key) ?? null
}
function findCompareChanceEdge(compareNode: ChanceNode | null, key: string): ChanceEdge | null {
  return compareNode?.edges.find((e) => edgeKey(e, 'chance') === key) ?? null
}

function Delta({ current, baseline, format }: { current: number; baseline: number | null; format: (n: number) => string }) {
  if (baseline === null) return null
  const diff = current - baseline
  if (Math.abs(diff) < 1e-9) return null
  return <span className={`mtx-delta ${diff > 0 ? 'mtx-delta-up' : 'mtx-delta-down'}`}>{diff > 0 ? '▲' : '▼'}{format(Math.abs(diff))}</span>
}

function DecisionNodeBlockView({
  node,
  compareNode,
  path,
  expandedPaths,
  onToggle,
  onFocusTrend,
}: Omit<NodeBlockProps, 'node' | 'compareNode'> & { node: DecisionNode; compareNode: DecisionNode | null }) {
  const [showValue, setShowValue] = useState(false)
  const { showWinRate } = useViewSettings()
  const edges = [...node.edges].sort((a, b) => b.visit_count - a.visit_count)
  const maxVisit = edges[0]?.visit_count ?? 0
  const bestPuctEdge = edges.reduce<DecisionEdge | null>((best, e) => (best === null || e.puct_score > best.puct_score ? e : best), null)
  const totalVisits = edges.reduce((sum, e) => sum + e.visit_count, 0) || 1
  const winRate = showWinRate ? node.rank_probs?.[node.current_player]?.[0] : undefined

  return (
    <div className="mtx-node-block">
      <div className="mtx-node-header">
        <PlayerBadge index={node.current_player} winRate={winRate} />
        <span className="mtx-pill mtx-pill-phase">{node.phase}</span>
        {node.is_terminal && <span className="mtx-pill mtx-pill-terminal">⏹ game over</span>}
        <span className="mtx-visit-badge">N={node.visit_count}</span>
        {node.value && <PlayerValueStrip values={node.value} currentPlayer={node.current_player} />}
        <span className="mtx-node-header-actions">
          {edges.length >= 2 && (
            <button type="button" className="mtx-btn mtx-btn-ghost mtx-btn-small" onClick={() => onFocusTrend(path)}>
              trend ↗
            </button>
          )}
          <button
            type="button"
            className="mtx-btn mtx-btn-ghost mtx-btn-small"
            onClick={() => setShowValue((s) => !s)}
            aria-expanded={showValue}
          >
            state {showValue ? '▴' : '▾'}
          </button>
        </span>
      </div>
      {showValue && <ValuePanel node={node} />}
      {edges.length === 0 && !node.is_terminal && (
        <div className="mtx-empty-note">No edges recorded (unexpanded, or truncated by --max-depth).</div>
      )}
      {edges.length > 0 && (
        <div className="mtx-edge-list">
          <div className="mtx-col-header mtx-cols-decision">
            <span />
            <span>Action</span>
            <span>Visit share</span>
            <span className="mtx-num">Prior</span>
            <span className="mtx-num">Q</span>
            <span className="mtx-num">U</span>
            <span className="mtx-num">PUCT</span>
            <span>Mean value</span>
          </div>
          {edges.map((edge) => {
            const key = edgeKey(edge, 'decision')
            const nextPath = [...path, key]
            return (
              <DecisionEdgeRowView
                key={key}
                edge={edge}
                compareEdge={findCompareDecisionEdge(compareNode, key)}
                totalVisits={totalVisits}
                isStar={maxVisit > 0 && edge.visit_count === maxVisit}
                isBolt={edge === bestPuctEdge}
                parentPlayer={node.current_player}
                path={nextPath}
                expandedPaths={expandedPaths}
                onToggle={onToggle}
                onFocusTrend={onFocusTrend}
              />
            )
          })}
        </div>
      )}
    </div>
  )
}

function ChanceNodeBlockView({
  node,
  compareNode,
  path,
  expandedPaths,
  onToggle,
  onFocusTrend,
}: Omit<NodeBlockProps, 'node' | 'compareNode'> & { node: ChanceNode; compareNode: ChanceNode | null }) {
  const edges = [...node.edges].sort((a, b) => b.visit_count - a.visit_count)
  const maxVisit = edges[0]?.visit_count ?? 0
  const totalVisits = edges.reduce((sum, e) => sum + e.visit_count, 0) || 1

  return (
    <div className="mtx-node-block">
      <div className="mtx-node-header">
        <span className="mtx-pill mtx-pill-chance">✲ chance — unresolved reveal</span>
      </div>
      {edges.length === 0 ? (
        <div className="mtx-empty-note">No sampled values yet.</div>
      ) : (
        <div className="mtx-edge-list">
          <div className="mtx-col-header mtx-cols-chance">
            <span />
            <span>Card</span>
            <span>Visit share</span>
            <span className="mtx-num">Prior</span>
            <span>Mean value</span>
          </div>
          {edges.map((edge) => {
            const key = edgeKey(edge, 'chance')
            const nextPath = [...path, key]
            return (
              <ChanceEdgeRowView
                key={key}
                edge={edge}
                compareEdge={findCompareChanceEdge(compareNode, key)}
                totalVisits={totalVisits}
                isStar={maxVisit > 0 && edge.visit_count === maxVisit}
                path={nextPath}
                expandedPaths={expandedPaths}
                onToggle={onToggle}
                onFocusTrend={onFocusTrend}
              />
            )
          })}
        </div>
      )}
    </div>
  )
}

interface DecisionEdgeRowProps {
  edge: DecisionEdge
  compareEdge: DecisionEdge | null
  totalVisits: number
  isStar: boolean
  isBolt: boolean
  parentPlayer: number
  path: string[]
  expandedPaths: Set<string>
  onToggle: (path: string[]) => void
  onFocusTrend: (path: string[]) => void
}

function DecisionEdgeRowView({
  edge,
  compareEdge,
  totalVisits,
  isStar,
  isBolt,
  parentPlayer,
  path,
  expandedPaths,
  onToggle,
  onFocusTrend,
}: DecisionEdgeRowProps) {
  const { filterText, hideZero } = useViewSettings()
  const hasChild = edge.child !== null
  const expanded = hasChild && expandedPaths.has(pathId(path))
  const matchesFilter = !filterText || edge.action.type.toUpperCase().includes(filterText)
  const hidden = hideZero && edge.visit_count === 0
  const share = edge.visit_count / totalVisits

  let title: string | undefined
  if (!hasChild) {
    title = edge.visit_count > 0 ? 'Visited but not expanded in this export — truncated by --max-depth.' : 'Never explored in this snapshot.'
  }

  if (hidden) return null

  return (
    <div className={`mtx-edge-row${matchesFilter ? '' : ' mtx-dimmed'}`}>
      <button
        type="button"
        className="mtx-cols-decision mtx-edge-row-main"
        disabled={!hasChild}
        aria-expanded={hasChild ? expanded : undefined}
        aria-label={`${actionLabel(edge.action)} — ${edge.visit_count} visits, ${fmtPct(share)} share, PUCT ${fmtNum(edge.puct_score)}`}
        title={title}
        onClick={() => onToggle(path)}
      >
        <span className={`mtx-chevron${expanded ? ' mtx-chevron-open' : ''}${hasChild ? '' : ' mtx-chevron-leaf'}`}>›</span>
        <span className="mtx-action-label">
          {isStar && <span className="mtx-badge-star" title="Most-visited — what live play would choose">★</span>}
          {isBolt && <span className="mtx-badge-bolt" title="Highest PUCT score right now">⚡</span>}
          <span className="mtx-action-txt">{actionLabel(edge.action)}</span>
        </span>
        <span className="mtx-bar-cell">
          <span className="mtx-bar-track">
            <span className="mtx-bar-fill" style={{ width: `${(share * 100).toFixed(1)}%` }} />
          </span>
          <span className="mtx-bar-num">{edge.visit_count}</span>
          <Delta current={edge.visit_count} baseline={compareEdge?.visit_count ?? null} format={(n) => String(Math.round(n))} />
        </span>
        <span className="mtx-num-cell">
          {fmtPct(edge.prior, 1)}
          {Math.abs(edge.prior - edge.prior_before_noise) > 1e-6 && (
            <span className="mtx-delta" title="prior before Dirichlet root noise">
              raw {fmtPct(edge.prior_before_noise, 1)}
            </span>
          )}
        </span>
        <span className={`mtx-num-cell ${edge.q >= 0 ? 'mtx-good-text' : 'mtx-critical-text'}`}>
          {fmtSigned(edge.q, 3)}
          <Delta current={edge.q} baseline={compareEdge?.q ?? null} format={(n) => n.toFixed(3)} />
        </span>
        <span className="mtx-num-cell mtx-muted">{fmtNum(edge.u, 3)}</span>
        <span className={`mtx-num-cell${isBolt ? ' mtx-best' : ''}`}>{fmtNum(edge.puct_score, 3)}</span>
        <PlayerValueStrip values={edge.mean_value} currentPlayer={parentPlayer} />
      </button>
      {expanded && edge.child && (
        <div className="mtx-node-children">
          <NodeBlock
            node={edge.child}
            compareNode={compareEdge?.child ?? null}
            path={path}
            expandedPaths={expandedPaths}
            onToggle={onToggle}
            onFocusTrend={onFocusTrend}
          />
        </div>
      )}
    </div>
  )
}

interface ChanceEdgeRowProps {
  edge: ChanceEdge
  compareEdge: ChanceEdge | null
  totalVisits: number
  isStar: boolean
  path: string[]
  expandedPaths: Set<string>
  onToggle: (path: string[]) => void
  onFocusTrend: (path: string[]) => void
}

function ChanceEdgeRowView({ edge, compareEdge, totalVisits, isStar, path, expandedPaths, onToggle, onFocusTrend }: ChanceEdgeRowProps) {
  const { hideZero } = useViewSettings()
  const hasChild = edge.child !== null
  const expanded = hasChild && expandedPaths.has(pathId(path))
  const share = edge.visit_count / totalVisits
  if (hideZero && edge.visit_count === 0) return null

  return (
    <div className="mtx-edge-row">
      <button
        type="button"
        className="mtx-cols-chance mtx-edge-row-main"
        disabled={!hasChild}
        aria-expanded={hasChild ? expanded : undefined}
        aria-label={`Card ${edge.card_value} — ${edge.visit_count} visits, ${fmtPct(share)} share`}
        onClick={() => onToggle(path)}
      >
        <span className={`mtx-chevron${expanded ? ' mtx-chevron-open' : ''}${hasChild ? '' : ' mtx-chevron-leaf'}`}>›</span>
        <span>
          <span className={`mtx-card-chip ${cardTone(edge.card_value)}`}>{edge.card_value}</span>
          {isStar && <span className="mtx-badge-star">★</span>}
        </span>
        <span className="mtx-bar-cell">
          <span className="mtx-bar-track">
            <span className="mtx-bar-fill" style={{ width: `${(share * 100).toFixed(1)}%` }} />
          </span>
          <span className="mtx-bar-num">{edge.visit_count}</span>
          <Delta current={edge.visit_count} baseline={compareEdge?.visit_count ?? null} format={(n) => String(Math.round(n))} />
        </span>
        <span className="mtx-num-cell">{fmtPct(edge.prior, 1)}</span>
        <PlayerValueStrip values={edge.mean_value} />
      </button>
      {expanded && edge.child && (
        <div className="mtx-node-children">
          <NodeBlock
            node={edge.child}
            compareNode={compareEdge?.child ?? null}
            path={path}
            expandedPaths={expandedPaths}
            onToggle={onToggle}
            onFocusTrend={onFocusTrend}
          />
        </div>
      )}
    </div>
  )
}
