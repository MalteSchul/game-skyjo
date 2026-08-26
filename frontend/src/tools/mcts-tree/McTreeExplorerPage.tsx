import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { ApiError, getMctsTree } from '../../api/matchClient'
import { normalizeSnapshots, sortedSnapshotKeys, TreeParseError } from './treeParse'
import type { SnapshotMap, TreeNode } from './types'
import { actionLabel, bestLinePrefixIds, buildTrendSeries, collectAllPathIds, computeBestLine, fmtPct, fmtSigned, pathId, topEdgeByVisits } from './treeUtils'
import { ViewSettingsContext } from './ViewSettingsContext'
import NodeBlock from './TreeNodeView'
import SnapshotScrubber from './SnapshotScrubber'
import MovePreferenceChart from './MovePreferenceChart'
import './mcts-tree.css'

const CARD_LEGEND: Array<{ label: string; tone: string }> = [
  { label: '-2 / -1', tone: 'tone-neg' },
  { label: '0', tone: 'tone-zero' },
  { label: '1–4', tone: 'tone-low' },
  { label: '5–8', tone: 'tone-mid' },
  { label: '9–12', tone: 'tone-high' },
]

function detectPlayerCount(snapshots: SnapshotMap): number {
  let max = 2
  for (const root of Object.values(snapshots)) {
    if (root.kind === 'decision' && root.value) max = Math.max(max, root.value.length)
  }
  return max
}

export default function McTreeExplorerPage() {
  // The game's own #root is a fixed 1126px centered column (see index.css) —
  // this page is a dense data table that wants full width instead, so it
  // opts out for as long as it's mounted (see mcts-tree.css).
  useEffect(() => {
    document.body.classList.add('mcts-tools-page')
    return () => document.body.classList.remove('mcts-tools-page')
  }, [])

  const [snapshots, setSnapshots] = useState<SnapshotMap | null>(null)
  const [order, setOrder] = useState<string[]>([])
  const [fileName, setFileName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  // ?matchId=...&nodeId=... points this page at one history node's move in a
  // live match, instead of a dropped file — see
  // api.matches.get_history_node_mcts_tree. Read once: this page has no
  // router, and re-parsing on every render would be pointless since the URL
  // never changes without a full reload here.
  const [liveMatch] = useState(() => {
    const params = new URLSearchParams(window.location.search)
    const matchId = params.get('matchId')
    const nodeId = params.get('nodeId')
    return matchId && nodeId ? { matchId, nodeId } : null
  })
  const [liveLoading, setLiveLoading] = useState(liveMatch !== null)

  const [activeIndex, setActiveIndex] = useState(0)
  const [compareKey, setCompareKey] = useState<string | null>(null)
  const [filterText, setFilterText] = useState('')
  const [hideZero, setHideZero] = useState(false)
  const [showWinRate, setShowWinRate] = useState(false)
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set())
  const [chartPath, setChartPath] = useState<string[]>([])

  const fileInputRef = useRef<HTMLInputElement>(null)

  const activeKey = order[activeIndex] ?? null
  const root: TreeNode | null = activeKey && snapshots ? snapshots[activeKey] : null
  const compareRoot: TreeNode | null = compareKey && snapshots ? snapshots[compareKey] : null

  // Each snapshot gets its own principal variation auto-expanded; manual
  // expand/collapse choices reset when you switch snapshots, same as
  // switching to a different position resets which lines you had open.
  useEffect(() => {
    if (!root) return
    setExpandedPaths(new Set(bestLinePrefixIds(computeBestLine(root))))
    setChartPath([])
  }, [activeKey, root])

  function applyData(data: unknown, label: string) {
    try {
      const normalized = normalizeSnapshots(data)
      const keys = sortedSnapshotKeys(normalized)
      setSnapshots(normalized)
      setOrder(keys)
      setFileName(label)
      setActiveIndex(keys.length - 1)
      setCompareKey(null)
      setError(null)
    } catch (err) {
      setError(err instanceof TreeParseError ? err.message : 'Something went wrong reading that data.')
    }
  }

  function loadFile(file: File) {
    setError(null)
    const reader = new FileReader()
    reader.onload = () => {
      let data: unknown
      try {
        data = JSON.parse(String(reader.result))
      } catch (err) {
        setError(`That file isn't valid JSON (${err instanceof Error ? err.message : String(err)}).`)
        return
      }
      applyData(data, file.name)
    }
    reader.onerror = () => setError("Couldn't read that file.")
    reader.readAsText(file)
  }

  const loadFromLiveMatch = useCallback(() => {
    if (!liveMatch) return
    setLiveLoading(true)
    setError(null)
    getMctsTree(liveMatch.matchId, liveMatch.nodeId)
      .then((data) => applyData(data, `match ${liveMatch.matchId} · node ${liveMatch.nodeId}`))
      .catch((err) => {
        setError(err instanceof ApiError ? err.message : "Couldn't load that match's search tree.")
      })
      .finally(() => setLiveLoading(false))
  }, [liveMatch])

  useEffect(() => {
    loadFromLiveMatch()
    // Runs once on mount only — loadFromLiveMatch is stable across the
    // lifetime of this page (liveMatch is read once, see above).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) loadFile(file)
  }

  const playerCount = snapshots ? detectPlayerCount(snapshots) : 2

  const trendSeries = useMemo(() => {
    if (!snapshots) return []
    return buildTrendSeries(snapshots, order, chartPath)
  }, [snapshots, order, chartPath])

  const chartTitle = chartPath.length === 0 ? 'Root move preference over time' : 'Move preference over time (focused node)'
  const chartSubtitle =
    chartPath.length === 0
      ? 'Each root action\'s share of total simulations at every exported snapshot.'
      : `${chartPath.length} step${chartPath.length === 1 ? '' : 's'} into the tree, same idea.`

  return (
    <div className="mcts-explorer">
      <header className="mtx-topbar">
        <div className="mtx-brand">
          <span className="mtx-mark" aria-hidden="true">🌳</span>
          <div>
            <h1>MCTS Tree Explorer</h1>
            <div className="mtx-tagline">for dump_mcts_tree.py exports</div>
          </div>
        </div>
        {snapshots && (
          <div className="mtx-topbar-actions">
            <span className="mtx-file-pill">
              <span aria-hidden="true">{liveMatch ? '🔴' : '📄'}</span>
              <span className="mtx-fname">{fileName}</span>
            </span>
            {liveMatch && (
              <button type="button" className="mtx-btn" onClick={loadFromLiveMatch} disabled={liveLoading}>
                {liveLoading ? 'Reloading…' : 'Reload latest'}
              </button>
            )}
            <button type="button" className="mtx-btn" onClick={() => fileInputRef.current?.click()}>
              Load different file
            </button>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          aria-label="Load tree export file"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) loadFile(file)
            e.target.value = ''
          }}
        />
      </header>

      {!snapshots && liveMatch ? (
        <div className="mtx-empty-state">
          <div className="mtx-dropzone">
            <div className="mtx-glyph" aria-hidden="true">🌳</div>
            <h2>{liveLoading ? 'Loading the match’s search tree…' : 'Could not load that match'}</h2>
            <p>
              match <code>{liveMatch.matchId}</code>, node <code>{liveMatch.nodeId}</code>
            </p>
            {error && (
              <div className="mtx-error-line" role="alert">
                {error}
              </div>
            )}
            {!liveLoading && (
              <button type="button" className="mtx-btn mtx-btn-primary" onClick={loadFromLiveMatch}>
                Retry
              </button>
            )}
          </div>
        </div>
      ) : !snapshots ? (
        <div className="mtx-empty-state">
          <div
            className={`mtx-dropzone${dragOver ? ' mtx-drag-over' : ''}`}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <div className="mtx-glyph" aria-hidden="true">🌳</div>
            <h2>Drop a tree export here</h2>
            <p>
              Load the JSON produced by <code>scripts/dump_mcts_tree.py</code> — a snapshot dict keyed by simulation
              count, or a single exported tree. Nothing leaves your browser; the file is only read locally.
            </p>
            <button type="button" className="mtx-btn mtx-btn-primary" onClick={() => fileInputRef.current?.click()}>
              Choose file…
            </button>
            {error && (
              <div className="mtx-error-line" role="alert">
                {error}
              </div>
            )}
            <div className="mtx-hint">
              Expects the schema from <code>tree_export.tree_to_dict</code>: nodes shaped{' '}
              <code>{'{"kind":"decision"|"chance", "edges":[...]}'}</code>. Run with <code>--network</code> to get
              non-uniform priors/values and populated <code>rank_probs</code>.
            </div>
          </div>
        </div>
      ) : root ? (
        <div className="mtx-workspace">
          <aside className="mtx-rail">
            <details className="mtx-rail-section" open>
              <summary>Compare against</summary>
              <label className="mtx-field-label" htmlFor="mtx-compare-select">
                Show deltas vs.
              </label>
              <select
                id="mtx-compare-select"
                value={compareKey ?? ''}
                onChange={(e) => setCompareKey(e.target.value || null)}
              >
                <option value="">None</option>
                {order
                  .filter((k) => k !== activeKey)
                  .map((k) => (
                    <option key={k} value={k}>
                      N={k}
                    </option>
                  ))}
              </select>
            </details>

            <details className="mtx-rail-section" open>
              <summary>Filter</summary>
              <label className="mtx-field-label" htmlFor="mtx-filter-input">
                Action type contains
              </label>
              <input
                id="mtx-filter-input"
                type="text"
                placeholder="e.g. PLACE, DRAW_STOCK…"
                value={filterText}
                onChange={(e) => setFilterText(e.target.value.toUpperCase())}
              />
              <label className="mtx-checkline">
                <input type="checkbox" checked={hideZero} onChange={(e) => setHideZero(e.target.checked)} />
                Hide never-visited edges
              </label>
              <label className="mtx-checkline">
                <input type="checkbox" checked={showWinRate} onChange={(e) => setShowWinRate(e.target.checked)} />
                Show win % on player badge
              </label>
            </details>

            <details className="mtx-rail-section">
              <summary>Legend</summary>
              <div className="mtx-legend-grid">
                <div className="mtx-legend-row">
                  <span className="mtx-legend-swatch" style={{ background: 'var(--accent)' }} />
                  Visit-share bar — fraction of the parent&rsquo;s simulations that went down this edge. This is what
                  live play actually picks by (argmax visits).
                </div>
                <div className="mtx-legend-row">
                  <span aria-hidden="true">★</span>
                  Most-visited edge at this node — the &ldquo;answer&rdquo; after search.
                </div>
                <div className="mtx-legend-row">
                  <span aria-hidden="true">⚡</span>
                  Highest <span className="mtx-legend-key">puct_score</span> right now — what selection would try
                  next, not necessarily the same edge.
                </div>
                <div className="mtx-legend-row">
                  <span className="mtx-legend-swatch" style={{ background: 'var(--mtx-good)' }} />
                  <span className="mtx-legend-swatch" style={{ background: 'var(--danger)' }} />
                  Diverging bars — <span className="mtx-legend-key">q</span> / value, from the acting player&rsquo;s
                  own view.
                </div>
                <div className="mtx-legend-row">
                  <div>
                    Chance-edge cards, by Skyjo&rsquo;s own colour bands:
                    <div className="mtx-chip-row">
                      {CARD_LEGEND.map(({ label, tone }) => (
                        <span key={label} className={`mtx-card-chip ${tone}`}>
                          {label}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="mtx-legend-row">
                  <div>
                    Players, by seat:
                    <div className="mtx-chip-row">
                      {Array.from({ length: playerCount }, (_, i) => (
                        <span key={i} className="mtx-pill mtx-pill-player" style={{ background: `var(--mtx-player-${i % 8})` }}>
                          P{i}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </details>
          </aside>

          <main className="mtx-canvas">
            <StatTiles root={root} />

            <SnapshotScrubber
              order={order}
              visitCounts={order.map((k) => {
                const node = snapshots[k]
                return node.kind === 'decision' ? node.visit_count : 0
              })}
              activeIndex={activeIndex}
              onChange={setActiveIndex}
            />

            <div className="mtx-toolbar">
              <button
                type="button"
                className="mtx-btn mtx-btn-small"
                onClick={() => setExpandedPaths(new Set(bestLinePrefixIds(computeBestLine(root))))}
              >
                Expand best line
              </button>
              <button type="button" className="mtx-btn mtx-btn-small mtx-btn-ghost" onClick={() => setExpandedPaths(new Set(collectAllPathIds(root)))}>
                Expand all
              </button>
              <button type="button" className="mtx-btn mtx-btn-small mtx-btn-ghost" onClick={() => setExpandedPaths(new Set())}>
                Collapse all
              </button>
              <div className="mtx-spacer" />
              <input
                className="mtx-search-input"
                type="text"
                placeholder="Filter by action…"
                value={filterText}
                onChange={(e) => setFilterText(e.target.value.toUpperCase())}
                aria-label="Filter by action type"
              />
            </div>

            <MovePreferenceChart
              title={chartTitle}
              subtitle={chartSubtitle}
              series={trendSeries}
              order={order}
              onClose={chartPath.length > 0 ? () => setChartPath([]) : undefined}
            />

            <div className="mtx-tree-shell">
              <ViewSettingsContext.Provider value={{ filterText, hideZero, showWinRate }}>
                <NodeBlock
                  node={root}
                  compareNode={compareRoot}
                  path={[]}
                  expandedPaths={expandedPaths}
                  onToggle={(path) =>
                    setExpandedPaths((prev) => {
                      const id = pathId(path)
                      const next = new Set(prev)
                      if (next.has(id)) next.delete(id)
                      else next.add(id)
                      return next
                    })
                  }
                  onFocusTrend={setChartPath}
                />
              </ViewSettingsContext.Provider>
            </div>
          </main>
        </div>
      ) : (
        <div className="mtx-empty-state">
          <p>That snapshot has no data.</p>
        </div>
      )}
    </div>
  )
}

function StatTiles({ root }: { root: TreeNode }) {
  if (root.kind !== 'decision') {
    return (
      <div className="mtx-stat-tiles">
        <div className="mtx-stat-tile">
          <div className="mtx-label">Root</div>
          <div className="mtx-value">chance node</div>
          <div className="mtx-sub">unexpected — trees should always root at a decision</div>
        </div>
      </div>
    )
  }
  const best = topEdgeByVisits(root.edges)
  const totalVisits = root.visit_count

  return (
    <div className="mtx-stat-tiles">
      <div className="mtx-stat-tile">
        <div className="mtx-label">Simulations run</div>
        <div className="mtx-value mtx-mono">{totalVisits}</div>
        <div className="mtx-sub">at this snapshot</div>
      </div>
      <div className="mtx-stat-tile">
        <div className="mtx-label">Legal actions at root</div>
        <div className="mtx-value mtx-mono">{root.edges.length}</div>
        <div className="mtx-sub">{root.phase}</div>
      </div>
      {best && (
        <div className="mtx-stat-tile">
          <div className="mtx-label">Most-visited root action</div>
          <div className="mtx-value">{actionLabel(best.action)}</div>
          <div className="mtx-sub">
            {best.visit_count} visits · {totalVisits > 0 ? fmtPct(best.visit_count / totalVisits, 0) : '—'} share
          </div>
        </div>
      )}
      {root.value && (
        <div className="mtx-stat-tile">
          <div className="mtx-label">Predicted value · P{root.current_player}</div>
          <div className={`mtx-value mtx-mono ${root.value[root.current_player] >= 0 ? 'mtx-good-text' : 'mtx-critical-text'}`}>
            {fmtSigned(root.value[root.current_player], 3)}
          </div>
          <div className="mtx-sub">network&rsquo;s own view</div>
        </div>
      )}
    </div>
  )
}
