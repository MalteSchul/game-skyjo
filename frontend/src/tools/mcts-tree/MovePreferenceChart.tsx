import { useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'
import {
  DEFAULT_LAYOUT,
  lastPlottedIndex,
  nearestIndexFromX,
  pathForSeries,
  selectTopSeries,
  xForIndex,
  yForShare,
} from './chartMath'
import { fmtPct } from './treeUtils'
import type { TrendSeries } from './treeUtils'

const SERIES_COLORS = ['#0f9b8e', '#c9971f', '#c23b82', '#4c5fd5', '#3f8f4f', '#c76b2e']
const OTHER_COLOR = 'var(--text-muted)'
const GRIDLINE_SHARES = [0, 0.25, 0.5, 0.75, 1]

interface MovePreferenceChartProps {
  title: string
  subtitle: string
  series: TrendSeries[]
  order: string[]
  onClose?: () => void
}

/** "How did the search's move preference evolve" — one line per action,
 * y = visit share of its parent node's total, x = snapshot in search-progress
 * order. This is the chart the whole export exists to answer (see
 * `dump_mcts_tree.py`'s own docstring: "so you can diff/inspect how it
 * actually grows"), just drawn instead of read off JSON by eye. */
export default function MovePreferenceChart({ title, subtitle, series, order, onClose }: MovePreferenceChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)
  const layout = DEFAULT_LAYOUT

  if (order.length === 0 || series.length === 0) {
    return (
      <div className="mtx-chart-card">
        <div className="mtx-chart-header">
          <span className="mtx-chart-title">{title}</span>
          {onClose && (
            <button type="button" className="mtx-btn mtx-btn-ghost mtx-btn-small" onClick={onClose}>
              close
            </button>
          )}
        </div>
        <p className="mtx-chart-subtitle">No actions to chart yet at this node.</p>
      </div>
    )
  }

  const { shown, other } = selectTopSeries(series, order, 6)
  const allSeries = other ? [...shown, other] : shown
  const colorFor = (index: number, isOther: boolean) => (isOther ? OTHER_COLOR : SERIES_COLORS[index % SERIES_COLORS.length])

  function handleMove(e: ReactMouseEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return
    const relativeX = ((e.clientX - rect.left) / rect.width) * layout.width
    setHoverIndex(nearestIndexFromX(relativeX, order.length, layout))
  }

  const hoverKey = hoverIndex !== null ? order[hoverIndex] : null
  const hoverX = hoverIndex !== null ? xForIndex(hoverIndex, order.length, layout) : 0

  // Direct end-labels only help when they land far enough apart to read —
  // once two series converge near the same final share, stacking labels
  // there just detaches them from their lines (see marks-and-anatomy: past a
  // few converging series, fall back to the legend + tooltip instead).
  const MIN_LABEL_GAP_PX = 14
  const endY = allSeries
    .map((s) => {
      const idx = lastPlottedIndex(s, order)
      const point = idx >= 0 ? s.points.find((p) => p.snapshotKey === order[idx]) : undefined
      return point ? yForShare(point.share, layout) : null
    })
    .filter((y): y is number => y !== null)
    .sort((a, b) => a - b)
  const labelsCollide = endY.some((y, i) => i > 0 && y - endY[i - 1] < MIN_LABEL_GAP_PX)
  const showDirectLabels = allSeries.length <= 4 && !labelsCollide

  return (
    <div className="mtx-chart-card">
      <div className="mtx-chart-header">
        <div>
          <div className="mtx-chart-title">{title}</div>
          <div className="mtx-chart-subtitle">{subtitle}</div>
        </div>
        {onClose && (
          <button type="button" className="mtx-btn mtx-btn-ghost mtx-btn-small" onClick={onClose}>
            close
          </button>
        )}
      </div>

      <div className="mtx-chart-svg-wrap" style={{ position: 'relative' }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${layout.width} ${layout.height + 20}`}
          width="100%"
          role="img"
          aria-label={`${title}: visit share of each action across ${order.length} snapshots`}
          onMouseMove={handleMove}
          onMouseLeave={() => setHoverIndex(null)}
        >
          {GRIDLINE_SHARES.map((share) => (
            <line
              key={share}
              x1={layout.padLeft}
              x2={layout.width - layout.padRight}
              y1={yForShare(share, layout)}
              y2={yForShare(share, layout)}
              stroke="var(--border)"
              strokeWidth={1}
            />
          ))}
          {GRIDLINE_SHARES.filter((s) => s === 0 || s === 0.5 || s === 1).map((share) => (
            <text key={share} x={2} y={yForShare(share, layout) - 3} fontSize={9} fill="var(--text-muted)">
              {fmtPct(share, 0)}
            </text>
          ))}

          {order.map((key, i) => (
            <text
              key={key}
              x={xForIndex(i, order.length, layout)}
              y={layout.height + 14}
              fontSize={9.5}
              textAnchor="middle"
              fill={i === hoverIndex ? 'var(--accent)' : 'var(--text-muted)'}
              fontFamily="var(--mono)"
            >
              N={key}
            </text>
          ))}

          {hoverIndex !== null && (
            <line
              x1={xForIndex(hoverIndex, order.length, layout)}
              x2={xForIndex(hoverIndex, order.length, layout)}
              y1={layout.padTop}
              y2={layout.height - layout.padBottom}
              stroke="var(--text-muted)"
              strokeWidth={1}
            />
          )}

          {allSeries.map((s, i) => {
            const isOther = other !== null && s.key === other.key
            const color = colorFor(i, isOther)
            const d = pathForSeries(s, order, layout)
            const lastIdx = lastPlottedIndex(s, order)
            const lastPoint = lastIdx >= 0 ? s.points.find((p) => p.snapshotKey === order[lastIdx]) : undefined
            return (
              <g key={s.key}>
                <path d={d} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" strokeDasharray={isOther ? '4 3' : undefined} />
                {lastPoint && (
                  <circle
                    cx={xForIndex(lastIdx, order.length, layout)}
                    cy={yForShare(lastPoint.share, layout)}
                    r={4}
                    fill={color}
                    stroke="var(--bg-panel)"
                    strokeWidth={2}
                  />
                )}
                {showDirectLabels &&
                  lastPoint &&
                  (() => {
                    // Anchor the label on whichever side actually has room —
                    // a fixed left-of-point offset clips the moment the
                    // series' most recent point sits near the right edge
                    // (see marks-and-anatomy: a label never gets clipped by
                    // its own mark; measure first).
                    const text = `${s.label} · ${fmtPct(lastPoint.share, 0)}`
                    const estimatedWidth = text.length * 5.6
                    const px = xForIndex(lastIdx, order.length, layout)
                    const fitsRight = px + 6 + estimatedWidth <= layout.width - 2
                    return (
                      <text
                        x={fitsRight ? px + 6 : px - 6}
                        y={yForShare(lastPoint.share, layout) + 3}
                        fontSize={10}
                        textAnchor={fitsRight ? 'start' : 'end'}
                        fill="var(--text-h)"
                      >
                        {text}
                      </text>
                    )
                  })()}
              </g>
            )
          })}
        </svg>

        {hoverKey && (
          <div
            className="mtx-chart-tooltip"
            style={{ left: `${(hoverX / layout.width) * 100}%`, top: 8 }}
          >
            <div className="mtx-chart-tooltip-n">N={hoverKey}</div>
            {allSeries.map((s, i) => {
              const isOther = other !== null && s.key === other.key
              const point = s.points.find((p) => p.snapshotKey === hoverKey)
              return (
                <div className="mtx-chart-tooltip-row" key={s.key}>
                  <span className="mtx-chart-tooltip-key">
                    <span className="mtx-chart-tooltip-swatch" style={{ background: colorFor(i, isOther) }} />
                    {s.label}
                  </span>
                  <span className="mtx-chart-tooltip-value">{point ? fmtPct(point.share, 0) : '—'}</span>
                </div>
              )
            })}
          </div>
        )}
      </div>

      <div className="mtx-chart-legend">
        {allSeries.map((s, i) => {
          const isOther = other !== null && s.key === other.key
          return (
            <span className="mtx-chart-legend-item" key={s.key}>
              <span className="mtx-chart-legend-swatch" style={{ background: colorFor(i, isOther) }} />
              {s.label}
            </span>
          )
        })}
      </div>
    </div>
  )
}
