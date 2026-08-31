import { useRef, useState } from 'react'
import type { MouseEvent as ReactMouseEvent } from 'react'
import { DEFAULT_LAYOUT, nearestIndexFromX, xForIndex, yForShare } from '../mcts-tree/chartMath'
import type { ChartLayout } from '../mcts-tree/chartMath'
import { fmtPct } from './replayUtils'
import type { WinProbPoint, WinProbSeries } from './replayUtils'

const SEAT_COLORS = ['#2a78d6', '#d9701f', '#1baf7a', '#b5860a']
const GRIDLINE_SHARES = [0, 0.25, 0.5, 0.75, 1]
// A third of mcts-tree's own DEFAULT_LAYOUT height - this chart is a quick
// glance-at-the-trend strip synced to the scrubber, not the main focus the
// way the tree explorer's own move-preference chart is, so it doesn't need
// nearly as much vertical room. Only height changes; width/padding stay
// shared with DEFAULT_LAYOUT so the x-axis geometry (xForIndex) matches.
const LAYOUT: ChartLayout = { ...DEFAULT_LAYOUT, height: Math.round(DEFAULT_LAYOUT.height / 3) }

interface WinProbabilityChartProps {
  series: WinProbSeries[]
  totalSteps: number
  activeStep: number
  onScrub?: (step: number) => void
}

function pathFor(points: WinProbPoint[], totalSteps: number, layout = LAYOUT): string {
  let d = ''
  for (const point of points) {
    const x = xForIndex(point.step, totalSteps, layout)
    const y = yForShare(point.p, layout)
    d += d === '' ? `M ${x} ${y}` : ` L ${x} ${y}`
  }
  return d
}

function nearestPoint(points: WinProbPoint[], step: number): WinProbPoint | null {
  if (points.length === 0) return null
  return points.reduce((best, p) => (Math.abs(p.step - step) < Math.abs(best.step - step) ? p : best))
}

/** "How did each seat's own confidence evolve across the whole game" - the
 * chart that replaces manually tabulating `raw_rank_probs[seat][0]` by hand
 * at every decision. Each seat only has a point at its *own* decisions (see
 * `buildWinProbSeries`), so for two different nets this also surfaces where
 * they'd have disagreed about the position had they evaluated it. */
export default function WinProbabilityChart({ series, totalSteps, activeStep, onScrub }: WinProbabilityChartProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [hoverStep, setHoverStep] = useState<number | null>(null)
  const layout = LAYOUT

  if (totalSteps === 0) {
    return (
      <div className="mtx-chart-card">
        <div className="mtx-chart-header">
          <span className="mtx-chart-title">Win probability over the game</span>
        </div>
        <p className="mtx-chart-subtitle">No decisions to chart.</p>
      </div>
    )
  }

  function handleMove(e: ReactMouseEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return
    const relativeX = ((e.clientX - rect.left) / rect.width) * layout.width
    setHoverStep(nearestIndexFromX(relativeX, totalSteps, layout))
  }

  return (
    <div className="mtx-chart-card">
      <div className="mtx-chart-header">
        <div>
          <div className="mtx-chart-title">Win probability over the game</div>
          <div className="mtx-chart-subtitle">Each seat&rsquo;s own network&rsquo;s P(finish rank 0) at its own decisions.</div>
        </div>
      </div>

      <div className="mtx-chart-svg-wrap" style={{ position: 'relative' }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${layout.width} ${layout.height + 20}`}
          width="100%"
          role="img"
          aria-label="Win probability per seat across the game"
          onMouseMove={handleMove}
          onMouseLeave={() => setHoverStep(null)}
          onClick={() => {
            if (onScrub && hoverStep !== null) onScrub(hoverStep)
          }}
          style={onScrub ? { cursor: 'pointer' } : undefined}
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
          {[0, 0.5, 1].map((share) => (
            <text key={share} x={2} y={yForShare(share, layout) - 3} fontSize={9} fill="var(--text-muted)">
              {fmtPct(share, 0)}
            </text>
          ))}

          <line
            x1={xForIndex(activeStep, totalSteps, layout)}
            x2={xForIndex(activeStep, totalSteps, layout)}
            y1={layout.padTop}
            y2={layout.height - layout.padBottom}
            stroke="var(--accent)"
            strokeWidth={1.5}
          />

          {series.map((s, i) => (
            <path
              key={s.seat}
              d={pathFor(s.points, totalSteps, layout)}
              fill="none"
              stroke={SEAT_COLORS[i % SEAT_COLORS.length]}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
        </svg>

        {hoverStep !== null && (
          <div className="mtx-chart-tooltip" style={{ left: `${(xForIndex(hoverStep, totalSteps, layout) / layout.width) * 100}%`, top: 8 }}>
            <div className="mtx-chart-tooltip-n">step {hoverStep}</div>
            {series.map((s, i) => {
              const point = nearestPoint(s.points, hoverStep)
              return (
                <div className="mtx-chart-tooltip-row" key={s.seat}>
                  <span className="mtx-chart-tooltip-key">
                    <span className="mtx-chart-tooltip-swatch" style={{ background: SEAT_COLORS[i % SEAT_COLORS.length] }} />
                    {s.seatName}
                  </span>
                  <span className="mtx-chart-tooltip-value">{point ? fmtPct(point.p, 0) : '—'}</span>
                </div>
              )
            })}
          </div>
        )}
      </div>

      <div className="mtx-chart-legend">
        {series.map((s, i) => (
          <span className="mtx-chart-legend-item" key={s.seat}>
            <span className="mtx-chart-legend-swatch" style={{ background: SEAT_COLORS[i % SEAT_COLORS.length] }} />
            {s.seatName}
          </span>
        ))}
      </div>
    </div>
  )
}
