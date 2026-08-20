import { useEffect, useState } from 'react'

const PLAY_INTERVAL_MS = 900

interface SnapshotScrubberProps {
  order: string[]
  visitCounts: number[]
  activeIndex: number
  onChange: (index: number) => void
}

/** A slider over the exported snapshots (not a continuous value — there's no
 * data between them) plus a play button that steps through them
 * automatically, so you can watch the search's preferences settle rather
 * than clicking through N=10, N=50, N=200 one at a time. Ticks are
 * positioned by each snapshot's actual simulation count, not evenly spaced
 * by index, since `--sim-points 5,30,120` are not evenly spaced themselves
 * and that spacing is itself informative (a wide sim gap between two ticks
 * is a wide gap in what the search had time to do). */
export default function SnapshotScrubber({ order, visitCounts, activeIndex, onChange }: SnapshotScrubberProps) {
  const [playing, setPlaying] = useState(false)
  const atEnd = activeIndex >= order.length - 1

  // Recursive setTimeout (re-scheduled each time activeIndex changes) rather
  // than setInterval, mirroring App.tsx's thinking-poll: it always reads the
  // current activeIndex instead of risking a stale closure, and there is
  // never more than one pending advance in flight.
  useEffect(() => {
    if (!playing || atEnd) return
    const timeoutId = window.setTimeout(() => onChange(activeIndex + 1), PLAY_INTERVAL_MS)
    return () => window.clearTimeout(timeoutId)
  }, [playing, atEnd, activeIndex, onChange])

  // Reaching the end stops playback rather than looping silently — arriving
  // there is itself the interesting moment ("this is where search stopped").
  useEffect(() => {
    if (playing && atEnd) setPlaying(false)
  }, [playing, atEnd])

  const values = order.map((key) => Number(key))
  const minN = Math.min(...values)
  const maxN = Math.max(...values)
  const span = maxN - minN || 1

  return (
    <div className="mtx-scrubber">
      <div className="mtx-scrubber-top">
        <button
          type="button"
          className="mtx-btn mtx-btn-small"
          onClick={() => setPlaying((p) => !p)}
          disabled={order.length <= 1}
          aria-label={playing ? 'Pause playback' : 'Play through snapshots'}
        >
          {playing ? '⏸' : '▶'}
        </button>
        <span className="mtx-scrubber-label mtx-mono">
          N={order[activeIndex]} · {visitCounts[activeIndex] ?? 0} visits
        </span>
        <input
          type="range"
          className="mtx-scrubber-range"
          min={0}
          max={Math.max(0, order.length - 1)}
          step={1}
          value={activeIndex}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-label="Simulation snapshot"
        />
      </div>
      <div className="mtx-scrubber-ticks">
        {order.map((key, i) => {
          const n = values[i]
          const pct = ((n - minN) / span) * 100
          return (
            <span
              key={key}
              className={`mtx-scrubber-tick${i === activeIndex ? ' mtx-scrubber-tick-active' : ''}`}
              style={{ left: `${pct}%` }}
            >
              {key}
            </span>
          )
        })}
      </div>
    </div>
  )
}
