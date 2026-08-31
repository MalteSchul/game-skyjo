import { useEffect, useState } from 'react'

const PLAY_INTERVAL_MS = 250

interface DecisionScrubberProps {
  count: number
  activeIndex: number
  overrideIndices: number[]
  roundBoundaryIndices: number[]
  /** Steps where what was actually played differs from what a fixed
   * rule-based reference would have played - rendered as a second lane below
   * the first, only when `showHeuristicDiffs` is true. Kept as its own lane
   * rather than merged into the first one so a step that's both an override
   * and a heuristic-diff shows two marks instead of one hiding the other. */
  heuristicDiffIndices?: number[]
  showHeuristicDiffs?: boolean
  /** Steps where the played action differs from what search's own visit
   * distribution most favored - a tau-sampling artifact, not a disagreement
   * with the raw prior (see `chosenDiffersFromTopVisit`). Its own third
   * lane, only when `showChosenDiffers` is true. */
  chosenDiffersIndices?: number[]
  showChosenDiffers?: boolean
  onChange: (index: number) => void
}

interface MarkerLaneProps {
  count: number
  lastIndex: number
  onChange: (index: number) => void
  markers: Array<{ index: number; kind: string; label: string }>
}

function MarkerLane({ count, lastIndex, onChange, markers }: MarkerLaneProps) {
  if (count === 0) return null
  return (
    <div className="gr-scrubber-markers">
      {markers.map(({ index, kind, label }) => {
        const pct = lastIndex === 0 ? 0 : (index / lastIndex) * 100
        return (
          <button
            key={`${kind}-${index}`}
            type="button"
            className={`gr-scrubber-marker gr-scrubber-marker-${kind}`}
            style={{ left: `${pct}%` }}
            onClick={() => onChange(index)}
            title={`Jump to step ${index}`}
            aria-label={`Jump to step ${index} — ${label}`}
          />
        )
      })}
    </div>
  )
}

/** A slider over a game's sequence of decisions - the game-replay
 * counterpart to `mcts-tree/SnapshotScrubber`, which scrubs across
 * simulation-count snapshots of one decision instead. Ticks are evenly
 * spaced by index here (unlike that scrubber, decisions don't carry a
 * second meaningful axis to space by), and one or two thin lanes below the
 * track flag every step of interest - search overrides and round boundaries
 * always, heuristic diffs when enabled - so patterns are visible before
 * scrubbing to any one of them. */
export default function DecisionScrubber({
  count,
  activeIndex,
  overrideIndices,
  roundBoundaryIndices,
  heuristicDiffIndices = [],
  showHeuristicDiffs = false,
  chosenDiffersIndices = [],
  showChosenDiffers = false,
  onChange,
}: DecisionScrubberProps) {
  const [playing, setPlaying] = useState(false)
  const atEnd = activeIndex >= count - 1

  useEffect(() => {
    if (!playing || atEnd) return
    const timeoutId = window.setTimeout(() => onChange(activeIndex + 1), PLAY_INTERVAL_MS)
    return () => window.clearTimeout(timeoutId)
  }, [playing, atEnd, activeIndex, onChange])

  useEffect(() => {
    if (playing && atEnd) setPlaying(false)
  }, [playing, atEnd])

  const overrideSet = new Set(overrideIndices)
  const roundBoundarySet = new Set(roundBoundaryIndices)
  const lastIndex = Math.max(0, count - 1)

  const primaryMarkers: Array<{ index: number; kind: string; label: string }> = []
  for (let i = 0; i < count; i++) {
    if (overrideSet.has(i)) primaryMarkers.push({ index: i, kind: 'override', label: 'search overrode the raw prior' })
    else if (roundBoundarySet.has(i)) primaryMarkers.push({ index: i, kind: 'round', label: 'round boundary' })
  }
  const heuristicMarkers = heuristicDiffIndices.map((i) => ({
    index: i,
    kind: 'heuristic',
    label: 'differs from the heuristic reference',
  }))
  const chosenDiffersMarkers = chosenDiffersIndices.map((i) => ({
    index: i,
    kind: 'sampling',
    label: 'played action differs from what search itself most favored',
  }))

  return (
    <div className="mtx-scrubber">
      <div className="gr-scrubber-controls">
        <button
          type="button"
          className="mtx-btn mtx-btn-small"
          onClick={() => setPlaying((p) => !p)}
          disabled={count <= 1}
          aria-label={playing ? 'Pause playback' : 'Play through the game'}
        >
          {playing ? '⏸' : '▶'}
        </button>
        <span className="mtx-scrubber-label mtx-mono">
          step {activeIndex} / {lastIndex}
        </span>
      </div>
      {/* The range input and its marker overlay share this one box (rather
       * than the input sharing a row with the button/label, whose width it
       * wouldn't match) so a marker's left offset and the input's value both
       * map onto the same pixel span. Still not pixel-perfect — a native
       * thumb is inset from the track's true 0%/100% by about half its own
       * width, which nothing here corrects for — but no longer off by the
       * width of the button and label, which was the visible bug. */}
      <div className="gr-scrubber-track-wrap">
        <input
          type="range"
          className="mtx-scrubber-range"
          min={0}
          max={lastIndex}
          step={1}
          value={activeIndex}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-label="Decision step"
        />
        <MarkerLane count={count} lastIndex={lastIndex} onChange={onChange} markers={primaryMarkers} />
        {showHeuristicDiffs && (
          <MarkerLane count={count} lastIndex={lastIndex} onChange={onChange} markers={heuristicMarkers} />
        )}
        {showChosenDiffers && (
          <MarkerLane count={count} lastIndex={lastIndex} onChange={onChange} markers={chosenDiffersMarkers} />
        )}
      </div>
    </div>
  )
}
