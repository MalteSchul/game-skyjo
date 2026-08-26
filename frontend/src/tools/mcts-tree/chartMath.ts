import type { TrendSeries } from './treeUtils'

/** Pure layout/selection math for `MovePreferenceChart`, kept separate from
 * the SVG-rendering component so it can be unit tested with plain numbers —
 * jsdom's `getBoundingClientRect` always reports zero size, which makes any
 * pixel-geometry test written against real DOM layout meaningless. */

export interface ChartLayout {
  width: number
  height: number
  padLeft: number
  padRight: number
  padTop: number
  padBottom: number
}

export const DEFAULT_LAYOUT: ChartLayout = { width: 640, height: 220, padLeft: 8, padRight: 8, padTop: 12, padBottom: 8 }

export function innerWidth(layout: ChartLayout): number {
  return Math.max(1, layout.width - layout.padLeft - layout.padRight)
}
export function innerHeight(layout: ChartLayout): number {
  return Math.max(1, layout.height - layout.padTop - layout.padBottom)
}

/** Evenly-spaced ordinal x position for snapshot index `i` of `count` total
 * — snapshots are plotted by search-progress order, not by their raw
 * simulation count, matching how training/eval curves are conventionally
 * read (see `SnapshotScrubber`'s docstring for the one place this repo
 * *does* want true N-proportional spacing — the scrubber ticks). */
export function xForIndex(i: number, count: number, layout: ChartLayout = DEFAULT_LAYOUT): number {
  if (count <= 1) return layout.padLeft + innerWidth(layout) / 2
  return layout.padLeft + (i / (count - 1)) * innerWidth(layout)
}

export function yForShare(share: number, layout: ChartLayout = DEFAULT_LAYOUT): number {
  const clamped = Math.max(0, Math.min(1, share))
  return layout.padTop + (1 - clamped) * innerHeight(layout)
}

/** Inverse of `xForIndex`, snapped to the nearest whole snapshot — there is
 * no data between two exported snapshots, so hovering always resolves to
 * one of them. */
export function nearestIndexFromX(x: number, count: number, layout: ChartLayout = DEFAULT_LAYOUT): number {
  if (count <= 1) return 0
  const ratio = (x - layout.padLeft) / innerWidth(layout)
  const index = Math.round(ratio * (count - 1))
  return Math.max(0, Math.min(count - 1, index))
}

export interface SelectedSeries {
  shown: TrendSeries[]
  other: TrendSeries | null
}

/** Caps the chart at `maxSeries` lines, keeping the ones with the highest
 * *final* visit share (the actions the search ended up preferring) and
 * folding the rest into a single grey "Other" line — more than ~6 lines on
 * one chart stops being readable regardless of how CVD-safe the palette is. */
export function selectTopSeries(series: TrendSeries[], order: string[], maxSeries = 6): SelectedSeries {
  const finalShare = (s: TrendSeries): number => {
    for (let i = order.length - 1; i >= 0; i--) {
      const point = s.points.find((p) => p.snapshotKey === order[i])
      if (point) return point.share
    }
    return 0
  }
  const sorted = [...series].sort((a, b) => finalShare(b) - finalShare(a))
  if (sorted.length <= maxSeries) return { shown: sorted, other: null }

  const shown = sorted.slice(0, maxSeries)
  const rest = sorted.slice(maxSeries)
  const other: TrendSeries = {
    key: '__other__',
    label: `Other (${rest.length})`,
    points: order
      .map((key) => {
        const points = rest.map((s) => s.points.find((p) => p.snapshotKey === key)).filter((p) => p !== undefined)
        if (points.length === 0) return null
        return {
          snapshotKey: key,
          n: points[0].n,
          visitCount: points.reduce((sum, p) => sum + p.visitCount, 0),
          share: points.reduce((sum, p) => sum + p.share, 0),
        }
      })
      .filter((p) => p !== null),
  }
  return { shown, other }
}

/** Builds an SVG path `d` string from a series' points, skipping the gap
 * before an action's first appearance rather than drawing a false "0%"
 * lead-in (see `buildTrendSeries`'s docstring in `treeUtils.ts`). */
export function pathForSeries(series: TrendSeries, order: string[], layout: ChartLayout = DEFAULT_LAYOUT): string {
  const byKey = new Map(series.points.map((p) => [p.snapshotKey, p]))
  let d = ''
  order.forEach((key, i) => {
    const point = byKey.get(key)
    if (!point) return
    const x = xForIndex(i, order.length, layout)
    const y = yForShare(point.share, layout)
    d += d === '' ? `M ${x} ${y}` : ` L ${x} ${y}`
  })
  return d
}

/** The index (into `order`) of a series' last plotted point — where its end
 * marker and, space permitting, its direct label go. */
export function lastPlottedIndex(series: TrendSeries, order: string[]): number {
  const keys = new Set(series.points.map((p) => p.snapshotKey))
  for (let i = order.length - 1; i >= 0; i--) {
    if (keys.has(order[i])) return i
  }
  return -1
}
