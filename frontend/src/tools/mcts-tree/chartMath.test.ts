import { describe, expect, it } from 'vitest'
import type { TrendSeries } from './treeUtils'
import {
  DEFAULT_LAYOUT,
  lastPlottedIndex,
  nearestIndexFromX,
  pathForSeries,
  selectTopSeries,
  xForIndex,
  yForShare,
} from './chartMath'

const LAYOUT = { width: 100, height: 50, padLeft: 0, padRight: 0, padTop: 0, padBottom: 0 }

describe('xForIndex / yForShare', () => {
  it('spaces points evenly across the inner width regardless of raw N spacing (happy path)', () => {
    expect(xForIndex(0, 3, LAYOUT)).toBe(0)
    expect(xForIndex(1, 3, LAYOUT)).toBe(50)
    expect(xForIndex(2, 3, LAYOUT)).toBe(100)
  })

  it('centers the single point when there is only one snapshot (bad path)', () => {
    expect(xForIndex(0, 1, LAYOUT)).toBe(50)
  })

  it('maps share 1 to the top and share 0 to the bottom (happy path)', () => {
    expect(yForShare(1, LAYOUT)).toBe(0)
    expect(yForShare(0, LAYOUT)).toBe(50)
  })

  it('clamps out-of-range shares instead of drawing off the chart (bad path)', () => {
    expect(yForShare(1.5, LAYOUT)).toBe(0)
    expect(yForShare(-0.2, LAYOUT)).toBe(50)
  })
})

describe('nearestIndexFromX', () => {
  it('snaps to the closest snapshot index (happy path)', () => {
    expect(nearestIndexFromX(0, 3, LAYOUT)).toBe(0)
    expect(nearestIndexFromX(49, 3, LAYOUT)).toBe(1)
    expect(nearestIndexFromX(100, 3, LAYOUT)).toBe(2)
  })

  it('clamps pointer positions outside the chart area (bad path)', () => {
    expect(nearestIndexFromX(-40, 3, LAYOUT)).toBe(0)
    expect(nearestIndexFromX(500, 3, LAYOUT)).toBe(2)
  })

  it('always resolves to index 0 for a single snapshot (bad path)', () => {
    expect(nearestIndexFromX(999, 1, LAYOUT)).toBe(0)
  })
})

function series(key: string, points: { snapshotKey: string; share: number; visitCount?: number }[]): TrendSeries {
  return { key, label: key, points: points.map((p) => ({ n: Number(p.snapshotKey), visitCount: p.visitCount ?? 0, ...p })) }
}

describe('selectTopSeries', () => {
  const order = ['0', '10']
  it('keeps every series unchanged when at or under the cap (happy path)', () => {
    const s = [series('a', [{ snapshotKey: '10', share: 0.6 }]), series('b', [{ snapshotKey: '10', share: 0.4 }])]
    const result = selectTopSeries(s, order, 6)
    expect(result.shown.map((x) => x.key)).toEqual(['a', 'b'])
    expect(result.other).toBeNull()
  })

  it('folds series beyond the cap into a single "Other" line summing their shares (sad path: too many actions)', () => {
    const s = [
      series('a', [{ snapshotKey: '10', share: 0.5 }]),
      series('b', [{ snapshotKey: '10', share: 0.3 }]),
      series('c', [{ snapshotKey: '10', share: 0.15 }]),
      series('d', [{ snapshotKey: '10', share: 0.05 }]),
    ]
    const result = selectTopSeries(s, order, 2)
    expect(result.shown.map((x) => x.key)).toEqual(['a', 'b'])
    expect(result.other?.label).toBe('Other (2)')
    expect(result.other?.points[0].share).toBeCloseTo(0.2)
  })

  it('ranks by the most recent snapshot each series actually has a point at, not strictly the last one (bad path)', () => {
    const s = [series('a', [{ snapshotKey: '0', share: 0.2 }]), series('b', [{ snapshotKey: '0', share: 0.1 }, { snapshotKey: '10', share: 0.9 }])]
    const result = selectTopSeries(s, order, 6)
    // 'a' has no point at '10' (e.g. a deeper node not yet visited then), so
    // its most recent known share ('0': 0.2) is what ranks it — still below
    // 'b's later, higher one.
    expect(result.shown.map((x) => x.key)).toEqual(['b', 'a'])
  })
})

describe('pathForSeries / lastPlottedIndex', () => {
  const order = ['0', '10', '50']

  it('draws a segment through every snapshot the action has a point at (happy path)', () => {
    const s = series('a', [{ snapshotKey: '0', share: 0 }, { snapshotKey: '10', share: 0.5 }, { snapshotKey: '50', share: 1 }])
    const d = pathForSeries(s, order, LAYOUT)
    expect(d).toBe('M 0 50 L 50 25 L 100 0')
    expect(lastPlottedIndex(s, order)).toBe(2)
  })

  it('skips the lead-in before an action first appears rather than drawing a false 0% (sad path)', () => {
    const s = series('a', [{ snapshotKey: '50', share: 1 }])
    const d = pathForSeries(s, order, LAYOUT)
    expect(d).toBe('M 100 0')
    expect(lastPlottedIndex(s, order)).toBe(2)
  })

  it('returns an empty path and -1 index for a series with no points at all (bad path)', () => {
    const s = series('a', [])
    expect(pathForSeries(s, order, LAYOUT)).toBe('')
    expect(lastPlottedIndex(s, order)).toBe(-1)
  })
})

it('DEFAULT_LAYOUT has positive inner dimensions', () => {
  expect(DEFAULT_LAYOUT.width - DEFAULT_LAYOUT.padLeft - DEFAULT_LAYOUT.padRight).toBeGreaterThan(0)
  expect(DEFAULT_LAYOUT.height - DEFAULT_LAYOUT.padTop - DEFAULT_LAYOUT.padBottom).toBeGreaterThan(0)
})
