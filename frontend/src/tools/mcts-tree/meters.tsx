import type { CSSProperties } from 'react'
import { fmtPct, fmtSigned } from './treeUtils'

/** A value in roughly [-1, 1] as a bar growing from a center baseline —
 * green to the right (favors whoever this is drawn for), red to the left.
 * Used for Q / mean_value, which are per-player utility estimates in that
 * range (see `AlphaZeroNet`'s rank-weighted value head). */
export function DivergingMeter({ value, label }: { value: number; label?: string }) {
  const clamped = Math.max(-1, Math.min(1, value))
  const isPositive = clamped >= 0
  const magnitude = Math.abs(clamped) * 50
  const style: CSSProperties = isPositive ? { width: `${magnitude}%`, left: '50%' } : { width: `${magnitude}%`, right: '50%' }
  return (
    <div className="mtx-div-meter" role="img" aria-label={label ?? `value ${fmtSigned(value)}`}>
      <div className={`mtx-div-meter-fill ${isPositive ? 'mtx-good' : 'mtx-critical'}`} style={style} />
    </div>
  )
}

/** One small square per player, colored by seat and filled proportional to
 * that player's component of a value/mean_value vector — lets a single cell
 * show every player's stake in an outcome, not just the acting player's. The
 * acting player's square gets an outline so it reads as "this is whose turn
 * it is" without needing a separate legend lookup. */
export function PlayerValueStrip({ values, currentPlayer }: { values: number[]; currentPlayer?: number }) {
  return (
    <div className="mtx-value-strip">
      {values.map((v, i) => {
        const isPositive = v >= 0
        const magnitude = Math.min(1, Math.abs(v)) * 100
        return (
          <div
            key={i}
            className={`mtx-value-chip${i === currentPlayer ? ' mtx-value-chip-active' : ''}`}
            style={{ '--player-color': `var(--mtx-player-${i % 8})` } as CSSProperties}
            title={`P${i}: ${fmtSigned(v, 3)}`}
          >
            <div
              className={`mtx-value-chip-fill ${isPositive ? 'mtx-good-fill' : 'mtx-critical-fill'}`}
              style={{ height: `${magnitude}%` }}
            />
          </div>
        )
      })}
    </div>
  )
}

/** `winRate`, when given, is P(this player finishes rank 0) — i.e. wins the
 * round — from the node's `rank_probs`; shown as a percentage appended to the
 * badge, behind the "show win %" view setting since it's only available once
 * `--network` populated `rank_probs` (see `ValuePanel`). */
export function PlayerBadge({ index, winRate }: { index: number; winRate?: number }) {
  return (
    <span className="mtx-pill mtx-pill-player" style={{ background: `var(--mtx-player-${index % 8})` }}>
      P{index}
      {winRate !== undefined && <span className="mtx-pill-winrate">{fmtPct(winRate, 0)}</span>}
    </span>
  )
}
