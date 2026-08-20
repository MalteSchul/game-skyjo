import type { CSSProperties } from 'react'
import type { DecisionNode } from './types'
import { fmtPct, fmtSigned } from './treeUtils'
import { DivergingMeter } from './meters'

/** The network's own prediction for one decision node: its raw `value`
 * vector and, when available, the `rank_probs` it's a linear weighting of.
 * Collapsed behind a toggle in the caller — this is the "why" behind a
 * node's numbers, read on demand rather than taking up a table row. */
export default function ValuePanel({ node }: { node: DecisionNode }) {
  return (
    <div className="mtx-value-panel">
      {node.value ? (
        <>
          <div className="mtx-value-panel-title">Network value · predicted utility per player</div>
          <div className="mtx-value-rows">
            {node.value.map((v, i) => (
              <div className="mtx-value-row" key={i}>
                <span className="mtx-value-row-label" style={{ color: `var(--mtx-player-${i % 8})` }}>
                  P{i}
                </span>
                <DivergingMeter value={v} label={`P${i} value ${fmtSigned(v)}`} />
                <span className="mtx-value-row-num">{fmtSigned(v, 3)}</span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="mtx-value-panel-title">Not expanded in this snapshot yet</div>
      )}

      {node.rank_probs && (
        <div className="mtx-rankprobs-wrap">
          <div className="mtx-rankprobs-caption">rank_probs — P(player i finishes at rank r)</div>
          <table className="mtx-rankprobs-table">
            <thead>
              <tr>
                <th />
                {node.rank_probs[0].map((_, r) => (
                  <th key={r}>rank {r}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {node.rank_probs.map((row, i) => (
                <tr key={i}>
                  <th style={{ color: `var(--mtx-player-${i % 8})` }}>P{i}</th>
                  {row.map((p, r) => (
                    <td key={r} style={{ '--w': p } as CSSProperties}>
                      {fmtPct(p, 0)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
