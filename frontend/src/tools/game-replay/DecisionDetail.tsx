import { DivergingMeter } from '../mcts-tree/meters'
import ValuePanel from '../mcts-tree/ValuePanel'
import type { DecisionNode } from '../mcts-tree/types'
import { actionLabel, buildPolicyComparison, buildValueComparison, fmtPct, fmtSigned, sameAction, searchOverrodeRawPrior } from './replayUtils'
import type { DecisionReplay } from './types'

interface DecisionDetailProps {
  decision: DecisionReplay
  seatNames: string[]
}

function PolicyBar({ share }: { share: number }) {
  return (
    <span className="mtx-bar-cell">
      <span className="mtx-bar-track">
        <span className="mtx-bar-fill" style={{ width: `${(share * 100).toFixed(1)}%` }} />
      </span>
      <span className="mtx-bar-num">{fmtPct(share, 0)}</span>
    </span>
  )
}

function ValueMeter({ value }: { value: number }) {
  return (
    <span className="mtx-bar-cell">
      <DivergingMeter value={value} label={`value ${fmtSigned(value)}`} />
      <span className="mtx-bar-num">{fmtSigned(value, 2)}</span>
    </span>
  )
}

/** Everything known about one decision: level 1 (raw policy prior) next to
 * level 2 (the MCTS visit distribution search actually settled on), plus the
 * network's value/rank_probs/points_pred reused as-is from `ValuePanel` —
 * the same "why" panel the tree explorer uses, since our root-level
 * `raw_rank_probs`/`raw_points_pred`/`mcts_root_value` are exactly the shape
 * it expects. For a training self-play recording (`dirichlet_noised_priors`/
 * `pi_target` present), the policy table grows two extra columns - the
 * noised prior PUCT actually searched, and the tau-tempered training
 * target - rather than losing that detail to fit the eval-style shape. */
export default function DecisionDetail({ decision, seatNames }: DecisionDetailProps) {
  const rows = buildPolicyComparison(decision)
  const valueRows = buildValueComparison(decision)
  const totalVisits = decision.mcts_visit_counts.reduce((sum, v) => sum + v.visit_count, 0)
  const hasSelfPlayDetail = decision.dirichlet_noised_priors != null || decision.pi_target != null

  const valuePanelNode: DecisionNode = {
    kind: 'decision',
    current_player: decision.actor_seat,
    phase: decision.phase,
    is_terminal: false,
    visit_count: totalVisits,
    value: decision.mcts_root_value,
    rank_probs: decision.raw_rank_probs,
    points_pred: decision.raw_points_pred,
    edges: [],
  }

  return (
    <div className="gr-decision-detail">
      <div className="mtx-node-header">
        <span className="mtx-pill mtx-pill-player" style={{ background: `var(--mtx-player-${decision.actor_seat % 8})` }}>
          {seatNames[decision.actor_seat] ?? `P${decision.actor_seat}`}
        </span>
        <span className="mtx-pill mtx-pill-phase">{decision.phase}</span>
        <span className="mtx-visit-badge">step {decision.step}</span>
        <span className="mtx-visit-badge">
          scores {decision.total_scores.map((s, i) => `${seatNames[i] ?? `P${i}`}:${s}`).join(' · ')}
        </span>
        {decision.reused_tree_visits > 0 && (
          <span className="mtx-visit-badge">{decision.reused_tree_visits} visits reused from prior turn</span>
        )}
        {decision.tau != null && <span className="mtx-visit-badge">tau={decision.tau}</span>}
        {decision.tied_group_size != null && decision.tied_group_size > 1 && (
          <span className="mtx-visit-badge" title="how many real board-position actions the chosen action was uniformly sampled from">
            {decision.tied_group_size} tied positions
          </span>
        )}
        {searchOverrodeRawPrior(decision) && (
          <span className="mtx-pill mtx-pill-terminal">search overrode prior</span>
        )}
      </div>

      <div className="mtx-empty-note">
        Chosen: <strong>{actionLabel(decision.chosen_action)}</strong>
        {' · '}Raw prior favorite: <strong>{actionLabel(decision.raw_prior_favorite)}</strong>
        {decision.heuristic_action != null && decision.heuristic_action_representative != null && (
          <>
            {' · '}Heuristic would play: <strong>{actionLabel(decision.heuristic_action)}</strong>{' '}
            <em>
              {sameAction(decision.heuristic_action_representative, decision.chosen_action)
                ? '(matches what was played)'
                : sameAction(decision.heuristic_action_representative, decision.raw_prior_favorite)
                  ? '(matches the raw prior, not what was played)'
                  : '(differs from both)'}
            </em>
          </>
        )}
      </div>

      <div className="mtx-edge-list">
        <div className={`mtx-col-header ${hasSelfPlayDetail ? 'gr-cols-policy-selfplay' : 'gr-cols-policy'}`}>
          <span>Action</span>
          <span>Raw prior</span>
          {hasSelfPlayDetail && <span>+Dirichlet noise</span>}
          <span>MCTS visit share</span>
          {hasSelfPlayDetail && <span>pi (tau={decision.tau})</span>}
        </div>
        {rows.map((row) => {
          const isChosen = sameAction(row.action, decision.chosen_action)
          const isPriorFavorite = sameAction(row.action, decision.raw_prior_favorite)
          const isHeuristic =
            decision.heuristic_action_representative != null &&
            sameAction(row.action, decision.heuristic_action_representative)
          return (
            <div className={hasSelfPlayDetail ? 'gr-cols-policy-selfplay' : 'gr-cols-policy'} key={`${row.action.type}:${row.action.position}`}>
              <span>
                {isChosen && <span title="chosen">★ </span>}
                {row.label}
                {isPriorFavorite && !isChosen && <span title="raw prior favorite"> (prior fav.)</span>}
                {isHeuristic && <span title="what the heuristic reference would have played"> (heuristic)</span>}
              </span>
              <PolicyBar share={row.priorShare} />
              {hasSelfPlayDetail && <PolicyBar share={row.noisedPriorShare ?? 0} />}
              <PolicyBar share={row.visitShare} />
              {hasSelfPlayDetail && <PolicyBar share={row.piShare ?? 0} />}
            </div>
          )
        })}
      </div>

      {valueRows.length > 0 && (
        <div className="gr-value-compare">
          <div className="gr-value-compare-title">
            How MCTS values each action — Q-value ({seatNames[decision.actor_seat] ?? `P${decision.actor_seat}`}&rsquo;s own view), early in search vs. after it finished
          </div>
          <div className="mtx-col-header gr-cols-value">
            <span>Action</span>
            <span>Initial</span>
            <span>Final</span>
          </div>
          {valueRows.map((row) => (
            <div className="gr-cols-value" key={`${row.action.type}:${row.action.position}`}>
              <span>{row.label}</span>
              <ValueMeter value={row.initialValue} />
              <ValueMeter value={row.finalValue} />
            </div>
          ))}
        </div>
      )}

      <ValuePanel node={valuePanelNode} />
    </div>
  )
}
