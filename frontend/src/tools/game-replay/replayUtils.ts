import type { DecisionReplay, GameReplay, ReplayAction } from './types'

const BOARD_COLUMNS = 4

/** 1-indexed "rRcC" for a board position (0-indexed, row-major over
 * BOARD_COLUMNS-wide rows) - the one position-naming scheme used everywhere
 * a position needs a human label. Duplicated from `game/HistoryPanel.tsx` /
 * `tools/mcts-tree/treeUtils.ts` for the same reason `actionLabel` below is
 * its own copy rather than an import - see that function's docstring. */
function rowColLabel(position: number): string {
  return `r${Math.floor(position / BOARD_COLUMNS) + 1}c${(position % BOARD_COLUMNS) + 1}`
}

const ACTION_LABELS: Record<string, (position: number | null) => string> = {
  DRAW_STOCK: () => 'Draw stock',
  DRAW_DISCARD: () => 'Draw discard',
  PLACE: (p) => `Place → ${rowColLabel(p!)}`,
  FLIP_INITIAL: (p) => `Flip ${rowColLabel(p!)}`,
  DISCARD_AND_REVEAL: (p) => `Discard & reveal ${rowColLabel(p!)}`,
}

/** Duplicated from `tools/mcts-tree/treeUtils.ts` rather than imported: that
 * module's `actionLabel` takes a `DecisionAction`, structurally identical to
 * `ReplayAction` here but a distinct type from a distinct export schema -
 * cross-importing would couple two independent JSON schemas that only
 * happen to agree on this one shape today. */
export function actionLabel(action: ReplayAction): string {
  const fn = ACTION_LABELS[action.type]
  if (fn) return fn(action.position)
  return action.position != null ? `${action.type} @ ${action.position}` : action.type
}

export function sameAction(a: ReplayAction, b: ReplayAction): boolean {
  return a.type === b.type && a.position === b.position
}

export function fmtPct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`
}

export function fmtSigned(x: number, digits = 3): string {
  const s = x.toFixed(digits)
  return x >= 0 ? `+${s}` : s
}

export interface WinProbPoint {
  step: number
  p: number
}

export interface WinProbSeries {
  seat: number
  seatName: string
  points: WinProbPoint[]
}

/** One series per seat: that seat's own network's P(finish rank 0) at each
 * of *that seat's own* decisions - the only steps where its net actually ran
 * a forward pass over the true state. There is no point for a seat on the
 * other seat's turn (see `raw_rank_probs`'s docstring in `types.ts`), so a
 * chart of this must skip gaps rather than interpolate/zero-fill them,
 * exactly like `mcts-tree/treeUtils.ts`'s `buildTrendSeries`. */
export function buildWinProbSeries(replay: GameReplay): WinProbSeries[] {
  return replay.seat_names.map((seatName, seat) => ({
    seat,
    seatName,
    points: replay.decisions
      .filter((d) => d.actor_seat === seat)
      .map((d) => ({ step: d.step, p: d.raw_rank_probs[seat]?.[0] ?? 0 })),
  }))
}

export interface PolicyRow {
  action: ReplayAction
  label: string
  priorShare: number
  visitShare: number
  /** Present only when the decision has `dirichlet_noised_priors`/`pi_target`
   * (a training self-play recording) - undefined, not 0, for an eval-style
   * recording, so callers can tell "no noise applied" apart from "this row's
   * noised share happens to be zero". */
  noisedPriorShare?: number
  piShare?: number
}

/** Pairs level-1 (raw prior) and level-2 (MCTS visit share) side by side for
 * every action either one considered, sorted by visit share (what search
 * actually settled on) - the data behind the disagreement view in
 * `DecisionDetail`. When the decision also carries training-self-play
 * detail (`dirichlet_noised_priors`/`pi_target`), each row additionally gets
 * `noisedPriorShare`/`piShare` for the same action set. */
export function buildPolicyComparison(decision: DecisionReplay): PolicyRow[] {
  const totalVisits = decision.mcts_visit_counts.reduce((sum, v) => sum + v.visit_count, 0) || 1
  const visitShareByKey = new Map<string, number>()
  const labelByKey = new Map<string, string>()
  const actionByKey = new Map<string, ReplayAction>()
  for (const v of decision.mcts_visit_counts) {
    const key = `${v.action.type}:${v.action.position}`
    visitShareByKey.set(key, v.visit_count / totalVisits)
    labelByKey.set(key, actionLabel(v.action))
    actionByKey.set(key, v.action)
  }
  const priorByKey = new Map<string, number>()
  for (const p of decision.raw_policy_priors) {
    const key = `${p.action.type}:${p.action.position}`
    priorByKey.set(key, p.prior)
    if (!labelByKey.has(key)) {
      labelByKey.set(key, actionLabel(p.action))
      actionByKey.set(key, p.action)
    }
  }
  const noisedByKey = new Map<string, number>()
  for (const p of decision.dirichlet_noised_priors ?? []) {
    const key = `${p.action.type}:${p.action.position}`
    noisedByKey.set(key, p.prior)
    if (!labelByKey.has(key)) {
      labelByKey.set(key, actionLabel(p.action))
      actionByKey.set(key, p.action)
    }
  }
  const piByKey = new Map<string, number>()
  for (const p of decision.pi_target ?? []) {
    const key = `${p.action.type}:${p.action.position}`
    piByKey.set(key, p.prior)
    if (!labelByKey.has(key)) {
      labelByKey.set(key, actionLabel(p.action))
      actionByKey.set(key, p.action)
    }
  }
  const hasSelfPlayDetail = decision.dirichlet_noised_priors != null || decision.pi_target != null
  const keys = new Set([...priorByKey.keys(), ...visitShareByKey.keys(), ...noisedByKey.keys(), ...piByKey.keys()])
  const rows: PolicyRow[] = [...keys].map((key) => ({
    action: actionByKey.get(key)!,
    label: labelByKey.get(key)!,
    priorShare: priorByKey.get(key) ?? 0,
    visitShare: visitShareByKey.get(key) ?? 0,
    ...(hasSelfPlayDetail
      ? { noisedPriorShare: noisedByKey.get(key) ?? 0, piShare: piByKey.get(key) ?? 0 }
      : {}),
  }))
  rows.sort((a, b) => b.visitShare - a.visitShare)
  return rows
}

export interface ValueRow {
  action: ReplayAction
  label: string
  initialValue: number
  finalValue: number
}

/** Pairs each action's Q-value (`mean_value()[actor_seat]`) early in this
 * decision's own search against its value after the full search, sorted by
 * final value descending (best action first) - a different axis entirely
 * from `buildPolicyComparison` (probability mass, not raw utility), so kept
 * as its own comparison rather than more columns bolted onto that table.
 * Empty when the decision has neither field (a replay exported before they
 * existed), not a list of zeros - callers should treat that as "not
 * available", the same convention `buildPolicyComparison`'s optional
 * columns use. */
export function buildValueComparison(decision: DecisionReplay): ValueRow[] {
  if (decision.initial_action_values == null && decision.final_action_values == null) return []
  const initialByKey = new Map<string, number>()
  const finalByKey = new Map<string, number>()
  const labelByKey = new Map<string, string>()
  const actionByKey = new Map<string, ReplayAction>()
  for (const v of decision.initial_action_values ?? []) {
    const key = `${v.action.type}:${v.action.position}`
    initialByKey.set(key, v.value)
    labelByKey.set(key, actionLabel(v.action))
    actionByKey.set(key, v.action)
  }
  for (const v of decision.final_action_values ?? []) {
    const key = `${v.action.type}:${v.action.position}`
    finalByKey.set(key, v.value)
    if (!labelByKey.has(key)) {
      labelByKey.set(key, actionLabel(v.action))
      actionByKey.set(key, v.action)
    }
  }
  const keys = new Set([...initialByKey.keys(), ...finalByKey.keys()])
  const rows: ValueRow[] = [...keys].map((key) => ({
    action: actionByKey.get(key)!,
    label: labelByKey.get(key)!,
    initialValue: initialByKey.get(key) ?? 0,
    finalValue: finalByKey.get(key) ?? 0,
  }))
  rows.sort((a, b) => b.finalValue - a.finalValue)
  return rows
}

/** Whether MCTS search's own most-visited action differs from, and actually
 * out-shares, the raw prior's favorite - recomputed from
 * `raw_policy_priors`/`mcts_visit_counts` directly rather than trusted from
 * the precomputed `search_overrode_prior` field, so this is correct even
 * for a replay exported before this distinction existed (or before the
 * backend's own definition was fixed to match it - see
 * `game_recorder._search_overrode_prior`'s docstring).
 *
 * Deliberately NOT "did `chosen_action` differ from the favorite": for a
 * training self-play recording, `chosen_action` is tau-sampled from `pi`, so
 * a low-probability action can get played by pure chance even when search's
 * own visit distribution fully agrees with the raw prior - that's sampling
 * variance, not search overriding anything. */
export function searchOverrodeRawPrior(decision: DecisionReplay): boolean {
  if (decision.mcts_visit_counts.length === 0) return false
  const topVisit = decision.mcts_visit_counts.reduce((best, v) => (v.visit_count > best.visit_count ? v : best))
  if (sameAction(topVisit.action, decision.raw_prior_favorite)) return false
  const totalVisits = decision.mcts_visit_counts.reduce((sum, v) => sum + v.visit_count, 0) || 1
  const topVisitShare = topVisit.visit_count / totalVisits
  const topVisitRawPriorShare = decision.raw_policy_priors.find((p) => sameAction(p.action, topVisit.action))?.prior ?? 0
  return topVisitShare > topVisitRawPriorShare
}

/** Every decision index (into `replay.decisions`) where search itself
 * (see `searchOverrodeRawPrior`) disagreed with the raw prior - the
 * divergence markers shown on the timeline. `seat`, if given, restricts this
 * to that seat's own decisions only (e.g. "just where iter5 overrode
 * itself"), since in a game between two different nets the two seats'
 * override patterns are usually the more interesting thing to compare, not
 * their sum. */
export function overrideStepIndices(replay: GameReplay, seat: number | null = null): number[] {
  const indices: number[] = []
  replay.decisions.forEach((d, i) => {
    if (searchOverrodeRawPrior(d) && (seat === null || d.actor_seat === seat)) indices.push(i)
  })
  return indices
}

/** Whether the action actually played (`chosen_action`) differs from what
 * search's own visit distribution most favored (its top-visited action) -
 * exactly the "a low-probability action got sampled by chance" case
 * `searchOverrodeRawPrior` deliberately excludes. Purely a tau-sampling
 * artifact, not a disagreement between search and the raw prior: the
 * top-visited action can (and often does) still agree with the raw prior
 * even when this is true. Always false for an eval-style recording (its
 * `chosen_action` is `greedy_action` - the top-visited action itself, by
 * construction), so this is only ever interesting for training self-play. */
export function chosenDiffersFromTopVisit(decision: DecisionReplay): boolean {
  if (decision.mcts_visit_counts.length === 0) return false
  const topVisit = decision.mcts_visit_counts.reduce((best, v) => (v.visit_count > best.visit_count ? v : best))
  return !sameAction(topVisit.action, decision.chosen_action)
}

/** Every decision index where `chosenDiffersFromTopVisit` holds - the
 * tau-sampling-deviation markers on the timeline. Same optional seat filter
 * as `overrideStepIndices`. */
export function chosenDiffersFromTopVisitStepIndices(replay: GameReplay, seat: number | null = null): number[] {
  const indices: number[] = []
  replay.decisions.forEach((d, i) => {
    if (chosenDiffersFromTopVisit(d) && (seat === null || d.actor_seat === seat)) indices.push(i)
  })
  return indices
}

/** Whether any decision carries training-self-play detail (`pi_target`) -
 * the signal that `chosenDiffersFromTopVisit` can ever be true here at all,
 * since an eval-style recording's `chosen_action` is always the top-visited
 * action by construction. */
export function hasTrainingSelfPlayData(replay: GameReplay): boolean {
  return replay.decisions.some((d) => d.pi_target != null)
}

/** Whether a decision has `heuristic_action_representative` - the field diff
 * logic actually needs - missing/null for a replay exported before it
 * existed (see `types.ts`), including one exported with the raw
 * `heuristic_action` alone but not yet its representative. */
export function hasHeuristicData(replay: GameReplay): boolean {
  return replay.decisions.some((d) => d.heuristic_action_representative != null)
}

/** Every decision index where what was actually played differs from what
 * the fixed rule-based reference would have played from that exact turn -
 * the "diffs from heuristic" markers. Compares `heuristic_action_representative`,
 * not the raw `heuristic_action` - see that field's docstring: `chosen_action`
 * is always itself a collapsed representative, so comparing against the raw
 * action would flag provably-equivalent choices (e.g. any initial-flip slot)
 * as disagreements. Same optional seat filter as `overrideStepIndices`, and
 * the same treatment of a replay with no heuristic data (returns no indices
 * rather than throwing). */
export function heuristicDiffStepIndices(replay: GameReplay, seat: number | null = null): number[] {
  const indices: number[] = []
  replay.decisions.forEach((d, i) => {
    if (
      d.heuristic_action_representative != null &&
      !sameAction(d.heuristic_action_representative, d.chosen_action) &&
      (seat === null || d.actor_seat === seat)
    ) {
      indices.push(i)
    }
  })
  return indices
}
