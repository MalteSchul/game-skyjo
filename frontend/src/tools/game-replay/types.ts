/** Mirrors `backend/src/skyjo/rl/game_record_export.py` - the JSON shape
 * produced by `scripts/play_and_record_game.py`. This module only describes
 * that shape; it doesn't validate it (see `gameReplayParse.ts` for the
 * minimal structural check applied to untrusted file input). */

import type { BoardOut } from '../../api/types'

export interface ReplayAction {
  type: string
  position: number | null
}

/** The true, un-redacted state before a decision (hidden card values
 * included - see `domain.state_json.game_state_to_dict`). Boards use the
 * same `{value, face_up}` shape as the live match API's `BoardOut`, so
 * `game/Card.tsx` renders them unmodified. */
export interface ReplayBoardState {
  boards: BoardOut[]
  stock: number[]
  discard: number[]
  current_player: number
  drawn_card: number | null
  finisher: number | null
  players_awaiting_final_turn: number[]
  round_scores: number[] | null
  total_scores: number[]
  phase: string
  reshuffle_seed: number
  target_score: number
}

export interface WeightedAction {
  action: ReplayAction
  prior: number
}

export interface VisitedAction {
  action: ReplayAction
  visit_count: number
}

export interface ValuedAction {
  action: ReplayAction
  value: number
}

export interface DecisionReplay {
  step: number
  actor_seat: number
  actor_name: string
  phase: string
  total_scores: number[]
  board_state: ReplayBoardState
  /** Level 1: the network's policy head alone, no search - sorted by
   * descending prior. */
  raw_policy_priors: WeightedAction[]
  raw_prior_favorite: ReplayAction
  /** rank_probs[i][r] = P(player i finishes at rank r), absolute seat order,
   * from the same no-search forward pass as raw_policy_priors. */
  raw_rank_probs: number[][]
  raw_points_pred: number[]
  mcts_num_simulations_requested: number
  /** Level 2: the full MCTS root visit distribution - sorted by descending
   * visit_count. */
  mcts_visit_counts: VisitedAction[]
  mcts_root_value: number[] | null
  reused_tree_visits: number
  /** Q-value (`mean_value()[actor_seat]`, this seat's own view) per action -
   * `initial_action_values` from partway into this decision's own search, an
   * early checkpoint (not just the first simulation), `final_action_values`
   * after all requested simulations - shows how much search revised the
   * network's early value guess for each candidate, not just which one it
   * ended up preferring (`mcts_visit_counts`). `null`/missing for a replay
   * exported before these fields existed. */
  initial_action_values?: ValuedAction[] | null
  final_action_values?: ValuedAction[] | null
  chosen_action: ReplayAction
  /** `chosen_action !== raw_prior_favorite` - search corrected the network's
   * own first instinct. */
  search_overrode_prior: boolean
  /** What a fixed rule-based reference (no search) would have played from
   * this exact turn - purely hypothetical, never actually played. `null`/
   * missing for a replay exported before this field existed, so callers
   * should treat both the same as "not available" rather than assume it's
   * always present. Shown to the user, but NOT what diff logic should
   * compare against directly - see `heuristic_action_representative`. */
  heuristic_action?: ReplayAction | null
  /** `heuristic_action`'s own equivalence-class representative (see
   * `domain.action_equivalence`) - what `chosen_action`/`raw_prior_favorite`
   * actually are, since both the network and MCTS only ever operate on
   * collapsed representatives, never a raw board-position-specific action.
   * Comparing `heuristic_action` itself would flag e.g. every initial-flip
   * decision as "different" merely because the heuristic picks a uniformly
   * random real slot while the net's choice always renders as the same
   * representative slot - not a genuine disagreement. Use this field for
   * any diff/match logic; use `heuristic_action` only for display. */
  heuristic_action_representative?: ReplayAction | null
  /** Training self-play only (`record_training_selfplay_game.py`) - null for
   * an eval-style recording (`play_and_record_game.py`), which plays greedy
   * with no root noise and never computes a tau-tempered target at all.
   * `raw_policy_priors` above is always the *pre*-noise prior (the network's
   * own output); this is the root prior actually searched by PUCT, after
   * Dirichlet(alpha) noise was mixed in at weight epsilon. */
  dirichlet_noised_priors?: WeightedAction[] | null
  /** The tau-tempered visit distribution actually used as this decision's
   * training target (`rl.selfplay.generate_episode`'s `pi`) - sharper or
   * softer than the raw visit-count ratio depending on `tau`. Training
   * self-play only, like the three fields above/below. */
  pi_target?: WeightedAction[] | null
  tau?: number | null
  /** How many real board-position actions `chosen_action` was uniformly
   * sampled from (`domain.action_equivalence.tied_actions`) - >1 whenever
   * multiple positions were still equivalent at search time (e.g. every
   * initial-flip slot before any card is known). */
  tied_group_size?: number | null
}

export interface GameReplay {
  schema_version: number
  seat_names: string[]
  checkpoint_paths: string[]
  seed: number
  num_simulations: number
  c_puct: number
  final_total_scores: number[] | null
  final_ranks: number[] | null
  winner_name: string | null
  rounds_played: number
  decisions: DecisionReplay[]
}
