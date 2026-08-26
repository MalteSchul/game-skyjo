/** Mirrors `backend/scripts/TREE_EXPORT_SCHEMA.md` — the JSON shape produced by
 * `rl.tree_export.tree_to_dict` / `scripts/dump_mcts_tree.py`. This module only
 * describes that shape; it doesn't validate it (see `treeParse.ts` for the
 * minimal structural check applied to untrusted file input). */

export interface DecisionAction {
  type: string
  position: number | null
}

export interface DecisionEdge {
  action: DecisionAction
  prior: number
  prior_before_noise: number
  visit_count: number
  mean_value: number[]
  q: number
  u: number
  puct_score: number
  child: TreeNode | null
}

export interface DecisionNode {
  kind: 'decision'
  current_player: number
  phase: string
  is_terminal: boolean
  visit_count: number
  value: number[] | null
  rank_probs: number[][] | null
  points_pred: number[] | null
  edges: DecisionEdge[]
}

export interface ChanceEdge {
  card_value: number
  prior: number
  visit_count: number
  mean_value: number[]
  child: DecisionNode | null
}

export interface ChanceNode {
  kind: 'chance'
  edges: ChanceEdge[]
}

export type TreeNode = DecisionNode | ChanceNode

/** Keyed by simulation count, as a string (the JSON keys `dump_mcts_tree.py` writes). */
export type SnapshotMap = Record<string, TreeNode>
