// Mirrors backend/src/skyjo/api/schemas.py — the backend is the source of truth for rules
// and wire shapes; this file only describes what already crosses the wire.

export type Phase = 'initial_flip' | 'awaiting_draw' | 'awaiting_placement' | 'round_over' | 'game_over'

export type ActionTypeName = 'flip_initial' | 'draw_stock' | 'draw_discard' | 'place' | 'discard_and_reveal'

export type PlayerTypeName = 'human' | 'random_bot' | 'thinking_bot' | 'heuristic_bot' | 'mcts_bot'

export type MatchStatus = 'idle' | 'thinking'

export interface CardOut {
  value: number | null
  face_up: boolean
}

export interface BoardOut {
  cards: (CardOut | null)[]
}

export interface ActionOut {
  type: ActionTypeName
  position: number | null
}

export interface MatchStateOut {
  match_id: string
  phase: Phase
  boards: BoardOut[]
  player_names: string[]
  player_types: PlayerTypeName[]
  stock_count: number
  discard_top: number | null
  current_player: number
  drawn_card: number | null
  finisher: number | null
  players_awaiting_final_turn: number[]
  round_scores: number[] | null
  total_scores: number[]
  target_score: number
  legal_actions: ActionOut[]
  status: MatchStatus
  thinking_player: number | null
  thinking_progress: number | null
}

export interface NewMatchRequest {
  player_count: number
  seed?: number
  player_names?: string[]
  player_types?: PlayerTypeName[]
  player_mcts_models?: (string | null)[]
  player_mcts_num_simulations?: (number | null)[]
}

export interface ActionRequest {
  type: ActionTypeName
  position?: number
}

export type HistoryEdgeKind = 'root' | 'action' | 'next_round'

export interface HistoryEdgeOut {
  kind: HistoryEdgeKind
  action_type: ActionTypeName | null
  position: number | null
}

export interface HistoryNodeOut {
  node_id: string
  parent_id: string | null
  seq: number
  round_index: number
  actor: number | null
  current_player: number
  phase: Phase
  edge: HistoryEdgeOut
  has_mcts_tree: boolean
}

export interface MatchHistoryOut {
  head_id: string
  nodes: HistoryNodeOut[]
}
