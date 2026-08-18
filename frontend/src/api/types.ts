// Mirrors backend/src/skyjo/api/schemas.py — the backend is the source of truth for rules
// and wire shapes; this file only describes what already crosses the wire.

export type Phase = 'initial_flip' | 'awaiting_draw' | 'awaiting_placement' | 'round_over' | 'game_over'

export type ActionTypeName = 'flip_initial' | 'draw_stock' | 'draw_discard' | 'place' | 'discard_and_reveal'

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
}

export interface NewMatchRequest {
  player_count: number
  seed?: number
  player_names?: string[]
}

export interface ActionRequest {
  type: ActionTypeName
  position?: number
}
