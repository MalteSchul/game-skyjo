import { describe, expect, it } from 'vitest'
import { GameReplayParseError, parseGameReplay } from './gameReplayParse'
import type { DecisionReplay } from './types'

const DECISION: DecisionReplay = {
  step: 0,
  actor_seat: 0,
  actor_name: 'p0',
  phase: 'awaiting_draw',
  total_scores: [0, 0],
  board_state: {
    boards: [{ cards: [] }, { cards: [] }],
    stock: [],
    discard: [],
    current_player: 0,
    drawn_card: null,
    finisher: null,
    players_awaiting_final_turn: [],
    round_scores: null,
    total_scores: [0, 0],
    phase: 'awaiting_draw',
    reshuffle_seed: 1,
    target_score: 100,
  },
  raw_policy_priors: [{ action: { type: 'DRAW_STOCK', position: null }, prior: 1 }],
  raw_prior_favorite: { type: 'DRAW_STOCK', position: null },
  raw_rank_probs: [[0.5, 0.5], [0.5, 0.5]],
  raw_points_pred: [0, 0],
  mcts_num_simulations_requested: 10,
  mcts_visit_counts: [{ action: { type: 'DRAW_STOCK', position: null }, visit_count: 10 }],
  mcts_root_value: [0, 0],
  reused_tree_visits: 0,
  chosen_action: { type: 'DRAW_STOCK', position: null },
  search_overrode_prior: false,
}

const REPLAY = {
  schema_version: 1,
  seat_names: ['a', 'b'],
  checkpoint_paths: ['a.pt', 'b.pt'],
  seed: 0,
  num_simulations: 10,
  c_puct: 1.5,
  final_total_scores: [80, 95],
  final_ranks: [0, 1],
  winner_name: 'a',
  rounds_played: 1,
  decisions: [DECISION],
}

describe('parseGameReplay', () => {
  it('accepts a well-formed game replay (happy path)', () => {
    expect(parseGameReplay(REPLAY)).toEqual(REPLAY)
  })

  it('throws for an object with no decisions array (bad path)', () => {
    expect(() => parseGameReplay({ ...REPLAY, decisions: undefined })).toThrow(GameReplayParseError)
  })

  it('throws for an empty decisions array (bad path)', () => {
    expect(() => parseGameReplay({ ...REPLAY, decisions: [] })).toThrow(GameReplayParseError)
  })

  it('throws for decisions that are not replay-shaped (bad path)', () => {
    expect(() => parseGameReplay({ ...REPLAY, decisions: [{ note: 'not a decision' }] })).toThrow(GameReplayParseError)
  })

  it('throws for primitives, arrays, and null rather than crashing (bad path)', () => {
    expect(() => parseGameReplay(null)).toThrow(GameReplayParseError)
    expect(() => parseGameReplay(42)).toThrow(GameReplayParseError)
    expect(() => parseGameReplay([REPLAY])).toThrow(GameReplayParseError)
  })
})
