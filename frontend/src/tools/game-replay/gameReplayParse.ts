import type { DecisionReplay, GameReplay } from './types'

export class GameReplayParseError extends Error {}

const EXPECTED_SHAPE_MESSAGE =
  'Couldn\'t find a game replay in this file — expected the JSON from scripts/play_and_record_game.py: an object with a non-empty "decisions" array.'

function looksLikeDecision(value: unknown): value is DecisionReplay {
  if (typeof value !== 'object' || value === null) return false
  const v = value as Record<string, unknown>
  return (
    typeof v.step === 'number' &&
    typeof v.actor_seat === 'number' &&
    typeof v.board_state === 'object' &&
    v.board_state !== null &&
    Array.isArray(v.raw_policy_priors) &&
    Array.isArray(v.mcts_visit_counts)
  )
}

/** Accepts the object `game_record_to_dict` produces: a `decisions` array of
 * per-step records, plus game-level summary fields. Only checks enough of
 * the shape to render safely - not a full schema validator - throwing
 * `GameReplayParseError` with a message meant to be shown directly to the
 * person loading the file. */
export function parseGameReplay(data: unknown): GameReplay {
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    throw new GameReplayParseError(EXPECTED_SHAPE_MESSAGE)
  }
  const obj = data as Record<string, unknown>
  const decisions = obj.decisions
  if (!Array.isArray(decisions) || decisions.length === 0 || !decisions.every(looksLikeDecision)) {
    throw new GameReplayParseError(EXPECTED_SHAPE_MESSAGE)
  }
  return data as GameReplay
}
