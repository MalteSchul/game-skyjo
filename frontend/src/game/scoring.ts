/** Mirrors `domain.engine._score_and_close_round`'s doubling rule: a round's
 * finisher gets their round score doubled unless it's the table's sole
 * lowest. `RoundResultOut.scores` (from the wire, via `MatchStateOut.
 * round_scores` / `round_history`) is always the raw, undoubled per-round
 * tally - only `total_scores` ever reflects the double - so this is the only
 * way the UI can tell whether it applied. `finisher === null` (a
 * `force_close_round` outcome - see that function's docstring) never
 * doubles, since nobody actually finished the round. */
export function finisherWasDoubled(scores: number[], finisher: number | null): boolean {
  if (finisher === null) return false
  const lowest = Math.min(...scores)
  const soleLowest = scores[finisher] === lowest && scores.filter((s) => s === lowest).length === 1
  return !soleLowest
}
