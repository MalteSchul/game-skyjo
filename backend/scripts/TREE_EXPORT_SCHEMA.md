# MCTS tree export schema

Every key produced by `rl.tree_export.tree_to_dict` (used by
`scripts/dump_mcts_tree.py`). There are two node shapes - `"kind": "decision"`
and `"kind": "chance"` - each with its own edge shape.

## Fixed vs. changing across simulations

Comparing two snapshots of the *same* node/edge (e.g. N=1 vs N=200 from one
`--sim-points` run) - which keys can differ, and which never do:

| Key | Behavior across simulations |
|---|---|
| `kind`, `current_player`, `phase`, `is_terminal`, `action`, `card_value` | **Always fixed.** Structural facts about the position/action, set once when the node/edge is created, never touched again. |
| `value`, `rank_probs`, `points_pred`, `prior_before_noise` | **Fixed once the node exists.** Captured from a single `evaluate()` call at first expansion (or `null` if the node hasn't been expanded yet in this snapshot). A node created later (in a higher-N snapshot) has its own fixed value; an already-existing node's doesn't change. |
| `prior` | **Fixed after root noise is applied**, which happens once, before simulation 1 - so for every snapshot at N>=1 it's already settled. Only ever differs from `prior_before_noise` on root edges. |
| `visit_count`, `mean_value`, `q`, `u`, `puct_score` | **Change every simulation** (for edges actually visited). `visit_count` only grows; `mean_value`/`q` refine toward the true backed-up average; `u` shrinks as `visit_count` grows relative to the parent's total; `puct_score` moves with both. |

So: everything about *what a node/edge represents* is fixed forever; everything about *how much the search trusts it* keeps changing as long as it keeps getting visited.

## Selection: two different mechanisms, don't confuse them

**During each simulation** (`mcts._select_edge`): a plain deterministic
argmax of `puct_score` (`q + u`) over a node's edges - not chance-weighted,
not sampled. The edge with the single highest `puct_score` gets it, every
time. Comparing `puct_score` across a snapshot's edges shows exactly what
would be picked *at that instant*, but the next simulation may pick
differently since `visit_count`/`mean_value` (and therefore `q`/`u`) just
changed. `prior` only matters here indirectly, through `u`'s exploration
term - it is not itself a "probability of being picked."

**After all simulations finish** ("the answer"), `puct_score` is never
consulted again - two different rules exist depending on why the search ran:

- **Live play** (`bots.mcts_bot.MctsBot.choose_action`): deterministic
  argmax of `visit_count` (ties broken randomly). "Most-visited wins,"
  full stop.
- **Self-play training data** (`mcts.visit_distribution` +
  `mcts.sample_action`): `visit_count` is turned into a probability
  distribution (`count ** (1/tau)`, normalized) and an action is *actually*
  sampled from it - the one place in this pipeline a genuinely
  chance-weighted pick happens, and it's weighted by visit share, not by
  `puct_score`.

## Decision node (`"kind": "decision"`)

A point where the acting player picks an `Action`. PUCT selects among its
`edges`.

| Key | Short | Long |
|---|---|---|
| `kind` | `"decision"` | Discriminates this node from a chance node - always the literal string `"decision"` here. |
| `current_player` | Whose turn | Index of the player who acts at this node (`GameState.current_player`). PUCT reads *this* player's own component out of every child edge's `mean_value` - see `q` below. |
| `phase` | Game phase | The engine's phase string at this node (`initial_flip`, `awaiting_draw`, `awaiting_placement`, `round_over`, `game_over`). |
| `is_terminal` | Game over? | `true` only for a `game_over` node - it has no edges, nothing left to search. |
| `visit_count` | Simulations through here | Sum of `visit_count` across this node's own edges - how many simulations have passed through this exact node so far. |
| `value` | Network's raw prediction | The `evaluate(state)` output captured **once**, the instant this node was first expanded - a per-player utility vector. Frozen forever after that: it is *not* re-evaluated on later visits, and it is *not* the same thing as the Monte-Carlo-refined `q` on the edges leading here. `null` for a node the search never reached. |
| `rank_probs` | Predicted finishing-rank distribution | `rank_probs[i][r]` = P(player i finishes at rank r), the un-reduced prediction `value` is a fixed linear weighting of (`value[i] = sum_r rank_probs[i][r] * (1 - 2r/(n-1))`, see `AlphaZeroNet.forward`). Only populated when the run passed a `rank_probs_by_state` lookup to `tree_to_dict` (`scripts/dump_mcts_tree.py --network` does this automatically) - `null` otherwise, and always `null` for the uniform stand-in, which has no such concept. |
| `points_pred` | Predicted final score (auxiliary head) | `points_pred[i]` = player i's predicted final total score, normalized by `target_score` - an auxiliary regression head trained alongside `rank_probs` (see `AlphaZeroNet.points_head`), giving a magnitude signal (`points_pred`) `rank_probs` can't express on its own (it's purely ordinal). Not consumed by search - `value`/`rank_probs` are what MCTS actually backs up. Only populated when the run passed a `points_pred_by_state` lookup to `tree_to_dict` (`scripts/dump_mcts_tree.py --network` does this automatically) - `null` otherwise. |
| `edges` | Legal actions from here | List of decision edges (see below), sorted by `visit_count` descending - so the most-explored line reads first. Equivalent actions (e.g. every position in the very first `initial_flip` decision) are collapsed onto one shared edge, so this can have fewer entries than the raw legal-action count. |

## Decision edge (one entry in a decision node's `edges`)

One legal `Action` out of a decision node, with everything PUCT used (or
would use) to score it.

| Key | Short | Long |
|---|---|---|
| `action` | `{type, position}` | The `Action` this edge represents - `type` is the `ActionType` name (`DRAW_STOCK`, `DRAW_DISCARD`, `PLACE`, `FLIP_INITIAL`, `DISCARD_AND_REVEAL`), `position` is the board slot involved (`null` when the action type doesn't need one). |
| `prior` | Prior used in search | The probability PUCT actually used for this action - the network's raw prior, mixed with Dirichlet root noise if this is a root edge and noise was enabled (`run_mcts(..., add_root_noise=True)`, the default). Equal to `prior_before_noise` everywhere noise wasn't applied (i.e. every non-root edge). |
| `prior_before_noise` | Prior before noise | The evaluator's un-perturbed prior for this action, before any root-noise mixing. Compare against `prior` to see how much noise moved it; equal to `prior` if this edge never got noise. |
| `visit_count` | Times selected | How many simulations picked this edge during selection. |
| `mean_value` | Average outcome (Q) | `value_sum / visit_count` per player - the Monte-Carlo average of every simulation's backed-up outcome through this edge. This is what actually improves as search runs longer; `[0, 0, ...]` if `visit_count` is 0 (never visited). |
| `q` | This edge's Q, from the acting player's view | `mean_value[current_player]` of the node this edge belongs to - the single number PUCT compares against other edges' `q` at this node. |
| `u` | Exploration bonus | `c_puct * prior * sqrt(parent.visit_count) / (1 + visit_count)` - rewards under-visited, high-prior edges. Shrinks toward 0 as `visit_count` grows relative to the parent's total. Computed fresh at export time from the snapshot's own numbers - it's not a value the search itself stores, since it changes on every simulation. |
| `puct_score` | `q + u` | The full PUCT selection score at the moment of this snapshot. The edge with the highest `puct_score` is the one `_select_edge` would pick next - but recomputing this off a snapshot only reflects *that instant*; it changes as more simulations run. |
| `child` | What this action leads to | The child node/chance-node this edge points to (see below), or `null` if never visited, or if `max_depth` truncated the export before reaching it. |

## Chance node (`"kind": "chance"`)

Sits between a reveal-triggering edge (e.g. `DRAW_STOCK`, where the drawn
card's value isn't yet known) and the decision node that follows once the
card's value is resolved. Not selected via PUCT - sampled by how many of
each value are still unknown to a real player, so it has no `current_player`,
`phase`, or `value` of its own.

| Key | Short | Long |
|---|---|---|
| `kind` | `"chance"` | Discriminates this node from a decision node - always `"chance"` here. |
| `edges` | Possible revealed values | List of chance edges (see below), sorted by `visit_count` descending. |

## Chance edge (one entry in a chance node's `edges`)

One possible card value the reveal could resolve to.

| Key | Short | Long |
|---|---|---|
| `card_value` | The revealed card value | The specific card value this branch represents (e.g. `4`, `12`), not an `Action` - chance edges are keyed by outcome, not by choice. Named `card_value`, not `value`, so it can't be confused with a decision node's `value` (the network's predicted utility) one level away in the same tree. |
| `prior` | Chance of this value | This value's share of the still-unknown cards at the moment this branch was first created (`count of this value / total unknown`) - a fixed probability, not touched by root noise. |
| `visit_count` | Times sampled | How many simulations resampled and landed on this exact value (repeat visits to the same value reuse the same cached child; different sampled values grow sibling branches). |
| `mean_value` | Average outcome | Same idea as a decision edge's `mean_value` - the Monte-Carlo average of backed-up outcomes through this specific revealed value. |
| `child` | The decision node once resolved | The decision `MCTSNode` reached once this value is revealed, or `null` if unvisited/truncated. Chance edges have no `q`/`u`/`puct_score` - there's no PUCT selection here, just probability-weighted sampling. |
