"""Bridges `AlphaZeroNet` to MCTS's `EvaluateFn` signature: one `GameState`
in, `(priors over the collapsed set of legal actions, utility vector)` out -
see `domain.action_equivalence`: provably-equivalent actions (e.g. every
FLIP_INITIAL position on a fresh board) share one representative, so the
policy head never has to spread probability mass, or learn a target of zero,
across options that can't currently differ.

Also hosts `evaluate_vs_heuristic`, a diagnostic harness that plays a net
against `HeuristicBot` and reports win rate/rank/points - the cheap "are we
still improving" signal `rl.loop.run_training_loop` checks periodically
during self-play training. It builds MCTS search directly (`rl.mcts.run_mcts`
+ `greedy_action`) rather than going through `bots.mcts_bot.MctsBot`: that
module imports this one to get its evaluator, so importing it back here would
be circular. `HeuristicBot` is imported lazily inside `_play_one_eval_game`
for the same reason: `skyjo.bots`' own `__init__.py` eagerly imports
`bots.factory`, which imports `bots.mcts_bot` - so even a top-level `from
skyjo.bots.heuristic_bot import HeuristicBot` here would trigger that same
cycle by way of the package `__init__`, despite `heuristic_bot.py` itself
having no dependency on this module.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from skyjo.domain.action_equivalence import distinct_actions
from skyjo.domain.engine import (
    Action,
    GameState,
    apply_action,
    force_close_round,
    new_match,
    start_next_round,
)
from skyjo.domain.observation import Turn
from skyjo.rl.action_space import ACTION_SPACE_SIZE, index_to_action
from skyjo.rl.encoding import encode_state
from skyjo.rl.mcts import (
    DEFAULT_C_PUCT,
    BatchEvaluateFn,
    EvaluateFn,
    LeafResult,
    greedy_action,
    run_mcts,
)
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MAX_STEPS,
    DEFAULT_ROUND_MAX_STEPS,
    final_ranks,
)


def make_network_evaluator(
    net: AlphaZeroNet,
    device: str | torch.device = "cpu",
    *,
    rank_probs_sink: MutableMapping[GameState, np.ndarray] | None = None,
) -> EvaluateFn:
    """`rank_probs_sink`, if given, is filled in as a side effect of every
    `evaluate` call: `sink[state]` becomes that state's `rank_probs[i, r]` =
    P(player i finishes at rank r), the same forward pass this function
    already runs to get `value` (`value` is in fact just `rank_probs`
    reduced by a fixed linear weighting - see `AlphaZeroNet.forward`'s
    docstring - so this is strictly more information from work already
    being done, not an extra network call).

    Exists only for diagnostics (`scripts/dump_mcts_tree.py --network`) - training/self-play
    never pass it, so `EvaluateFn`'s contract for every other caller is
    untouched, and the `.cpu().numpy()` conversion for `rank_probs` doesn't
    even run when no sink is given.
    """
    net.to(device)
    net.eval()

    def evaluate(
        state: GameState, *, turn: Turn | None = None
    ) -> tuple[dict[Action, float], np.ndarray]:
        encoding = encode_state(state, legal_actions=None if turn is None else distinct_actions(turn))
        with torch.no_grad():
            features = torch.from_numpy(encoding.features).unsqueeze(0).to(device)
            mask = torch.from_numpy(encoding.legal_action_mask).unsqueeze(0).to(device)
            active_count = torch.tensor([encoding.active_count], dtype=torch.long, device=device)
            policy_probs, rank_probs, utility = net(features, mask, active_count)

        policy_probs = policy_probs[0].cpu().numpy()
        priors = {
            index_to_action(i): float(policy_probs[i])
            for i in range(ACTION_SPACE_SIZE)
            if encoding.legal_action_mask[i]
        }
        value = utility[0, : encoding.active_count].cpu().numpy()
        if rank_probs_sink is not None:
            n = encoding.active_count
            rank_probs_sink[state] = rank_probs[0, :n, :n].cpu().numpy()
        return priors, value

    return evaluate


def make_batch_network_evaluator(
    net: AlphaZeroNet,
    device: str | torch.device = "cpu",
) -> BatchEvaluateFn:
    """Batched sibling of `make_network_evaluator`: encodes every given state
    and runs exactly one forward pass over all of them, regardless of how
    many there are - a batch-of-N call is far cheaper per-state than N
    batch-of-1 calls, which is the entire reason this exists (see
    `rl.mcts.run_mcts_batch`, the intended caller). Returns results in the
    same order as `states`; `states=[]` short-circuits without touching the
    network at all.
    """
    net.to(device)
    net.eval()

    def evaluate_batch(
        states: Sequence[GameState], *, turns: Sequence[Turn] | None = None
    ) -> list[LeafResult]:
        if not states:
            return []

        if turns is None:
            encodings = [encode_state(state) for state in states]
        else:
            encodings = [
                encode_state(state, legal_actions=distinct_actions(turn))
                for state, turn in zip(states, turns, strict=True)
            ]
        with torch.no_grad():
            features = torch.from_numpy(np.stack([e.features for e in encodings])).to(device)
            mask = torch.from_numpy(np.stack([e.legal_action_mask for e in encodings])).to(device)
            active_count = torch.tensor(
                [e.active_count for e in encodings], dtype=torch.long, device=device
            )
            policy_probs, _rank_probs, utility = net(features, mask, active_count)

        policy_probs = policy_probs.cpu().numpy()
        utility = utility.cpu().numpy()

        results: list[LeafResult] = []
        for i, encoding in enumerate(encodings):
            priors = {
                index_to_action(j): float(policy_probs[i, j])
                for j in range(ACTION_SPACE_SIZE)
                if encoding.legal_action_mask[j]
            }
            results.append((priors, utility[i, : encoding.active_count]))
        return results

    return evaluate_batch


@dataclass(frozen=True)
class HeuristicEvalResult:
    games_played: int
    win_rate: float  # fraction of games the net finished rank 0 (best)
    avg_rank: float  # 0 (best) .. player_count - 1 (worst)
    avg_points: float  # the net's mean total_scores at game end


def _play_one_eval_game(
    evaluate: EvaluateFn,
    *,
    net_seat: int,
    seed: int,
    num_simulations: int,
    c_puct: float,
    max_steps: int,
    round_max_steps: int = DEFAULT_ROUND_MAX_STEPS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> tuple[int, int]:
    """Returns (net's final rank, net's final points) for one 2-player game
    of net-driven MCTS vs `HeuristicBot`, with `net_seat` controlling which
    seat the network plays (so callers can alternate seats across a batch of
    games, matching `scripts/tournament.py`'s approach).

    Same round_max_steps/max_rounds safety valves as `rl.selfplay.generate_episode`
    (`domain.engine.force_close_round`) - a low-simulation or early-untrained
    net can get the same non-progressing tie-lock here as in self-play.
    """
    from skyjo.bots.heuristic_bot import (
        HeuristicBot,  # deferred to dodge a circular import, see module docstring
    )

    heuristic_seat = 1 - net_seat
    heuristic_bot = HeuristicBot(seed=seed)
    rng = np.random.default_rng(seed)
    state = new_match(player_count=2, seed=seed)
    round_step = 0
    round_count = 0
    for _ in range(max_steps):
        if state.phase == "round_over":
            round_count += 1
            if round_count >= max_rounds:
                break
            state = start_next_round(state)
            round_step = 0
            continue
        if state.phase == "game_over":
            break
        if round_step >= round_max_steps:
            state = force_close_round(state)
            round_step = 0
            continue
        turn = Turn.from_state(state)
        if turn.acting_player == heuristic_seat:
            action = heuristic_bot.choose_action(turn)
        else:
            root = run_mcts(
                turn, evaluate, num_simulations=num_simulations, c_puct=c_puct, add_root_noise=False, rng=rng
            )
            action = greedy_action(root, rng)
        state = apply_action(state, action)
        round_step += 1
    else:
        raise RuntimeError(f"_play_one_eval_game: seed={seed} did not finish within {max_steps} steps")
    return final_ranks(state.total_scores)[net_seat], state.total_scores[net_seat]


def evaluate_vs_heuristic(
    net: AlphaZeroNet,
    num_games: int,
    *,
    num_simulations: int,
    c_puct: float = DEFAULT_C_PUCT,
    max_steps: int = DEFAULT_MAX_STEPS,
    round_max_steps: int = DEFAULT_ROUND_MAX_STEPS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    seed: int = 0,
) -> HeuristicEvalResult:
    """Plays `net`-driven MCTS against `HeuristicBot` for `num_games`
    2-player games, alternating which seat the network takes so neither
    always moves first - same idea as `scripts/tournament.py`'s pairings.

    Uses a fixed seed sequence derived from `seed`: repeated calls with the
    same `seed` replay the exact same game set regardless of the net's
    weights, so results across training iterations reflect the net actually
    improving (or not), not a different random sample of games each time.
    """
    if num_games <= 0:
        raise ValueError("evaluate_vs_heuristic: num_games must be > 0")

    evaluate = make_network_evaluator(net)
    ranks: list[int] = []
    points: list[int] = []
    for g in range(num_games):
        rank, game_points = _play_one_eval_game(
            evaluate,
            net_seat=g % 2,
            seed=seed + g,
            num_simulations=num_simulations,
            c_puct=c_puct,
            max_steps=max_steps,
            round_max_steps=round_max_steps,
            max_rounds=max_rounds,
        )
        ranks.append(rank)
        points.append(game_points)

    return HeuristicEvalResult(
        games_played=num_games,
        win_rate=sum(1 for rank in ranks if rank == 0) / num_games,
        avg_rank=sum(ranks) / num_games,
        avg_points=sum(points) / num_games,
    )
