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
having no dependency on this module. It does, however, reuse its search tree
across turns the same way `MctsBot` does live play - both share `rl.mcts
.advance_cached_root`'s tree-walk rather than duplicating it.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

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
from skyjo.rl.hidden_info import gamestate_from_turn
from skyjo.rl.mcts import (
    DEFAULT_C_PUCT,
    BatchEvaluateFn,
    EvaluateFn,
    LeafResult,
    MCTSNode,
    advance_cached_root,
    greedy_action,
    run_mcts,
    run_mcts_batch,
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
    points_pred_sink: MutableMapping[GameState, np.ndarray] | None = None,
) -> EvaluateFn:
    """`rank_probs_sink`, if given, is filled in as a side effect of every
    `evaluate` call: `sink[state]` becomes that state's `rank_probs[i, r]` =
    P(player i finishes at rank r), the same forward pass this function
    already runs to get `value` (`value` is in fact just `rank_probs`
    reduced by a fixed linear weighting - see `AlphaZeroNet.forward`'s
    docstring - so this is strictly more information from work already
    being done, not an extra network call). `points_pred_sink`, if given, is
    filled in the same way with `points_pred` - each player's predicted
    final normalized score (see `AlphaZeroNet.points_head`), the auxiliary
    head's own take on "how well is this player doing" rather than
    `rank_probs`'s ordinal one.

    Exists only for diagnostics (`scripts/dump_mcts_tree.py --network`) - training/self-play
    never pass either sink, so `EvaluateFn`'s contract for every other caller
    is untouched, and the `.cpu().numpy()` conversion for `rank_probs`/
    `points_pred` doesn't even run when no sink is given for it.
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
            policy_probs, rank_probs, utility, points_pred = net(features, mask, active_count)

        policy_probs = policy_probs[0].cpu().numpy()
        priors = {
            index_to_action(i): float(policy_probs[i])
            for i in range(ACTION_SPACE_SIZE)
            if encoding.legal_action_mask[i]
        }
        n = encoding.active_count
        perm = encoding.perm[:n]
        value_canonical = utility[0, :n].cpu().numpy()
        value = np.empty_like(value_canonical)
        value[perm] = value_canonical  # canonical slot -> absolute player id
        if rank_probs_sink is not None:
            rank_probs_canonical = rank_probs[0, :n, :n].cpu().numpy()  # [canonical slot, rank]
            rank_probs_abs = np.empty_like(rank_probs_canonical)
            rank_probs_abs[perm] = rank_probs_canonical
            rank_probs_sink[state] = rank_probs_abs
        if points_pred_sink is not None:
            points_pred_canonical = points_pred[0, :n].cpu().numpy()
            points_pred_abs = np.empty_like(points_pred_canonical)
            points_pred_abs[perm] = points_pred_canonical
            points_pred_sink[state] = points_pred_abs
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
            policy_probs, _rank_probs, utility, _points_pred = net(features, mask, active_count)

        policy_probs = policy_probs.cpu().numpy()
        utility = utility.cpu().numpy()

        results: list[LeafResult] = []
        for i, encoding in enumerate(encodings):
            priors = {
                index_to_action(j): float(policy_probs[i, j])
                for j in range(ACTION_SPACE_SIZE)
                if encoding.legal_action_mask[j]
            }
            n = encoding.active_count
            perm = encoding.perm[:n]
            value_canonical = utility[i, :n]
            value = np.empty_like(value_canonical)
            value[perm] = value_canonical  # canonical slot -> absolute player id
            results.append((priors, value))
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

    Reuses the net's search tree across its own turns exactly as
    `bots.mcts_bot.MctsBot` does live (via the same `rl.mcts
    .advance_cached_root`), advancing it through the heuristic opponent's
    moves too - a real single-line game is exactly the case reuse is safe
    for. `num_simulations` is kept a *target total* per decision rather than
    "always this many more": a reused root only runs the shortfall (`max(0,
    num_simulations - cached_root.visit_count)`), so a net decision never
    ends up backed by more total visits than a from-scratch search would've
    given it - the whole point being that this diagnostic's numbers stay
    governed by the same `num_simulations` knob whether or not reuse
    actually fires for a given decision, not that reuse is free to make
    later decisions in a game strictly more informed than earlier ones.
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
    cached_root: MCTSNode | None = None
    for _ in range(max_steps):
        if state.phase == "round_over":
            round_count += 1
            if round_count >= max_rounds:
                break
            state = start_next_round(state)
            round_step = 0
            cached_root = None  # a fresh round is an independent shuffle - never cacheable across it
            continue
        if state.phase == "game_over":
            break
        if round_step >= round_max_steps:
            state = force_close_round(state)
            round_step = 0
            cached_root = None  # force-closing bypasses the real transition the cache tracks
            continue
        turn = Turn.from_state(state)
        if turn.acting_player == heuristic_seat:
            action = heuristic_bot.choose_action(turn)
        else:
            if cached_root is not None and cached_root.state != gamestate_from_turn(turn):
                cached_root = None
            already_visited = cached_root.visit_count if cached_root is not None else 0
            root = run_mcts(
                turn,
                evaluate,
                num_simulations=max(0, num_simulations - already_visited),
                c_puct=c_puct,
                add_root_noise=False,
                rng=rng,
                reuse_root=cached_root,
            )
            action = greedy_action(root, rng)
            cached_root = root
        state = apply_action(state, action)
        round_step += 1
        cached_root = (
            None
            if state.phase in ("round_over", "game_over")
            else advance_cached_root(cached_root, turn, action, Turn.from_state(state))
        )
    else:
        raise RuntimeError(f"_play_one_eval_game: seed={seed} did not finish within {max_steps} steps")
    return final_ranks(state.total_scores)[net_seat], state.total_scores[net_seat]


def _play_eval_games_batch(
    evaluate_batch: BatchEvaluateFn,
    *,
    net_seats: Sequence[int],
    seeds: Sequence[int],
    num_simulations: int,
    c_puct: float,
    max_steps: int,
    round_max_steps: int,
    max_rounds: int,
) -> list[tuple[int, int]]:
    """Batched sibling of `_play_one_eval_game`: plays `len(seeds)` 2-player
    net-vs-`HeuristicBot` games concurrently, batching every decision round's
    net-controlled leaf evaluations into one `evaluate_batch` call via
    `run_mcts_batch` - mirrors `rl.selfplay.generate_episodes_batch`, the
    self-play sibling this is modeled on (same "resolve round_over/force-close
    before this round's turn, then batch the still-active decisions" shape).

    Unlike `_play_one_eval_game`, there's no cached-root tree reuse across a
    game's own turns: `run_mcts_batch` has no `reuse_root` support (batched
    self-play doesn't use it either - see `generate_episodes_batch`'s
    docstring), so every net decision here is a fresh root. That's the same
    throughput-over-reuse tradeoff batching already makes elsewhere, not a
    new one introduced here.

    `max_steps` bounds the number of decision *rounds* (each round advances
    every still-active game by exactly one decision), not raw loop
    iterations the way `_play_one_eval_game`'s `max_steps` does - same
    distinction `generate_episodes_batch` documents.

    Returns (net's final rank, net's final points) per game, in `seeds` order.
    """
    from skyjo.bots.heuristic_bot import (
        HeuristicBot,  # deferred to dodge a circular import, see module docstring
    )

    n = len(seeds)
    heuristic_bots = [HeuristicBot(seed=seed) for seed in seeds]
    rngs = [np.random.default_rng(seed) for seed in seeds]
    states = [new_match(player_count=2, seed=seed) for seed in seeds]
    finished = [False] * n
    round_steps = [0] * n
    round_counts = [0] * n

    for _ in range(max_steps):
        for i in range(n):
            if finished[i]:
                continue
            while True:
                if states[i].phase == "round_over":
                    round_counts[i] += 1
                    if round_counts[i] >= max_rounds:
                        finished[i] = True
                        break
                    states[i] = start_next_round(states[i])
                    round_steps[i] = 0
                    continue
                if states[i].phase == "game_over":
                    finished[i] = True
                    break
                if round_steps[i] >= round_max_steps:
                    states[i] = force_close_round(states[i])
                    round_steps[i] = 0
                    continue
                break

        active = [i for i in range(n) if not finished[i]]
        if not active:
            break

        turns = {i: Turn.from_state(states[i]) for i in active}
        net_active = [i for i in active if turns[i].acting_player == net_seats[i]]
        heuristic_active = [i for i in active if turns[i].acting_player != net_seats[i]]

        for i in heuristic_active:
            action = heuristic_bots[i].choose_action(turns[i])
            states[i] = apply_action(states[i], action)
            round_steps[i] += 1

        if net_active:
            roots = run_mcts_batch(
                [turns[i] for i in net_active],
                evaluate_batch,
                num_simulations=num_simulations,
                c_puct=c_puct,
                add_root_noise=False,
                rngs=[rngs[i] for i in net_active],
            )
            for i, root in zip(net_active, roots, strict=True):
                action = greedy_action(root, rngs[i])
                states[i] = apply_action(states[i], action)
                round_steps[i] += 1
    else:
        unfinished = sum(1 for f in finished if not f)
        raise RuntimeError(
            f"_play_eval_games_batch: {unfinished} of {n} games did not reach game_over "
            f"within {max_steps} decision rounds"
        )

    return [
        (final_ranks(states[i].total_scores)[net_seats[i]], states[i].total_scores[net_seats[i]])
        for i in range(n)
    ]


@dataclass(frozen=True)
class _EvalBatchJob:
    seeds: tuple[int, ...]
    net_seats: tuple[int, ...]
    num_simulations: int
    c_puct: float
    max_steps: int
    round_max_steps: int
    max_rounds: int


# Set by `_init_eval_worker` in a subprocess - see `evaluate_vs_heuristic`'s
# `workers > 1` path, mirrors `rl.loop`'s `_worker_evaluate_batch`.
_eval_worker_evaluate_batch: BatchEvaluateFn | None = None


def _init_eval_worker(state_dict: dict[str, torch.Tensor], network_kwargs: dict[str, Any]) -> None:
    global _eval_worker_evaluate_batch
    net = AlphaZeroNet(**network_kwargs)
    net.load_state_dict(state_dict)
    _eval_worker_evaluate_batch = make_batch_network_evaluator(net)


def _play_eval_batch_job(job: _EvalBatchJob) -> list[tuple[int, int]]:
    if _eval_worker_evaluate_batch is None:
        raise RuntimeError("_play_eval_batch_job: worker was not initialized via _init_eval_worker")
    return _play_eval_games_batch(
        _eval_worker_evaluate_batch,
        net_seats=job.net_seats,
        seeds=job.seeds,
        num_simulations=job.num_simulations,
        c_puct=job.c_puct,
        max_steps=job.max_steps,
        round_max_steps=job.round_max_steps,
        max_rounds=job.max_rounds,
    )


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
    workers: int = 0,
    eval_batch_size: int = 1,
    network_kwargs: dict[str, Any] | None = None,
) -> HeuristicEvalResult:
    """Plays `net`-driven MCTS against `HeuristicBot` for `num_games`
    2-player games, alternating which seat the network takes so neither
    always moves first - same idea as `scripts/tournament.py`'s pairings.

    Uses a fixed seed sequence derived from `seed`: repeated calls with the
    same `seed` replay the exact same game set regardless of the net's
    weights, so results across training iterations reflect the net actually
    improving (or not), not a different random sample of games each time.

    `eval_batch_size <= 1` (default) is the original path: one game at a
    time, each with its own tree-reuse across turns (`_play_one_eval_game`).
    `eval_batch_size > 1` groups games into batches of this many, played
    concurrently via `_play_eval_games_batch` so every decision round's net
    evaluations across the group become one batched network call instead of
    one per game - at the cost of no tree reuse within a game (see that
    function's docstring). `workers` shards those groups across processes
    exactly as `rl.loop`'s self-play does, requiring `network_kwargs` (the
    `AlphaZeroNet` constructor kwargs `net` was built with) so each worker
    can reconstruct the network from a plain state dict.
    """
    if num_games <= 0:
        raise ValueError("evaluate_vs_heuristic: num_games must be > 0")
    if eval_batch_size < 1:
        raise ValueError("evaluate_vs_heuristic: eval_batch_size must be >= 1")
    if workers < 0:
        raise ValueError("evaluate_vs_heuristic: workers must be >= 0")

    if eval_batch_size <= 1 and workers <= 1:
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
    else:
        seeds = [seed + g for g in range(num_games)]
        net_seats = [g % 2 for g in range(num_games)]
        jobs = [
            _EvalBatchJob(
                seeds=tuple(seeds[start : start + eval_batch_size]),
                net_seats=tuple(net_seats[start : start + eval_batch_size]),
                num_simulations=num_simulations,
                c_puct=c_puct,
                max_steps=max_steps,
                round_max_steps=round_max_steps,
                max_rounds=max_rounds,
            )
            for start in range(0, num_games, eval_batch_size)
        ]

        if workers <= 1:
            evaluate_batch = make_batch_network_evaluator(net)
            group_results = [
                _play_eval_games_batch(
                    evaluate_batch,
                    net_seats=job.net_seats,
                    seeds=job.seeds,
                    num_simulations=job.num_simulations,
                    c_puct=job.c_puct,
                    max_steps=job.max_steps,
                    round_max_steps=job.round_max_steps,
                    max_rounds=job.max_rounds,
                )
                for job in jobs
            ]
        else:
            if network_kwargs is None:
                raise ValueError("evaluate_vs_heuristic: network_kwargs is required when workers > 1")
            state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_eval_worker, initargs=(state_dict, network_kwargs)
            ) as pool:
                group_results = list(pool.map(_play_eval_batch_job, jobs))

        ranks = [rank for group in group_results for rank, _ in group]
        points = [game_points for group in group_results for _, game_points in group]

    return HeuristicEvalResult(
        games_played=num_games,
        win_rate=sum(1 for rank in ranks if rank == 0) / num_games,
        avg_rank=sum(ranks) / num_games,
        avg_points=sum(points) / num_games,
    )
