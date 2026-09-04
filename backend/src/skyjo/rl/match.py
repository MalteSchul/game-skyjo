"""Shared "decider" abstraction for playing batched games against a frozen
opponent - a checkpoint-backed net (searched via MCTS) or a simple bot
(`HeuristicBot`/`RandomBot`, no search). Used wherever something needs a
batched opponent that isn't the net currently being trained:
`scripts/tournament.py` (both sides are deciders), `rl.selfplay`'s
pool-opponent self-play (opponent side only - the net_seat side trains via
its own noisy, tau-sampled, sample-recording search and never goes through a
`Decider`), and this module's own `evaluate_vs_decider` (opponent side only,
mirroring `rl.evaluator.evaluate_vs_heuristic` but against a checkpoint
instead of a fixed heuristic).

Deliberately no root noise, no tau sampling, and greedy (most-visited) action
selection throughout - every caller here is either match play or a frozen
opponent's move in someone else's self-play, never the side whose own
training data is being recorded, so there's no reason to explore off the
search's own best line (see `rl.selfplay.generate_episode_vs_bot`/
`generate_episodes_batch_vs_decider` for the seat that does explore/record).
For the same reason, ties aren't widened back out across their full
equivalence class the way self-play's recorded seat does (`domain
.action_equivalence.tied_actions`) - that widening exists to keep training
data's real board layouts diverse, which doesn't matter for a side that
never produces a `ReplaySample`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from skyjo.bots.base import Bot
from skyjo.domain.engine import (
    Action,
    apply_action,
    force_close_round,
    new_match,
    start_next_round,
)
from skyjo.domain.observation import Turn
from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.mcts import (
    DEFAULT_C_PUCT,
    BatchEvaluateFn,
    greedy_action,
    run_mcts_batch,
)
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MAX_STEPS,
    DEFAULT_ROUND_MAX_STEPS,
    final_ranks,
)


@dataclass
class MctsDecider:
    """One shared network/evaluator for the whole batch, searched via
    `run_mcts_batch` so every still-active game's leaf evaluations land in a
    single network call per decision round."""

    evaluate_batch: BatchEvaluateFn
    num_simulations: int
    c_puct: float
    cap_root_lead: bool = False


@dataclass
class SimpleDecider:
    """No network, so no batching benefit - just a per-game bot instance
    (indexed the same way as the batch it was built for) called directly.
    `bots[i]` must correspond to game index `i` of whatever batch this
    decider is used with."""

    bots: list[Bot]


Decider = MctsDecider | SimpleDecider


def decider_from_net(
    net: AlphaZeroNet, *, num_simulations: int, c_puct: float = DEFAULT_C_PUCT, cap_root_lead: bool = False
) -> MctsDecider:
    """Wraps an in-memory net (as opposed to `build_decider`'s checkpoint-file
    loading) into an `MctsDecider` - the case where the caller already holds
    the net (e.g. the live net currently being trained) rather than a spec
    string naming where to load one from."""
    from skyjo.rl.evaluator import (
        make_batch_network_evaluator,  # deferred - see module docstring on why rl.evaluator is never imported at module load time here
    )

    return MctsDecider(
        evaluate_batch=make_batch_network_evaluator(net), num_simulations=num_simulations, c_puct=c_puct,
        cap_root_lead=cap_root_lead,
    )


def build_decider(
    spec: str,
    seeds: list[int],
    *,
    num_simulations: int,
    c_puct: float = DEFAULT_C_PUCT,
    network_kwargs: dict[str, Any] | None = None,
    cap_root_lead: bool = False,
) -> Decider:
    """`spec` is `"random"`, `"heuristic"`, or a checkpoint path (loaded into
    a fresh `AlphaZeroNet(**network_kwargs)`). `seeds` gives one per-game seed
    for a `random`/`heuristic` decider's bot instances (ignored for a
    checkpoint decider, which has no per-game state); its length must match
    whatever batch this decider will be used with.

    `HeuristicBot`/`RandomBot` are imported lazily here, not at module level,
    for the same circular-import reason `rl.evaluator._play_one_eval_game`
    does it - see `rl.evaluator`'s module docstring.
    """
    if spec == "random":
        from skyjo.bots.random_bot import RandomBot

        return SimpleDecider(bots=[RandomBot(seed=s) for s in seeds])
    if spec == "heuristic":
        from skyjo.bots.heuristic_bot import HeuristicBot

        return SimpleDecider(bots=[HeuristicBot(seed=s) for s in seeds])
    net = AlphaZeroNet(**(network_kwargs or {}))
    load_checkpoint(spec, net)
    return decider_from_net(net, num_simulations=num_simulations, c_puct=c_puct, cap_root_lead=cap_root_lead)


def decide_batch(
    decider: Decider, indices: list[int], turns: list[Turn], rngs: list[np.random.Generator]
) -> list[Action]:
    """`indices` are into the batch's own per-game arrays (e.g. `decider.bots`
    for a `SimpleDecider`); `turns`/`rngs` are already the subset for just
    these indices, in the same order."""
    if isinstance(decider, SimpleDecider):
        return [decider.bots[i].choose_action(turn) for i, turn in zip(indices, turns, strict=True)]
    roots = run_mcts_batch(
        turns,
        decider.evaluate_batch,
        num_simulations=decider.num_simulations,
        c_puct=decider.c_puct,
        add_root_noise=False,
        rngs=rngs,
        cap_root_lead=decider.cap_root_lead,
    )
    tie_break = "value" if decider.cap_root_lead else "random"
    return [greedy_action(root, rng, tie_break=tie_break) for root, rng in zip(roots, rngs, strict=True)]


def play_batch(
    decider_a: Decider,
    decider_b: Decider,
    seeds: list[int],
    forward: list[bool],
    max_steps: int,
    round_max_steps: int = DEFAULT_ROUND_MAX_STEPS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> list[tuple[tuple[int, int], tuple[int, int]] | None]:
    """Plays `len(seeds)` 2-player games concurrently - `forward[i]` says
    whether decider_a occupies seat 0 (True) or seat 1 (False) in game `i`.
    Returns one `(ranks, points)` per game, in `seeds` order, or `None` for a
    game that didn't reach `game_over` within `max_steps` decision rounds -
    isolated per game rather than aborting the whole batch.
    """
    n = len(seeds)
    states = [new_match(player_count=2, seed=s) for s in seeds]
    rngs = [np.random.default_rng(s) for s in seeds]
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

        turns = [Turn.from_state(states[i]) for i in active]
        seat_of_a = [0 if forward[i] else 1 for i in active]
        group_a = [
            (i, t, rngs[i]) for i, t, sa in zip(active, turns, seat_of_a, strict=True) if t.acting_player == sa
        ]
        group_b = [
            (i, t, rngs[i]) for i, t, sa in zip(active, turns, seat_of_a, strict=True) if t.acting_player != sa
        ]

        actions: dict[int, Action] = {}
        for decider, group in ((decider_a, group_a), (decider_b, group_b)):
            if not group:
                continue
            idxs = [g[0] for g in group]
            group_turns = [g[1] for g in group]
            group_rngs = [g[2] for g in group]
            for i, action in zip(idxs, decide_batch(decider, idxs, group_turns, group_rngs), strict=True):
                actions[i] = action

        for i in active:
            states[i] = apply_action(states[i], actions[i])
            round_steps[i] += 1

    results: list[tuple[tuple[int, int], tuple[int, int]] | None] = []
    for i in range(n):
        if not finished[i]:
            results.append(None)
            continue
        results.append((tuple(final_ranks(states[i].total_scores)), tuple(states[i].total_scores)))
    return results


@dataclass(frozen=True)
class MatchEvalResult:
    games_played: int
    win_rate: float  # fraction of games the net finished rank 0 (best)
    avg_rank: float  # 0 (best) .. player_count - 1 (worst)
    avg_points: float  # the net's mean total_scores at game end


def evaluate_vs_decider(
    net: AlphaZeroNet,
    opponent_checkpoint_path: str,
    num_games: int,
    *,
    num_simulations: int,
    opponent_num_simulations: int,
    c_puct: float = DEFAULT_C_PUCT,
    network_kwargs: dict[str, Any] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    round_max_steps: int = DEFAULT_ROUND_MAX_STEPS,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    seed: int = 0,
    batch_size: int = 1,
) -> MatchEvalResult:
    """Plays `net`-driven MCTS against a fixed checkpoint for `num_games`
    2-player games, alternating which seat the net takes - the
    checkpoint-opponent sibling of `rl.evaluator.evaluate_vs_heuristic`,
    for tracking progress against a real (and non-saturating, unlike a
    heuristic bot a trained net quickly starts beating every game) reference
    point instead.

    `opponent_checkpoint_path` must be a checkpoint file, not `"heuristic"`/
    `"random"` - those already have a dedicated, cheaper harness in
    `evaluate_vs_heuristic`; this one always builds an `MctsDecider` (no
    per-game bot state), so it can be built once and reused across every
    batch instead of rebuilding it (and reloading the checkpoint) per group.

    Unlike `evaluate_vs_heuristic`, there is no `workers` multiprocess
    sharding yet - every group runs in-process, one after another. Fine for
    the occasional (`eval_every`-gated) progress check this exists for; worth
    adding if this ever needs to scale to `evaluate_vs_heuristic`-sized
    `num_games`.
    """
    if num_games <= 0:
        raise ValueError("evaluate_vs_decider: num_games must be > 0")
    if batch_size < 1:
        raise ValueError("evaluate_vs_decider: batch_size must be >= 1")

    net_decider = decider_from_net(net, num_simulations=num_simulations, c_puct=c_puct)
    opponent_decider = build_decider(
        opponent_checkpoint_path,
        [],
        num_simulations=opponent_num_simulations,
        c_puct=c_puct,
        network_kwargs=network_kwargs,
    )
    if isinstance(opponent_decider, SimpleDecider):
        # ValueError, not TypeError: this is an invalid spec *value* ("heuristic"/
        # "random" instead of a checkpoint path), not a Python type mismatch.
        raise ValueError(  # noqa: TRY004
            "evaluate_vs_decider: opponent_checkpoint_path must be a checkpoint file, not 'heuristic'/'random' "
            "- use evaluate_vs_heuristic for a heuristic opponent"
        )

    ranks: list[int] = []
    points: list[int] = []
    for start in range(0, num_games, batch_size):
        group = list(range(start, min(start + batch_size, num_games)))
        seeds = [seed + g for g in group]
        net_seats = [g % 2 for g in group]
        forward = [net_seat == 0 for net_seat in net_seats]
        outcomes = play_batch(net_decider, opponent_decider, seeds, forward, max_steps, round_max_steps, max_rounds)
        for net_seat, outcome in zip(net_seats, outcomes, strict=True):
            if outcome is None:
                continue
            game_ranks, game_points = outcome
            ranks.append(game_ranks[net_seat])
            points.append(game_points[net_seat])

    if not ranks:
        raise RuntimeError(f"evaluate_vs_decider: all {num_games} games failed to finish within max_steps")

    played = len(ranks)
    return MatchEvalResult(
        games_played=played,
        win_rate=sum(1 for r in ranks if r == 0) / played,
        avg_rank=sum(ranks) / played,
        avg_points=sum(points) / played,
    )
