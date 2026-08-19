"""Plays one match to completion under MCTS, recording (state, N_act, pi)
at every decision and back-filling the true final ranks once it ends.

`round_over` is passed through automatically (same non-decision transition
`mcts._advance_state` uses), so no replay sample is ever recorded for it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from skyjo.domain.engine import GameState, apply_action, start_next_round
from skyjo.rl.action_space import pi_to_vector
from skyjo.rl.mcts import (
    DEFAULT_C_PUCT,
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DIRICHLET_EPSILON,
    EvaluateFn,
    run_mcts,
    sample_action,
    visit_distribution,
)

DEFAULT_MAX_STEPS = 5000


@dataclass(frozen=True)
class ReplaySample:
    # pi/y are excluded from eq/hash: numpy arrays don't support the boolean
    # `==` a dataclass-generated __eq__ needs, so two non-identical samples
    # would raise ValueError ("truth value of an array...") on comparison.
    state: GameState
    n_act: int
    pi: np.ndarray = field(compare=False)  # shape (ACTION_SPACE_SIZE,)
    y: np.ndarray = field(compare=False)  # shape (n_act,) int64, final rank per player (0 = best)


def final_ranks(total_scores: Sequence[int]) -> list[int]:
    """Integer rank per player (0 = best); ties broken by lower player index.

    A hard label, unlike MCTS's fractional tie handling for search values -
    training needs one categorical target class per row, not a continuous one.
    """
    order = sorted(range(len(total_scores)), key=lambda i: (total_scores[i], i))
    ranks = [0] * len(total_scores)
    for rank, player in enumerate(order):
        ranks[player] = rank
    return ranks


def generate_episode(
    initial_state: GameState,
    evaluate: EvaluateFn,
    *,
    num_simulations: int,
    tau_schedule: Callable[[int], float] | float = 1.0,
    c_puct: float = DEFAULT_C_PUCT,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON,
    rng: np.random.Generator | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[ReplaySample]:
    rng = rng if rng is not None else np.random.default_rng()
    state = initial_state
    n_act = len(state.boards)
    pending: list[tuple[GameState, np.ndarray]] = []

    for step in range(max_steps):
        if state.phase == "round_over":
            state = start_next_round(state)
            continue
        if state.phase == "game_over":
            break

        tau = tau_schedule(step) if callable(tau_schedule) else tau_schedule
        root = run_mcts(
            state,
            evaluate,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            rng=rng,
        )
        pi = visit_distribution(root, tau=tau)
        pending.append((state, pi_to_vector(pi)))

        action = sample_action(pi, rng)
        state = apply_action(state, action)
    else:
        raise RuntimeError(f"generate_episode: match did not reach game_over within {max_steps} steps")

    y = np.asarray(final_ranks(state.total_scores), dtype=np.int64)
    return [ReplaySample(state=s, n_act=n_act, pi=pi_vector, y=y) for s, pi_vector in pending]
