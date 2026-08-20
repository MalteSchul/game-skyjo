"""Plays one match to completion under MCTS, recording (state, N_act, pi)
at every decision and back-filling the true final ranks once it ends.

`round_over` is passed through automatically (same non-decision transition
`mcts._advance_state` uses), so no replay sample is ever recorded for it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from skyjo.domain.engine import Action, GameState, apply_action, start_next_round
from skyjo.domain.observation import Turn
from skyjo.rl.action_space import pi_to_vector
from skyjo.rl.mcts import (
    DEFAULT_C_PUCT,
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DIRICHLET_EPSILON,
    BatchEvaluateFn,
    EvaluateFn,
    run_mcts,
    run_mcts_batch,
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
            Turn.from_state(state),
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


def generate_episodes_batch(
    initial_states: Sequence[GameState],
    evaluate_batch: BatchEvaluateFn,
    *,
    num_simulations: int,
    tau_schedule: Callable[[int], float] | float = 1.0,
    c_puct: float = DEFAULT_C_PUCT,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON,
    rngs: Sequence[np.random.Generator] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[list[ReplaySample]]:
    """Batched sibling of `generate_episode`: plays `len(initial_states)`
    games to completion concurrently instead of one after another, using
    `run_mcts_batch` for every decision round so every still-in-progress
    game's leaf evaluations get batched into one evaluator call - see
    `rl.mcts.run_mcts_batch`'s docstring for why. Each game's own play is
    otherwise identical to `generate_episode`: same tau schedule, same
    action-sampling, same replay-sample recording, and finished games simply
    stop contributing to later rounds' batches (no refill - a game that ends
    early just shrinks the batch for the remaining games' rounds).

    Returns one list of `ReplaySample`s per game, in `initial_states` order.

    `max_steps` bounds the number of decision *rounds* (each round advances
    every still-active game by exactly one decision), not raw loop iterations
    the way `generate_episode`'s `max_steps` does - a `round_over` transition
    is resolved before it can consume a round. Immaterial in practice given
    `DEFAULT_MAX_STEPS`'s size, but not byte-for-byte the same bound.
    """
    n = len(initial_states)
    if n == 0:
        return []
    rngs = list(rngs) if rngs is not None else [np.random.default_rng() for _ in range(n)]
    if len(rngs) != n:
        raise ValueError("generate_episodes_batch: rngs must be the same length as initial_states")

    states = list(initial_states)
    n_acts = [len(s.boards) for s in states]
    pending: list[list[tuple[GameState, np.ndarray]]] = [[] for _ in range(n)]
    finished = [False] * n
    steps = [0] * n

    for _ in range(max_steps):
        for i in range(n):
            while not finished[i] and states[i].phase == "round_over":
                states[i] = start_next_round(states[i])
            if not finished[i] and states[i].phase == "game_over":
                finished[i] = True

        active = [i for i in range(n) if not finished[i]]
        if not active:
            break

        turns = [Turn.from_state(states[i]) for i in active]
        taus = [tau_schedule(steps[i]) if callable(tau_schedule) else tau_schedule for i in active]

        roots = run_mcts_batch(
            turns,
            evaluate_batch,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            rngs=[rngs[i] for i in active],
        )

        for slot, i in enumerate(active):
            pi = visit_distribution(roots[slot], tau=taus[slot])
            pending[i].append((states[i], pi_to_vector(pi)))
            action = sample_action(pi, rngs[i])
            states[i] = apply_action(states[i], action)
            steps[i] += 1
    else:
        unfinished = sum(1 for f in finished if not f)
        raise RuntimeError(
            f"generate_episodes_batch: {unfinished} of {n} games did not reach game_over "
            f"within {max_steps} decision rounds"
        )

    results = []
    for i in range(n):
        y = np.asarray(final_ranks(states[i].total_scores), dtype=np.int64)
        results.append([ReplaySample(state=s, n_act=n_acts[i], pi=pi_vector, y=y) for s, pi_vector in pending[i]])
    return results


def generate_bot_episode(
    initial_state: GameState,
    choose_action: Sequence[Callable[[Turn], Action]],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[ReplaySample]:
    """Plays one game to completion with `choose_action[i]` deciding player
    i's moves directly - e.g. `HeuristicBot(seed=s).choose_action` - instead
    of MCTS, recording a one-hot `pi` on each bot's actual choice. No search,
    no network evaluator, so games are cheap and always finish: a fast,
    reliable source of real win/loss outcomes (`y`, exactly as in
    `generate_episode`) to warm-start the value head on before switching to
    real MCTS self-play, without needing to tune MCTS-specific knobs
    (`num_simulations`, `max_steps`, tau) just to get a first dataset.

    Takes a plain callable per player rather than the `Bot` protocol so
    `skyjo.rl` doesn't have to depend on `skyjo.bots` - any bot's
    `choose_action` method already satisfies `Callable[[Turn], Action]` when
    called positionally (its `report_progress` parameter is keyword-only
    with a default).
    """
    state = initial_state
    n_act = len(state.boards)
    pending: list[tuple[GameState, np.ndarray]] = []

    for _ in range(max_steps):
        if state.phase == "round_over":
            state = start_next_round(state)
            continue
        if state.phase == "game_over":
            break

        turn = Turn.from_state(state)
        action = choose_action[turn.acting_player](turn)
        pending.append((state, pi_to_vector({action: 1.0})))
        state = apply_action(state, action)
    else:
        raise RuntimeError(f"generate_bot_episode: match did not reach game_over within {max_steps} steps")

    y = np.asarray(final_ranks(state.total_scores), dtype=np.int64)
    return [ReplaySample(state=s, n_act=n_act, pi=pi_vector, y=y) for s, pi_vector in pending]
