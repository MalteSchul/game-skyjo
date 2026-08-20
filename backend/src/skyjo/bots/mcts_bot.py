"""A bot backed by AlphaZero-style MCTS (`skyjo.rl.mcts`) instead of a fixed
strategy - it runs a real search, respecting exactly the same hidden-info
boundary a human sees (see `rl.hidden_info`), and picks the most-visited root
action.

The network every seat shares is built once per process by
`default_evaluator()` (see below), not per bot instance - constructing
`AlphaZeroNet` isn't free, and a fresh one per seat would also mean two
mcts_bot seats in the same match couldn't share warm state.

Previously, over a long enough real game, `rl.hidden_info`'s public/unknown
card bookkeeping could become inconsistent and trip the AssertionError in
`hidden_info._unknown_from_boards_and_discard` - root-caused to
`unknown_card_counts` never excluding the drawn card (already resolved to a
real value, but not yet in `boards`/`discard`) from the "still unknown" pool,
letting the same value get sampled again for another hidden card. Fixed by
excluding it (see `hidden_info.unknown_card_counts`).
"""

from __future__ import annotations

import functools
import os

import numpy as np

from skyjo.bots.base import ProgressReporter
from skyjo.domain.engine import Action
from skyjo.domain.observation import Turn
from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.mcts import DEFAULT_C_PUCT, EvaluateFn, run_mcts
from skyjo.rl.network import AlphaZeroNet

# If set, default_evaluator() loads this checkpoint (the format
# skyjo.rl.checkpoint.save_checkpoint writes, e.g. scripts/train_mcts.py's
# --checkpoint-dir/latest.pt) instead of using a freshly-initialized
# (untrained) network.
CHECKPOINT_PATH_ENV_VAR = "SKYJO_MCTS_CHECKPOINT_PATH"

# Overrides DEFAULT_NUM_SIMULATIONS below - the right value depends on host
# CPU speed and how much per-move latency is acceptable, so it's left
# runtime-configurable rather than hardcoded.
NUM_SIMULATIONS_ENV_VAR = "SKYJO_MCTS_NUM_SIMULATIONS"

DEFAULT_NUM_SIMULATIONS = 200


@functools.lru_cache(maxsize=1)
def default_evaluator() -> EvaluateFn:
    """The process-wide evaluator every mcts_bot seat shares. Cached so the
    (untrained, unless a checkpoint is configured) network is built once."""
    net = AlphaZeroNet()
    checkpoint_path = os.environ.get(CHECKPOINT_PATH_ENV_VAR)
    if checkpoint_path:
        load_checkpoint(checkpoint_path, net)
    return make_network_evaluator(net)


def default_num_simulations() -> int:
    raw = os.environ.get(NUM_SIMULATIONS_ENV_VAR)
    return DEFAULT_NUM_SIMULATIONS if raw is None else int(raw)


class MctsBot:
    def __init__(
        self,
        evaluate: EvaluateFn,
        *,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        c_puct: float = DEFAULT_C_PUCT,
        seed: int | None = None,
    ) -> None:
        if seed is not None and not isinstance(seed, int):
            raise TypeError("MctsBot: seed must be an int if provided")
        if num_simulations < 0:
            raise ValueError("MctsBot: num_simulations must be >= 0")
        self._evaluate = evaluate
        self._num_simulations = num_simulations
        self._c_puct = c_puct
        self._rng = np.random.default_rng(seed)

    def choose_action(self, turn: Turn, *, report_progress: ProgressReporter | None = None) -> Action:
        def on_simulation(step: int) -> None:
            if report_progress is not None:
                report_progress(step / self._num_simulations)

        root = run_mcts(
            turn,
            self._evaluate,
            num_simulations=self._num_simulations,
            c_puct=self._c_puct,
            # Root noise exists to diversify self-play training data - a live
            # bot should always search its true best line, not an
            # artificially perturbed one.
            add_root_noise=False,
            rng=self._rng,
            on_simulation=on_simulation if self._num_simulations > 0 else None,
        )
        # Most-visited (strongest) action, not a tau-sampled one - but ties
        # (common with an uninformative/untrained evaluator, e.g. every edge
        # visited once) are broken randomly rather than always taking the
        # same "first" edge: a deterministic tie-break can lock two such
        # bots into repeating the same tied pair of actions forever (e.g.
        # draw-then-discard, never placing) since nothing about the state
        # that matters to the tie ever changes turn to turn.
        visits = {action: edge.visit_count for action, edge in root.edges.items()}
        max_visits = max(visits.values())
        best_actions = [action for action, count in visits.items() if count == max_visits]
        best_action = best_actions[0] if len(best_actions) == 1 else best_actions[self._rng.integers(len(best_actions))]

        if report_progress is not None and self._num_simulations == 0:
            report_progress(1.0)
        return best_action
