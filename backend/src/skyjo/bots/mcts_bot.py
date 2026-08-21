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
from skyjo.domain.action_equivalence import group_representatives
from skyjo.domain.engine import Action, ActionType
from skyjo.domain.observation import Turn
from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.hidden_info import gamestate_from_turn, is_reveal, will_close_round
from skyjo.rl.mcts import DEFAULT_C_PUCT, ChanceNode, EvaluateFn, MCTSNode, greedy_action, run_mcts
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
        # This bot's own last fully-searched tree, kept around so the *next*
        # choose_action - once the real game has genuinely advanced to a
        # position this tree already explored - can resume search from that
        # position instead of expanding a fresh root. Advanced turn-by-turn
        # by `observe_transition`, and re-verified against the incoming
        # `turn` here regardless (see `choose_action`) - if anything broke
        # that walk (a rewound history, a chance outcome never visited, a
        # round transition), this is simply None and search falls back to
        # today's from-scratch behavior. Never causes a wrong answer, only
        # sometimes misses the chance to reuse work.
        self._cached_root: MCTSNode | None = None

    def choose_action(self, turn: Turn, *, report_progress: ProgressReporter | None = None) -> Action:
        def on_simulation(step: int) -> None:
            if report_progress is not None:
                report_progress(step / self._num_simulations)

        reuse_root = self._cached_root
        if reuse_root is not None and reuse_root.state != gamestate_from_turn(turn):
            reuse_root = None

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
            reuse_root=reuse_root,
        )
        # Most-visited (strongest) action, not a tau-sampled one - see
        # `greedy_action`'s docstring for why ties are broken randomly.
        best_action = greedy_action(root, self._rng)
        self._cached_root = root

        if report_progress is not None and self._num_simulations == 0:
            report_progress(1.0)
        return best_action

    def observe_transition(self, turn_before: Turn, action: Action, turn_after: Turn) -> None:
        """Advances `self._cached_root` by one real transition - `action`,
        taken by whichever seat `turn_before.acting_player` was - so the
        cache reflects the match's actual path rather than just this bot's
        own last search. See `ObservesActions`: called for *every* seat's
        action, not just this bot's own turns, since another player's move
        (or a chance reveal it triggers) advances the position this bot will
        next be asked to search from just the same.

        Every branch below either finds the exact already-visited child that
        `action` (and, for a reveal, its real resolved value) leads to, or
        clears the cache - there's no partial/uncertain middle ground, since
        an incorrect reuse would corrupt the search silently while a missed
        one only costs some redundant computation next time.
        """
        if self._cached_root is None:
            return
        state = self._cached_root.state
        if gamestate_from_turn(turn_before) != state:
            self._cached_root = None
            return

        representative = group_representatives(turn_before).get(action, action)
        edge = self._cached_root.edges.get(representative)
        if edge is None or edge.child is None or will_close_round(state, action):
            # Round-closing is never cached on its edge in the first place
            # (see rl.mcts's module docstring) - nothing to carry forward.
            self._cached_root = None
            return

        if is_reveal(state, action):
            chance = edge.child
            revealed_value = _infer_revealed_value(turn_before, action, turn_after)
            chance_edge = (
                chance.edges.get(revealed_value)
                if isinstance(chance, ChanceNode) and revealed_value is not None
                else None
            )
            child = chance_edge.child if chance_edge is not None else None
        else:
            child = edge.child if isinstance(edge.child, MCTSNode) else None

        self._cached_root = child if isinstance(child, MCTSNode) and not child.is_terminal else None


def _infer_revealed_value(turn_before: Turn, action: Action, turn_after: Turn) -> int | None:
    """Best-effort recovery of the real value `action` revealed, purely from
    public `Turn` fields turn_after already exposes - never anything this
    bot wasn't already entitled to see. Returns None whenever it can't be
    recovered with confidence (e.g. an immediate column-clear already wiped
    the very position that would have shown it) - always a safe answer, that
    `observe_transition` treats exactly like any other cache miss.
    """
    if action.type is ActionType.DRAW_STOCK:
        # Still the same acting player's turn (awaiting_placement next), and
        # a drawn card sits in `drawn_card` until it's placed or discarded.
        return turn_after.drawn_card
    if action.position is None:
        return None
    card = turn_after.boards[turn_before.acting_player].cards[action.position]
    return card.value if card is not None and card.face_up else None
