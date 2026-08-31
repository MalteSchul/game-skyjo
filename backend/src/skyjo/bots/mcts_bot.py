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
from pathlib import Path

import numpy as np

from skyjo.bots.base import ProgressReporter
from skyjo.domain.engine import Action
from skyjo.domain.observation import Turn
from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.hidden_info import gamestate_from_turn
from skyjo.rl.mcts import (
    DEFAULT_C_PUCT,
    EvaluateFn,
    MCTSNode,
    advance_cached_root,
    greedy_action,
    run_mcts,
)
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.tree_export import tree_to_dict

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

# `last_tree_snapshots` progress checkpoints, as a fraction of `num_simulations`
# - mirrors `scripts/dump_mcts_tree.py --sim-points` so the "why did it pick
# this" UI can show the tree actually growing across a decision instead of
# only its finished state.
_SNAPSHOT_PROGRESS_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Depth captured in each `last_tree_snapshots` snapshot - matches the old
# `last_tree_dict(max_depth=2)` default, the only value any caller ever used.
_SNAPSHOT_MAX_DEPTH = 2

# Directory scanned for selectable named checkpoints - every "*.pt" file in it
# becomes a model the UI can offer for an mcts_bot seat (see list_available_models
# / evaluator_for_model), keyed by filename stem. Separate from
# CHECKPOINT_PATH_ENV_VAR, which points at a single fixed checkpoint used when no
# model is explicitly selected.
MODELS_DIR_ENV_VAR = "SKYJO_MCTS_MODELS_DIR"
DEFAULT_MODELS_DIR = "models"


def _models_dir() -> Path:
    return Path(os.environ.get(MODELS_DIR_ENV_VAR, DEFAULT_MODELS_DIR))


def list_available_models() -> list[str]:
    """Names of the checkpoints in the models directory, sorted alphabetically.
    Each name is a ".pt" file's stem and can be passed to `evaluator_for_model`."""
    directory = _models_dir()
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.pt"))


def save_uploaded_model(filename: str, content: bytes) -> str:
    """Saves an uploaded checkpoint into the models directory, making it
    selectable via `list_available_models` / `evaluator_for_model` under a name
    derived from `filename` (its stem, ignoring any directory components).
    Rejects anything that isn't a loadable `save_checkpoint` payload so a bad
    upload can't silently become an unusable "model" in the list. Clears the
    `evaluator_for_model` cache so re-uploading an existing name picks up the
    new weights instead of serving the previously cached network."""
    name = Path(filename).stem
    if not name or Path(filename).suffix.lower() != ".pt":
        raise ValueError(f"save_uploaded_model: expected a .pt file, got {filename!r}")
    directory = _models_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pt"
    path.write_bytes(content)
    try:
        load_checkpoint(path, AlphaZeroNet())
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise ValueError(f"save_uploaded_model: {filename!r} isn't a valid checkpoint") from exc
    evaluator_for_model.cache_clear()
    return name


@functools.cache
def evaluator_for_model(model_name: str) -> EvaluateFn:
    """Evaluator for a named checkpoint from the models directory. Cached per
    model name (like `default_evaluator`) so selecting the same model for
    multiple seats reuses one loaded network."""
    path = _models_dir() / f"{model_name}.pt"
    if not path.is_file():
        raise ValueError(f"evaluator_for_model: unknown model {model_name!r}")
    net = AlphaZeroNet()
    load_checkpoint(path, net)
    return make_network_evaluator(net)


@functools.lru_cache(maxsize=1)
def default_evaluator() -> EvaluateFn:
    """The process-wide evaluator an mcts_bot seat uses when no model is
    explicitly selected. Cached so the (untrained, unless a checkpoint is
    configured) network is built once."""
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
        cap_root_lead: bool = False,
    ) -> None:
        if seed is not None and not isinstance(seed, int):
            raise TypeError("MctsBot: seed must be an int if provided")
        if num_simulations < 0:
            raise ValueError("MctsBot: num_simulations must be >= 0")
        self._evaluate = evaluate
        self._num_simulations = num_simulations
        self._c_puct = c_puct
        # See `rl.mcts.run_mcts`'s own `cap_root_lead` docstring: an eval-only
        # root selection policy, off by default. Paired with `greedy_action`'s
        # `tie_break="value"` in `choose_action` below - that's the hedge for
        # the exact-tie case this policy can produce.
        self._cap_root_lead = cap_root_lead
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
        # sometimes misses the chance to reuse work. `num_simulations` is a
        # *target total* per decision, not "always this many more" (see
        # `choose_action`) - reuse's actual payoff for a live bot is lower
        # latency (fewer new simulations needed to reach that target), not
        # just a stronger search at the same cost.
        self._cached_root: MCTSNode | None = None
        # Progress-checkpoint exports (see `_SNAPSHOT_PROGRESS_FRACTIONS`) of
        # the most recent `choose_action`'s search, captured *during* that
        # search - not just derived from the finished root afterwards, since
        # `_cached_root` gets walked forward (and its nodes' visit counts keep
        # growing in place) the moment `observe_transition` sees an action
        # applied, which would otherwise lose both the tree's growth over the
        # search *and* every rejected root-level alternative (the very thing
        # a "why did it pick this" view needs). Keyed by the actual visit
        # count at that checkpoint (a string, same convention as
        # `dump_mcts_tree.py`'s output).
        self._last_decision_snapshots: dict[str, dict] | None = None

    def choose_action(self, turn: Turn, *, report_progress: ProgressReporter | None = None) -> Action:
        # cap_root_lead only bounds selection at whichever node is currently
        # the *root* - a reused subtree spent its entire prior life as a
        # non-root descendant, governed by plain uncapped `_select_edge`, so
        # promoting it via reuse would carry over whatever lead it already
        # had. cap_root_lead can only stop that lead from growing further
        # from here, not retroactively unwind it - so a capped bot searches
        # fresh every turn instead, trading reuse's latency benefit for the
        # cap actually holding on every single decision.
        reuse_root = None if self._cap_root_lead else self._cached_root
        if reuse_root is not None and reuse_root.state != gamestate_from_turn(turn):
            reuse_root = None
        already_visited = reuse_root.visit_count if reuse_root is not None else 0
        remaining = max(0, self._num_simulations - already_visited)

        # Visit counts (absolute, not step-relative) at which to freeze a
        # snapshot - deduped/sorted since small `num_simulations` can round
        # several fractions to the same integer.
        target_visit_counts = sorted({round(f * self._num_simulations) for f in _SNAPSHOT_PROGRESS_FRACTIONS})
        snapshots: dict[str, dict] = {}
        root_box: list[MCTSNode] = []

        def snapshot(root: MCTSNode) -> dict:
            return tree_to_dict(root, max_depth=_SNAPSHOT_MAX_DEPTH, c_puct=self._c_puct)

        def on_root_ready(root: MCTSNode) -> None:
            root_box.append(root)
            if already_visited in target_visit_counts:
                snapshots[str(already_visited)] = snapshot(root)

        def on_simulation(step: int) -> None:
            visits = already_visited + step
            if visits in target_visit_counts:
                snapshots[str(visits)] = snapshot(root_box[0])
            if report_progress is not None:
                report_progress(visits / self._num_simulations)

        root = run_mcts(
            turn,
            self._evaluate,
            num_simulations=remaining,
            c_puct=self._c_puct,
            # Root noise exists to diversify self-play training data - a live
            # bot should always search its true best line, not an
            # artificially perturbed one.
            add_root_noise=False,
            rng=self._rng,
            on_root_ready=on_root_ready,
            on_simulation=on_simulation if remaining > 0 else None,
            reuse_root=reuse_root,
            cap_root_lead=self._cap_root_lead,
        )
        # Guarantees at least one (the final) snapshot even when none of
        # target_visit_counts landed exactly on the search's actual visit
        # counts - e.g. a reused root already past every checkpoint but 100%.
        if str(root.visit_count) not in snapshots:
            snapshots[str(root.visit_count)] = snapshot(root)

        # Most-visited (strongest) action, not a tau-sampled one - see
        # `greedy_action`'s docstring for why ties are broken randomly by
        # default, and by value instead when `cap_root_lead` is on.
        best_action = greedy_action(
            root, self._rng, tie_break="value" if self._cap_root_lead else "random"
        )
        self._cached_root = root
        self._last_decision_snapshots = snapshots

        if report_progress is not None and remaining == 0:
            report_progress(1.0)
        return best_action

    def last_tree_snapshots(self) -> dict[str, dict] | None:
        """JSON-ready exports (see `tree_export.tree_to_dict`), keyed by
        visit count as a string, of the search tree behind this bot's most
        recent `choose_action` decision at ~0/20/40/60/80/100% progress - for
        a "why did it pick this" view, e.g. in the live match UI, that can
        show the tree actually growing rather than only its finished state.
        Each snapshot reflects the full root (every legal action considered,
        not just the one taken), since they're captured before
        `observe_transition` walks `_cached_root` past it. `None` if
        `choose_action` hasn't been called yet."""
        return self._last_decision_snapshots

    def observe_transition(self, turn_before: Turn, action: Action, turn_after: Turn) -> None:
        """Advances `self._cached_root` by one real transition - see
        `rl.mcts.advance_cached_root`, which does the actual tree-walk this
        just delegates to (shared with `evaluator._play_one_eval_game`'s own
        reuse). See `ObservesActions`: called for *every* seat's action, not
        just this bot's own turns, since another player's move (or a chance
        reveal it triggers) advances the position this bot will next be
        asked to search from just the same.
        """
        self._cached_root = advance_cached_root(self._cached_root, turn_before, action, turn_after)
