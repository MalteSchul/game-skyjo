"""Runs one continuous MCTS search and dumps the tree as JSON at several
simulation counts along the way, so you can diff/inspect how it actually
grows (structure, visit counts, priors, Q-values) without a debugger.

Usage:
  uv run python scripts/dump_mcts_tree.py --sim-points 10,50,200 --out tree.json
  uv run python scripts/dump_mcts_tree.py --sim-points 10,50,200 --max-depth 2

  # search a specific position instead of a fresh deal - a checked-in example:
  uv run python scripts/dump_mcts_tree.py --state-file example_states/midgame_awaiting_draw.json --sim-points 50

  # or build your own: write a template, hand-edit it, then point at it
  uv run python scripts/dump_mcts_tree.py --dump-example-state state.json --seed 1
  uv run python scripts/dump_mcts_tree.py --state-file state.json --sim-points 50

`--state-file` takes a JSON file produced by `domain.state_json.game_state_to_dict`
(the true, un-redacted `GameState` - hidden card values included) and roots the
search there instead of calling `new_match`. See `example_states/` for
checked-in positions (and how they were generated). `--dump-example-state
<path>` writes a fresh `new_match(seed=..., player_count=...)` state to that
path in the same format, as a starting template, and exits without running a
search. `--state-file` and `--seed`/`--players` are mutually exclusive.

By default the search runs against a uniform stand-in (equal prior on every
legal action, zero value for every state) - deliberately state-blind, so it's
useless for judging *what the network thinks* but fine for looking at how the
tree's shape/visit counts grow. Pass `--network` to swap in the real
`AlphaZeroNet` (same architecture `bots.mcts_bot.default_evaluator` builds,
random-init unless `SKYJO_MCTS_CHECKPOINT_PATH` is set) instead: even
untrained, its priors and per-state `value` are non-uniform and non-zero,
because they actually depend on the encoded state. `--network` also captures
each visited node's `rank_probs` (P(player i finishes at rank r), the
un-reduced prediction `value` is a linear weighting of - see
`evaluator.make_network_evaluator`'s `rank_probs_sink` param) from that same
forward pass, at no extra network-call cost.

Output is a JSON object keyed by simulation count (as a string). Every
snapshot comes from the *same* search: `run_mcts`'s `on_root_ready` hands
back the live root once (before any simulation), then `on_simulation`
serializes that same object at each requested step - so the N=50 snapshot is
a real continuation of the N=10 one, not a second independent run. `--max-depth`
truncates how many plies deep each snapshot descends (unset = whole built
tree, which grows large fast). See `TREE_EXPORT_SCHEMA.md` for what every key
in the output means.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from skyjo.bots.mcts_bot import CHECKPOINT_PATH_ENV_VAR
from skyjo.domain.engine import GameState, new_match
from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.domain.observation import Turn
from skyjo.domain.state_json import game_state_from_dict, game_state_to_dict
from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.mcts import EvaluateFn, MCTSNode, run_mcts
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.tree_export import tree_to_dict


def _uniform_evaluate(state: GameState):
    """Untrained stand-in: uniform priors, zero value - see `--network` for
    the real thing."""
    actions = engine_legal_actions(state)
    priors = {a: 1.0 / len(actions) for a in actions}
    return priors, np.zeros(len(state.boards))


def _build_network_evaluator(rank_probs_sink: dict[GameState, np.ndarray]) -> EvaluateFn:
    """Builds its own `AlphaZeroNet` rather than reusing
    `bots.mcts_bot.default_evaluator()` (which is `lru_cache`d and has no way
    to pass a `rank_probs_sink`) - fine here since this script runs once and
    exits, unlike a live bot process serving many seats off one warm net."""
    net = AlphaZeroNet()
    checkpoint_path = os.environ.get(CHECKPOINT_PATH_ENV_VAR)
    if checkpoint_path:
        load_checkpoint(checkpoint_path, net)
    return make_network_evaluator(net, rank_probs_sink=rank_probs_sink)


def _parse_sim_points(raw: str) -> list[int]:
    points = sorted({int(p) for p in raw.split(",")})
    if any(p < 0 for p in points):
        raise ValueError("--sim-points: all values must be >= 0")
    return points


def _parse_max_depth(raw: str) -> int | None:
    """`--max-depth full` (or `none`/`unbounded`) maps to `tree_to_dict`'s
    `max_depth=None` - "traverse the whole built tree" - which plain
    `type=int` can't express since argparse has no way to fall through to a
    non-int default."""
    if raw.lower() in ("full", "none", "unbounded"):
        return None
    return int(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--state-file", help="JSON GameState (see domain.state_json) to root the search at, instead of a fresh new_match()")
    parser.add_argument("--dump-example-state", metavar="PATH", help="write a fresh new_match() state to PATH as a template, then exit")
    parser.add_argument(
        "--network",
        action="store_true",
        help="use the real AlphaZeroNet (random-init unless SKYJO_MCTS_CHECKPOINT_PATH is set) instead of the uniform stand-in",
    )
    parser.add_argument("--sim-points", type=_parse_sim_points, default=_parse_sim_points("10,50,200"))
    parser.add_argument("--max-depth", type=_parse_max_depth, default=2, help="int, or 'full'/'none'/'unbounded' for the whole built tree")
    parser.add_argument("--out", default="scripts/output/mcts_tree.json", help="gitignored by default (scripts/output/) - dumps are regenerable, not meant to be committed")
    args = parser.parse_args()

    if args.dump_example_state is not None:
        template_state = new_match(player_count=args.players, seed=args.seed)
        with open(args.dump_example_state, "w") as f:
            json.dump(game_state_to_dict(template_state), f, indent=2)
        print(f"wrote an example GameState to {args.dump_example_state}")
        return

    if args.state_file is not None:
        if args.seed != parser.get_default("seed") or args.players != parser.get_default("players"):
            raise SystemExit("--state-file is mutually exclusive with --seed/--players")
        with open(args.state_file) as f:
            state = game_state_from_dict(json.load(f))
    else:
        state = new_match(player_count=args.players, seed=args.seed)
    turn = Turn.from_state(state)
    rank_probs_by_state: dict[GameState, np.ndarray] = {}
    evaluate = _build_network_evaluator(rank_probs_by_state) if args.network else _uniform_evaluate

    snapshots: dict[str, dict] = {}
    target_points = set(args.sim_points)
    root_box: list[MCTSNode] = []

    def dump_snapshot(root: MCTSNode) -> dict:
        # rank_probs_by_state keeps growing as the search visits more states -
        # passing the same live dict at every snapshot means each one reports
        # rank_probs for whatever's been visited *so far*, same as visit_count.
        return tree_to_dict(root, max_depth=args.max_depth, rank_probs_by_state=rank_probs_by_state)

    def on_root_ready(root: MCTSNode) -> None:
        root_box.append(root)
        if 0 in target_points:
            snapshots["0"] = dump_snapshot(root)

    def on_simulation(step: int) -> None:
        if step in target_points:
            snapshots[str(step)] = dump_snapshot(root_box[0])

    run_mcts(
        turn,
        evaluate,
        num_simulations=max(args.sim_points),
        rng=np.random.default_rng(args.seed),
        on_root_ready=on_root_ready,
        on_simulation=on_simulation,
    )

    with open(args.out, "w") as f:
        json.dump(snapshots, f, indent=2)

    print(f"wrote {len(snapshots)} snapshot(s) at N={sorted(target_points)} to {args.out}")


if __name__ == "__main__":
    main()
