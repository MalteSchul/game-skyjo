"""Run a single MCTS search and drop into the debugger with the built tree
still in memory, for stepping through `root`/`edge`/`child` by hand.

Usage: uv run python scripts/debug_mcts.py [--seed N] [--simulations N] [--players N]

Once stopped at the breakpoint, useful things to inspect/evaluate:
  root.edges                                    # dict[Action, MCTSEdge] at the root
  sorted(root.edges.values(), key=lambda e: -e.visit_count)[:5]   # most-explored edges
  edge = root.edges[some_action]
  edge.visit_count, edge.prior, edge.mean_value()
  edge.child                                     # MCTSNode or ChanceNode (reveal) below it
  edge.child.edges                               # descend one more ply
  chance_edge = edge.child.edges[revealed_value]  # for a ChanceNode, keyed by card value
"""

from __future__ import annotations

import argparse

import numpy as np

from skyjo.domain.engine import GameState, new_match
from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.domain.observation import Turn
from skyjo.rl.mcts import run_mcts


def _uniform_evaluate(state: GameState):
    """Untrained stand-in: uniform priors, zero value. Swap in
    `make_network_evaluator(AlphaZeroNet())` (or a loaded checkpoint) to
    inspect a trained network's tree instead."""
    actions = engine_legal_actions(state)
    priors = {a: 1.0 / len(actions) for a in actions}
    return priors, np.zeros(len(state.boards))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--players", type=int, default=2)
    args = parser.parse_args()

    state = new_match(player_count=args.players, seed=args.seed)
    turn = Turn.from_state(state)

    root = run_mcts(
        turn,
        _uniform_evaluate,
        num_simulations=args.simulations,
        rng=np.random.default_rng(args.seed),
    )

    print(f"root visit_count={root.visit_count}, {len(root.edges)} legal actions")
    breakpoint()  # noqa: T100 - intentional: this script exists to stop here


if __name__ == "__main__":
    main()
