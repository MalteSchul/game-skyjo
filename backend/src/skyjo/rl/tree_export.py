"""Serializes a built MCTS tree (`rl.mcts.MCTSNode`/`ChanceNode`) to a plain,
JSON-ready dict - for dumping snapshots at different simulation counts and
diffing/inspecting them outside a debugger (a text/graph viewer, `jq`, a
notebook, whatever), rather than walking the live object graph by hand.

Every field here is read-only, presentation-layer computation done at export
time - none of it touches `rl.mcts`'s search-time data structures or hot
path, so calling this (or not) has zero effect on training/self-play
performance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from skyjo.domain.engine import Action, GameState
from skyjo.rl.mcts import DEFAULT_C_PUCT, ChanceEdge, ChanceNode, MCTSEdge, MCTSNode

RankProbsByState = Mapping[GameState, np.ndarray]


def _action_to_dict(action: Action) -> dict[str, Any]:
    return {"type": action.type.name, "position": action.position}


def tree_to_dict(
    node: MCTSNode | ChanceNode,
    *,
    max_depth: int | None = None,
    c_puct: float = DEFAULT_C_PUCT,
    rank_probs_by_state: RankProbsByState | None = None,
) -> dict[str, Any]:
    """`max_depth` counts edges from `node`: 0 means describe `node` itself
    with no children traversed (edges listed but each `child` omitted).
    `None` traverses the whole built tree. `c_puct` must match what the
    search itself used (`run_mcts`'s default unless overridden) for the
    exported `q`/`u`/`puct_score` fields to mean anything - it's not stored
    on the tree itself, so there's no way to recover it after the fact.
    `rank_probs_by_state`, if given, is looked up by each decision node's
    `state` to fill in that node's `rank_probs` field - see
    `evaluator.make_network_evaluator`'s `rank_probs_sink` param, which is
    how such a mapping gets built in the first place. `None`/omitted means
    every node's `rank_probs` comes back `null`."""
    if isinstance(node, ChanceNode):
        return _chance_node_to_dict(node, max_depth, c_puct, rank_probs_by_state)
    return _mcts_node_to_dict(node, max_depth, c_puct, rank_probs_by_state)


def _descend(
    child: MCTSNode | ChanceNode | None,
    max_depth: int | None,
    c_puct: float,
    rank_probs_by_state: RankProbsByState | None,
) -> dict[str, Any] | None:
    if child is None:
        return None
    if max_depth is not None and max_depth <= 0:
        return None
    next_depth = None if max_depth is None else max_depth - 1
    return tree_to_dict(child, max_depth=next_depth, c_puct=c_puct, rank_probs_by_state=rank_probs_by_state)


def _mcts_edge_to_dict(
    edge: MCTSEdge,
    max_depth: int | None,
    c_puct: float,
    rank_probs_by_state: RankProbsByState | None,
    *,
    current_player: int,
    parent_visit_count: int,
) -> dict[str, Any]:
    # Same formula as `mcts._select_edge` - reconstructed here rather than
    # read off the tree because the actual PUCT score used at selection time
    # was never a fixed property of the edge: it depends on the *parent's*
    # total visit count, which keeps changing across simulations. This is
    # only ever "the score as of this snapshot", not a value the search
    # itself stored anywhere.
    q = edge.mean_value()[current_player]
    u = c_puct * edge.prior * math.sqrt(parent_visit_count) / (1 + edge.visit_count)
    return {
        "action": _action_to_dict(edge.action),
        "prior": edge.prior,
        "prior_before_noise": edge.prior_before_noise,
        "visit_count": edge.visit_count,
        "mean_value": edge.mean_value().tolist(),
        "q": q,
        "u": u,
        "puct_score": q + u,
        "child": _descend(edge.child, max_depth, c_puct, rank_probs_by_state),
    }


def _chance_edge_to_dict(
    edge: ChanceEdge, max_depth: int | None, c_puct: float, rank_probs_by_state: RankProbsByState | None
) -> dict[str, Any]:
    # Exported as "card_value", not "value": a decision node's "value" means
    # the network's predicted utility vector (see _mcts_node_to_dict) - an
    # unrelated concept that happens to sit one level below this same edge
    # (edge.child), so reusing the same key here would be actively
    # misleading rather than just redundant.
    return {
        "card_value": edge.value,
        "prior": edge.prior,
        "visit_count": edge.visit_count,
        "mean_value": edge.value_sum.tolist()
        if edge.visit_count == 0
        else (edge.value_sum / edge.visit_count).tolist(),
        "child": _descend(edge.child, max_depth, c_puct, rank_probs_by_state),
    }


def _mcts_node_to_dict(
    node: MCTSNode, max_depth: int | None, c_puct: float, rank_probs_by_state: RankProbsByState | None
) -> dict[str, Any]:
    edges = sorted(node.edges.values(), key=lambda e: e.visit_count, reverse=True)
    parent_visit_count = node.visit_count
    current_player = node.state.current_player
    rank_probs = None if rank_probs_by_state is None else rank_probs_by_state.get(node.state)
    return {
        "kind": "decision",
        "current_player": current_player,
        "phase": node.state.phase,
        "is_terminal": node.is_terminal,
        "visit_count": parent_visit_count,
        # The network's own prediction for this node's state (None until
        # expanded) - not the Monte-Carlo average of visits through it, see
        # `MCTSNode.value`'s docstring in rl.mcts.
        "value": node.value.tolist() if node.value is not None else None,
        # rank_probs[i][r] = P(player i finishes at rank r), the un-reduced
        # prediction `value` is a fixed linear weighting of - only available
        # when the evaluator was given a `rank_probs_sink` (see docstring
        # above), otherwise always null.
        "rank_probs": None if rank_probs is None else rank_probs.tolist(),
        "edges": [
            _mcts_edge_to_dict(
                edge,
                max_depth,
                c_puct,
                rank_probs_by_state,
                current_player=current_player,
                parent_visit_count=parent_visit_count,
            )
            for edge in edges
        ],
    }


def _chance_node_to_dict(
    node: ChanceNode, max_depth: int | None, c_puct: float, rank_probs_by_state: RankProbsByState | None
) -> dict[str, Any]:
    edges = sorted(node.edges.values(), key=lambda e: e.visit_count, reverse=True)
    return {
        "kind": "chance",
        "edges": [_chance_edge_to_dict(edge, max_depth, c_puct, rank_probs_by_state) for edge in edges],
    }
