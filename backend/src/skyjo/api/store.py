"""In-memory match storage.

Each match is kept as a tree of `GameState` nodes, not a single mutable state:
every action or round transition adds a node, and the store's "head" pointer is
just wherever in that tree the match currently is. Since `apply_action` and
`start_next_round` are deterministic given a state, going back to an earlier
node (`goto`) and diverging from it reuses the existing branch if that exact
move was taken before, or grows a new one otherwise — so the tree is the
match's full history, browsable and replayable from any point.

A restart loses all matches; swap for a real store (Redis, DB) if the API
needs to survive restarts or run multi-process.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from skyjo.domain.engine import Action, GameState


class MatchNotFoundError(Exception):
    pass


class NodeNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class Edge:
    """What was done to move from a node's parent to the node itself. `action`
    is set only when `kind == "action"`; the root node has no edge at all."""

    kind: Literal["action", "next_round"]
    action: Action | None = None


@dataclass(frozen=True)
class MatchNode:
    node_id: str
    parent_id: str | None
    edge: Edge | None  # None only for the root node
    actor: int | None  # player who triggered this transition; None for root/next_round
    state: GameState
    seq: int  # creation order, stable across branches
    round_index: int  # rounds completed strictly before this node


class _MatchTree:
    """Mutable per-match tree. Only ever touched while MatchStore._lock is held."""

    def __init__(self, player_names: tuple[str, ...], root_state: GameState) -> None:
        self.player_names = player_names
        self.nodes: dict[str, MatchNode] = {}
        self.children: dict[str, dict[Edge, str]] = {}
        self._next_seq = 0
        root = self._add_node(parent_id=None, edge=None, actor=None, state=root_state)
        self.root_id = root.node_id
        self.head_id = root.node_id

    def _add_node(
        self, parent_id: str | None, edge: Edge | None, actor: int | None, state: GameState
    ) -> MatchNode:
        round_index = 0
        if parent_id is not None:
            parent = self.nodes[parent_id]
            round_index = parent.round_index + (1 if edge is not None and edge.kind == "next_round" else 0)

        node = MatchNode(
            node_id=uuid4().hex,
            parent_id=parent_id,
            edge=edge,
            actor=actor,
            state=state,
            seq=self._next_seq,
            round_index=round_index,
        )
        self._next_seq += 1
        self.nodes[node.node_id] = node
        self.children[node.node_id] = {}
        if parent_id is not None:
            self.children[parent_id][edge] = node.node_id
        return node

    def node(self, node_id: str) -> MatchNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise NodeNotFoundError(node_id) from exc

    def head(self) -> MatchNode:
        return self.nodes[self.head_id]

    def advance(self, edge: Edge, compute_state) -> MatchNode:
        """Move the head forward via `edge`, reusing the existing child if this
        exact edge was already taken from the head before instead of recomputing
        (state transitions are deterministic given a starting state and edge)."""
        parent = self.head()
        existing = self.children[parent.node_id].get(edge)
        if existing is not None:
            self.head_id = existing
            return self.nodes[existing]

        new_state = compute_state(parent.state)
        actor = parent.state.current_player if edge.kind == "action" else None
        node = self._add_node(parent_id=parent.node_id, edge=edge, actor=actor, state=new_state)
        self.head_id = node.node_id
        return node

    def goto(self, node_id: str) -> MatchNode:
        node = self.node(node_id)
        self.head_id = node.node_id
        return node


class MatchStore:
    def __init__(self) -> None:
        self._matches: dict[str, _MatchTree] = {}
        self._lock = threading.Lock()

    def create(self, state: GameState, player_names: tuple[str, ...]) -> str:
        match_id = uuid4().hex
        with self._lock:
            self._matches[match_id] = _MatchTree(player_names, state)
        return match_id

    def _tree(self, match_id: str) -> _MatchTree:
        tree = self._matches.get(match_id)
        if tree is None:
            raise MatchNotFoundError(match_id)
        return tree

    def get_head(self, match_id: str) -> tuple[MatchNode, tuple[str, ...]]:
        with self._lock:
            tree = self._tree(match_id)
            return tree.head(), tree.player_names

    def get_history(self, match_id: str) -> tuple[list[MatchNode], str, tuple[str, ...]]:
        with self._lock:
            tree = self._tree(match_id)
            return list(tree.nodes.values()), tree.head_id, tree.player_names

    def apply_action(self, match_id: str, action: Action, compute_state) -> tuple[MatchNode, tuple[str, ...]]:
        """`compute_state(parent_state) -> new_state`; only called when this
        exact action hasn't been taken from the head before. May propagate
        whatever `compute_state` raises (e.g. IllegalActionError)."""
        with self._lock:
            tree = self._tree(match_id)
            node = tree.advance(Edge(kind="action", action=action), compute_state)
            return node, tree.player_names

    def start_next_round(self, match_id: str, compute_state) -> tuple[MatchNode, tuple[str, ...]]:
        with self._lock:
            tree = self._tree(match_id)
            node = tree.advance(Edge(kind="next_round"), compute_state)
            return node, tree.player_names

    def goto(self, match_id: str, node_id: str) -> tuple[MatchNode, tuple[str, ...]]:
        with self._lock:
            tree = self._tree(match_id)
            node = tree.goto(node_id)
            return node, tree.player_names
