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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from skyjo.bots.base import Bot, ObservesActions
from skyjo.domain.engine import Action, GameState
from skyjo.domain.observation import Turn


class MatchNotFoundError(Exception):
    pass


class NodeNotFoundError(Exception):
    pass


class MatchBusyError(Exception):
    """Raised when a human action or history navigation is attempted while a
    bot is still deciding. Without this guard, such a call could mutate
    `head_id` out from under the in-flight autoplay thread, which resolves
    the tree's head only once its bot's `choose_action` finally returns -
    silently attaching that move to the wrong node."""


@dataclass(frozen=True)
class AutoplayStatus:
    status: Literal["idle", "thinking"]
    player: int | None
    progress: float | None


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

    def __init__(
        self,
        player_names: tuple[str, ...],
        player_types: tuple[str, ...],
        bots: tuple[Bot | None, ...],
        root_state: GameState,
    ) -> None:
        self.player_names = player_names
        self.player_types = player_types
        # A live Bot per seat (None for human seats), one instance for the
        # match's whole lifetime so a seeded bot's randomness actually
        # progresses turn over turn instead of resetting every request.
        self.bots = bots
        self.nodes: dict[str, MatchNode] = {}
        self.children: dict[str, dict[Edge, str]] = {}
        self._next_seq = 0
        root = self._add_node(parent_id=None, edge=None, actor=None, state=root_state)
        self.root_id = root.node_id
        self.head_id = root.node_id
        self.autoplay_status = AutoplayStatus(status="idle", player=None, progress=None)
        self.autoplay_thread: threading.Thread | None = None
        self.autoplay_event: threading.Event | None = None

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

    def create(
        self,
        state: GameState,
        player_names: tuple[str, ...],
        player_types: tuple[str, ...],
        bots: tuple[Bot | None, ...],
    ) -> str:
        match_id = uuid4().hex
        with self._lock:
            self._matches[match_id] = _MatchTree(player_names, player_types, bots, state)
        return match_id

    def _tree(self, match_id: str) -> _MatchTree:
        tree = self._matches.get(match_id)
        if tree is None:
            raise MatchNotFoundError(match_id)
        return tree

    def get_head(self, match_id: str) -> tuple[MatchNode, tuple[str, ...], tuple[str, ...]]:
        with self._lock:
            tree = self._tree(match_id)
            return tree.head(), tree.player_names, tree.player_types

    def get_bot(self, match_id: str, seat: int) -> Bot | None:
        with self._lock:
            tree = self._tree(match_id)
            return tree.bots[seat]

    def get_history(self, match_id: str) -> tuple[list[MatchNode], str, tuple[str, ...]]:
        with self._lock:
            tree = self._tree(match_id)
            return list(tree.nodes.values()), tree.head_id, tree.player_names

    def apply_action(
        self, match_id: str, action: Action, compute_state
    ) -> tuple[MatchNode, tuple[str, ...], tuple[str, ...]]:
        """Applies a human-initiated action. `compute_state(parent_state) ->
        new_state`; only called when this exact action hasn't been taken from
        the head before. May propagate whatever `compute_state` raises (e.g.
        IllegalActionError). Raises MatchBusyError while a bot is still
        deciding for this match - use `apply_autoplay_action` from within the
        match's own autoplay thread instead."""
        with self._lock:
            tree = self._tree(match_id)
            self._require_idle(tree)
            return self._advance(tree, Edge(kind="action", action=action), compute_state)

    def apply_autoplay_action(
        self, match_id: str, action: Action, compute_state
    ) -> tuple[MatchNode, tuple[str, ...], tuple[str, ...]]:
        """Like `apply_action`, but for use only from within the match's own
        autoplay thread (see `trigger_autoplay`), which legitimately holds
        "thinking" status while it runs."""
        with self._lock:
            tree = self._tree(match_id)
            return self._advance(tree, Edge(kind="action", action=action), compute_state)

    def start_next_round(self, match_id: str, compute_state) -> tuple[MatchNode, tuple[str, ...], tuple[str, ...]]:
        with self._lock:
            tree = self._tree(match_id)
            self._require_idle(tree)
            return self._advance(tree, Edge(kind="next_round"), compute_state)

    def goto(self, match_id: str, node_id: str) -> tuple[MatchNode, tuple[str, ...], tuple[str, ...]]:
        with self._lock:
            tree = self._tree(match_id)
            self._require_idle(tree)
            node = tree.goto(node_id)
            return node, tree.player_names, tree.player_types

    def _advance(
        self, tree: _MatchTree, edge: Edge, compute_state
    ) -> tuple[MatchNode, tuple[str, ...], tuple[str, ...]]:
        parent = tree.head()
        node = tree.advance(edge, compute_state)
        # Only a real action, landing on a non-terminal state, is a transition
        # any bot's cached search state could meaningfully advance through -
        # a next_round edge carries nothing (a fresh round is an independent
        # shuffle) and a round/game-ending state has no Turn to build at all.
        if edge.kind == "action" and node.state.phase not in ("round_over", "game_over"):
            assert edge.action is not None
            self._notify_bots(tree, parent.state, edge.action, node.state)
        return node, tree.player_names, tree.player_types

    def _notify_bots(
        self, tree: _MatchTree, state_before: GameState, action: Action, state_after: GameState
    ) -> None:
        """Tells every seat's bot - not just whoever acted - that `action`
        was just taken, so a bot keeping state across turns (`MctsBot`'s
        cached search tree) can advance it to match the match's real path.
        See `ObservesActions`."""
        turn_before = Turn.from_state(state_before)
        turn_after = Turn.from_state(state_after)
        for bot in tree.bots:
            if isinstance(bot, ObservesActions):
                bot.observe_transition(turn_before, action, turn_after)

    def _require_idle(self, tree: _MatchTree) -> None:
        if tree.autoplay_status.status == "thinking":
            raise MatchBusyError

    def trigger_autoplay(self, match_id: str, work: Callable[[str], None]) -> threading.Event:
        """Starts `work(match_id)` on a background thread unless one is
        already running for this match, in which case its existing event is
        returned instead of starting a second one. `work` always runs to
        completion - even if it raises - before status is reset to idle and
        the event is set, so a match can never get stuck reporting "thinking"
        forever."""
        with self._lock:
            tree = self._tree(match_id)
            if tree.autoplay_thread is not None and tree.autoplay_thread.is_alive():
                assert tree.autoplay_event is not None
                return tree.autoplay_event

            event = threading.Event()
            tree.autoplay_event = event

            def run() -> None:
                try:
                    work(match_id)
                finally:
                    with self._lock:
                        tree.autoplay_status = AutoplayStatus(status="idle", player=None, progress=None)
                    event.set()

            thread = threading.Thread(target=run, daemon=True)
            tree.autoplay_thread = thread
            thread.start()
            return event

    def set_thinking(self, match_id: str, player: int) -> None:
        with self._lock:
            tree = self._tree(match_id)
            tree.autoplay_status = AutoplayStatus(status="thinking", player=player, progress=None)

    def set_thinking_progress(self, match_id: str, progress: float) -> None:
        with self._lock:
            tree = self._tree(match_id)
            # A stale callback from a superseded run (e.g. autoplay already
            # finished) should be a no-op rather than resurrecting "thinking".
            if tree.autoplay_status.status != "thinking":
                return
            tree.autoplay_status = AutoplayStatus(
                status="thinking", player=tree.autoplay_status.player, progress=progress
            )

    def get_status(self, match_id: str) -> AutoplayStatus:
        with self._lock:
            tree = self._tree(match_id)
            return tree.autoplay_status
