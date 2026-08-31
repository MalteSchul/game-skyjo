"""Request/response shapes for the matches API and their mapping to/from domain types.

Face-down cards are redacted to value=None on the way out — the wire format is the
only place that needs to hide information, so it lives here rather than in the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from skyjo.api.store import AutoplayStatus, Edge, MatchNode
from skyjo.domain.engine import (
    Action,
    ActionType,
    Card,
    GameState,
    Phase,
    PlayerBoard,
    legal_actions,
)

ActionTypeName = Literal["flip_initial", "draw_stock", "draw_discard", "place", "discard_and_reveal"]
PlayerTypeName = Literal["human", "random_bot", "thinking_bot", "heuristic_bot", "mcts_bot"]
MatchStatus = Literal["idle", "thinking"]


def action_type_from_name(name: str) -> ActionType:
    try:
        return ActionType[name.upper()]
    except KeyError as exc:
        raise ValueError(f"unknown action type {name!r}") from exc


def action_type_to_name(action_type: ActionType) -> str:
    return action_type.name.lower()


class NewMatchRequest(BaseModel):
    player_count: int
    seed: int | None = None
    # One name per player, in seat order. Omitted or blank entries fall back to
    # "Player N" — see api.matches._resolve_player_names.
    player_names: list[str] | None = None
    # One type per player, in seat order. Omitted defaults every seat to
    # "human" — see api.matches._resolve_player_types.
    player_types: list[PlayerTypeName] | None = None
    # One entry per player, in seat order — a name from GET /matches/mcts-models
    # to use for that seat's mcts_bot, or None for the untrained default
    # network. Ignored for seats whose player_type isn't "mcts_bot". Omitted
    # defaults every seat to None — see api.matches._resolve_mcts_models.
    player_mcts_models: list[str | None] | None = None
    # One entry per player, in seat order — overrides that seat's mcts_bot
    # search depth (rollouts per move), or None for the process default (see
    # bots.mcts_bot.default_num_simulations). Ignored for seats whose
    # player_type isn't "mcts_bot". Omitted defaults every seat to None — see
    # api.matches._resolve_mcts_num_simulations.
    player_mcts_num_simulations: list[int | None] | None = None
    # One entry per player, in seat order - whether that seat's mcts_bot
    # caps how far its search's leading move can pull ahead in visits (see
    # rl.mcts.run_mcts's cap_root_lead). Ignored for seats whose player_type
    # isn't "mcts_bot". Omitted defaults every seat to False - see
    # api.matches._resolve_mcts_cap_root_lead.
    player_mcts_cap_root_lead: list[bool] | None = None


class ActionRequest(BaseModel):
    type: ActionTypeName
    position: int | None = None


class CardOut(BaseModel):
    value: int | None
    face_up: bool

    @classmethod
    def from_card(cls, card: Card) -> CardOut:
        return cls(value=card.value if card.face_up else None, face_up=card.face_up)


class BoardOut(BaseModel):
    cards: list[CardOut | None]

    @classmethod
    def from_board(cls, board: PlayerBoard) -> BoardOut:
        return cls(cards=[CardOut.from_card(c) if c is not None else None for c in board.cards])


class ActionOut(BaseModel):
    type: ActionTypeName
    position: int | None

    @classmethod
    def from_action(cls, action: Action) -> ActionOut:
        return cls(type=action_type_to_name(action.type), position=action.position)


class RoundResultOut(BaseModel):
    """One completed round's contribution to the match: the raw (undoubled)
    per-player scores and who finished it - see `api.store.MatchStore.
    get_round_history`. Lets the UI show a running per-round breakdown
    instead of only the cumulative `total_scores`, and work out for itself
    (mirroring `domain.engine._score_and_close_round`'s rule) whether the
    finisher's score was doubled. `finisher` is None only for a
    `force_close_round` outcome (an escape hatch with no real finisher - see
    that function's docstring), which never doubles."""

    scores: list[int]
    finisher: int | None

    @classmethod
    def from_node(cls, node: MatchNode) -> RoundResultOut:
        assert node.state.round_scores is not None
        return cls(scores=list(node.state.round_scores), finisher=node.state.finisher)


class MatchStateOut(BaseModel):
    match_id: str
    phase: Phase
    boards: list[BoardOut]
    player_names: list[str]
    player_types: list[PlayerTypeName]
    stock_count: int
    discard_top: int | None
    current_player: int
    drawn_card: int | None
    finisher: int | None
    players_awaiting_final_turn: list[int]
    round_scores: list[int] | None
    total_scores: list[int]
    # Every round completed so far on the way to this state, oldest first -
    # see RoundResultOut.
    round_history: list[RoundResultOut]
    target_score: int
    legal_actions: list[ActionOut]
    status: MatchStatus
    thinking_player: int | None
    thinking_progress: float | None

    @classmethod
    def from_state(
        cls,
        match_id: str,
        state: GameState,
        player_names: Sequence[str],
        player_types: Sequence[str],
        autoplay_status: AutoplayStatus,
        round_history: Sequence[MatchNode] = (),
    ) -> MatchStateOut:
        return cls(
            match_id=match_id,
            phase=state.phase,
            boards=[BoardOut.from_board(b) for b in state.boards],
            player_names=list(player_names),
            player_types=list(player_types),
            stock_count=len(state.stock),
            discard_top=state.discard[-1] if state.discard else None,
            current_player=state.current_player,
            drawn_card=state.drawn_card,
            finisher=state.finisher,
            players_awaiting_final_turn=sorted(state.players_awaiting_final_turn),
            round_scores=list(state.round_scores) if state.round_scores is not None else None,
            total_scores=list(state.total_scores),
            round_history=[RoundResultOut.from_node(n) for n in round_history],
            target_score=state.target_score,
            legal_actions=[ActionOut.from_action(a) for a in legal_actions(state)],
            status=autoplay_status.status,
            thinking_player=autoplay_status.player,
            thinking_progress=autoplay_status.progress,
        )


class HistoryEdgeOut(BaseModel):
    kind: Literal["root", "action", "next_round"]
    action_type: ActionTypeName | None = None
    position: int | None = None

    @classmethod
    def from_edge(cls, edge: Edge | None) -> HistoryEdgeOut:
        if edge is None:
            return cls(kind="root")
        if edge.kind == "next_round":
            return cls(kind="next_round")
        assert edge.action is not None
        return cls(kind="action", action_type=action_type_to_name(edge.action.type), position=edge.action.position)


def _mcts_decision_summary(mcts_tree: dict | None) -> tuple[float | None, bool | None]:
    """Two cheap-to-compute signals summarizing a node's recorded search
    (see `MatchNode.mcts_tree`), read off its finished (highest-visit-count)
    snapshot - lets the history panel show "how certain was this" and "did
    search change its mind" per node without shipping the whole tree down the
    wire.

    `mcts_visit_share`: the chosen (most-visited) action's share of total
    root visits - 1.0 means every simulation agreed, lower means the search
    seriously considered an alternative.

    `mcts_prior_overridden`: whether that chosen action differs from the
    network's own raw top prior - i.e. search's conclusion overrode the
    network's first instinct rather than just confirming it.

    Both `None` for a node with no recorded search, or a root with no visited
    edges (can happen for `num_simulations=0`)."""
    if not mcts_tree:
        return None, None
    final = mcts_tree[max(mcts_tree, key=int)]
    edges = final.get("edges", [])
    total_visits = sum(e["visit_count"] for e in edges)
    if not edges or total_visits == 0:
        return None, None
    # `tree_to_dict` sorts a decision node's edges by visit_count descending,
    # so edges[0] is already the chosen action.
    top_by_visits = edges[0]
    top_by_prior = max(edges, key=lambda e: e["prior"])
    return top_by_visits["visit_count"] / total_visits, top_by_visits["action"] != top_by_prior["action"]


class HistoryNodeOut(BaseModel):
    node_id: str
    parent_id: str | None
    seq: int
    round_index: int
    actor: int | None
    current_player: int
    phase: Phase
    edge: HistoryEdgeOut
    # Whether GET /matches/{id}/history/{node_id}/mcts-tree has something to
    # return for this node - lets the history panel show a tree affordance
    # only where one actually exists, without probing every node.
    has_mcts_tree: bool
    # See `_mcts_decision_summary` above. Both None whenever has_mcts_tree is False.
    mcts_visit_share: float | None
    mcts_prior_overridden: bool | None

    @classmethod
    def from_node(cls, node: MatchNode) -> HistoryNodeOut:
        visit_share, prior_overridden = _mcts_decision_summary(node.mcts_tree)
        return cls(
            node_id=node.node_id,
            parent_id=node.parent_id,
            seq=node.seq,
            round_index=node.round_index,
            actor=node.actor,
            current_player=node.state.current_player,
            phase=node.state.phase,
            edge=HistoryEdgeOut.from_edge(node.edge),
            has_mcts_tree=node.mcts_tree is not None,
            mcts_visit_share=visit_share,
            mcts_prior_overridden=prior_overridden,
        )


class MatchHistoryOut(BaseModel):
    head_id: str
    nodes: list[HistoryNodeOut]

    @classmethod
    def from_nodes(cls, nodes: list[MatchNode], head_id: str) -> MatchHistoryOut:
        ordered = sorted(nodes, key=lambda n: n.seq)
        return cls(head_id=head_id, nodes=[HistoryNodeOut.from_node(n) for n in ordered])
