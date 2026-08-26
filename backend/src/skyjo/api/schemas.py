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

    @classmethod
    def from_node(cls, node: MatchNode) -> HistoryNodeOut:
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
        )


class MatchHistoryOut(BaseModel):
    head_id: str
    nodes: list[HistoryNodeOut]

    @classmethod
    def from_nodes(cls, nodes: list[MatchNode], head_id: str) -> MatchHistoryOut:
        ordered = sorted(nodes, key=lambda n: n.seq)
        return cls(head_id=head_id, nodes=[HistoryNodeOut.from_node(n) for n in ordered])
