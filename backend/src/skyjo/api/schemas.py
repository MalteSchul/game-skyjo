"""Request/response shapes for the matches API and their mapping to/from domain types.

Face-down cards are redacted to value=None on the way out — the wire format is the
only place that needs to hide information, so it lives here rather than in the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

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

    @classmethod
    def from_state(cls, match_id: str, state: GameState, player_names: Sequence[str]) -> MatchStateOut:
        return cls(
            match_id=match_id,
            phase=state.phase,
            boards=[BoardOut.from_board(b) for b in state.boards],
            player_names=list(player_names),
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
        )
