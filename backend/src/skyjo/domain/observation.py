"""Redacted views of GameState for consumers outside the referee.

GameState itself hides nothing - every face-down Card still carries its true
value, because the engine needs it (dealing, column-clear checks, scoring).
Observation and Turn are the two redacted projections built from it:

- Observation is the full public snapshot: everything any viewer could fairly
  know, including round_scores. Used for the API, UI, spectators, and as the
  reward signal source for training.
- Turn is only the fields a policy conditions on to pick from legal_actions.
  round_scores is deliberately absent: it's only ever populated when
  legal_actions is empty (round_over/game_over), so no decision can ever
  depend on it. Turn.from_state raises IllegalActionError in that case,
  same as start_next_round raises when called outside phase == "round_over" -
  both are "there's nothing valid to do here" rather than a normal result.

The two are built independently from GameState rather than one from the
other, so Turn's shape is never accidentally widened by fields that only
make sense for the general Observation.
"""

from __future__ import annotations

from dataclasses import dataclass

from skyjo.domain.engine import (
    Action,
    Card,
    GameState,
    IllegalActionError,
    Phase,
    PlayerBoard,
    legal_actions,
)


@dataclass(frozen=True)
class CardView:
    # value is None whenever face_up is False - nobody, including the card's
    # owner, knows a face-down card's value until it's flipped.
    value: int | None
    face_up: bool

    @classmethod
    def from_card(cls, card: Card) -> CardView:
        return cls(value=card.value if card.face_up else None, face_up=card.face_up)


@dataclass(frozen=True)
class BoardView:
    # None = a cleared column slot, same as PlayerBoard.cards.
    cards: tuple[CardView | None, ...]

    @classmethod
    def from_board(cls, board: PlayerBoard) -> BoardView:
        return cls(cards=tuple(CardView.from_card(c) if c is not None else None for c in board.cards))


@dataclass(frozen=True)
class Observation:
    boards: tuple[BoardView, ...]
    discard_top: int | None
    discard_count: int
    stock_count: int
    current_player: int
    phase: Phase
    drawn_card: int | None
    finisher: int | None
    players_awaiting_final_turn: frozenset[int]
    round_scores: tuple[int, ...] | None
    total_scores: tuple[int, ...]
    target_score: int
    legal_actions: tuple[Action, ...]

    @classmethod
    def from_state(cls, state: GameState) -> Observation:
        return cls(
            boards=tuple(BoardView.from_board(b) for b in state.boards),
            discard_top=state.discard[-1] if state.discard else None,
            discard_count=len(state.discard),
            stock_count=len(state.stock),
            current_player=state.current_player,
            phase=state.phase,
            drawn_card=state.drawn_card,
            finisher=state.finisher,
            players_awaiting_final_turn=state.players_awaiting_final_turn,
            round_scores=state.round_scores,
            total_scores=state.total_scores,
            target_score=state.target_score,
            legal_actions=tuple(legal_actions(state)),
        )


@dataclass(frozen=True)
class Turn:
    acting_player: int
    phase: Phase
    boards: tuple[BoardView, ...]
    discard_top: int | None
    discard_count: int
    stock_count: int
    drawn_card: int | None
    finisher: int | None
    players_awaiting_final_turn: frozenset[int]
    total_scores: tuple[int, ...]
    target_score: int
    legal_actions: tuple[Action, ...]

    @classmethod
    def from_state(cls, state: GameState) -> Turn:
        actions = tuple(legal_actions(state))
        if not actions:
            raise IllegalActionError(f"Turn.from_state: no legal actions in phase {state.phase!r}")
        return cls(
            acting_player=state.current_player,
            phase=state.phase,
            boards=tuple(BoardView.from_board(b) for b in state.boards),
            discard_top=state.discard[-1] if state.discard else None,
            discard_count=len(state.discard),
            stock_count=len(state.stock),
            drawn_card=state.drawn_card,
            finisher=state.finisher,
            players_awaiting_final_turn=state.players_awaiting_final_turn,
            total_scores=state.total_scores,
            target_score=state.target_score,
            legal_actions=actions,
        )
