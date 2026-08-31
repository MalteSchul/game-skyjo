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
from typing import Literal

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
    # The full pile, not just the top: every discard is a public event, so
    # its entire history is fair game for a viewer (or a policy) to know -
    # unlike stock order, nothing here is hidden. discard_top/discard_count
    # are kept alongside it as the two projections most consumers actually need.
    discard: tuple[int, ...]
    discard_top: int | None
    discard_count: int
    stock_count: int
    current_player: int
    phase: Phase
    drawn_card: int | None
    # Where drawn_card came from - not hidden info (the acting player always
    # knows this about their own draw), so it's exposed here even though it
    # only affects legality, not what's shown to a viewer. See
    # engine.legal_actions's discard-swap restriction.
    drawn_card_source: Literal["stock", "discard"] | None
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
            discard=state.discard,
            discard_top=state.discard[-1] if state.discard else None,
            discard_count=len(state.discard),
            stock_count=len(state.stock),
            current_player=state.current_player,
            phase=state.phase,
            drawn_card=state.drawn_card,
            drawn_card_source=state.drawn_card_source,
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
    # Full pile - see Observation.discard. This is what makes Turn alone
    # enough to compute the odds behind a hidden card or a stock draw: the
    # only genuinely secret things left are stock order and face-down values,
    # and both are exactly deck composition minus every card counted here
    # (revealed board cards) or in this pile.
    discard: tuple[int, ...]
    discard_top: int | None
    discard_count: int
    stock_count: int
    drawn_card: int | None
    # See Observation.drawn_card_source - not hidden info, and specifically
    # needed here: a policy conditions on legal_actions, and this is what
    # makes engine.legal_actions's discard-swap restriction reproducible
    # from a Turn alone (see rl.hidden_info.gamestate_from_turn).
    drawn_card_source: Literal["stock", "discard"] | None
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
            discard=state.discard,
            discard_top=state.discard[-1] if state.discard else None,
            discard_count=len(state.discard),
            stock_count=len(state.stock),
            drawn_card=state.drawn_card,
            drawn_card_source=state.drawn_card_source,
            finisher=state.finisher,
            players_awaiting_final_turn=state.players_awaiting_final_turn,
            total_scores=state.total_scores,
            target_score=state.target_score,
            legal_actions=actions,
        )
