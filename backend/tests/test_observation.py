import dataclasses
from collections.abc import Sequence
from dataclasses import replace

import pytest

from skyjo.domain.engine import (
    BOARD_SIZE,
    Action,
    ActionType,
    Card,
    GameState,
    IllegalActionError,
    PlayerBoard,
    apply_action,
)
from skyjo.domain.observation import CardView, Observation, Turn

# --- fixture helpers ---------------------------------------------------------


def _board_from_values(values: Sequence[int], *, face_up: bool = False) -> PlayerBoard:
    assert len(values) == BOARD_SIZE
    return PlayerBoard(cards=tuple(Card(value=v, face_up=face_up) for v in values))


def _reveal(board: PlayerBoard, *positions: int) -> PlayerBoard:
    cards = list(board.cards)
    for p in positions:
        cards[p] = replace(cards[p], face_up=True)
    return replace(board, cards=tuple(cards))


def _clear(board: PlayerBoard, *positions: int) -> PlayerBoard:
    cards = list(board.cards)
    for p in positions:
        cards[p] = None
    return replace(board, cards=tuple(cards))


def _state(
    boards: tuple[PlayerBoard, ...],
    *,
    stock: tuple[int, ...] = (),
    discard: tuple[int, ...] = (1,),
    current_player: int = 0,
    drawn_card: int | None = None,
    finisher: int | None = None,
    players_awaiting_final_turn: frozenset[int] = frozenset(),
    round_scores: tuple[int, ...] | None = None,
    total_scores: tuple[int, ...] | None = None,
    phase: str = "awaiting_draw",
    reshuffle_seed: int | None = None,
    target_score: int = 100,
) -> GameState:
    return GameState(
        boards=boards,
        stock=stock,
        discard=discard,
        current_player=current_player,
        drawn_card=drawn_card,
        finisher=finisher,
        players_awaiting_final_turn=players_awaiting_final_turn,
        round_scores=round_scores,
        total_scores=total_scores if total_scores is not None else tuple(0 for _ in boards),
        phase=phase,
        reshuffle_seed=reshuffle_seed,
        target_score=target_score,
    )


# --- Observation --------------------------------------------------------------


def test_observation_redacts_face_down_cards_and_exposes_public_state():
    board0 = _reveal(_board_from_values([5, -2, 2, 3, 5, 4, 6, 7, 5, 8, 9, 10]), 0, 4, 8)
    board1 = _clear(_board_from_values(list(range(12))), 0, 4, 8)
    state = _state(
        (board0, board1),
        stock=(1, 2, 3),
        discard=(9, 7),
        current_player=1,
        phase="awaiting_draw",
    )

    obs = Observation.from_state(state)

    assert obs.boards[0].cards[0] == CardView(value=5, face_up=True)
    assert obs.boards[0].cards[1] == CardView(value=None, face_up=False)
    assert obs.boards[1].cards[0] is None
    assert obs.discard_top == 7
    assert obs.discard_count == 2
    assert obs.stock_count == 3
    assert obs.current_player == 1
    assert obs.legal_actions == (Action(ActionType.DRAW_STOCK), Action(ActionType.DRAW_DISCARD))


def test_observation_includes_round_scores_once_the_round_has_closed():
    board0 = _board_from_values(list(range(12)), face_up=True)
    board1 = _board_from_values(list(range(12)), face_up=True)
    state = _state((board0, board1), phase="round_over", round_scores=(52, 67), total_scores=(52, 67))

    obs = Observation.from_state(state)

    assert obs.round_scores == (52, 67)
    assert obs.legal_actions == ()


def test_observation_does_not_redact_the_drawn_card():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), drawn_card=7, phase="awaiting_placement")

    obs = Observation.from_state(state)

    assert obs.drawn_card == 7


# --- Turn -------------------------------------------------------------


def test_turn_mirrors_observation_for_the_fields_a_policy_needs():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), stock=(9,), drawn_card=None, current_player=0, phase="awaiting_draw")

    turn = Turn.from_state(state)

    assert turn.acting_player == 0
    assert turn.phase == "awaiting_draw"
    assert turn.boards[0].cards[0] == CardView(value=None, face_up=False)
    assert turn.stock_count == 1
    assert turn.legal_actions == (Action(ActionType.DRAW_STOCK), Action(ActionType.DRAW_DISCARD))


def test_turn_has_no_round_scores_field_at_all():
    field_names = {f.name for f in dataclasses.fields(Turn)}
    assert "round_scores" not in field_names


def test_turn_raises_once_the_round_or_match_is_over():
    board0 = _board_from_values(list(range(12)), face_up=True)
    board1 = _board_from_values(list(range(12)), face_up=True)
    state = _state((board0, board1), phase="round_over", round_scores=(52, 67), total_scores=(52, 67))

    with pytest.raises(IllegalActionError):
        Turn.from_state(state)
    with pytest.raises(IllegalActionError):
        Turn.from_state(replace(state, phase="game_over"))


def test_turn_reflects_state_after_an_action_is_applied():
    board0 = _board_from_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2])
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), stock=(99,), discard=(5,), current_player=0, phase="awaiting_draw")

    state = apply_action(state, Action(ActionType.DRAW_STOCK))
    turn = Turn.from_state(state)

    assert turn.drawn_card == 99
    assert turn.phase == "awaiting_placement"
    assert Action(ActionType.PLACE, position=0) in turn.legal_actions
