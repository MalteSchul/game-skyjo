from collections.abc import Sequence
from dataclasses import replace

import pytest

from skyjo.domain.deck import DECK_SIZE
from skyjo.domain.engine import (
    BOARD_SIZE,
    Action,
    ActionType,
    Card,
    GameState,
    IllegalActionError,
    PlayerBoard,
    apply_action,
    legal_actions,
    new_match,
    start_next_round,
)

# --- fixture helpers -------------------------------------------------------


def _board_from_values(values: Sequence[int], *, face_up: bool = False) -> PlayerBoard:
    assert len(values) == BOARD_SIZE
    return PlayerBoard(cards=tuple(Card(value=v, face_up=face_up) for v in values))


def _reveal(board: PlayerBoard, *positions: int) -> PlayerBoard:
    cards = list(board.cards)
    for p in positions:
        cards[p] = replace(cards[p], face_up=True)
    return replace(board, cards=tuple(cards))


def _hide(board: PlayerBoard, *positions: int) -> PlayerBoard:
    cards = list(board.cards)
    for p in positions:
        cards[p] = replace(cards[p], face_up=False)
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
        round_scores=None,
        total_scores=total_scores if total_scores is not None else tuple(0 for _ in boards),
        phase=phase,
        reshuffle_seed=reshuffle_seed,
        target_score=target_score,
    )


# --- new_match / dealing ----------------------------------------------------


def test_new_match_deals_valid_round_awaiting_initial_flips():
    state = new_match(player_count=3, seed=42)

    assert state.phase == "initial_flip"
    assert len(state.boards) == 3
    assert all(len(b.cards) == BOARD_SIZE for b in state.boards)
    assert all(not c.face_up for b in state.boards for c in b.cards)
    assert len(state.discard) == 1
    dealt = sum(len(b.cards) for b in state.boards) + len(state.stock) + len(state.discard)
    assert dealt == DECK_SIZE


def test_new_match_rejects_out_of_range_player_count():
    with pytest.raises(ValueError):
        new_match(player_count=1)
    with pytest.raises(ValueError):
        new_match(player_count=9)


def test_new_match_rejects_non_int_seed():
    with pytest.raises(TypeError):
        new_match(player_count=2, seed="nope")  # type: ignore[arg-type]


# --- initial flip / starting player -----------------------------------------


def test_initial_flip_alternates_players_and_starter_is_highest_sum():
    state = new_match(player_count=2, seed=7)
    p0_sum = state.boards[0].cards[0].value + state.boards[0].cards[1].value
    p1_sum = state.boards[1].cards[0].value + state.boards[1].cards[1].value

    state = apply_action(state, Action(ActionType.FLIP_INITIAL, 0))
    assert state.current_player == 1
    state = apply_action(state, Action(ActionType.FLIP_INITIAL, 0))
    assert state.current_player == 0
    state = apply_action(state, Action(ActionType.FLIP_INITIAL, 1))
    assert state.current_player == 1
    state = apply_action(state, Action(ActionType.FLIP_INITIAL, 1))

    assert state.phase == "awaiting_draw"
    expected_starter = 0 if p0_sum >= p1_sum else 1
    assert state.current_player == expected_starter


# --- core turn loop ----------------------------------------------------------


def test_draw_stock_then_place_swaps_card_and_discards_old():
    board0 = _board_from_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2])
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), stock=(99,), discard=(5,), current_player=0, phase="awaiting_draw")

    state = apply_action(state, Action(ActionType.DRAW_STOCK))
    assert state.drawn_card == 99
    assert state.phase == "awaiting_placement"
    assert state.stock == ()

    state = apply_action(state, Action(ActionType.PLACE, position=0))

    assert state.boards[0].cards[0] == Card(value=99, face_up=True)
    assert state.discard[-1] == 1
    assert state.drawn_card is None
    assert state.current_player == 1
    assert state.phase == "awaiting_draw"


def test_discard_and_reveal_clears_a_completed_column():
    board0 = _reveal(_board_from_values([5, 1, 2, 3, 5, 4, 6, 7, 5, 8, 9, 10]), 0, 4)
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1), stock=(1,), discard=(0, 7), current_player=0, drawn_card=42, phase="awaiting_placement"
    )

    state = apply_action(state, Action(ActionType.DISCARD_AND_REVEAL, position=8))

    assert state.boards[0].cards[0] is None
    assert state.boards[0].cards[4] is None
    assert state.boards[0].cards[8] is None
    # The drawn card is discarded first, then the cleared column goes on top of it.
    assert state.discard == (0, 7, 42, 5, 5, 5)
    assert state.current_player == 1


def test_place_clears_a_completed_column_and_discards_the_swapped_out_card():
    board0 = _reveal(_board_from_values([9, 1, 2, 3, 9, 4, 6, 7, 0, 8, 9, 10]), 0, 4)
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1), stock=(1,), discard=(0, 7), current_player=0, drawn_card=9, phase="awaiting_placement"
    )

    state = apply_action(state, Action(ActionType.PLACE, position=8))

    assert state.boards[0].cards[0] is None
    assert state.boards[0].cards[4] is None
    assert state.boards[0].cards[8] is None
    # The card swapped out of position 8 (a 0) is discarded first, then the cleared column.
    assert state.discard == (0, 7, 0, 9, 9, 9)
    assert state.current_player == 1


def test_place_without_completing_a_column_only_discards_the_swapped_out_card():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1), stock=(1,), discard=(0, 7), current_player=0, drawn_card=99, phase="awaiting_placement"
    )

    state = apply_action(state, Action(ActionType.PLACE, position=0))

    assert state.discard == (0, 7, 0)


# --- round end / scoring -----------------------------------------------------


def _finish_round(board0_values, board1_values, *, total_scores=(0, 0), target_score=100):
    board0 = _hide(_board_from_values(board0_values, face_up=True), 11)
    board1 = _hide(_board_from_values(board1_values, face_up=True), 11)
    state = _state(
        (board0, board1),
        stock=(7, 3),
        discard=(0,),
        current_player=0,
        phase="awaiting_draw",
        total_scores=total_scores,
        target_score=target_score,
        reshuffle_seed=1,
    )
    state = apply_action(state, Action(ActionType.DRAW_STOCK))
    state = apply_action(state, Action(ActionType.DISCARD_AND_REVEAL, position=11))
    assert state.finisher == 0
    assert state.players_awaiting_final_turn == frozenset({1})

    state = apply_action(state, Action(ActionType.DRAW_STOCK))
    state = apply_action(state, Action(ActionType.DISCARD_AND_REVEAL, position=11))
    return state


def test_round_ends_and_scores_without_doubling_when_finisher_has_sole_lowest():
    state = _finish_round(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
    )

    assert state.phase == "round_over"
    assert state.round_scores == (52, 67)
    assert state.total_scores == (52, 67)


def test_finisher_score_is_doubled_on_tie_for_lowest():
    state = _finish_round(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, -2, -1],
    )

    assert state.round_scores == (52, 52)
    assert state.total_scores == (104, 52)


def test_round_that_pushes_a_player_over_target_score_ends_the_match():
    state = _finish_round(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        total_scores=(50, 0),
        target_score=100,
    )

    assert state.phase == "game_over"
    assert state.total_scores == (102, 67)
    with pytest.raises(IllegalActionError):
        start_next_round(state)


def test_start_next_round_carries_scores_forward_and_deals_fresh_hands():
    state = _finish_round(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
    )
    assert state.phase == "round_over"
    assert state.reshuffle_seed == 1

    next_state = start_next_round(state)

    assert next_state.phase == "initial_flip"
    assert next_state.total_scores == (52, 67)
    assert all(not c.face_up for b in next_state.boards for c in b.cards)
    assert next_state.reshuffle_seed == 2


# --- stock exhaustion / reshuffle --------------------------------------------


def test_draw_stock_reshuffles_discard_when_stock_is_empty_deterministically():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1),
        stock=(),
        discard=(1, 2, 3, 4, 5),
        current_player=0,
        phase="awaiting_draw",
        reshuffle_seed=99,
    )

    result_a = apply_action(state, Action(ActionType.DRAW_STOCK))
    result_b = apply_action(state, Action(ActionType.DRAW_STOCK))

    assert result_a.discard == (5,)
    assert result_a.stock == result_b.stock
    assert result_a.drawn_card == result_b.drawn_card
    assert result_a.reshuffle_seed == 100
    assert set(result_a.stock) | {result_a.drawn_card} == {1, 2, 3, 4}


# --- illegal actions ----------------------------------------------------------


def test_place_without_a_drawn_card_is_illegal():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), stock=(1,), discard=(2,), phase="awaiting_draw")

    with pytest.raises(IllegalActionError):
        apply_action(state, Action(ActionType.PLACE, position=0))


def test_drawing_while_already_holding_a_card_is_illegal():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1), stock=(1,), discard=(2,), drawn_card=9, phase="awaiting_placement"
    )

    with pytest.raises(IllegalActionError):
        apply_action(state, Action(ActionType.DRAW_STOCK))


def test_discard_and_reveal_targeting_an_already_face_up_card_is_illegal():
    board0 = _reveal(_board_from_values(list(range(12))), 0)
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1), stock=(1,), discard=(2,), drawn_card=9, phase="awaiting_placement"
    )

    with pytest.raises(IllegalActionError):
        apply_action(state, Action(ActionType.DISCARD_AND_REVEAL, position=0))


def test_place_targeting_a_cleared_slot_is_illegal():
    board0 = _clear(_board_from_values(list(range(12))), 0, 4, 8)
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1), stock=(1,), discard=(2,), drawn_card=9, phase="awaiting_placement"
    )

    with pytest.raises(IllegalActionError):
        apply_action(state, Action(ActionType.PLACE, position=0))


def test_legal_actions_is_empty_once_round_or_match_is_over():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), phase="round_over")

    assert legal_actions(state) == []
    assert legal_actions(replace(state, phase="game_over")) == []
