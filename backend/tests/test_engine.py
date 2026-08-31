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
    force_close_round,
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
    drawn_card_source: str | None = None,
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
        drawn_card_source=drawn_card_source,
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


# --- discard-sourced placement restriction ------------------------------------


def test_discard_sourced_place_cannot_downgrade_a_face_up_card():
    board0 = _reveal(_board_from_values([5, 1, 2, 3, 7, 4, 6, 7, 9, 8, 9, 10]), 0)
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1),
        stock=(1,),
        discard=(2,),
        current_player=0,
        drawn_card=5,
        phase="awaiting_placement",
        drawn_card_source="discard",
    )

    assert Action(ActionType.PLACE, position=0) not in legal_actions(state)
    with pytest.raises(IllegalActionError):
        apply_action(state, Action(ActionType.PLACE, position=0))


def test_discard_sourced_place_can_still_strictly_improve_a_face_up_card():
    board0 = _reveal(_board_from_values([5, 1, 2, 3, 7, 4, 6, 7, 9, 8, 9, 10]), 0)
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1),
        stock=(1,),
        discard=(2,),
        current_player=0,
        drawn_card=4,
        phase="awaiting_placement",
        drawn_card_source="discard",
    )

    assert Action(ActionType.PLACE, position=0) in legal_actions(state)
    state = apply_action(state, Action(ActionType.PLACE, position=0))
    assert state.boards[0].cards[0] == Card(value=4, face_up=True)


def test_discard_sourced_place_completing_a_column_is_exempt_from_the_restriction():
    # Column (0, 4, 8) shows 5, 5, 2 face-up; the discard-sourced 5 downgrades
    # position 8's own value (2 <= 5) but completes the column, so it must
    # stay legal despite the restriction.
    board0 = _reveal(_board_from_values([5, 1, 2, 3, 5, 4, 6, 7, 2, 8, 9, 10]), 0, 4, 8)
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1),
        stock=(1,),
        discard=(7,),
        current_player=0,
        drawn_card=5,
        phase="awaiting_placement",
        drawn_card_source="discard",
    )

    assert Action(ActionType.PLACE, position=8) in legal_actions(state)
    state = apply_action(state, Action(ActionType.PLACE, position=8))
    assert state.boards[0].cards[0] is None
    assert state.boards[0].cards[4] is None
    assert state.boards[0].cards[8] is None


def test_stock_sourced_place_is_never_restricted():
    board0 = _reveal(_board_from_values([5, 1, 2, 3, 7, 4, 6, 7, 9, 8, 9, 10]), 0)
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1),
        stock=(1,),
        discard=(2,),
        current_player=0,
        drawn_card=5,
        phase="awaiting_placement",
        drawn_card_source="stock",
    )

    assert Action(ActionType.PLACE, position=0) in legal_actions(state)


def test_discard_sourced_restriction_never_applies_to_face_down_targets_or_discard_and_reveal():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state(
        (board0, board1),
        stock=(1,),
        discard=(2,),
        current_player=0,
        drawn_card=0,
        phase="awaiting_placement",
        drawn_card_source="discard",
    )

    assert Action(ActionType.PLACE, position=1) in legal_actions(state)
    assert Action(ActionType.DISCARD_AND_REVEAL, position=1) in legal_actions(state)


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


def test_round_end_reveals_remaining_face_down_cards_but_leaves_cleared_slots_alone():
    board0 = _hide(_board_from_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2], face_up=True), 11)
    board1 = _clear(_board_from_values([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]), 0, 4, 8)
    state = _state(
        (board0, board1),
        stock=(7, 3),
        discard=(0,),
        current_player=0,
        phase="awaiting_draw",
        reshuffle_seed=1,
    )

    state = apply_action(state, Action(ActionType.DRAW_STOCK))
    state = apply_action(state, Action(ActionType.DISCARD_AND_REVEAL, position=11))
    assert state.finisher == 0
    assert state.players_awaiting_final_turn == frozenset({1})

    # Player 1's single final turn only reveals position 1; the rest of their
    # non-cleared cards are still face-down at the moment the round closes.
    state = apply_action(state, Action(ActionType.DRAW_STOCK))
    state = apply_action(state, Action(ActionType.DISCARD_AND_REVEAL, position=1))

    assert state.phase == "round_over"
    board1_after = state.boards[1]
    assert board1_after.cards[0] is None
    assert board1_after.cards[4] is None
    assert board1_after.cards[8] is None
    assert all(c.face_up for c in board1_after.cards if c is not None)
    assert all(c.face_up for c in state.boards[0].cards if c is not None)
    assert state.round_scores == (52, 55)
    assert state.total_scores == (52, 55)


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


def test_force_close_round_reveals_everything_and_scores_with_no_finisher_penalty():
    # board0's sum (52) is not the sole lowest (board1 is lower at 47), which
    # would double board0's score under a *natural* close if board0 were the
    # finisher - force_close_round must not apply that penalty to anyone,
    # since nobody actually finished.
    board0 = _hide(_board_from_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2]), 0, 5, 11)
    board1 = _board_from_values([0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, -2], face_up=True)
    state = _state((board0, board1), current_player=0, phase="awaiting_draw")

    closed = force_close_round(state)

    assert all(c.face_up for c in closed.boards[0].cards if c is not None)
    assert closed.round_scores == (52, 47)
    assert closed.total_scores == (52, 47)  # no doubling, unlike _score_and_close_round
    assert closed.finisher is None
    assert closed.players_awaiting_final_turn == frozenset()
    assert closed.phase == "round_over"


def test_force_close_round_ends_the_match_if_it_crosses_target_score():
    board0 = _board_from_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2], face_up=True)
    board1 = _board_from_values([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], face_up=True)
    state = _state((board0, board1), current_player=0, phase="awaiting_draw", total_scores=(50, 0), target_score=100)

    closed = force_close_round(state)

    assert closed.phase == "game_over"
    assert closed.total_scores == (102, 67)


@pytest.mark.parametrize("phase", ["round_over", "game_over"])
def test_force_close_round_rejects_a_round_that_is_already_over(phase):
    board = _board_from_values([0] * BOARD_SIZE, face_up=True)
    state = _state((board, board), phase=phase)

    with pytest.raises(IllegalActionError):
        force_close_round(state)


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


def test_start_next_round_hands_the_first_flip_to_the_next_seat():
    # new_match's round 0 always starts at seat 0 (see
    # test_initial_flip_alternates_players_and_starter_is_highest_sum) - the
    # very next round must not repeat that, or seat 0 would always flip
    # first, round after round, for the whole match.
    state = _finish_round(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
    )
    assert state.round_number == 0

    next_state = start_next_round(state)

    assert next_state.round_number == 1
    assert next_state.current_player == 1


def test_round_starter_rotation_wraps_around_for_any_player_count():
    board = _board_from_values([0] * BOARD_SIZE, face_up=True)
    state = _state((board, board, board), phase="round_over", total_scores=(0, 0, 0))

    # Round 2 (0-indexed) is the last seat for 3 players; round 3 must wrap
    # back to seat 0 rather than indexing past the end.
    round_2 = start_next_round(replace(state, round_number=1))
    assert round_2.current_player == 2

    round_3 = start_next_round(replace(state, round_number=2))
    assert round_3.current_player == 0


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


def test_apply_action_validate_false_skips_the_legality_check():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), stock=(1,), discard=(2,), phase="awaiting_draw")
    action = Action(ActionType.DRAW_STOCK)  # legal here, so the two calls must agree

    validated = apply_action(state, action)
    unvalidated = apply_action(state, action, validate=False)

    assert validated == unvalidated


def test_apply_action_validate_false_does_not_raise_on_an_illegal_action():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), stock=(1,), discard=(2,), phase="awaiting_draw")

    # PLACE is illegal in "awaiting_draw" (no drawn_card yet) - the default
    # (validate=True) path raises for it; validate=False is a trusted-caller
    # opt-out of that check, not a guarantee the result is meaningful.
    with pytest.raises(IllegalActionError):
        apply_action(state, Action(ActionType.PLACE, position=0))
    apply_action(state, Action(ActionType.PLACE, position=0), validate=False)


def test_legal_actions_is_empty_once_round_or_match_is_over():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), phase="round_over")

    assert legal_actions(state) == []
    assert legal_actions(replace(state, phase="game_over")) == []
