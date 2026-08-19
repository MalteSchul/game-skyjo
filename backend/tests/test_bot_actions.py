from collections.abc import Sequence
from dataclasses import replace

from skyjo.bots.actions import distinct_actions
from skyjo.domain.engine import (
    BOARD_SIZE,
    Action,
    ActionType,
    Card,
    GameState,
    PlayerBoard,
)
from skyjo.domain.observation import Turn

# --- fixture helpers ---------------------------------------------------------


def _board_from_values(values: Sequence[int], *, face_up: bool = False) -> PlayerBoard:
    assert len(values) == BOARD_SIZE
    return PlayerBoard(cards=tuple(Card(value=v, face_up=face_up) for v in values))


def _reveal(board: PlayerBoard, *positions: int) -> PlayerBoard:
    cards = list(board.cards)
    for p in positions:
        cards[p] = replace(cards[p], face_up=True)
    return replace(board, cards=tuple(cards))


def _state(boards: tuple[PlayerBoard, ...], *, drawn_card: int, current_player: int = 0) -> GameState:
    return GameState(
        boards=boards,
        stock=(1, 2, 3),
        discard=(9,),
        current_player=current_player,
        drawn_card=drawn_card,
        finisher=None,
        players_awaiting_final_turn=frozenset(),
        round_scores=None,
        total_scores=tuple(0 for _ in boards),
        phase="awaiting_placement",
        reshuffle_seed=None,
    )


def _positions(actions: Sequence[Action], action_type: ActionType) -> set[int]:
    return {a.position for a in actions if a.type is action_type}


# --- happy path: hidden-slot collapsing --------------------------------------


def test_distinct_actions_collapses_hidden_slots_by_column_context():
    # Column 0 (0,4,8) anchored by a known 4; column 1 (1,5,9) anchored by a
    # *different* known value, 11; columns 2 and 3 (2,6,10 / 3,7,11) are both
    # fully blank. Filler values on hidden positions are irrelevant - they're
    # redacted to None regardless of what's underneath.
    board0 = _reveal(_board_from_values([4, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 0, 1)
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), drawn_card=7)

    turn = Turn.from_state(state)
    actions = distinct_actions(turn)

    place_positions = _positions(actions, ActionType.PLACE)
    reveal_positions = _positions(actions, ActionType.DISCARD_AND_REVEAL)

    # {4, 8} (column 0, anchored by 4) and {5, 9} (column 1, anchored by 11)
    # are each their own equivalence class, and distinct from each other.
    assert len(place_positions & {4, 8}) == 1
    assert len(place_positions & {5, 9}) == 1
    assert len(place_positions & {4, 5, 8, 9}) == 2
    assert len(reveal_positions & {4, 8}) == 1
    assert len(reveal_positions & {5, 9}) == 1
    assert len(reveal_positions & {4, 5, 8, 9}) == 2

    # {2, 3, 6, 7, 10, 11} are all in blank columns - one shared class.
    assert len(place_positions & {2, 3, 6, 7, 10, 11}) == 1
    assert len(reveal_positions & {2, 3, 6, 7, 10, 11}) == 1

    # The already-revealed positions (0 and 1) hold different known values
    # (4 vs 11), only ever offer PLACE, and both survive as their own class.
    assert place_positions & {0, 1} == {0, 1}


# --- edge path: revealed-slot collapsing (the corrected part of the rule) ---


def test_distinct_actions_collapses_revealed_slots_with_matching_value_and_context():
    # 0 and 1 both hold a known 4 with nothing else revealed in their column.
    # 2, 3, 6, 7 all hold a known 5, each paired with exactly one other known
    # 5 in its own column. These two groups must not merge with each other,
    # or with the still-hidden 4/5/8/9 and 10/11 groups from the AB example.
    board0 = _board_from_values([4, 4, 5, 5, 0, 0, 5, 5, 0, 0, 0, 0])
    board0 = _reveal(board0, 0, 1, 2, 3, 6, 7)
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), drawn_card=7)

    turn = Turn.from_state(state)
    actions = distinct_actions(turn)
    place_positions = _positions(actions, ActionType.PLACE)

    assert len(place_positions & {0, 1}) == 1
    assert len(place_positions & {2, 3, 6, 7}) == 1
    assert len(place_positions & {4, 5, 8, 9}) == 1
    assert len(place_positions & {10, 11}) == 1
    # Four distinct classes total among these positions.
    assert len(place_positions) == 4


def test_distinct_actions_never_merges_a_revealed_slot_with_a_different_value():
    # Same column shape (nothing else revealed in either column), but the
    # known values being given up (9 vs 2) differ - must stay separate.
    board0 = _reveal(_board_from_values([9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 0, 1)
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), drawn_card=7)

    turn = Turn.from_state(state)
    place_positions = _positions(distinct_actions(turn), ActionType.PLACE)

    assert {0, 1}.issubset(place_positions)


# --- edge path: action types never merge -------------------------------------


def test_distinct_actions_never_merges_place_and_discard_and_reveal():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), drawn_card=7)

    turn = Turn.from_state(state)
    actions = distinct_actions(turn)

    assert any(a.type is ActionType.PLACE for a in actions)
    assert any(a.type is ActionType.DISCARD_AND_REVEAL for a in actions)


# --- bad path: nothing to collapse -------------------------------------------


def test_distinct_actions_passes_draw_actions_through_unchanged():
    board0 = _board_from_values(list(range(12)))
    board1 = _board_from_values(list(range(12)))
    state = replace(
        _state((board0, board1), drawn_card=0),
        drawn_card=None,
        phase="awaiting_draw",
    )

    turn = Turn.from_state(state)

    assert distinct_actions(turn) == turn.legal_actions
