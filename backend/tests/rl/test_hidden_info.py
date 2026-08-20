from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from skyjo.domain.deck import CARD_COUNTS, DECK_SIZE
from skyjo.domain.engine import (
    BOARD_SIZE,
    Action,
    ActionType,
    Card,
    GameState,
    PlayerBoard,
    apply_action,
    legal_actions,
    new_match,
)
from skyjo.domain.observation import Turn
from skyjo.rl.hidden_info import (
    HIDDEN_SENTINEL,
    gamestate_from_turn,
    is_reveal,
    rescrub,
    resolve_drawn_stock_card,
    resolve_reveal,
    resolve_round_close,
    sample_reveal,
    unknown_card_counts,
    will_close_round,
)

# --- fixture helpers -----------------------------------------------------------


def _board_from_values(values, *, face_up: bool = False) -> PlayerBoard:
    assert len(values) == BOARD_SIZE
    return PlayerBoard(cards=tuple(Card(value=v, face_up=face_up) for v in values))


def _reveal(board: PlayerBoard, *positions: int) -> PlayerBoard:
    cards = list(board.cards)
    for p in positions:
        cards[p] = replace(cards[p], face_up=True)
    return replace(board, cards=tuple(cards))


def _state(
    boards,
    *,
    stock=(),
    discard=(1,),
    current_player=0,
    drawn_card=None,
    finisher=None,
    players_awaiting_final_turn=frozenset(),
    total_scores=None,
    phase="awaiting_draw",
    reshuffle_seed=None,
    target_score=100,
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


# --- unknown_card_counts --------------------------------------------------------


def test_unknown_card_counts_is_deck_minus_public_cards():
    board0 = _reveal(_board_from_values([5, -2] + [0] * 10), 0, 1)
    board1 = _board_from_values(list(range(12)))
    state = _state((board0, board1), discard=(9, 7))

    turn = Turn.from_state(state)
    counts = unknown_card_counts(turn)

    expected = Counter(dict(CARD_COUNTS))
    expected[5] -= 1
    expected[-2] -= 1
    expected[9] -= 1
    expected[7] -= 1
    assert counts == expected
    assert sum(counts.values()) == DECK_SIZE - 4


def test_unknown_card_counts_matches_hidden_plus_stock_by_conservation():
    state = new_match(player_count=3, seed=5)
    turn = Turn.from_state(state)

    total_unknown = sum(unknown_card_counts(turn).values())

    hidden_board_cards = sum(
        1 for board in state.boards for card in board.cards if card is not None and not card.face_up
    )
    assert total_unknown == hidden_board_cards + len(state.stock)


def test_unknown_card_counts_excludes_the_drawn_card():
    # Regression: the drawn card is already resolved to a real value (via
    # its own earlier reveal), but sits in `drawn_card` rather than `boards`
    # or `discard` until placed/discarded. Omitting it here let the same
    # value be sampled again for some other hidden card - see
    # resolve_round_close's counterpart test below for the concrete failure.
    board = _board_from_values([HIDDEN_SENTINEL] * 12)
    state = _state((board, board), discard=(1, 2), drawn_card=5, phase="awaiting_placement")

    counts = unknown_card_counts(Turn.from_state(state))

    expected = Counter(dict(CARD_COUNTS))
    expected[1] -= 1
    expected[2] -= 1
    expected[5] -= 1
    assert counts == expected


def test_sample_reveal_raises_when_the_pool_is_exhausted():
    with pytest.raises(ValueError):
        sample_reveal(Counter({5: 0, 7: -1}), np.random.default_rng(0))


def test_sample_reveal_only_ever_returns_a_value_with_positive_count():
    counts = Counter({-2: 0, 0: 3, 4: 1})
    rng = np.random.default_rng(0)

    for _ in range(50):
        assert sample_reveal(counts, rng) in (0, 4)


# --- is_reveal / will_close_round -----------------------------------------------


def test_is_reveal_true_for_flip_discard_reveal_and_draw_stock():
    board = _board_from_values(list(range(12)))
    state = _state((board, board), phase="awaiting_draw")

    assert is_reveal(state, Action(ActionType.DRAW_STOCK))
    assert is_reveal(replace(state, phase="initial_flip"), Action(ActionType.FLIP_INITIAL, position=0))
    assert is_reveal(
        replace(state, phase="awaiting_placement", drawn_card=3),
        Action(ActionType.DISCARD_AND_REVEAL, position=0),
    )


def test_is_reveal_false_for_draw_discard_and_place_on_a_face_up_position():
    board = _reveal(_board_from_values(list(range(12))), 0)
    state = _state((board, board), phase="awaiting_placement", drawn_card=3)

    assert not is_reveal(state, Action(ActionType.PLACE, position=0))
    assert not is_reveal(replace(state, phase="awaiting_draw", drawn_card=None), Action(ActionType.DRAW_DISCARD))


def test_is_reveal_true_for_place_on_a_still_hidden_position():
    board = _board_from_values(list(range(12)))
    state = _state((board, board), phase="awaiting_placement", drawn_card=3)

    assert is_reveal(state, Action(ActionType.PLACE, position=0))


def test_will_close_round_true_only_for_the_last_awaiting_players_closing_action():
    board = _board_from_values(list(range(12)))
    state = _state(
        (board, board, board),
        phase="awaiting_placement",
        drawn_card=3,
        current_player=1,
        finisher=0,
        players_awaiting_final_turn=frozenset({1}),
    )

    assert will_close_round(state, Action(ActionType.PLACE, position=0))
    assert will_close_round(state, Action(ActionType.DISCARD_AND_REVEAL, position=0))


def test_will_close_round_false_when_other_players_are_still_awaiting():
    board = _board_from_values(list(range(12)))
    state = _state(
        (board, board, board),
        phase="awaiting_placement",
        drawn_card=3,
        current_player=1,
        finisher=0,
        players_awaiting_final_turn=frozenset({1, 2}),
    )

    assert not will_close_round(state, Action(ActionType.PLACE, position=0))


def test_will_close_round_false_for_draw_actions_even_when_this_is_the_last_awaiting_player():
    board = _board_from_values(list(range(12)))
    state = _state(
        (board, board),
        phase="awaiting_draw",
        current_player=1,
        finisher=0,
        players_awaiting_final_turn=frozenset({1}),
    )

    assert not will_close_round(state, Action(ActionType.DRAW_STOCK))


# --- resolve_reveal / resolve_drawn_stock_card ----------------------------------


def test_resolve_reveal_writes_the_sampled_value_without_flipping_the_card():
    board = _board_from_values([HIDDEN_SENTINEL] * 12)
    state = _state((board, board), phase="initial_flip")

    resolved = resolve_reveal(state, Action(ActionType.FLIP_INITIAL, position=3), value=7)

    card = resolved.boards[0].cards[3]
    assert card.value == 7
    assert card.face_up is False


def test_resolve_reveal_rejects_draw_stock():
    board = _board_from_values([HIDDEN_SENTINEL] * 12)
    state = _state((board, board), phase="awaiting_draw")

    with pytest.raises(AssertionError):
        resolve_reveal(state, Action(ActionType.DRAW_STOCK), value=7)


def test_resolve_drawn_stock_card_overwrites_drawn_card_only():
    board = _board_from_values(list(range(12)))
    state = _state((board, board), phase="awaiting_placement", drawn_card=HIDDEN_SENTINEL)

    resolved = resolve_drawn_stock_card(state, value=11)

    assert resolved.drawn_card == 11
    assert resolved.boards == state.boards


# --- resolve_round_close ---------------------------------------------------------


def test_resolve_round_close_fills_every_hidden_slot_on_every_board():
    board0 = _board_from_values([HIDDEN_SENTINEL] * 12)
    board0 = replace(board0, cards=(replace(board0.cards[0], value=5, face_up=True),) + board0.cards[1:])
    board1 = _board_from_values([HIDDEN_SENTINEL] * 12)
    state = _state((board0, board1), discard=(1, 2))

    resolved = resolve_round_close(state, np.random.default_rng(0))

    for board in resolved.boards:
        for card in board.cards:
            assert card.value != HIDDEN_SENTINEL, "every still-hidden slot must get a real sampled value"
    # The already-revealed slot on board0 is untouched.
    assert resolved.boards[0].cards[0] == Card(value=5, face_up=True)


def test_resolve_round_close_excludes_an_already_resolved_slot_from_the_pool():
    board0 = _board_from_values([HIDDEN_SENTINEL] * 12)
    board0 = replace(board0, cards=(replace(board0.cards[0], value=-2),) + board0.cards[1:])
    board1 = _board_from_values([HIDDEN_SENTINEL] * 12)
    # Leave exactly one unknown -2 available (deck has 5 total: 4 already
    # public via discard, the 5th is the pre-resolved slot itself).
    state = _state((board0, board1), discard=(-2, -2, -2, -2))

    resolved = resolve_round_close(state, np.random.default_rng(0), already_resolved=(0, 0))

    # The pre-resolved slot must survive untouched...
    assert resolved.boards[0].cards[0].value == -2
    # ...and no other slot can have been handed the same now-exhausted -2.
    other_values = [
        card.value
        for player, board in enumerate(resolved.boards)
        for position, card in enumerate(board.cards)
        if (player, position) != (0, 0)
    ]
    assert -2 not in other_values


def test_resolve_round_close_excludes_the_drawn_card_from_the_pool():
    board0 = _board_from_values([HIDDEN_SENTINEL] * 12)
    board1 = _board_from_values([HIDDEN_SENTINEL] * 12)
    # -2 has 5 total copies: 4 already in discard, the 5th is the drawn card
    # in the current player's hand - known, but not yet placed or discarded,
    # so none should be left for the pool that fills every other hidden slot.
    state = _state((board0, board1), discard=(-2, -2, -2, -2), drawn_card=-2)

    resolved = resolve_round_close(state, np.random.default_rng(0))

    all_values = [card.value for board in resolved.boards for card in board.cards]
    assert -2 not in all_values


# --- rescrub -----------------------------------------------------------------


def test_rescrub_blanks_hidden_cards_and_stock_but_preserves_public_fields():
    board0 = _reveal(_board_from_values([5, -1] + [3] * 10), 0)
    state = _state((board0, board0), stock=(1, 2, 3), discard=(9,), reshuffle_seed=42)

    scrubbed = rescrub(state)

    assert scrubbed.boards[0].cards[0] == Card(value=5, face_up=True)  # public, untouched
    assert scrubbed.boards[0].cards[1] == Card(value=HIDDEN_SENTINEL, face_up=False)
    assert scrubbed.stock == (HIDDEN_SENTINEL, HIDDEN_SENTINEL, HIDDEN_SENTINEL)
    assert scrubbed.discard == (9,)
    assert scrubbed.reshuffle_seed is None


def test_rescrub_is_idempotent():
    state = new_match(player_count=2, seed=3)

    once = rescrub(state)
    twice = rescrub(once)

    assert once == twice


# --- gamestate_from_turn ----------------------------------------------------------


def test_gamestate_from_turn_round_trips_back_to_an_identical_turn():
    state = new_match(player_count=3, seed=9)
    while state.phase == "initial_flip":
        state = apply_action(state, legal_actions(state)[0])
    original_turn = Turn.from_state(state)

    redacted = gamestate_from_turn(original_turn)
    round_tripped_turn = Turn.from_state(redacted)

    assert round_tripped_turn == original_turn


def test_gamestate_from_turn_preserves_legal_actions():
    state = new_match(player_count=4, seed=11)
    while state.phase == "initial_flip":
        state = apply_action(state, legal_actions(state)[0])
    turn = Turn.from_state(state)

    redacted = gamestate_from_turn(turn)

    assert legal_actions(redacted) == list(turn.legal_actions)


def test_gamestate_from_turn_never_carries_a_real_hidden_value():
    state = new_match(player_count=2, seed=1)  # still all face-down
    turn = Turn.from_state(state)

    redacted = gamestate_from_turn(turn)

    for board in redacted.boards:
        for card in board.cards:
            assert card is not None
            if not card.face_up:
                assert card.value == HIDDEN_SENTINEL
    assert set(redacted.stock) <= {HIDDEN_SENTINEL}
