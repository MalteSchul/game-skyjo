from dataclasses import replace

import numpy as np
import pytest

from skyjo.domain.engine import (
    MAX_PLAYERS,
    MIN_PLAYERS,
    Action,
    ActionType,
    apply_action,
    legal_actions,
    new_match,
)
from skyjo.rl.action_space import ACTION_SPACE_SIZE
from skyjo.rl.encoding import (
    _ABSENT_VALUE,
    _BOARD_FEATURES,
    GLOBAL_FEATURES,
    INPUT_DIM,
    N_MAX_PLAYERS,
    _normalize_value,
    encode_state,
)

_BOARD_BLOCK_SIZE = N_MAX_PLAYERS * _BOARD_FEATURES
# Offsets into the global-feature block, matching encode_state's build order.
_DISCARD_TOP_PRESENT = _BOARD_BLOCK_SIZE + 0
_DISCARD_TOP_VALUE = _BOARD_BLOCK_SIZE + 1
_DRAWN_PRESENT = _BOARD_BLOCK_SIZE + 9
_DRAWN_VALUE = _BOARD_BLOCK_SIZE + 10

# --- shape / dtype stability across every legal player count -----------------


@pytest.mark.parametrize("player_count", range(MIN_PLAYERS, MAX_PLAYERS + 1))
def test_encode_state_produces_a_fixed_size_finite_feature_vector(player_count):
    state = new_match(player_count=player_count, seed=1)

    encoding = encode_state(state)

    assert encoding.features.shape == (INPUT_DIM,)
    assert encoding.features.dtype == np.float32
    assert np.isfinite(encoding.features).all()
    assert encoding.legal_action_mask.shape == (ACTION_SPACE_SIZE,)
    assert encoding.active_count == player_count
    assert encoding.active_player == state.current_player


def test_input_dim_does_not_depend_on_player_count():
    small = encode_state(new_match(player_count=2, seed=1))
    large = encode_state(new_match(player_count=8, seed=1))

    assert small.features.shape == large.features.shape


# --- padding for players beyond N_act -----------------------------------------


def test_board_features_for_players_beyond_active_count_are_zero_padded():
    state = new_match(player_count=3, seed=2)

    encoding = encode_state(state)

    board_block = encoding.features[: N_MAX_PLAYERS * _BOARD_FEATURES].reshape(N_MAX_PLAYERS, _BOARD_FEATURES)
    padded_players = board_block[3:]

    assert np.all(padded_players == 0.0)
    assert np.any(board_block[0] != 0.0)  # sanity: real players actually produce features


def test_legal_action_mask_matches_between_encoding_and_action_space_module():
    from skyjo.rl.action_space import legal_action_mask

    state = new_match(player_count=4, seed=3)

    assert np.array_equal(encode_state(state).legal_action_mask, legal_action_mask(state))


# --- bad path ------------------------------------------------------------------


def test_encode_state_rejects_a_state_with_out_of_range_player_count():
    state = new_match(player_count=2, seed=1)
    too_many_boards = state.boards * 5  # 10 boards, past MAX_PLAYERS
    bad_state = replace(state, boards=too_many_boards, total_scores=(0,) * 10)

    with pytest.raises(ValueError):
        encode_state(bad_state)


# --- absent-value sentinel: -1 must never collide with a real card value -----


def test_input_dim_matches_the_documented_layout():
    assert _BOARD_BLOCK_SIZE + GLOBAL_FEATURES + ACTION_SPACE_SIZE == INPUT_DIM


def test_drawn_card_value_feature_is_the_sentinel_when_nothing_is_drawn():
    state = new_match(player_count=2, seed=1)
    while state.phase == "initial_flip":
        state = apply_action(state, legal_actions(state)[0])
    assert state.drawn_card is None  # sanity: still awaiting_draw

    features = encode_state(state).features

    assert features[_DRAWN_PRESENT] == 0.0
    assert features[_DRAWN_VALUE] == _ABSENT_VALUE


def test_drawn_card_value_feature_is_distinguishable_from_the_lowest_real_card():
    # -2 is the lowest real card value and normalizes to 0.0 - the exact
    # collision the sentinel exists to avoid, so pin both endpoints.
    state = new_match(player_count=2, seed=1)
    while state.phase == "initial_flip":
        state = apply_action(state, legal_actions(state)[0])
    state = replace(state, drawn_card=-2, phase="awaiting_placement")

    features = encode_state(state).features

    assert features[_DRAWN_PRESENT] == 1.0
    assert features[_DRAWN_VALUE] == pytest.approx(0.0)
    assert features[_DRAWN_VALUE] != _ABSENT_VALUE


def test_discard_top_value_feature_is_the_sentinel_when_the_discard_pile_is_empty():
    state = new_match(player_count=2, seed=42)
    while state.phase == "initial_flip":
        state = apply_action(state, legal_actions(state)[0])
    assert len(state.discard) == 1  # sanity: exactly the starting discard card
    state = apply_action(state, Action(ActionType.DRAW_DISCARD))
    assert state.discard == ()  # sanity: taking the sole card empties the pile

    features = encode_state(state).features

    assert features[_DISCARD_TOP_PRESENT] == 0.0
    assert features[_DISCARD_TOP_VALUE] == _ABSENT_VALUE


def test_board_card_value_feature_is_the_sentinel_while_face_down():
    # Every card starts face-down: its value feature must be the sentinel,
    # never 0.0 (which is what a face-up -2 would normalize to).
    state = new_match(player_count=2, seed=1)

    board_block = encode_state(state).features[:_BOARD_BLOCK_SIZE].reshape(N_MAX_PLAYERS, _BOARD_FEATURES)
    value_features = board_block[0].reshape(-1, 3)[:, 2]

    assert np.all(value_features == _ABSENT_VALUE)


def test_board_card_value_feature_is_distinguishable_from_the_lowest_real_card_once_flipped():
    state = new_match(player_count=2, seed=1)
    board0 = state.boards[0]
    lowest_card = replace(board0.cards[0], value=-2, face_up=True)
    boards = (replace(board0, cards=(lowest_card,) + board0.cards[1:]),) + state.boards[1:]
    state = replace(state, boards=boards)

    board_block = encode_state(state).features[:_BOARD_BLOCK_SIZE].reshape(N_MAX_PLAYERS, _BOARD_FEATURES)
    first_card_features = board_block[0].reshape(-1, 3)[0]

    assert first_card_features[1] == 1.0  # face_up
    assert first_card_features[2] == pytest.approx(_normalize_value(-2))
    assert first_card_features[2] != _ABSENT_VALUE
