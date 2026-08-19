from dataclasses import replace

import numpy as np
import pytest

from skyjo.domain.engine import MAX_PLAYERS, MIN_PLAYERS, new_match
from skyjo.rl.action_space import ACTION_SPACE_SIZE
from skyjo.rl.encoding import _BOARD_FEATURES, INPUT_DIM, N_MAX_PLAYERS, encode_state

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
