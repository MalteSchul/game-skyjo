from dataclasses import replace

import numpy as np
import pytest

from skyjo.domain.action_equivalence import distinct_actions
from skyjo.domain.engine import (
    MAX_PLAYERS,
    MIN_PLAYERS,
    Action,
    ActionType,
    apply_action,
    legal_actions,
    new_match,
)
from skyjo.domain.observation import Turn
from skyjo.rl.action_space import ACTION_SPACE_SIZE
from skyjo.rl.encoding import (
    _ABSENT_VALUE,
    _BOARD_FEATURES,
    GLOBAL_FEATURES,
    INPUT_DIM,
    N_MAX_PLAYERS,
    _normalize_value,
    encode_state,
    rotation_perm,
)

_BOARD_BLOCK_SIZE = N_MAX_PLAYERS * _BOARD_FEATURES
# Offsets into the global-feature block, matching encode_state's build order.
_DISCARD_TOP_PRESENT = _BOARD_BLOCK_SIZE + 0
_DISCARD_TOP_VALUE = _BOARD_BLOCK_SIZE + 1
_DRAWN_PRESENT = _BOARD_BLOCK_SIZE + 9
_DRAWN_VALUE = _BOARD_BLOCK_SIZE + 10
_FINISHER_PRESENT = _DRAWN_VALUE + 1
_FINISHER_ONEHOT = _FINISHER_PRESENT + 1
_AWAITING_MULTIHOT = _FINISHER_ONEHOT + N_MAX_PLAYERS
_TOTAL_SCORES = _AWAITING_MULTIHOT + N_MAX_PLAYERS

# --- rotation_perm: canonical (current-player-at-slot-0) permutation ---------


@pytest.mark.parametrize("n_act", range(MIN_PLAYERS, MAX_PLAYERS + 1))
def test_rotation_perm_puts_current_player_at_slot_zero(n_act):
    for current_player in range(n_act):
        perm = rotation_perm(current_player, n_act)
        assert perm[0] == current_player


@pytest.mark.parametrize("n_act", range(MIN_PLAYERS, MAX_PLAYERS + 1))
def test_rotation_perm_is_a_bijection_on_real_players(n_act):
    for current_player in range(n_act):
        perm = rotation_perm(current_player, n_act)
        assert sorted(perm[:n_act].tolist()) == list(range(n_act))


@pytest.mark.parametrize("n_act", range(MIN_PLAYERS, MAX_PLAYERS + 1))
def test_rotation_perm_round_trips_through_rotate_then_scatter(n_act):
    for current_player in range(n_act):
        perm = rotation_perm(current_player, n_act)
        original = np.arange(n_act) * 10  # distinct values so misplacement is obvious
        rotated = original[perm[:n_act]]
        recovered = np.empty_like(rotated)
        recovered[perm[:n_act]] = rotated
        np.testing.assert_array_equal(recovered, original)


def test_rotation_perm_leaves_padding_slots_as_identity():
    perm = rotation_perm(current_player=1, n_act=3)
    np.testing.assert_array_equal(perm[3:], np.arange(3, N_MAX_PLAYERS))


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
    assert encoding.perm[0] == state.current_player


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


# --- canonical ordering: encode_state output rotates with current_player -----


def test_board_and_total_scores_blocks_rotate_with_current_player():
    # Same underlying boards/scores, only current_player differs - board_block
    # and total_scores_norm must shift by exactly the resulting permutation,
    # since that's the whole point of canonical (rotated) encoding. Player 0
    # as current_player is the identity permutation, so its encoding gives us
    # each absolute player's board/score features as an unrotated baseline.
    state = new_match(player_count=4, seed=7)
    state = replace(state, total_scores=(3, 5, 7, 11), current_player=0)

    baseline = encode_state(state)
    baseline_board_block = baseline.features[:_BOARD_BLOCK_SIZE].reshape(N_MAX_PLAYERS, _BOARD_FEATURES)
    baseline_total_scores = baseline.features[_TOTAL_SCORES : _TOTAL_SCORES + N_MAX_PLAYERS]

    for current_player in range(1, 4):
        rotated_state = replace(state, current_player=current_player)
        encoding = encode_state(rotated_state)
        perm = encoding.perm

        board_block = encoding.features[:_BOARD_BLOCK_SIZE].reshape(N_MAX_PLAYERS, _BOARD_FEATURES)
        for canonical_slot in range(4):
            np.testing.assert_array_equal(board_block[canonical_slot], baseline_board_block[perm[canonical_slot]])

        total_scores_norm = encoding.features[_TOTAL_SCORES : _TOTAL_SCORES + N_MAX_PLAYERS]
        for canonical_slot in range(4):
            assert total_scores_norm[canonical_slot] == pytest.approx(baseline_total_scores[perm[canonical_slot]])


def test_finisher_onehot_rotates_with_current_player():
    state = new_match(player_count=4, seed=7)
    state = replace(state, finisher=2)

    for current_player in range(4):
        rotated_state = replace(state, current_player=current_player)
        encoding = encode_state(rotated_state)

        expected_slot = (state.finisher - current_player) % 4
        finisher_onehot = encoding.features[_FINISHER_ONEHOT : _FINISHER_ONEHOT + N_MAX_PLAYERS]
        assert finisher_onehot[expected_slot] == 1.0
        assert finisher_onehot.sum() == 1.0


def test_encode_state_defaults_to_the_collapsed_representative_set_not_every_raw_legal_action():
    # Fresh initial_flip: every FLIP_INITIAL position is provably equivalent
    # (nothing revealed anywhere yet) - domain.action_equivalence collapses
    # all of them onto one representative, and that's what the network's
    # policy head should see, not all of the raw per-position actions.
    state = new_match(player_count=4, seed=3)
    turn = Turn.from_state(state)
    assert len(turn.legal_actions) > 1  # sanity: there's real collapsing to observe

    mask = encode_state(state).legal_action_mask

    assert int(mask.sum()) == len(distinct_actions(turn))
    assert int(mask.sum()) < len(turn.legal_actions)


def test_encode_state_mask_matches_the_raw_legal_action_mask_when_nothing_can_collapse():
    # awaiting_draw: DRAW_STOCK vs DRAW_DISCARD are never equivalent to each
    # other, so collapsing is a no-op here - the two masks should agree exactly.
    from skyjo.rl.action_space import legal_action_mask

    state = new_match(player_count=2, seed=1)
    while state.phase == "initial_flip":
        state = apply_action(state, legal_actions(state)[0])

    assert np.array_equal(encode_state(state).legal_action_mask, legal_action_mask(state))


def test_encode_state_with_precomputed_legal_actions_matches_recomputing_them():
    state = new_match(player_count=3, seed=1)
    turn = Turn.from_state(state)

    recomputed = encode_state(state)
    precomputed = encode_state(state, legal_actions=distinct_actions(turn))

    np.testing.assert_array_equal(precomputed.features, recomputed.features)
    np.testing.assert_array_equal(precomputed.legal_action_mask, recomputed.legal_action_mask)


def test_encode_state_actually_uses_the_precomputed_legal_actions_not_just_ignoring_them():
    state = new_match(player_count=3, seed=1)

    encoding = encode_state(state, legal_actions=[])

    assert not encoding.legal_action_mask.any()  # would be non-empty if the override were ignored


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
