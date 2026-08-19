import numpy as np
import pytest

from skyjo.domain.engine import Action, ActionType, new_match
from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.rl.action_space import (
    ACTION_SPACE_SIZE,
    action_to_index,
    index_to_action,
    legal_action_mask,
    pi_to_vector,
)

# --- action_to_index / index_to_action ---------------------------------------


def test_every_index_round_trips_through_index_to_action_and_back():
    for index in range(ACTION_SPACE_SIZE):
        action = index_to_action(index)
        assert action_to_index(action) == index


def test_indices_are_unique_across_the_whole_action_space():
    indices = {action_to_index(index_to_action(i)) for i in range(ACTION_SPACE_SIZE)}
    assert len(indices) == ACTION_SPACE_SIZE


def test_non_positional_action_types_do_not_require_a_position():
    assert action_to_index(Action(ActionType.DRAW_STOCK)) != action_to_index(Action(ActionType.DRAW_DISCARD))


def test_action_to_index_rejects_positional_type_missing_a_position():
    with pytest.raises(ValueError):
        action_to_index(Action(type=ActionType.PLACE, position=None))


def test_index_to_action_rejects_out_of_range_index():
    with pytest.raises(ValueError):
        index_to_action(-1)
    with pytest.raises(ValueError):
        index_to_action(ACTION_SPACE_SIZE)


# --- legal_action_mask --------------------------------------------------------


def test_legal_action_mask_matches_engine_legal_actions():
    state = new_match(player_count=3, seed=1)
    mask = legal_action_mask(state)

    expected_indices = {action_to_index(a) for a in engine_legal_actions(state)}
    actual_indices = set(np.flatnonzero(mask).tolist())

    assert actual_indices == expected_indices
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype == bool


def test_legal_action_mask_is_all_false_at_round_over():
    state = new_match(player_count=2, seed=1)
    state = _play_until_round_over(state)

    mask = legal_action_mask(state)

    assert not mask.any()


def _play_until_round_over(state):
    import random

    rng = random.Random(0)
    for _ in range(5000):
        if state.phase in ("round_over", "game_over"):
            return state
        actions = engine_legal_actions(state)
        from skyjo.domain.engine import apply_action

        state = apply_action(state, rng.choice(actions))
    raise AssertionError("did not reach round_over")


# --- pi_to_vector --------------------------------------------------------------


def test_pi_to_vector_places_mass_at_the_right_indices_and_sums_to_one():
    a1 = Action(ActionType.DRAW_STOCK)
    a2 = Action(ActionType.DRAW_DISCARD)
    pi = {a1: 0.75, a2: 0.25}

    vector = pi_to_vector(pi)

    assert vector.shape == (ACTION_SPACE_SIZE,)
    assert vector[action_to_index(a1)] == pytest.approx(0.75)
    assert vector[action_to_index(a2)] == pytest.approx(0.25)
    assert vector.sum() == pytest.approx(1.0)


def test_pi_to_vector_of_empty_distribution_is_all_zero():
    vector = pi_to_vector({})

    assert vector.sum() == 0.0
