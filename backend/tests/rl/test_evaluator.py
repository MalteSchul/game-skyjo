import numpy as np
import pytest

from skyjo.domain.engine import GameState, legal_actions, new_match
from skyjo.rl.encoding import N_MAX_PLAYERS
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.network import AlphaZeroNet

# --- fixture helpers -----------------------------------------------------------


def _tiny_net() -> AlphaZeroNet:
    return AlphaZeroNet(trunk_dim=16, num_residual_blocks=1)


# --- happy path -----------------------------------------------------------------


def test_returns_priors_over_legal_actions_and_a_per_player_value():
    state = new_match(player_count=3, seed=1)
    evaluate = make_network_evaluator(_tiny_net())

    priors, value = evaluate(state)

    assert set(priors.keys()) == set(legal_actions(state))
    assert value.shape == (3,)


def test_rank_probs_sink_is_untouched_when_not_given():
    state = new_match(player_count=2, seed=1)
    evaluate = make_network_evaluator(_tiny_net())

    evaluate(state)  # no sink passed - nothing to assert on, just must not raise


def test_rank_probs_sink_is_filled_in_as_a_side_effect():
    state = new_match(player_count=3, seed=1)
    sink: dict[GameState, np.ndarray] = {}
    evaluate = make_network_evaluator(_tiny_net(), rank_probs_sink=sink)

    evaluate(state)

    assert state in sink
    assert sink[state].shape == (3, 3)


def test_rank_probs_rows_sum_to_one_for_each_active_player():
    state = new_match(player_count=4, seed=2)
    sink: dict[GameState, np.ndarray] = {}
    evaluate = make_network_evaluator(_tiny_net(), rank_probs_sink=sink)

    evaluate(state)

    for row in sink[state]:
        assert row.sum() == pytest.approx(1.0, abs=1e-4)


def test_value_is_the_rank_probs_weighted_by_the_fixed_rank_utility_mapping():
    # value[i] = sum_r rank_probs[i, r] * w[r], w[r] = 1 - 2r/(n-1) - see
    # AlphaZeroNet.forward's docstring. Confirms the sink exposes the exact
    # un-reduced distribution `value` is computed from, not something else.
    state = new_match(player_count=3, seed=1)
    sink: dict[GameState, np.ndarray] = {}
    evaluate = make_network_evaluator(_tiny_net(), rank_probs_sink=sink)

    _, value = evaluate(state)
    rank_probs = sink[state]

    n = 3
    weights = np.array([1.0 - 2.0 * r / (n - 1) for r in range(n)])
    reconstructed = rank_probs @ weights
    np.testing.assert_allclose(reconstructed, value, atol=1e-4)


def test_rank_probs_sink_only_holds_the_active_n_by_n_slice():
    # active_count for a 2-player match is 2 - the network's rank head is
    # sized for up to N_MAX_PLAYERS, but the sink should only ever expose the
    # slice that's actually meaningful for this state.
    state = new_match(player_count=2, seed=1)
    assert N_MAX_PLAYERS > 2
    sink: dict[GameState, np.ndarray] = {}
    evaluate = make_network_evaluator(_tiny_net(), rank_probs_sink=sink)

    evaluate(state)

    assert sink[state].shape == (2, 2)


# --- bad/edge path ---------------------------------------------------------------


def test_two_different_states_get_two_separate_sink_entries():
    state_a = new_match(player_count=2, seed=1)
    state_b = new_match(player_count=2, seed=2)
    sink: dict[GameState, np.ndarray] = {}
    evaluate = make_network_evaluator(_tiny_net(), rank_probs_sink=sink)

    evaluate(state_a)
    evaluate(state_b)

    assert len(sink) == 2
