import numpy as np
import pytest

from skyjo.domain.engine import GameState, legal_actions, new_match
from skyjo.rl.encoding import N_MAX_PLAYERS
from skyjo.rl.evaluator import (
    _play_one_eval_game,
    evaluate_vs_heuristic,
    make_batch_network_evaluator,
    make_network_evaluator,
)
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


# --- make_batch_network_evaluator ------------------------------------------


def test_batch_evaluator_matches_single_evaluator_state_by_state():
    # A batched forward pass must produce, per row, exactly what evaluating
    # that same state alone would - no cross-row mixing anywhere in
    # AlphaZeroNet (LayerNorm/softmax/einsum are all per-row), so batching
    # states together should never change any individual state's result.
    net = _tiny_net()
    state_a = new_match(player_count=2, seed=1)
    state_b = new_match(player_count=4, seed=2)
    single_evaluate = make_network_evaluator(net)
    batch_evaluate = make_batch_network_evaluator(net)

    priors_a, value_a = single_evaluate(state_a)
    priors_b, value_b = single_evaluate(state_b)
    [(batch_priors_a, batch_value_a), (batch_priors_b, batch_value_b)] = batch_evaluate([state_a, state_b])

    assert batch_priors_a.keys() == priors_a.keys()
    for action in priors_a:
        assert batch_priors_a[action] == pytest.approx(priors_a[action], abs=1e-5)
    np.testing.assert_allclose(batch_value_a, value_a, atol=1e-5)

    assert batch_priors_b.keys() == priors_b.keys()
    for action in priors_b:
        assert batch_priors_b[action] == pytest.approx(priors_b[action], abs=1e-5)
    np.testing.assert_allclose(batch_value_b, value_b, atol=1e-5)


def test_batch_evaluator_returns_results_in_the_same_order_as_states():
    net = _tiny_net()
    states = [new_match(player_count=2, seed=i) for i in range(5)]
    batch_evaluate = make_batch_network_evaluator(net)
    single_evaluate = make_network_evaluator(net)

    results = batch_evaluate(states)

    for state, (priors, value) in zip(states, results, strict=True):
        expected_priors, expected_value = single_evaluate(state)
        assert priors.keys() == expected_priors.keys()
        np.testing.assert_allclose(value, expected_value, atol=1e-5)


def test_batch_evaluator_returns_priors_over_legal_actions_and_a_per_player_value():
    state = new_match(player_count=3, seed=1)
    batch_evaluate = make_batch_network_evaluator(_tiny_net())

    [(priors, value)] = batch_evaluate([state])

    assert set(priors.keys()) == set(legal_actions(state))
    assert value.shape == (3,)


# --- bad/edge path ---------------------------------------------------------------


def test_batch_evaluator_with_no_states_returns_an_empty_list_without_touching_the_network():
    batch_evaluate = make_batch_network_evaluator(_tiny_net())

    assert batch_evaluate([]) == []


def test_two_different_states_get_two_separate_sink_entries():
    state_a = new_match(player_count=2, seed=1)
    state_b = new_match(player_count=2, seed=2)
    sink: dict[GameState, np.ndarray] = {}
    evaluate = make_network_evaluator(_tiny_net(), rank_probs_sink=sink)

    evaluate(state_a)
    evaluate(state_b)

    assert len(sink) == 2


# --- evaluate_vs_heuristic ---------------------------------------------------


def test_evaluate_vs_heuristic_plays_the_requested_number_of_games():
    result = evaluate_vs_heuristic(_tiny_net(), 4, num_simulations=2, max_steps=2000, seed=0)

    assert result.games_played == 4
    assert 0.0 <= result.win_rate <= 1.0
    assert 0.0 <= result.avg_rank <= 1.0  # 2-player: 0 (best) .. 1 (worst)


def test_evaluate_vs_heuristic_is_deterministic_for_the_same_seed_and_weights():
    net = _tiny_net()

    first = evaluate_vs_heuristic(net, 3, num_simulations=2, max_steps=2000, seed=5)
    second = evaluate_vs_heuristic(net, 3, num_simulations=2, max_steps=2000, seed=5)

    assert first == second


def test_evaluate_vs_heuristic_supports_zero_simulations():
    # num_simulations=0 still expands the root once via `run_mcts`, so
    # `greedy_action` always has at least one visited edge to pick from -
    # exercised here since it's the cheapest possible eval config.
    result = evaluate_vs_heuristic(_tiny_net(), 2, num_simulations=0, max_steps=2000, seed=0)

    assert result.games_played == 2


# --- bad path ------------------------------------------------------------


def test_evaluate_vs_heuristic_rejects_non_positive_num_games():
    with pytest.raises(ValueError):
        evaluate_vs_heuristic(_tiny_net(), 0, num_simulations=2)


# --- evaluate_vs_heuristic: round_max_steps / max_rounds --------------------


def _degenerate_evaluate(state: GameState):
    actions = legal_actions(state)
    preferred = next((a for a in actions if a.type.name == "DRAW_STOCK"), None)
    if preferred is None:
        preferred = next((a for a in actions if a.type.name == "PLACE" and a.position == 0), None)
    if preferred is None:
        preferred = actions[0]
    priors = {a: (1.0 if a == preferred else 0.0) for a in actions}
    return priors, np.zeros(len(state.boards))


def _degenerate_bot_choose_action(self, turn, *, report_progress=None):
    actions = turn.legal_actions
    preferred = next((a for a in actions if a.type.name == "DRAW_STOCK"), None)
    if preferred is None:
        preferred = next((a for a in actions if a.type.name == "PLACE" and a.position == 0), None)
    return preferred if preferred is not None else actions[0]


def test_play_one_eval_game_force_closes_a_stuck_round_instead_of_raising(monkeypatch):
    # Both seats degenerate (net via a rigged evaluator, HeuristicBot via a
    # monkeypatched choose_action) so neither ever completes their board and
    # the round genuinely never closes on its own - same trap as self-play's.
    monkeypatch.setattr("skyjo.bots.heuristic_bot.HeuristicBot.choose_action", _degenerate_bot_choose_action)

    rank, points = _play_one_eval_game(
        _degenerate_evaluate,
        net_seat=0,
        seed=1,
        num_simulations=2,
        c_puct=1.5,
        max_steps=5000,
        round_max_steps=20,
        max_rounds=3,
    )

    assert rank in (0, 1)
    assert isinstance(points, int)


def test_play_one_eval_game_without_round_max_steps_hits_the_same_trap(monkeypatch):
    monkeypatch.setattr("skyjo.bots.heuristic_bot.HeuristicBot.choose_action", _degenerate_bot_choose_action)

    with pytest.raises(RuntimeError):
        _play_one_eval_game(
            _degenerate_evaluate,
            net_seat=0,
            seed=1,
            num_simulations=2,
            c_puct=1.5,
            max_steps=50,
            round_max_steps=100_000,
            max_rounds=100_000,
        )


def test_evaluate_vs_heuristic_raises_if_a_game_exceeds_max_steps():
    with pytest.raises(RuntimeError):
        evaluate_vs_heuristic(_tiny_net(), 1, num_simulations=1, max_steps=1, seed=0)
