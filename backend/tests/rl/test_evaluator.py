from dataclasses import replace

import numpy as np
import pytest
import torch

from skyjo.domain.action_equivalence import distinct_actions
from skyjo.domain.engine import GameState, apply_action, legal_actions, new_match
from skyjo.domain.observation import Turn
from skyjo.rl.encoding import N_MAX_PLAYERS, encode_state
from skyjo.rl.evaluator import (
    _play_one_eval_game,
    evaluate_vs_heuristic,
    make_batch_network_evaluator,
    make_network_evaluator,
)
from skyjo.rl.mcts import DEFAULT_C_PUCT
from skyjo.rl.network import AlphaZeroNet

# --- fixture helpers -----------------------------------------------------------


def _tiny_net() -> AlphaZeroNet:
    return AlphaZeroNet(trunk_dim=16, num_residual_blocks=1)


# --- happy path -----------------------------------------------------------------


def test_returns_priors_over_the_collapsed_legal_actions_and_a_per_player_value():
    # A fresh match's initial_flip: every raw FLIP_INITIAL position is
    # provably equivalent, so the network's priors should span only the
    # collapsed representative set - never the full raw legal-action list.
    state = new_match(player_count=3, seed=1)
    evaluate = make_network_evaluator(_tiny_net())

    priors, value = evaluate(state)

    assert set(priors.keys()) == set(distinct_actions(Turn.from_state(state)))
    assert set(priors.keys()) < set(legal_actions(state))
    assert value.shape == (3,)


def test_evaluate_with_a_precomputed_turn_matches_recomputing_it():
    state = new_match(player_count=3, seed=1)
    net = _tiny_net()
    evaluate = make_network_evaluator(net)

    recomputed_priors, recomputed_value = evaluate(state)
    precomputed_priors, precomputed_value = evaluate(state, turn=Turn.from_state(state))

    assert precomputed_priors.keys() == recomputed_priors.keys()
    for action in precomputed_priors:
        assert precomputed_priors[action] == pytest.approx(recomputed_priors[action])
    np.testing.assert_allclose(precomputed_value, recomputed_value)


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


def test_rank_probs_sink_is_un_rotated_to_absolute_player_order():
    # `evaluate` runs the net on canonical (current-player-at-slot-0) input,
    # but the sink is documented to expose `rank_probs[i, r]` for absolute
    # player i - if the un-rotation in evaluator.py ever got dropped or
    # inverted, this state's rotation (current_player=1, a genuine rotation
    # for n_act=3) would make every other rank_probs test above still pass
    # (they only check shape/sums/the value-reconstruction identity, which
    # are rotation-invariant) while silently mislabeling which row belongs
    # to which player.
    state = new_match(player_count=3, seed=1)
    while state.current_player == 0:
        action = next(iter(legal_actions(state)))
        state = apply_action(state, action)
    net = _tiny_net()
    net.eval()

    encoding = encode_state(state)
    assert encoding.perm[0] == state.current_player
    assert state.current_player != 0  # non-trivial rotation, not the identity perm

    sink: dict[GameState, np.ndarray] = {}
    evaluate = make_network_evaluator(net, rank_probs_sink=sink)
    evaluate(state)

    n = 3
    perm = encoding.perm[:n]
    with torch.no_grad():
        features = torch.from_numpy(encoding.features).unsqueeze(0)
        mask = torch.from_numpy(encoding.legal_action_mask).unsqueeze(0)
        active_count = torch.tensor([n], dtype=torch.long)
        _policy_probs, rank_probs, _utility, _points_pred = net(features, mask, active_count)
    rank_probs_canonical = rank_probs[0, :n, :n].numpy()

    expected_abs = np.empty_like(rank_probs_canonical)
    expected_abs[perm] = rank_probs_canonical
    np.testing.assert_allclose(sink[state], expected_abs)
    # And directly: the row for absolute player `perm[0]` (== current_player,
    # canonical slot 0) must equal the canonical slot-0 row, not some other
    # player's row unless perm happens to be the identity.
    np.testing.assert_allclose(sink[state][state.current_player], rank_probs_canonical[0])


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


def test_points_pred_sink_is_untouched_when_not_given():
    state = new_match(player_count=2, seed=1)
    evaluate = make_network_evaluator(_tiny_net())

    evaluate(state)  # no sink passed - nothing to assert on, just must not raise


def test_points_pred_sink_is_filled_in_as_a_side_effect():
    state = new_match(player_count=3, seed=1)
    sink: dict[GameState, np.ndarray] = {}
    evaluate = make_network_evaluator(_tiny_net(), points_pred_sink=sink)

    evaluate(state)

    assert state in sink
    assert sink[state].shape == (3,)


def test_points_pred_sink_only_holds_the_active_slice():
    # Same reasoning as test_rank_probs_sink_only_holds_the_active_n_by_n_slice:
    # points_head is sized for up to N_MAX_PLAYERS, but the sink should only
    # ever expose the slice that's actually meaningful for this state.
    state = new_match(player_count=2, seed=1)
    assert N_MAX_PLAYERS > 2
    sink: dict[GameState, np.ndarray] = {}
    evaluate = make_network_evaluator(_tiny_net(), points_pred_sink=sink)

    evaluate(state)

    assert sink[state].shape == (2,)


def test_evaluate_un_rotates_points_pred_sink_back_to_absolute_player_index():
    # Same construction as test_evaluate_un_rotates_value_back_to_absolute_player_index.
    net = _tiny_net()
    state = new_match(player_count=3, seed=1)
    rotated_sink: dict[GameState, np.ndarray] = {}
    mirror_sink: dict[GameState, np.ndarray] = {}
    evaluate_rotated = make_network_evaluator(net, points_pred_sink=rotated_sink)
    evaluate_mirror = make_network_evaluator(net, points_pred_sink=mirror_sink)

    rotated_state = replace(state, current_player=1)
    mirror_state = replace(state, boards=(state.boards[1], state.boards[2], state.boards[0]), current_player=0)

    evaluate_rotated(rotated_state)
    evaluate_mirror(mirror_state)

    for slot in range(3):
        assert rotated_sink[rotated_state][(1 + slot) % 3] == pytest.approx(mirror_sink[mirror_state][slot])


# --- output un-rotation: canonical (rotated) network output must map back ---
# --- to the correct absolute player index, not just pass through identity ---


def test_evaluate_un_rotates_value_back_to_absolute_player_index():
    # A fresh match has all-zero total_scores/finisher/awaiting features, so
    # permuting `boards` alone produces a state whose canonical-order feature
    # vector is identical to the original's - letting us predict exactly how
    # `evaluate`'s absolute-indexed output must relate between the two,
    # without needing to inspect the network's internals.
    net = _tiny_net()
    state = new_match(player_count=3, seed=1)
    evaluate = make_network_evaluator(net)

    rotated_state = replace(state, current_player=1)
    mirror_state = replace(state, boards=(state.boards[1], state.boards[2], state.boards[0]), current_player=0)

    _, rotated_value = evaluate(rotated_state)
    _, mirror_value = evaluate(mirror_state)

    # rotated_state's canonical slot `s` holds absolute player (1 + s) % 3,
    # and mirror_state (current_player=0, so canonical order == its own
    # `boards` order) holds the same boards at slot `s` directly.
    for slot in range(3):
        assert rotated_value[(1 + slot) % 3] == pytest.approx(mirror_value[slot])


def test_batch_evaluate_un_rotates_value_back_to_absolute_player_index():
    net = _tiny_net()
    state = new_match(player_count=3, seed=1)
    batch_evaluate = make_batch_network_evaluator(net)

    rotated_state = replace(state, current_player=1)
    mirror_state = replace(state, boards=(state.boards[1], state.boards[2], state.boards[0]), current_player=0)

    [(_, rotated_value)] = batch_evaluate([rotated_state])
    [(_, mirror_value)] = batch_evaluate([mirror_state])

    for slot in range(3):
        assert rotated_value[(1 + slot) % 3] == pytest.approx(mirror_value[slot])


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


def test_batch_evaluator_returns_priors_over_the_collapsed_legal_actions_and_a_per_player_value():
    state = new_match(player_count=3, seed=1)
    batch_evaluate = make_batch_network_evaluator(_tiny_net())

    [(priors, value)] = batch_evaluate([state])

    assert set(priors.keys()) == set(distinct_actions(Turn.from_state(state)))
    assert set(priors.keys()) < set(legal_actions(state))
    assert value.shape == (3,)


def test_batch_evaluator_with_precomputed_turns_matches_recomputing_them():
    net = _tiny_net()
    states = [new_match(player_count=2, seed=1), new_match(player_count=3, seed=2)]
    batch_evaluate = make_batch_network_evaluator(net)

    recomputed = batch_evaluate(states)
    precomputed = batch_evaluate(states, turns=[Turn.from_state(s) for s in states])

    for (r_priors, r_value), (p_priors, p_value) in zip(recomputed, precomputed, strict=True):
        assert p_priors.keys() == r_priors.keys()
        np.testing.assert_allclose(p_value, r_value)


# --- bad/edge path ---------------------------------------------------------------


def test_batch_evaluator_with_no_states_returns_an_empty_list_without_touching_the_network():
    batch_evaluate = make_batch_network_evaluator(_tiny_net())

    assert batch_evaluate([]) == []


def test_batch_evaluator_rejects_mismatched_turns_length():
    states = [new_match(player_count=2, seed=1), new_match(player_count=2, seed=2)]
    batch_evaluate = make_batch_network_evaluator(_tiny_net())

    with pytest.raises(ValueError):
        batch_evaluate(states, turns=[Turn.from_state(states[0])])


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

    first = evaluate_vs_heuristic(
        net, 3, num_simulations=2, max_steps=2000, round_max_steps=200, max_rounds=10, seed=5
    )
    second = evaluate_vs_heuristic(
        net, 3, num_simulations=2, max_steps=2000, round_max_steps=200, max_rounds=10, seed=5
    )

    assert first == second


def test_play_one_eval_game_caps_each_net_decision_at_the_requested_total_visits(monkeypatch):
    # _play_one_eval_game reuses its search tree across the net's own turns
    # (see its docstring), which means a later decision's cached_root can
    # already carry visits from an earlier one. num_simulations must stay a
    # *target total* per decision in that case - the actual run_mcts call
    # should ask for fewer new simulations accordingly, never letting a
    # decision end up backed by more total visits than a from-scratch
    # search would've given it.
    import skyjo.rl.evaluator as evaluator_module

    real_run_mcts = evaluator_module.run_mcts
    seen_num_simulations: list[int] = []

    def spying_run_mcts(*args, **kwargs):
        seen_num_simulations.append(kwargs["num_simulations"])
        return real_run_mcts(*args, **kwargs)

    monkeypatch.setattr(evaluator_module, "run_mcts", spying_run_mcts)

    _play_one_eval_game(
        make_network_evaluator(_tiny_net()),
        net_seat=0,
        seed=1,
        num_simulations=5,
        c_puct=DEFAULT_C_PUCT,
        max_steps=2000,
    )

    assert all(n <= 5 for n in seen_num_simulations)
    # Reuse must have actually kicked in for at least one later decision,
    # shrinking the request below the full budget - otherwise this test
    # isn't exercising the new behavior at all.
    assert any(n < 5 for n in seen_num_simulations)


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


def test_evaluate_vs_heuristic_rejects_non_positive_eval_batch_size():
    with pytest.raises(ValueError):
        evaluate_vs_heuristic(_tiny_net(), 2, num_simulations=2, eval_batch_size=0)


def test_evaluate_vs_heuristic_rejects_negative_workers():
    with pytest.raises(ValueError):
        evaluate_vs_heuristic(_tiny_net(), 2, num_simulations=2, workers=-1)


def test_evaluate_vs_heuristic_requires_network_kwargs_when_workers_greater_than_one():
    with pytest.raises(ValueError):
        evaluate_vs_heuristic(_tiny_net(), 2, num_simulations=2, workers=2, eval_batch_size=1)


# --- evaluate_vs_heuristic: batched path (eval_batch_size > 1) --------------


def test_evaluate_vs_heuristic_batched_plays_the_requested_number_of_games():
    result = evaluate_vs_heuristic(
        _tiny_net(), 5, num_simulations=2, max_steps=2000, seed=0, eval_batch_size=2
    )

    assert result.games_played == 5
    assert 0.0 <= result.win_rate <= 1.0
    assert 0.0 <= result.avg_rank <= 1.0


def test_evaluate_vs_heuristic_batched_is_deterministic_for_the_same_seed_and_weights():
    net = _tiny_net()

    first = evaluate_vs_heuristic(net, 4, num_simulations=2, max_steps=2000, seed=5, eval_batch_size=3)
    second = evaluate_vs_heuristic(net, 4, num_simulations=2, max_steps=2000, seed=5, eval_batch_size=3)

    assert first == second


def test_evaluate_vs_heuristic_batched_raises_if_a_game_exceeds_max_steps():
    with pytest.raises(RuntimeError):
        evaluate_vs_heuristic(_tiny_net(), 2, num_simulations=1, max_steps=1, seed=0, eval_batch_size=2)


# --- evaluate_vs_heuristic: workers > 1 (subprocess pool) --------------------


def test_evaluate_vs_heuristic_with_workers_plays_the_requested_number_of_games():
    network_kwargs = {"trunk_dim": 16, "num_residual_blocks": 1}
    result = evaluate_vs_heuristic(
        _tiny_net(),
        4,
        num_simulations=2,
        max_steps=2000,
        seed=0,
        eval_batch_size=2,
        workers=2,
        network_kwargs=network_kwargs,
    )

    assert result.games_played == 4
    assert 0.0 <= result.win_rate <= 1.0


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
