from dataclasses import replace

import numpy as np
import pytest

from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.domain.engine import new_match
from skyjo.rl.mcts import (
    MCTSEdge,
    MCTSNode,
    _select_edge,
    _terminal_utility,
    run_mcts,
    sample_action,
    visit_distribution,
)
from skyjo.rl.selfplay import final_ranks

# --- fixture helpers -----------------------------------------------------------


def _uniform_evaluate(state):
    actions = engine_legal_actions(state)
    priors = {a: 1.0 / len(actions) for a in actions}
    return priors, np.zeros(len(state.boards))


# --- run_mcts: simulation bookkeeping ------------------------------------------


def test_root_total_visit_count_equals_num_simulations():
    state = new_match(player_count=3, seed=1)

    root = run_mcts(state, _uniform_evaluate, num_simulations=25, rng=np.random.default_rng(0))

    assert root.visit_count == 25


def test_run_mcts_on_a_terminal_state_returns_immediately_with_no_edges():
    state = new_match(player_count=2, seed=1)
    game_over_state = replace(state, phase="game_over", total_scores=(50, 10))

    root = run_mcts(game_over_state, _uniform_evaluate, num_simulations=10)

    assert root.is_terminal
    assert root.edges == {}
    assert root.visit_count == 0  # simulations never run against a terminal root


def test_run_mcts_rejects_negative_simulation_count():
    state = new_match(player_count=2, seed=1)

    with pytest.raises(ValueError):
        run_mcts(state, _uniform_evaluate, num_simulations=-1)


# --- Dirichlet root noise: verification criterion 4 ----------------------------


def test_root_noise_only_perturbs_priors_for_legal_actions_and_still_sums_to_one():
    state = new_match(player_count=3, seed=7)

    noisy = run_mcts(state, _uniform_evaluate, num_simulations=0, add_root_noise=True, rng=np.random.default_rng(1))
    clean = run_mcts(
        state, _uniform_evaluate, num_simulations=0, add_root_noise=False, rng=np.random.default_rng(1)
    )

    assert set(noisy.edges.keys()) == set(clean.edges.keys())
    assert any(
        abs(noisy.edges[a].prior - clean.edges[a].prior) > 1e-9 for a in clean.edges
    ), "noise should change at least one legal action's prior"
    assert sum(e.prior for e in noisy.edges.values()) == pytest.approx(1.0)


# --- PUCT selection uses the acting player's own Q component: criterion 2 -----


def test_select_edge_uses_the_node_current_players_q_component_not_always_player_zero():
    state = new_match(player_count=2, seed=1)

    def node_with_edges(current_player: int) -> MCTSNode:
        node_state = replace(state, current_player=current_player)
        node = MCTSNode(state=node_state, n_act=2, is_terminal=False)
        favors_player_0 = MCTSEdge(action=engine_legal_actions(node_state)[0], prior=0.5, n_act=2)
        favors_player_0.visit_count = 1
        favors_player_0.value_sum = np.array([10.0, 0.0])
        favors_player_1 = MCTSEdge(action=engine_legal_actions(node_state)[1], prior=0.5, n_act=2)
        favors_player_1.visit_count = 1
        favors_player_1.value_sum = np.array([0.0, 10.0])
        node.edges = {favors_player_0.action: favors_player_0, favors_player_1.action: favors_player_1}
        return node

    root_node = node_with_edges(current_player=0)
    opponent_node = node_with_edges(current_player=1)

    root_actions = list(root_node.edges)
    assert _select_edge(root_node, c_puct=1.5).action == root_actions[0]
    assert _select_edge(opponent_node, c_puct=1.5).action == root_actions[1]


# --- terminal utility / final ranks ---------------------------------------------


def test_terminal_utility_gives_best_payoff_to_the_lowest_score():
    utility = _terminal_utility((30, 10, 20))

    assert utility[1] > utility[2] > utility[0]
    assert utility[1] == pytest.approx(1.0)
    assert utility[0] == pytest.approx(-1.0)


def test_terminal_utility_splits_payoff_evenly_across_a_tie():
    utility = _terminal_utility((10, 10))

    assert utility[0] == pytest.approx(utility[1])


def test_final_ranks_orders_lowest_score_first():
    assert final_ranks((30, 10, 20)) == [2, 0, 1]


def test_final_ranks_breaks_ties_by_lower_player_index():
    assert final_ranks((10, 10, 5)) == [1, 2, 0]


# --- visit_distribution / sample_action -----------------------------------------


def test_visit_distribution_sums_to_one_and_favors_more_visited_actions():
    state = new_match(player_count=2, seed=3)
    root = run_mcts(state, _uniform_evaluate, num_simulations=20, rng=np.random.default_rng(0))

    pi = visit_distribution(root, tau=1.0)

    assert sum(pi.values()) == pytest.approx(1.0)
    most_visited = max(root.edges, key=lambda a: root.edges[a].visit_count)
    assert pi[most_visited] == max(pi.values())


def test_visit_distribution_near_zero_tau_is_one_hot_on_the_most_visited_action():
    state = new_match(player_count=2, seed=3)
    root = run_mcts(state, _uniform_evaluate, num_simulations=20, rng=np.random.default_rng(0))

    pi = visit_distribution(root, tau=1e-6)

    most_visited = max(root.edges, key=lambda a: root.edges[a].visit_count)
    assert pi[most_visited] == 1.0
    assert sum(v for a, v in pi.items() if a != most_visited) == 0.0


def test_visit_distribution_raises_on_a_root_with_no_edges():
    state = new_match(player_count=2, seed=1)
    game_over_state = replace(state, phase="game_over", total_scores=(1, 2))
    root = run_mcts(game_over_state, _uniform_evaluate, num_simulations=5)

    with pytest.raises(ValueError):
        visit_distribution(root)


def test_sample_action_always_returns_a_key_from_the_distribution():
    rng = np.random.default_rng(0)
    pi = {"a": 0.9, "b": 0.1}

    for _ in range(20):
        assert sample_action(pi, rng) in pi
