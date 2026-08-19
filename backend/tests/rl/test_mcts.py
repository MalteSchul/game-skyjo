from dataclasses import replace

import numpy as np
import pytest

from skyjo.domain.engine import (
    Action,
    ActionType,
    Card,
    GameState,
    PlayerBoard,
    apply_action,
    new_match,
)
from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.domain.observation import Turn
from skyjo.rl import hidden_info, mcts
from skyjo.rl.mcts import (
    ChanceNode,
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


def _board_from_values(values, *, face_up: bool = False) -> PlayerBoard:
    return PlayerBoard(cards=tuple(Card(value=v, face_up=face_up) for v in values))


# --- run_mcts: simulation bookkeeping ------------------------------------------


def test_root_total_visit_count_equals_num_simulations():
    state = new_match(player_count=3, seed=1)

    root = run_mcts(Turn.from_state(state), _uniform_evaluate, num_simulations=25, rng=np.random.default_rng(0))

    assert root.visit_count == 25


def test_on_simulation_is_called_once_per_simulation_with_a_1_indexed_step():
    state = new_match(player_count=2, seed=1)
    steps: list[int] = []

    run_mcts(
        Turn.from_state(state),
        _uniform_evaluate,
        num_simulations=5,
        rng=np.random.default_rng(0),
        on_simulation=steps.append,
    )

    assert steps == [1, 2, 3, 4, 5]


def test_on_simulation_is_never_called_for_zero_simulations():
    state = new_match(player_count=2, seed=1)
    steps: list[int] = []

    run_mcts(
        Turn.from_state(state),
        _uniform_evaluate,
        num_simulations=0,
        rng=np.random.default_rng(0),
        on_simulation=steps.append,
    )

    assert steps == []


def test_on_root_ready_fires_once_with_an_already_expanded_root_before_any_simulation():
    state = new_match(player_count=2, seed=1)
    call_count = 0
    visit_count_at_ready: int | None = None

    def on_root_ready(root: MCTSNode) -> None:
        nonlocal call_count, visit_count_at_ready
        call_count += 1
        visit_count_at_ready = root.visit_count
        assert root.expanded

    run_mcts(
        Turn.from_state(state),
        _uniform_evaluate,
        num_simulations=3,
        rng=np.random.default_rng(0),
        on_root_ready=on_root_ready,
    )

    assert call_count == 1
    assert visit_count_at_ready == 0


def test_on_root_ready_hands_back_the_same_live_object_run_mcts_returns():
    state = new_match(player_count=2, seed=1)
    ready_roots: list[MCTSNode] = []

    returned_root = run_mcts(
        Turn.from_state(state),
        _uniform_evaluate,
        num_simulations=5,
        rng=np.random.default_rng(0),
        on_root_ready=ready_roots.append,
    )

    assert ready_roots[0] is returned_root
    assert returned_root.visit_count == 5


def test_on_root_ready_is_optional():
    state = new_match(player_count=2, seed=1)

    root = run_mcts(Turn.from_state(state), _uniform_evaluate, num_simulations=2, rng=np.random.default_rng(0))

    assert root.visit_count == 2


def test_run_mcts_rejects_negative_simulation_count():
    state = new_match(player_count=2, seed=1)

    with pytest.raises(ValueError):
        run_mcts(Turn.from_state(state), _uniform_evaluate, num_simulations=-1)


def test_run_mcts_cannot_be_rooted_at_a_terminal_state():
    # Turn.from_state is the only way to build run_mcts's input, and it
    # refuses round_over/game_over states outright - there's no decision to
    # search for there. Covered fully in test_observation.py; this just
    # confirms run_mcts's own entry point relies on that guarantee.
    state = new_match(player_count=2, seed=1)
    game_over_state = replace(state, phase="game_over", total_scores=(50, 10))

    with pytest.raises(Exception):  # noqa: B017 - IllegalActionError, re-asserted at the boundary
        Turn.from_state(game_over_state)


# --- Dirichlet root noise: verification criterion 4 ----------------------------


def test_root_noise_only_perturbs_priors_for_legal_actions_and_still_sums_to_one():
    # Past initial_flip, not collapsed to a single action: initial_flip's 12
    # FLIP_INITIAL positions are provably equivalent on a fresh board and
    # collapse to one edge, which would make Dirichlet noise a no-op here.
    state = new_match(player_count=3, seed=7)
    while state.phase == "initial_flip":
        state = apply_action(state, engine_legal_actions(state)[0])
    turn = Turn.from_state(state)

    noisy = run_mcts(turn, _uniform_evaluate, num_simulations=0, add_root_noise=True, rng=np.random.default_rng(1))
    clean = run_mcts(turn, _uniform_evaluate, num_simulations=0, add_root_noise=False, rng=np.random.default_rng(1))

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
    root = run_mcts(Turn.from_state(state), _uniform_evaluate, num_simulations=20, rng=np.random.default_rng(0))

    pi = visit_distribution(root, tau=1.0)

    assert sum(pi.values()) == pytest.approx(1.0)
    most_visited = max(root.edges, key=lambda a: root.edges[a].visit_count)
    assert pi[most_visited] == max(pi.values())


def test_visit_distribution_near_zero_tau_is_one_hot_on_the_most_visited_action():
    state = new_match(player_count=2, seed=3)
    root = run_mcts(Turn.from_state(state), _uniform_evaluate, num_simulations=20, rng=np.random.default_rng(0))

    pi = visit_distribution(root, tau=1e-6)

    most_visited = max(root.edges, key=lambda a: root.edges[a].visit_count)
    assert pi[most_visited] == 1.0
    assert sum(v for a, v in pi.items() if a != most_visited) == 0.0


def test_visit_distribution_raises_on_a_root_with_no_edges():
    root = MCTSNode(state=new_match(player_count=2, seed=1), n_act=2, is_terminal=False)

    with pytest.raises(ValueError):
        visit_distribution(root)


def test_sample_action_always_returns_a_key_from_the_distribution():
    rng = np.random.default_rng(0)
    pi = {"a": 0.9, "b": 0.1}

    for _ in range(20):
        assert sample_action(pi, rng) in pi


# --- chance nodes: reveals are resampled, not frozen ----------------------------


def test_repeated_visits_to_a_reveal_edge_sample_more_than_one_value():
    # A fresh match's initial_flip: every FLIP_INITIAL position is provably
    # equivalent (nothing revealed anywhere yet), so _expand collapses all
    # 12 into a single representative edge - concentrating every simulation
    # onto the same ChanceNode and making the resample behavior observable.
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)

    root = run_mcts(turn, _uniform_evaluate, num_simulations=60, rng=np.random.default_rng(0))

    flip_action = next(a for a in root.edges if a.type is ActionType.FLIP_INITIAL)
    chance = root.edges[flip_action].child
    assert isinstance(chance, ChanceNode)
    assert len(chance.edges) > 1, "60 simulations through ~149 unknown cards should not all sample the same value"


def test_chance_node_never_offers_a_value_with_zero_remaining_count():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)

    root = run_mcts(turn, _uniform_evaluate, num_simulations=40, rng=np.random.default_rng(2))

    flip_action = next(a for a in root.edges if a.type is ActionType.FLIP_INITIAL)
    chance = root.edges[flip_action].child
    assert isinstance(chance, ChanceNode)
    for value in chance.edges:
        assert chance.counts[value] > 0


# --- action collapsing: distinct_actions is wired into expansion ---------------


def test_expand_collapses_equivalent_flip_actions_at_a_fresh_root():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)  # initial_flip, fully hidden board -> all 12 flips equivalent

    root = run_mcts(turn, _uniform_evaluate, num_simulations=0, add_root_noise=False)

    assert len(root.edges) == 1
    assert len(turn.legal_actions) == 12


def test_expand_keeps_distinguishable_actions_separate():
    # awaiting_draw offers exactly two genuinely different actions - collapsing
    # must never merge across action types.
    state = new_match(player_count=2, seed=1)
    while state.phase == "initial_flip":
        state = apply_action(state, engine_legal_actions(state)[0])
    turn = Turn.from_state(state)

    root = run_mcts(turn, _uniform_evaluate, num_simulations=0, add_root_noise=False)

    assert {a.type for a in root.edges} == {ActionType.DRAW_STOCK, ActionType.DRAW_DISCARD}


# --- round close: the bulk reveal resolves to a plausible, finite outcome ------


def _near_closing_state(*, target_score: int) -> GameState:
    # Player 0 is already the finisher with a fully-revealed board; player 1
    # is the sole remaining awaiting player, mid-turn, placing into position
    # 11 (this action's own single-card reveal) while positions 9 and 10
    # stay hidden - those can only be resolved by the round-close bulk fill.
    board0 = _board_from_values(list(range(12)), face_up=True)
    board1 = _board_from_values([1] * 9 + [3, 4, 5], face_up=True)
    board1 = replace(
        board1, cards=board1.cards[:9] + tuple(replace(c, face_up=False) for c in board1.cards[9:])
    )
    return GameState(
        boards=(board0, board1),
        stock=(5, 6, 7),
        discard=(2,),
        current_player=1,
        drawn_card=9,
        finisher=0,
        players_awaiting_final_turn=frozenset({1}),
        round_scores=None,
        total_scores=(66, 0),
        phase="awaiting_placement",
        reshuffle_seed=None,
        target_score=target_score,
    )


def test_advance_round_closing_never_leaks_the_hidden_sentinel_into_a_revealed_card():
    # target_score=50 forces game_over once this round's score is folded in
    # (player 0's prior total, 66, already exceeds it), so the reveal-all
    # scoring state is returned directly rather than folded into a fresh
    # (freshly-hidden-again) round.
    state = _near_closing_state(target_score=50)

    next_state = mcts._advance_round_closing(
        state, Action(ActionType.PLACE, position=11), np.random.default_rng(0)
    )

    assert next_state.phase == "game_over"
    for board in next_state.boards:
        for card in board.cards:
            if card is not None and card.face_up:
                assert card.value != hidden_info.HIDDEN_SENTINEL
    assert all(-1000 < s < 1000 for s in next_state.total_scores)


def test_search_through_a_round_closing_edge_backs_up_bounded_utility():
    state = _near_closing_state(target_score=50)
    turn = Turn.from_state(state)

    root = run_mcts(turn, _uniform_evaluate, num_simulations=15, rng=np.random.default_rng(0))

    assert root.visit_count == 15
    for edge in root.edges.values():
        assert np.all(np.isfinite(edge.mean_value()))
        assert np.all(edge.mean_value() >= -1.0 - 1e-6)
        assert np.all(edge.mean_value() <= 1.0 + 1e-6)
