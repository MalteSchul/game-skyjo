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
    _accepts_keyword,
    _select_edge,
    _terminal_utility,
    greedy_action,
    run_mcts,
    run_mcts_batch,
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

    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=25, rng=np.random.default_rng(0)
    )

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

    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=2, rng=np.random.default_rng(0)
    )

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

    noisy = run_mcts(
        turn,
        _uniform_evaluate,
        num_simulations=0,
        add_root_noise=True,
        rng=np.random.default_rng(1),
    )
    clean = run_mcts(
        turn,
        _uniform_evaluate,
        num_simulations=0,
        add_root_noise=False,
        rng=np.random.default_rng(1),
    )

    assert set(noisy.edges.keys()) == set(clean.edges.keys())
    assert any(abs(noisy.edges[a].prior - clean.edges[a].prior) > 1e-9 for a in clean.edges), (
        "noise should change at least one legal action's prior"
    )
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
        node.edges = {
            favors_player_0.action: favors_player_0,
            favors_player_1.action: favors_player_1,
        }
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
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=20, rng=np.random.default_rng(0)
    )

    pi = visit_distribution(root, tau=1.0)

    assert sum(pi.values()) == pytest.approx(1.0)
    most_visited = max(root.edges, key=lambda a: root.edges[a].visit_count)
    assert pi[most_visited] == max(pi.values())


def test_visit_distribution_near_zero_tau_is_one_hot_on_the_most_visited_action():
    state = new_match(player_count=2, seed=3)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=20, rng=np.random.default_rng(0)
    )

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
    assert len(chance.edges) > 1, (
        "60 simulations through ~149 unknown cards should not all sample the same value"
    )


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


# --- run_mcts_batch --------------------------------------------------------


def _batch_evaluate(states):
    return [_uniform_evaluate(s) for s in states]


def test_run_mcts_batch_matches_running_run_mcts_once_per_tree():
    # Batching only changes *when* the evaluator is called, never what it's
    # called with or what comes back - so a tree searched inside a batch of
    # several must land on exactly the same edges/visit-counts as the same
    # (turn, rng-seed) pair searched alone via run_mcts.
    state_a = new_match(player_count=2, seed=1)
    state_b = new_match(player_count=3, seed=2)
    turn_a, turn_b = Turn.from_state(state_a), Turn.from_state(state_b)

    root_a = run_mcts(turn_a, _uniform_evaluate, num_simulations=30, rng=np.random.default_rng(10), add_root_noise=False)
    root_b = run_mcts(turn_b, _uniform_evaluate, num_simulations=30, rng=np.random.default_rng(20), add_root_noise=False)

    batched_a, batched_b = run_mcts_batch(
        [turn_a, turn_b],
        _batch_evaluate,
        num_simulations=30,
        rngs=[np.random.default_rng(10), np.random.default_rng(20)],
        add_root_noise=False,
    )

    assert batched_a.visit_count == root_a.visit_count == 30
    assert batched_b.visit_count == root_b.visit_count == 30
    assert {a: e.visit_count for a, e in batched_a.edges.items()} == {
        a: e.visit_count for a, e in root_a.edges.items()
    }
    assert {a: e.visit_count for a, e in batched_b.edges.items()} == {
        a: e.visit_count for a, e in root_b.edges.items()
    }
    for action, edge in batched_a.edges.items():
        assert np.allclose(edge.mean_value(), root_a.edges[action].mean_value())


def test_run_mcts_batch_handles_a_mix_of_cached_terminal_and_live_leaves():
    # _near_closing_state's board is one PLACE away from ending the game
    # outright (see test_advance_round_closing_never_leaks_the_hidden_sentinel...
    # above) - pairing it with a fresh match means, within the same batch,
    # some rounds' pending leaves resolve without any evaluation (cached
    # terminal) while others still need one, exercising run_mcts_batch's
    # per-round "batch can shrink" path.
    closing_state = _near_closing_state(target_score=50)
    fresh_state = new_match(player_count=2, seed=5)
    turns = [Turn.from_state(closing_state), Turn.from_state(fresh_state)]

    roots = run_mcts_batch(
        turns,
        _batch_evaluate,
        num_simulations=20,
        rngs=[np.random.default_rng(1), np.random.default_rng(2)],
        add_root_noise=False,
    )

    assert len(roots) == 2
    for root in roots:
        assert root.visit_count == 20
        for edge in root.edges.values():
            assert np.all(np.isfinite(edge.mean_value()))


def test_run_mcts_batch_with_zero_simulations_still_expands_every_root():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)

    roots = run_mcts_batch(
        [turn, turn], _batch_evaluate, num_simulations=0, rngs=[np.random.default_rng(0), np.random.default_rng(1)]
    )

    assert len(roots) == 2
    for root in roots:
        assert root.visit_count == 0
        assert root.expanded
        assert len(root.edges) > 0


def test_run_mcts_batch_with_no_root_turns_returns_an_empty_list():
    assert run_mcts_batch([], _batch_evaluate, num_simulations=10) == []


# --- run_mcts_batch: bad path ----------------------------------------------


def test_run_mcts_batch_rejects_negative_num_simulations():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)

    with pytest.raises(ValueError):
        run_mcts_batch([turn], _batch_evaluate, num_simulations=-1)


def test_run_mcts_batch_rejects_mismatched_rngs_length():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)

    with pytest.raises(ValueError):
        run_mcts_batch([turn, turn], _batch_evaluate, num_simulations=5, rngs=[np.random.default_rng(0)])


# --- _accepts_keyword: detects an evaluator's optional turn(s) fast path -------


def test_accepts_keyword_true_for_an_explicit_keyword_only_param():
    def evaluate(state, *, turn=None):
        return None

    assert _accepts_keyword(evaluate, "turn") is True


def test_accepts_keyword_true_for_a_function_accepting_arbitrary_kwargs():
    def evaluate(state, **kwargs):
        return None

    assert _accepts_keyword(evaluate, "turn") is True


def test_accepts_keyword_false_for_a_function_without_that_param():
    def evaluate(state):
        return None

    assert _accepts_keyword(evaluate, "turn") is False


# --- turn-aware evaluators: the legal_actions dedup fast path -------------------


def test_run_mcts_passes_each_leafs_own_turn_to_an_evaluator_that_accepts_it():
    state = new_match(player_count=2, seed=1)
    seen: list[tuple] = []

    def recording_evaluate(state, *, turn=None):
        seen.append((state, turn))
        return _uniform_evaluate(state)

    run_mcts(Turn.from_state(state), recording_evaluate, num_simulations=10, rng=np.random.default_rng(0))

    assert seen  # at least the root was evaluated
    for seen_state, turn in seen:
        assert turn is not None
        assert turn.legal_actions == Turn.from_state(seen_state).legal_actions


def test_run_mcts_search_is_identical_whether_or_not_the_evaluator_is_turn_aware():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)

    def turn_aware_evaluate(state, *, turn=None):
        return _uniform_evaluate(state)

    blind_root = run_mcts(turn, _uniform_evaluate, num_simulations=20, rng=np.random.default_rng(0))
    aware_root = run_mcts(turn, turn_aware_evaluate, num_simulations=20, rng=np.random.default_rng(0))

    assert {a: e.visit_count for a, e in blind_root.edges.items()} == {
        a: e.visit_count for a, e in aware_root.edges.items()
    }


def test_run_mcts_batch_passes_each_leafs_own_turn_to_an_evaluator_that_accepts_it():
    turn = Turn.from_state(new_match(player_count=2, seed=1))
    seen: list[tuple] = []

    def recording_evaluate_batch(states, *, turns=None):
        assert turns is not None
        seen.extend(zip(states, turns, strict=True))
        return _batch_evaluate(states)

    run_mcts_batch([turn], recording_evaluate_batch, num_simulations=10, rngs=[np.random.default_rng(0)])

    assert seen
    for seen_state, seen_turn in seen:
        assert seen_turn.legal_actions == Turn.from_state(seen_state).legal_actions


def test_run_mcts_batch_search_is_identical_whether_or_not_the_evaluator_is_turn_aware():
    turn = Turn.from_state(new_match(player_count=2, seed=1))

    def turn_aware_batch_evaluate(states, *, turns=None):
        return _batch_evaluate(states)

    [blind_root] = run_mcts_batch(
        [turn], _batch_evaluate, num_simulations=20, rngs=[np.random.default_rng(0)]
    )
    [aware_root] = run_mcts_batch(
        [turn], turn_aware_batch_evaluate, num_simulations=20, rngs=[np.random.default_rng(0)]
    )

    assert {a: e.visit_count for a, e in blind_root.edges.items()} == {
        a: e.visit_count for a, e in aware_root.edges.items()
    }


# --- run_mcts: reuse_root ----------------------------------------------------


def test_reuse_root_continues_accumulating_visits_on_top_of_the_existing_tree():
    state = new_match(player_count=2, seed=1)
    while state.phase == "initial_flip":
        state = apply_action(state, engine_legal_actions(state)[0])
    turn = Turn.from_state(state)

    first_root = run_mcts(turn, _uniform_evaluate, num_simulations=10, rng=np.random.default_rng(0), add_root_noise=False)
    assert first_root.visit_count == 10

    reused_root = run_mcts(
        turn, _uniform_evaluate, num_simulations=5, rng=np.random.default_rng(1), add_root_noise=False,
        reuse_root=first_root,
    )

    assert reused_root is first_root
    assert reused_root.visit_count == 15


def test_reuse_root_produces_a_well_formed_tree_matching_a_longer_fresh_search_in_shape():
    # Not bit-identical to a single 15-simulation run (rng is consumed in a
    # different order across two calls than one), but reusing a 10-simulation
    # root and adding 5 more must land on the exact same *total* - nothing
    # lost, nothing double-counted.
    state = new_match(player_count=2, seed=1)
    while state.phase == "initial_flip":
        state = apply_action(state, engine_legal_actions(state)[0])
    turn = Turn.from_state(state)

    root = run_mcts(turn, _uniform_evaluate, num_simulations=10, rng=np.random.default_rng(0), add_root_noise=False)
    root = run_mcts(
        turn, _uniform_evaluate, num_simulations=5, rng=np.random.default_rng(0), add_root_noise=False,
        reuse_root=root,
    )

    assert root.visit_count == 15
    assert sum(edge.visit_count for edge in root.edges.values()) == 15


def test_reuse_root_rejects_a_root_whose_state_does_not_match_the_given_turn():
    state_a = new_match(player_count=2, seed=1)
    state_b = new_match(player_count=2, seed=2)
    turn_a, turn_b = Turn.from_state(state_a), Turn.from_state(state_b)
    root_a = run_mcts(turn_a, _uniform_evaluate, num_simulations=1, add_root_noise=False)

    with pytest.raises(ValueError):
        run_mcts(turn_b, _uniform_evaluate, num_simulations=1, reuse_root=root_a)


# --- greedy_action -----------------------------------------------------------


def test_greedy_action_returns_the_single_most_visited_action():
    action_low = Action(type=ActionType.DRAW_STOCK)
    action_high = Action(type=ActionType.DRAW_DISCARD)
    root = MCTSNode(state=new_match(player_count=2, seed=1), n_act=2, is_terminal=False)
    root.edges[action_low] = MCTSEdge(action=action_low, prior=0.5, n_act=2, visit_count=1)
    root.edges[action_high] = MCTSEdge(action=action_high, prior=0.5, n_act=2, visit_count=5)

    assert greedy_action(root, np.random.default_rng(0)) == action_high


def test_greedy_action_breaks_ties_randomly_among_the_max_visited_actions():
    action_a = Action(type=ActionType.DRAW_STOCK)
    action_b = Action(type=ActionType.DRAW_DISCARD)
    root = MCTSNode(state=new_match(player_count=2, seed=1), n_act=2, is_terminal=False)
    root.edges[action_a] = MCTSEdge(action=action_a, prior=0.5, n_act=2, visit_count=3)
    root.edges[action_b] = MCTSEdge(action=action_b, prior=0.5, n_act=2, visit_count=3)

    chosen = {greedy_action(root, np.random.default_rng(seed)) for seed in range(20)}

    assert chosen == {action_a, action_b}


def test_greedy_action_raises_on_a_root_with_no_edges():
    root = MCTSNode(state=new_match(player_count=2, seed=1), n_act=2, is_terminal=False)

    with pytest.raises(ValueError):
        greedy_action(root, np.random.default_rng(0))
