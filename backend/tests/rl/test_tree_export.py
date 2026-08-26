import json
import math

import numpy as np

from skyjo.domain.engine import ActionType, new_match
from skyjo.domain.engine import apply_action as engine_apply_action
from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.domain.observation import Turn
from skyjo.rl.mcts import DEFAULT_C_PUCT, run_mcts
from skyjo.rl.tree_export import tree_to_dict

# --- fixture helpers -----------------------------------------------------------


def _uniform_evaluate(state):
    actions = engine_legal_actions(state)
    priors = {a: 1.0 / len(actions) for a in actions}
    return priors, np.zeros(len(state.boards))


def _fixed_value_evaluate(value):
    """A stand-in whose value is the same known vector for every state, so
    tests can assert exactly what `MCTSNode.value` should hold without
    depending on search dynamics."""

    def evaluate(state):
        actions = engine_legal_actions(state)
        priors = {a: 1.0 / len(actions) for a in actions}
        return priors, np.array(value, dtype=np.float64)

    return evaluate


def _awaiting_draw_state(seed: int):
    """Advances a fresh 2-player match past both players' initial flips, so
    the root has >1 distinct (non-collapsed) legal action - unlike the very
    first `initial_flip` decision, where every position is equivalent."""
    state = new_match(player_count=2, seed=seed)
    while state.phase == "initial_flip":
        action = next(a for a in engine_legal_actions(state) if a.type is ActionType.FLIP_INITIAL)
        state = engine_apply_action(state, action)
    return state


# --- happy path -----------------------------------------------------------------


def test_root_dict_has_one_edge_per_root_edge_sorted_by_visit_count():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)
    root = run_mcts(turn, _uniform_evaluate, num_simulations=20, rng=np.random.default_rng(0))

    tree = tree_to_dict(root)

    assert tree["kind"] == "decision"
    # Equivalent actions (e.g. all initial flips, before any card is known)
    # collapse onto one shared edge - see domain.action_equivalence - so this
    # can be fewer than len(turn.legal_actions), never more.
    assert len(tree["edges"]) == len(root.edges)
    visit_counts = [edge["visit_count"] for edge in tree["edges"]]
    assert visit_counts == sorted(visit_counts, reverse=True)
    assert sum(visit_counts) == root.visit_count


def test_dict_is_json_serializable_end_to_end():
    state = new_match(player_count=3, seed=2)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=15, rng=np.random.default_rng(1)
    )

    dumped = json.dumps(tree_to_dict(root))
    reloaded = json.loads(dumped)

    assert reloaded["kind"] == "decision"


def test_edge_fields_match_the_live_edge_they_came_from():
    state = new_match(player_count=2, seed=3)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=10, rng=np.random.default_rng(2)
    )
    edge = next(iter(root.edges.values()))

    tree = tree_to_dict(root)
    dumped = next(
        e
        for e in tree["edges"]
        if e["action"]["type"] == edge.action.type.name
        and e["action"]["position"] == edge.action.position
    )

    assert dumped["prior"] == edge.prior
    assert dumped["visit_count"] == edge.visit_count
    assert dumped["mean_value"] == edge.mean_value().tolist()


# --- max_depth ---------------------------------------------------------------------


def test_max_depth_zero_lists_edges_but_omits_children():
    state = new_match(player_count=2, seed=1)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=20, rng=np.random.default_rng(0)
    )

    tree = tree_to_dict(root, max_depth=0)

    assert len(tree["edges"]) > 0
    assert all(edge["child"] is None for edge in tree["edges"])


def test_max_depth_none_traverses_until_a_leaf_or_unexpanded_edge():
    state = new_match(player_count=2, seed=1)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=30, rng=np.random.default_rng(0)
    )

    tree = tree_to_dict(root, max_depth=None)

    visited_edge = max(tree["edges"], key=lambda e: e["visit_count"])
    assert visited_edge["child"] is not None


# --- edge / bad path -----------------------------------------------------------


def test_zero_simulation_root_has_no_visited_children():
    state = new_match(player_count=2, seed=1)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=0, rng=np.random.default_rng(0)
    )

    tree = tree_to_dict(root)

    assert tree["visit_count"] == 0
    assert all(edge["visit_count"] == 0 for edge in tree["edges"])
    assert all(edge["child"] is None for edge in tree["edges"])


# --- node/edge diagnostics: value, prior_before_noise, puct_score --------------


def test_root_value_is_none_before_expansion_is_impossible_and_matches_evaluate_after():
    state = _awaiting_draw_state(seed=5)
    evaluate = _fixed_value_evaluate([0.25, -0.25])
    root = run_mcts(
        Turn.from_state(state), evaluate, num_simulations=1, rng=np.random.default_rng(0)
    )

    tree = tree_to_dict(root)

    # root is always expanded before run_mcts returns, so its value is never None
    assert tree["value"] == [0.25, -0.25]


def test_visited_childs_value_matches_what_evaluate_returned_for_it():
    # DRAW_DISCARD is deterministic (the discard top is already face up, no
    # reveal/ChanceNode involved) - unlike DRAW_STOCK, its child is always a
    # plain decision MCTSNode with its own "value", which is what this test
    # needs to isolate.
    state = _awaiting_draw_state(seed=5)
    evaluate = _fixed_value_evaluate([0.25, -0.25])
    root = run_mcts(
        Turn.from_state(state), evaluate, num_simulations=10, rng=np.random.default_rng(0)
    )

    tree = tree_to_dict(root, max_depth=None)

    discard_edge = next(e for e in tree["edges"] if e["action"]["type"] == "DRAW_DISCARD")
    assert discard_edge["visit_count"] > 0
    assert discard_edge["child"] is not None
    assert discard_edge["child"]["value"] == [0.25, -0.25]


def test_chance_edges_use_card_value_not_value_to_avoid_colliding_with_node_value():
    # DRAW_STOCK's drawn card isn't known yet, so its child is a ChanceNode
    # keyed by the revealed card value - exported as "card_value", not
    # "value", since a decision node's "value" means something entirely
    # different (the network's predicted utility, see the test above) and
    # the two sit right next to each other in the same tree.
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=20, rng=np.random.default_rng(0)
    )

    tree = tree_to_dict(root, max_depth=None)

    stock_edge = next(e for e in tree["edges"] if e["action"]["type"] == "DRAW_STOCK")
    assert stock_edge["child"] is not None
    chance_node = stock_edge["child"]
    assert chance_node["kind"] == "chance"
    assert len(chance_node["edges"]) > 0
    for chance_edge in chance_node["edges"]:
        assert "card_value" in chance_edge
        assert isinstance(chance_edge["card_value"], int)
        assert "value" not in chance_edge


def test_prior_before_noise_equals_prior_when_root_noise_is_disabled():
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state),
        _uniform_evaluate,
        num_simulations=1,
        rng=np.random.default_rng(0),
        add_root_noise=False,
    )

    tree = tree_to_dict(root)

    for edge in tree["edges"]:
        assert edge["prior"] == edge["prior_before_noise"]
        assert edge["prior_before_noise"] == 0.5  # uniform over the 2 legal actions


def test_prior_before_noise_differs_from_prior_when_root_noise_is_applied():
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state),
        _uniform_evaluate,
        num_simulations=1,
        rng=np.random.default_rng(0),
        add_root_noise=True,
    )

    tree = tree_to_dict(root)

    assert any(edge["prior"] != edge["prior_before_noise"] for edge in tree["edges"])
    assert all(edge["prior_before_noise"] == 0.5 for edge in tree["edges"])


def test_puct_score_matches_the_select_edge_formula():
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state),
        _uniform_evaluate,
        num_simulations=20,
        rng=np.random.default_rng(0),
        c_puct=2.0,
    )

    tree = tree_to_dict(root, c_puct=2.0)

    parent_visits = tree["visit_count"]
    for edge in tree["edges"]:
        expected_u = 2.0 * edge["prior"] * math.sqrt(parent_visits) / (1 + edge["visit_count"])
        expected_q = edge["mean_value"][tree["current_player"]]
        assert edge["u"] == expected_u
        assert edge["q"] == expected_q
        assert edge["puct_score"] == expected_q + expected_u


def test_puct_score_uses_default_c_puct_when_not_specified():
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=10, rng=np.random.default_rng(0)
    )

    tree = tree_to_dict(root)

    parent_visits = tree["visit_count"]
    for edge in tree["edges"]:
        expected_u = (
            DEFAULT_C_PUCT * edge["prior"] * math.sqrt(parent_visits) / (1 + edge["visit_count"])
        )
        assert edge["u"] == expected_u


# --- rank_probs_by_state ---------------------------------------------------------


def test_rank_probs_is_none_when_no_lookup_is_given():
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=1, rng=np.random.default_rng(0)
    )

    tree = tree_to_dict(root)

    assert tree["rank_probs"] is None


def test_rank_probs_is_looked_up_by_the_nodes_own_state():
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=1, rng=np.random.default_rng(0)
    )
    # keyed by root.state, not the original `state` passed to run_mcts: the
    # search operates on a redacted GameState (hidden cards scrubbed), so
    # they're deliberately not the same object - see rl.hidden_info.
    lookup = {root.state: np.array([[0.7, 0.3], [0.3, 0.7]])}

    tree = tree_to_dict(root, rank_probs_by_state=lookup)

    assert tree["rank_probs"] == [[0.7, 0.3], [0.3, 0.7]]


def test_rank_probs_is_none_for_a_state_missing_from_the_lookup():
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=1, rng=np.random.default_rng(0)
    )
    lookup: dict = {}  # deliberately doesn't contain `state`

    tree = tree_to_dict(root, rank_probs_by_state=lookup)

    assert tree["rank_probs"] is None


# --- points_pred_by_state ---------------------------------------------------------


def test_points_pred_is_none_when_no_lookup_is_given():
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=1, rng=np.random.default_rng(0)
    )

    tree = tree_to_dict(root)

    assert tree["points_pred"] is None


def test_points_pred_is_looked_up_by_the_nodes_own_state():
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=1, rng=np.random.default_rng(0)
    )
    # keyed by root.state, not the original `state` passed to run_mcts - same
    # reasoning as rank_probs_by_state above.
    lookup = {root.state: np.array([0.4, 0.9])}

    tree = tree_to_dict(root, points_pred_by_state=lookup)

    assert tree["points_pred"] == [0.4, 0.9]


def test_points_pred_is_none_for_a_state_missing_from_the_lookup():
    state = _awaiting_draw_state(seed=5)
    root = run_mcts(
        Turn.from_state(state), _uniform_evaluate, num_simulations=1, rng=np.random.default_rng(0)
    )
    lookup: dict = {}  # deliberately doesn't contain `state`

    tree = tree_to_dict(root, points_pred_by_state=lookup)

    assert tree["points_pred"] is None
