import json

import numpy as np

from skyjo.domain.engine import Action, ActionType, new_match
from skyjo.domain.state_json import game_state_to_dict
from skyjo.rl.encoding import encode_state
from skyjo.rl.game_record_export import decision_record_to_dict, game_record_to_dict
from skyjo.rl.game_recorder import DecisionRecord, GameRecord

_DRAW_STOCK = Action(type=ActionType.DRAW_STOCK, position=None)
_DRAW_DISCARD = Action(type=ActionType.DRAW_DISCARD, position=None)


def _decision(state, step: int = 0) -> DecisionRecord:
    return DecisionRecord(
        step=step,
        actor_seat=0,
        actor_name="p0",
        phase=state.phase,
        total_scores=state.total_scores,
        state=state,
        encoding=encode_state(state),
        raw_policy_priors={_DRAW_STOCK: 0.7, _DRAW_DISCARD: 0.3},
        raw_prior_favorite=_DRAW_STOCK,
        raw_rank_probs=np.array([[0.6, 0.4], [0.4, 0.6]]),
        raw_points_pred=np.array([0.1, -0.1]),
        mcts_num_simulations_requested=10,
        mcts_visit_counts={_DRAW_DISCARD: 3, _DRAW_STOCK: 7},
        mcts_root_value=np.array([0.2, -0.2]),
        reused_tree_visits=0,
        initial_action_values={_DRAW_STOCK: 0.1, _DRAW_DISCARD: -0.2},
        final_action_values={_DRAW_STOCK: 0.3, _DRAW_DISCARD: -0.4},
        chosen_action=_DRAW_STOCK,
        search_overrode_prior=False,
        heuristic_action=_DRAW_DISCARD,
        heuristic_action_representative=_DRAW_DISCARD,
    )


def test_decision_record_to_dict_sorts_priors_and_visits_by_descending_weight():
    state = new_match(player_count=2, seed=0)
    d = decision_record_to_dict(_decision(state))

    assert [p["action"]["type"] for p in d["raw_policy_priors"]] == ["DRAW_STOCK", "DRAW_DISCARD"]
    assert [v["action"]["type"] for v in d["mcts_visit_counts"]] == ["DRAW_STOCK", "DRAW_DISCARD"]
    assert d["mcts_visit_counts"][0]["visit_count"] == 7
    assert d["chosen_action"] == {"type": "DRAW_STOCK", "position": None}
    assert d["search_overrode_prior"] is False
    assert d["heuristic_action"] == {"type": "DRAW_DISCARD", "position": None}
    assert d["heuristic_action_representative"] == {"type": "DRAW_DISCARD", "position": None}


def test_decision_record_to_dict_sorts_action_values_descending_and_keeps_negatives():
    state = new_match(player_count=2, seed=0)
    d = decision_record_to_dict(_decision(state))

    # descending by value, not by prior/visit share (DRAW_STOCK's own prior
    # and visit share are both LOWER than DRAW_DISCARD's, but its value is
    # higher - a Q-value chart cares about a different ordering entirely).
    assert d["initial_action_values"] == [
        {"action": {"type": "DRAW_STOCK", "position": None}, "value": 0.1},
        {"action": {"type": "DRAW_DISCARD", "position": None}, "value": -0.2},
    ]
    assert d["final_action_values"] == [
        {"action": {"type": "DRAW_STOCK", "position": None}, "value": 0.3},
        {"action": {"type": "DRAW_DISCARD", "position": None}, "value": -0.4},
    ]


def test_decision_record_to_dict_embeds_the_true_board_state():
    state = new_match(player_count=2, seed=0)
    d = decision_record_to_dict(_decision(state))

    assert d["board_state"] == game_state_to_dict(state)


def test_decision_record_to_dict_is_json_serializable():
    state = new_match(player_count=2, seed=0)
    d = decision_record_to_dict(_decision(state))

    # Round-trips cleanly - no numpy scalars/arrays or Action objects leaking
    # through un-converted.
    json.dumps(d)


def test_decision_record_to_dict_handles_no_heuristic_reference():
    # record_training_selfplay_game's shape: no HeuristicBot query at all.
    state = new_match(player_count=2, seed=0)
    decision = DecisionRecord(
        step=0,
        actor_seat=0,
        actor_name="p0",
        phase=state.phase,
        total_scores=state.total_scores,
        state=state,
        encoding=encode_state(state),
        raw_policy_priors={_DRAW_STOCK: 0.7, _DRAW_DISCARD: 0.3},
        raw_prior_favorite=_DRAW_STOCK,
        raw_rank_probs=np.array([[0.6, 0.4], [0.4, 0.6]]),
        raw_points_pred=np.array([0.1, -0.1]),
        mcts_num_simulations_requested=10,
        mcts_visit_counts={_DRAW_DISCARD: 3, _DRAW_STOCK: 7},
        mcts_root_value=np.array([0.2, -0.2]),
        reused_tree_visits=0,
        initial_action_values={_DRAW_STOCK: 0.1, _DRAW_DISCARD: -0.2},
        final_action_values={_DRAW_STOCK: 0.3, _DRAW_DISCARD: -0.4},
        chosen_action=_DRAW_STOCK,
        search_overrode_prior=False,
    )

    d = decision_record_to_dict(decision)

    assert d["heuristic_action"] is None
    assert d["heuristic_action_representative"] is None
    json.dumps(d)


def test_decision_record_to_dict_includes_training_selfplay_fields_when_present():
    state = new_match(player_count=2, seed=0)
    decision = DecisionRecord(
        step=0,
        actor_seat=0,
        actor_name="p0",
        phase=state.phase,
        total_scores=state.total_scores,
        state=state,
        encoding=encode_state(state),
        raw_policy_priors={_DRAW_STOCK: 0.7, _DRAW_DISCARD: 0.3},
        raw_prior_favorite=_DRAW_STOCK,
        raw_rank_probs=np.array([[0.6, 0.4], [0.4, 0.6]]),
        raw_points_pred=np.array([0.1, -0.1]),
        mcts_num_simulations_requested=10,
        mcts_visit_counts={_DRAW_DISCARD: 3, _DRAW_STOCK: 7},
        mcts_root_value=np.array([0.2, -0.2]),
        reused_tree_visits=0,
        initial_action_values={_DRAW_STOCK: 0.1, _DRAW_DISCARD: -0.2},
        final_action_values={_DRAW_STOCK: 0.3, _DRAW_DISCARD: -0.4},
        chosen_action=_DRAW_STOCK,
        search_overrode_prior=False,
        dirichlet_noised_priors={_DRAW_STOCK: 0.6, _DRAW_DISCARD: 0.4},
        pi_target={_DRAW_STOCK: 0.9, _DRAW_DISCARD: 0.1},
        tau=0.1,
        tied_group_size=3,
    )

    d = decision_record_to_dict(decision)

    assert [p["action"]["type"] for p in d["dirichlet_noised_priors"]] == ["DRAW_STOCK", "DRAW_DISCARD"]
    assert [p["action"]["type"] for p in d["pi_target"]] == ["DRAW_STOCK", "DRAW_DISCARD"]
    assert d["tau"] == 0.1
    assert d["tied_group_size"] == 3
    json.dumps(d)


def test_decision_record_to_dict_defaults_training_selfplay_fields_to_none():
    state = new_match(player_count=2, seed=0)
    d = decision_record_to_dict(_decision(state))

    assert d["dirichlet_noised_priors"] is None
    assert d["pi_target"] is None
    assert d["tau"] is None
    assert d["tied_group_size"] is None


def test_game_record_to_dict_wraps_decisions_with_summary_fields():
    state = new_match(player_count=2, seed=0)
    record = GameRecord(
        seat_names=("bootstrap", "iter5"),
        checkpoint_paths=("a.pt", "b.pt"),
        seed=0,
        num_simulations=10,
        c_puct=1.5,
        decisions=[_decision(state, step=0)],
        final_total_scores=(80, 95),
        final_ranks=[0, 1],
        winner_name="bootstrap",
        rounds_played=3,
    )

    d = game_record_to_dict(record)

    assert d["seat_names"] == ["bootstrap", "iter5"]
    assert d["final_total_scores"] == [80, 95]
    assert d["winner_name"] == "bootstrap"
    assert d["rounds_played"] == 3
    assert len(d["decisions"]) == 1
    assert d["decisions"][0]["step"] == 0
    json.dumps(d)


def test_game_record_to_dict_handles_a_final_total_scores_of_none():
    record = GameRecord(
        seat_names=("a", "b"),
        checkpoint_paths=("a.pt", "b.pt"),
        seed=0,
        num_simulations=10,
        c_puct=1.5,
    )

    d = game_record_to_dict(record)

    assert d["final_total_scores"] is None
    assert d["decisions"] == []
