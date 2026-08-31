"""Serializes a `game_recorder.GameRecord` to a plain, JSON-ready dict for
the frontend's game-replay tool - the game-level counterpart to
`tree_export.tree_to_dict` (which serializes one decision's search tree, not
a whole game's sequence of decisions).

Every field here is read-only, presentation-layer computation done at export
time - none of it touches `game_recorder`'s data structures beyond reading
them, so calling this has no effect on anything upstream.
"""

from __future__ import annotations

from typing import Any

from skyjo.domain.engine import Action
from skyjo.domain.state_json import game_state_to_dict
from skyjo.rl.game_recorder import DecisionRecord, GameRecord
from skyjo.rl.tree_export import action_to_dict

SCHEMA_VERSION = 1


def _priors_to_list(priors: dict[Action, float]) -> list[dict[str, Any]]:
    return [
        {"action": action_to_dict(action), "prior": p}
        for action, p in sorted(priors.items(), key=lambda kv: -kv[1])
    ]


def _visit_counts_to_list(visit_counts: dict[Action, int]) -> list[dict[str, Any]]:
    return [
        {"action": action_to_dict(action), "visit_count": n}
        for action, n in sorted(visit_counts.items(), key=lambda kv: -kv[1])
    ]


def _values_to_list(values: dict[Action, float]) -> list[dict[str, Any]]:
    # Unlike priors/visit-shares, a Q-value can be negative (a losing
    # action) - sorted descending (best first) same as the others, just
    # without an implicit "these are all >= 0" assumption anywhere.
    return [
        {"action": action_to_dict(action), "value": v}
        for action, v in sorted(values.items(), key=lambda kv: -kv[1])
    ]


def decision_record_to_dict(decision: DecisionRecord) -> dict[str, Any]:
    return {
        "step": decision.step,
        "actor_seat": decision.actor_seat,
        "actor_name": decision.actor_name,
        "phase": decision.phase,
        "total_scores": list(decision.total_scores),
        # The true, un-redacted state (hidden card values included) - this is
        # an offline analysis tool, not live play, so there is no player to
        # hide information from. See domain.state_json for the shape.
        "board_state": game_state_to_dict(decision.state),
        "raw_policy_priors": _priors_to_list(decision.raw_policy_priors),
        "raw_prior_favorite": action_to_dict(decision.raw_prior_favorite),
        "raw_rank_probs": decision.raw_rank_probs.tolist(),
        "raw_points_pred": decision.raw_points_pred.tolist(),
        "mcts_num_simulations_requested": decision.mcts_num_simulations_requested,
        "mcts_visit_counts": _visit_counts_to_list(decision.mcts_visit_counts),
        "mcts_root_value": None if decision.mcts_root_value is None else decision.mcts_root_value.tolist(),
        "reused_tree_visits": decision.reused_tree_visits,
        # Q-value (mean_value()[actor_seat]) per action, early in this
        # decision's own search vs after all requested simulations - see
        # game_recorder.DecisionRecord's docstring on these two fields.
        "initial_action_values": _values_to_list(decision.initial_action_values),
        "final_action_values": _values_to_list(decision.final_action_values),
        "chosen_action": action_to_dict(decision.chosen_action),
        "search_overrode_prior": decision.search_overrode_prior,
        # None for a training self-play recording (record_training_selfplay_game),
        # which has no heuristic reference to query.
        "heuristic_action": None if decision.heuristic_action is None else action_to_dict(decision.heuristic_action),
        # What to actually compare heuristic_action against
        # chosen_action/raw_prior_favorite - see the field's docstring in
        # game_recorder.DecisionRecord for why the raw action isn't enough.
        "heuristic_action_representative": (
            None
            if decision.heuristic_action_representative is None
            else action_to_dict(decision.heuristic_action_representative)
        ),
        # Training self-play only (record_training_selfplay_game) - all None
        # for an eval-style play_and_record game. See these fields' own
        # docstrings on game_recorder.DecisionRecord.
        "dirichlet_noised_priors": (
            None if decision.dirichlet_noised_priors is None else _priors_to_list(decision.dirichlet_noised_priors)
        ),
        "pi_target": None if decision.pi_target is None else _priors_to_list(decision.pi_target),
        "tau": decision.tau,
        "tied_group_size": decision.tied_group_size,
    }


def game_record_to_dict(record: GameRecord) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seat_names": list(record.seat_names),
        "checkpoint_paths": list(record.checkpoint_paths),
        "seed": record.seed,
        "num_simulations": record.num_simulations,
        "c_puct": record.c_puct,
        "final_total_scores": None if record.final_total_scores is None else list(record.final_total_scores),
        "final_ranks": record.final_ranks,
        "winner_name": record.winner_name,
        "rounds_played": record.rounds_played,
        "decisions": [decision_record_to_dict(d) for d in record.decisions],
    }
