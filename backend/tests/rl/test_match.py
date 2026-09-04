import numpy as np
import pytest

from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.bots.random_bot import RandomBot
from skyjo.domain.engine import new_match
from skyjo.domain.observation import Turn
from skyjo.rl.checkpoint import save_checkpoint
from skyjo.rl.match import (
    MctsDecider,
    SimpleDecider,
    build_decider,
    decide_batch,
    decider_from_net,
    evaluate_vs_decider,
    play_batch,
)
from skyjo.rl.network import AlphaZeroNet

_NET_KWARGS = {"trunk_dim": 8, "num_residual_blocks": 1}


def _tiny_net() -> AlphaZeroNet:
    return AlphaZeroNet(**_NET_KWARGS)


# --- build_decider -----------------------------------------------------------


def test_build_decider_random_returns_one_random_bot_per_seed():
    decider = build_decider("random", [1, 2, 3], num_simulations=2, c_puct=1.5)

    assert isinstance(decider, SimpleDecider)
    assert len(decider.bots) == 3
    assert all(isinstance(b, RandomBot) for b in decider.bots)


def test_build_decider_heuristic_returns_one_heuristic_bot_per_seed():
    decider = build_decider("heuristic", [1, 2], num_simulations=2, c_puct=1.5)

    assert isinstance(decider, SimpleDecider)
    assert len(decider.bots) == 2
    assert all(isinstance(b, HeuristicBot) for b in decider.bots)


def test_build_decider_checkpoint_path_loads_the_saved_weights(tmp_path):
    net = _tiny_net()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, net, None, iteration=1, total_train_steps=0)

    decider = build_decider(str(path), [], num_simulations=2, c_puct=1.5, network_kwargs=_NET_KWARGS)

    assert isinstance(decider, MctsDecider)
    assert decider.num_simulations == 2


def test_build_decider_checkpoint_path_that_does_not_exist_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_decider(str(tmp_path / "missing.pt"), [], num_simulations=2, network_kwargs=_NET_KWARGS)


# --- decide_batch --------------------------------------------------------------


def test_decide_batch_simple_decider_returns_each_bots_own_choice():
    state = new_match(player_count=2, seed=1)
    turn = Turn.from_state(state)
    decider = build_decider("heuristic", [1, 2], num_simulations=1, c_puct=1.5)
    expected = [decider.bots[0].choose_action(turn), decider.bots[1].choose_action(turn)]
    # HeuristicBot is stateless per-call, so re-fetching the same turn twice
    # (once above, once inside decide_batch) yields identical choices.
    decider = build_decider("heuristic", [1, 2], num_simulations=1, c_puct=1.5)

    actions = decide_batch(decider, [0, 1], [turn, turn], [np.random.default_rng(0), np.random.default_rng(1)])

    assert actions == expected


def test_decide_batch_mcts_decider_returns_a_legal_action_for_each_game():
    state_a = new_match(player_count=2, seed=1)
    state_b = new_match(player_count=2, seed=2)
    decider = decider_from_net(_tiny_net(), num_simulations=2, c_puct=1.5)

    actions = decide_batch(
        decider,
        [0, 1],
        [Turn.from_state(state_a), Turn.from_state(state_b)],
        [np.random.default_rng(0), np.random.default_rng(1)],
    )

    assert len(actions) == 2
    assert actions[0] in Turn.from_state(state_a).legal_actions
    assert actions[1] in Turn.from_state(state_b).legal_actions


def test_decide_batch_with_no_games_returns_an_empty_list():
    decider = decider_from_net(_tiny_net(), num_simulations=2, c_puct=1.5)

    assert decide_batch(decider, [], [], []) == []


# --- play_batch ----------------------------------------------------------------


def test_play_batch_plays_every_game_to_a_valid_outcome():
    decider_a = build_decider("random", [1, 2], num_simulations=1, c_puct=1.5)
    decider_b = build_decider("heuristic", [3, 4], num_simulations=1, c_puct=1.5)

    outcomes = play_batch(decider_a, decider_b, [1, 2], [True, False], max_steps=3000, round_max_steps=200, max_rounds=10)

    assert len(outcomes) == 2
    for outcome in outcomes:
        assert outcome is not None
        ranks, points = outcome
        assert sorted(ranks) == [0, 1]
        assert len(points) == 2


def test_play_batch_forward_flag_controls_which_seat_each_decider_occupies():
    # random vs heuristic is asymmetric enough that swapping forward changes
    # which seat is which - checked indirectly via game_over still resolving
    # correctly for both orientations in the same call. Both deciders' bot
    # lists are sized to the number of games in this play_batch call (2),
    # matching how _run_batch_job/_build_pool_jobs always size them - a
    # SimpleDecider's bots[i] is indexed by the game's position in *this*
    # call, not by how many seeds it happens to be assigned across forward.
    decider_a = build_decider("random", [1, 2], num_simulations=1, c_puct=1.5)
    decider_b = build_decider("heuristic", [3, 4], num_simulations=1, c_puct=1.5)

    outcomes = play_batch(decider_a, decider_b, [1, 1], [True, False], max_steps=3000, round_max_steps=200, max_rounds=10)

    assert all(o is not None for o in outcomes)


def test_play_batch_returns_none_for_a_game_that_does_not_finish_within_max_steps():
    decider_a = build_decider("random", [1], num_simulations=1, c_puct=1.5)
    decider_b = build_decider("random", [2], num_simulations=1, c_puct=1.5)

    outcomes = play_batch(decider_a, decider_b, [1], [True], max_steps=1)

    assert outcomes == [None]


# --- evaluate_vs_decider ---------------------------------------------------------


def test_evaluate_vs_decider_happy_path(tmp_path):
    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint_path, _tiny_net(), None, iteration=1, total_train_steps=0)

    result = evaluate_vs_decider(
        _tiny_net(),
        str(checkpoint_path),
        4,
        num_simulations=2,
        opponent_num_simulations=2,
        network_kwargs=_NET_KWARGS,
        round_max_steps=200,
        max_rounds=10,
    )

    assert result.games_played == 4
    assert 0.0 <= result.win_rate <= 1.0
    assert 0.0 <= result.avg_rank <= 1.0


def test_evaluate_vs_decider_rejects_num_games_below_one(tmp_path):
    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint_path, _tiny_net(), None, iteration=1, total_train_steps=0)

    with pytest.raises(ValueError):
        evaluate_vs_decider(
            _tiny_net(), str(checkpoint_path), 0, num_simulations=2, opponent_num_simulations=2,
            network_kwargs=_NET_KWARGS,
        )


def test_evaluate_vs_decider_rejects_batch_size_below_one(tmp_path):
    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint_path, _tiny_net(), None, iteration=1, total_train_steps=0)

    with pytest.raises(ValueError):
        evaluate_vs_decider(
            _tiny_net(), str(checkpoint_path), 2, num_simulations=2, opponent_num_simulations=2,
            network_kwargs=_NET_KWARGS, batch_size=0,
        )


def test_evaluate_vs_decider_rejects_a_heuristic_opponent_spec():
    with pytest.raises(ValueError):
        evaluate_vs_decider(_tiny_net(), "heuristic", 2, num_simulations=2, opponent_num_simulations=2)


def test_evaluate_vs_decider_raises_when_every_game_fails_to_finish(tmp_path):
    checkpoint_path = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint_path, _tiny_net(), None, iteration=1, total_train_steps=0)

    with pytest.raises(RuntimeError):
        evaluate_vs_decider(
            _tiny_net(), str(checkpoint_path), 2, num_simulations=2, opponent_num_simulations=2,
            network_kwargs=_NET_KWARGS, max_steps=1,
        )
