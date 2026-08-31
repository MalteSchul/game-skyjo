from pathlib import Path

from skyjo.rl.checkpoint import save_checkpoint
from skyjo.rl.game_recorder import record_training_selfplay_game
from skyjo.rl.network import AlphaZeroNet

_NETWORK_KWARGS = {"trunk_dim": 8, "num_residual_blocks": 1}
_TINY_CONFIG = {
    "num_simulations": 2,
    "c_puct": 1.5,
    "tau": 0.1,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,
    "round_max_steps": 30,
    "max_rounds": 1,
    "max_steps_per_episode": 200,
    "network_kwargs": _NETWORK_KWARGS,
}


def _tiny_checkpoint(tmp_path: Path) -> str:
    net = AlphaZeroNet(**_NETWORK_KWARGS)
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, net, None, iteration=3, total_train_steps=600, extra={"config": _TINY_CONFIG})
    return str(path)


def test_record_training_selfplay_game_plays_with_noise_and_tau(tmp_path: Path):
    checkpoint = _tiny_checkpoint(tmp_path)

    record = record_training_selfplay_game(checkpoint, seed=0)

    assert len(record.decisions) > 0
    assert record.num_simulations == _TINY_CONFIG["num_simulations"]
    assert record.seat_names == ("tiny.pt seat0", "tiny.pt seat1")
    assert record.checkpoint_paths == (checkpoint, checkpoint)
    assert record.final_total_scores is not None
    assert record.final_ranks is not None
    assert record.winner_name in record.seat_names

    first = record.decisions[0]
    # This is the training-faithful path: every decision carries the
    # self-play-specific fields play_and_record's eval-style path never sets.
    assert first.dirichlet_noised_priors is not None
    assert first.pi_target is not None
    assert first.tau == _TINY_CONFIG["tau"]
    assert first.tied_group_size is not None and first.tied_group_size >= 1
    # No heuristic reference in this recording mode.
    assert first.heuristic_action is None
    assert first.heuristic_action_representative is None
    # raw_policy_priors is the pre-noise prior, distinct from the noised one
    # captured separately (they may coincide numerically at a single-edge
    # root, but the two dicts are populated independently either way).
    assert first.raw_policy_priors.keys() == first.dirichlet_noised_priors.keys()
    # initial/final Q-values are always populated (unlike the noise/tau
    # fields above, these aren't training-self-play-specific) - same action
    # set as the visit distribution they're read off the same root as.
    assert first.initial_action_values.keys() == first.mcts_visit_counts.keys()
    assert first.final_action_values.keys() == first.mcts_visit_counts.keys()


def test_record_training_selfplay_game_reads_config_from_the_checkpoint(tmp_path: Path):
    checkpoint = _tiny_checkpoint(tmp_path)

    record = record_training_selfplay_game(checkpoint, seed=1)

    assert record.c_puct == _TINY_CONFIG["c_puct"]
    for decision in record.decisions:
        assert decision.tau == _TINY_CONFIG["tau"]
        assert decision.mcts_num_simulations_requested == _TINY_CONFIG["num_simulations"]


def test_record_training_selfplay_game_tau_override_replaces_the_checkpoints_own_value(tmp_path: Path):
    checkpoint = _tiny_checkpoint(tmp_path)

    record = record_training_selfplay_game(checkpoint, seed=0, tau=1.0)

    assert _TINY_CONFIG["tau"] != 1.0  # sanity: the override actually differs from the checkpoint's own value
    for decision in record.decisions:
        assert decision.tau == 1.0
