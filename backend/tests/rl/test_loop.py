import json

import numpy as np
import pytest
import torch

import skyjo.rl.loop as loop_module
from skyjo.domain.engine import new_match
from skyjo.rl.action_space import ACTION_SPACE_SIZE
from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.loop import LoopState, TrainingConfig, make_tau_schedule, run_training_loop
from skyjo.rl.metrics import MetricsLogger
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import ReplaySample


def _tiny_config(**overrides) -> TrainingConfig:
    defaults = {
        "iterations": 2,
        "games_per_iteration": 1,
        "num_simulations": 2,
        "min_players": 2,
        "max_players": 2,
        "tau_moves": 1,
        "buffer_capacity": 200,
        "batch_size": 2,
        "train_steps_per_iteration": 1,
        "network_kwargs": {"trunk_dim": 8, "num_residual_blocks": 1},
        "workers": 0,
        "seed": 0,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# --- make_tau_schedule -----------------------------------------------------


def test_make_tau_schedule_switches_from_high_to_low_at_tau_moves():
    schedule = make_tau_schedule(tau_moves=3, high=1.0, low=0.0)

    assert [schedule(i) for i in range(5)] == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_make_tau_schedule_rejects_negative_tau_moves():
    with pytest.raises(ValueError):
        make_tau_schedule(tau_moves=-1)


# --- TrainingConfig validation ------------------------------------------------


def test_training_config_accepts_valid_values():
    config = _tiny_config()
    assert config.iterations == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"iterations": 0},
        {"games_per_iteration": 0},
        {"min_players": 1, "max_players": 2},
        {"min_players": 3, "max_players": 2},
        {"num_simulations": -1},
        {"batch_size": 0},
        {"workers": -1},
        {"checkpoint_every": 0},
    ],
)
def test_training_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        _tiny_config(**overrides)


# --- run_training_loop: happy path -----------------------------------------


def test_run_training_loop_trains_net_and_logs_metrics(tmp_path):
    torch.manual_seed(0)
    config = _tiny_config()
    net = AlphaZeroNet(**config.network_kwargs)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    before = [p.clone() for p in net.parameters()]

    with MetricsLogger(tmp_path / "logs") as metrics:
        trained_net, final_state = run_training_loop(config, metrics, net=net, optimizer=optimizer)

    assert trained_net is net
    assert isinstance(final_state, LoopState)
    assert final_state.iteration == config.iterations
    assert final_state.total_train_steps == config.iterations * config.train_steps_per_iteration
    assert any(not torch.equal(b, a) for b, a in zip(before, net.parameters(), strict=True))

    lines = (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()
    # one self_play log + one train log per iteration
    assert len(lines) == config.iterations * 2


def test_run_training_loop_skips_training_until_buffer_has_a_full_batch(tmp_path):
    # A single short game yields at most a few hundred decision points, well
    # under this batch_size, so the first iteration should record zero train
    # steps rather than raising a "not enough samples" error from the replay
    # buffer.
    config = _tiny_config(games_per_iteration=1, batch_size=5000, train_steps_per_iteration=1, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.total_train_steps == 0


def test_run_training_loop_writes_resumable_checkpoints(tmp_path):
    config = _tiny_config(checkpoint_dir=str(tmp_path / "checkpoints"), checkpoint_every=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        trained_net, final_state = run_training_loop(config, metrics)

    latest_path = tmp_path / "checkpoints" / "latest.pt"
    assert latest_path.exists()
    assert (tmp_path / "checkpoints" / f"checkpoint_{config.iterations:06d}.pt").exists()

    resumed_net = AlphaZeroNet(**config.network_kwargs)
    loaded = load_checkpoint(latest_path, resumed_net)
    assert loaded.iteration == final_state.iteration
    assert loaded.total_train_steps == final_state.total_train_steps
    for trained_param, resumed_param in zip(trained_net.parameters(), resumed_net.parameters(), strict=True):
        assert torch.equal(trained_param, resumed_param)


def test_run_training_loop_resumes_from_a_given_start_state(tmp_path):
    config = _tiny_config(iterations=3)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics, start_state=LoopState(iteration=2, total_train_steps=99))

    # only the remaining iteration (2 -> 3) should have run
    assert final_state.iteration == 3
    assert final_state.total_train_steps == 99 + config.train_steps_per_iteration


# --- run_training_loop: parallel self-play ---------------------------------


def test_run_training_loop_with_multiple_workers_produces_samples(tmp_path):
    config = _tiny_config(games_per_iteration=2, workers=2, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1


# --- self-play resilience: one bad game must not kill the run --------------


def _dummy_replay_samples(n_act: int, count: int) -> list[ReplaySample]:
    state = new_match(player_count=n_act, seed=0)
    pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    pi[: n_act + 1] = 1.0 / (n_act + 1)
    y = np.arange(n_act, dtype=np.int64)
    return [ReplaySample(state=state, n_act=n_act, pi=pi, y=y) for _ in range(count)]


def test_run_training_loop_survives_every_game_failing(tmp_path, monkeypatch):
    def always_fails(*args, **kwargs):
        raise AssertionError("simulated hidden_info bookkeeping bug")

    monkeypatch.setattr(loop_module, "generate_episode", always_fails)
    config = _tiny_config(games_per_iteration=3, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1
    assert final_state.total_train_steps == 0

    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/failed_games"] == 3
    assert record["self_play/samples_generated"] == 0


def test_run_training_loop_keeps_samples_from_games_that_succeed(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("simulated max_steps timeout")
        return _dummy_replay_samples(n_act=2, count=5)

    monkeypatch.setattr(loop_module, "generate_episode", flaky)
    config = _tiny_config(games_per_iteration=4, batch_size=2, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/failed_games"] == 2
    assert record["self_play/samples_generated"] == 10  # 2 successful games * 5 samples each
    assert final_state.total_train_steps > 0  # enough samples made it into the buffer to train on


def test_run_training_loop_survives_an_exception_type_not_specifically_anticipated(tmp_path, monkeypatch):
    # `_play_one_game` used to only catch AssertionError/RuntimeError - the
    # two failure modes already known about. A game failing with anything
    # else (a KeyError from a bookkeeping bug, an IllegalActionError from
    # engine.py, ...) would propagate out of the worker pool and crash the
    # whole run. It should degrade the same way any other per-game failure
    # does: skipped, counted, logged - not fatal.
    def always_fails(*args, **kwargs):
        raise KeyError("simulated unanticipated bug")

    monkeypatch.setattr(loop_module, "generate_episode", always_fails)
    config = _tiny_config(games_per_iteration=2, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1
    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/failed_games"] == 2


def test_run_training_loop_writes_failure_tracebacks_to_the_log_dir(tmp_path, monkeypatch):
    def always_fails(*args, **kwargs):
        raise RuntimeError("simulated max_steps timeout")

    monkeypatch.setattr(loop_module, "generate_episode", always_fails)
    config = _tiny_config(games_per_iteration=2, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    failure_log = tmp_path / "logs" / "self_play_failures.log"
    assert failure_log.exists()
    contents = failure_log.read_text()
    # one full traceback per failed game, not just the one-line summary
    assert contents.count("Traceback (most recent call last):") == 2
    assert "simulated max_steps timeout" in contents
