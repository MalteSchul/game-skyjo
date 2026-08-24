import json
import pickle

import numpy as np
import pytest
import torch

import skyjo.rl.loop as loop_module
from skyjo.domain.engine import new_match
from skyjo.rl.action_space import ACTION_SPACE_SIZE
from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.evaluator import HeuristicEvalResult
from skyjo.rl.loop import LoopState, TrainingConfig, run_training_loop
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
        "tau": 1.0,
        "buffer_capacity": 200,
        "batch_size": 2,
        "train_steps_per_iteration": 1,
        "network_kwargs": {"trunk_dim": 8, "num_residual_blocks": 1},
        "workers": 0,
        "selfplay_batch_size": 1,
        "seed": 0,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


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
        {"selfplay_batch_size": 0},
        {"round_max_steps": 0},
        {"max_rounds": 0},
        {"checkpoint_every": 0},
        {"eval_every": 0},
        {"eval_games": 0},
        {"eval_num_simulations": -1},
    ],
)
def test_training_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        _tiny_config(**overrides)


def test_training_config_rejects_unknown_selfplay_opponent():
    with pytest.raises(ValueError):
        _tiny_config(selfplay_opponent="bogus")


def test_training_config_rejects_heuristic_opponent_with_non_two_player():
    with pytest.raises(ValueError):
        _tiny_config(selfplay_opponent="heuristic", min_players=2, max_players=3)


# --- round_max_steps/max_rounds wiring --------------------------------------


def test_run_training_loop_passes_round_max_steps_and_max_rounds_to_generate_episode(tmp_path, monkeypatch):
    seen_kwargs = {}

    def fake_generate_episode(*args, **kwargs):
        seen_kwargs["round_max_steps"] = kwargs["round_max_steps"]
        seen_kwargs["max_rounds"] = kwargs["max_rounds"]
        return _dummy_replay_samples(n_act=2, count=3)

    monkeypatch.setattr(loop_module, "generate_episode", fake_generate_episode)
    config = _tiny_config(
        games_per_iteration=1, iterations=1, round_max_steps=17, max_rounds=3, selfplay_batch_size=1
    )

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    assert seen_kwargs == {"round_max_steps": 17, "max_rounds": 3}


def test_run_training_loop_passes_round_max_steps_and_max_rounds_to_generate_episodes_batch(tmp_path, monkeypatch):
    seen_kwargs = {}

    def fake_generate_episodes_batch(*args, **kwargs):
        seen_kwargs["round_max_steps"] = kwargs["round_max_steps"]
        seen_kwargs["max_rounds"] = kwargs["max_rounds"]
        return [_dummy_replay_samples(n_act=2, count=3)]

    monkeypatch.setattr(loop_module, "generate_episodes_batch", fake_generate_episodes_batch)
    config = _tiny_config(
        games_per_iteration=2, selfplay_batch_size=2, iterations=1, round_max_steps=23, max_rounds=4
    )

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    assert seen_kwargs == {"round_max_steps": 23, "max_rounds": 4}


# --- self_play/avg_points_per_round -----------------------------------------


def test_run_training_loop_logs_avg_points_per_round(tmp_path, monkeypatch):
    def fake_generate_episode(*args, round_stats_sink=None, **kwargs):
        if round_stats_sink is not None:
            round_stats_sink.append((4, (20, 30)))  # round_count=4, final points mean=25
        return _dummy_replay_samples(n_act=2, count=3)

    monkeypatch.setattr(loop_module, "generate_episode", fake_generate_episode)
    config = _tiny_config(games_per_iteration=1, iterations=1, selfplay_batch_size=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/avg_points_per_round"] == pytest.approx(25.0 / 4)


def test_run_training_loop_avg_points_per_round_is_zero_when_every_game_fails(tmp_path, monkeypatch):
    def always_fails(*args, **kwargs):
        raise RuntimeError("simulated max_steps timeout")

    monkeypatch.setattr(loop_module, "generate_episode", always_fails)
    config = _tiny_config(games_per_iteration=2, iterations=1, selfplay_batch_size=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/avg_points_per_round"] == 0.0


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


def test_run_training_loop_seeds_buffer_from_initial_samples(tmp_path, monkeypatch):
    # Same shape as test_run_training_loop_skips_training_until_buffer_has_a_full_batch
    # (self-play alone can't fill a batch_size=5000 buffer from one tiny game),
    # but here initial_samples pre-fills the buffer so training should proceed
    # on iteration 1 instead of being skipped.
    monkeypatch.setattr(loop_module, "generate_episode", lambda *a, **k: [])
    config = _tiny_config(games_per_iteration=1, buffer_capacity=5000, batch_size=5000, train_steps_per_iteration=1, iterations=1)
    seed_samples = _dummy_replay_samples(n_act=2, count=5000)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics, initial_samples=seed_samples)

    assert final_state.total_train_steps == 1


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


def test_run_training_loop_saves_buffer_alongside_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "generate_episode", lambda *a, **k: [])
    seed_samples = _dummy_replay_samples(n_act=2, count=50)
    config = _tiny_config(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_every=1,
        buffer_capacity=50,
        batch_size=50,
        train_steps_per_iteration=1,
        iterations=1,
    )

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics, initial_samples=seed_samples)

    buffer_path = tmp_path / "checkpoints" / "buffer_latest.pkl"
    assert buffer_path.exists()
    with open(buffer_path, "rb") as f:
        saved_samples = pickle.load(f)
    assert len(saved_samples) == 50
    assert all(isinstance(s, ReplaySample) for s in saved_samples)


def test_run_training_loop_resumes_from_a_given_start_state(tmp_path):
    config = _tiny_config(iterations=3)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics, start_state=LoopState(iteration=2, total_train_steps=99))

    # only the remaining iteration (2 -> 3) should have run
    assert final_state.iteration == 3
    assert final_state.total_train_steps == 99 + config.train_steps_per_iteration


# --- run_training_loop: periodic heuristic eval -----------------------------


def test_run_training_loop_does_not_evaluate_by_default(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(loop_module, "evaluate_vs_heuristic", lambda *a, **k: calls.append((a, k)))
    config = _tiny_config(iterations=2)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    assert calls == []
    lines = (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()
    assert not any("eval/" in line for line in lines)


def test_run_training_loop_logs_eval_metrics_every_eval_every_iterations(tmp_path, monkeypatch):
    calls = []

    def fake_eval(net, num_games, **kwargs):
        calls.append((num_games, kwargs))
        return HeuristicEvalResult(games_played=num_games, win_rate=0.75, avg_rank=0.25, avg_points=12.5)

    monkeypatch.setattr(loop_module, "evaluate_vs_heuristic", fake_eval)
    config = _tiny_config(
        iterations=1, eval_every=1, eval_games=7, eval_num_simulations=3, round_max_steps=11, max_rounds=2
    )

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    assert len(calls) == 1
    num_games, kwargs = calls[0]
    assert num_games == 7
    assert kwargs["num_simulations"] == 3
    # eval should use the same self-play safety-valve config, not silently fall back to
    # evaluate_vs_heuristic's own defaults - a user who tunes these for self-play but not eval
    # would otherwise hit the exact outer-max_steps-too-tight trap this valve exists to avoid.
    assert kwargs["round_max_steps"] == 11
    assert kwargs["max_rounds"] == 2

    lines = [json.loads(line) for line in (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()]
    eval_records = [line for line in lines if "eval/win_rate_vs_heuristic" in line]
    assert len(eval_records) == 1
    assert eval_records[0]["eval/win_rate_vs_heuristic"] == 0.75
    assert eval_records[0]["eval/avg_rank_vs_heuristic"] == 0.25
    assert eval_records[0]["eval/avg_points_vs_heuristic"] == 12.5


def test_run_training_loop_only_evaluates_on_eval_every_boundaries(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        loop_module,
        "evaluate_vs_heuristic",
        lambda *a, **k: calls.append(1) or HeuristicEvalResult(games_played=1, win_rate=1.0, avg_rank=0.0, avg_points=0.0),
    )
    config = _tiny_config(iterations=3, eval_every=2)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    # iterations 1, 2, 3 completed; only iteration 2 is a multiple of eval_every=2
    assert len(calls) == 1


# --- run_training_loop: parallel self-play ---------------------------------


def test_run_training_loop_with_multiple_workers_produces_samples(tmp_path):
    config = _tiny_config(games_per_iteration=2, workers=2, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1


# --- run_training_loop: batched self-play -----------------------------------


def test_run_training_loop_with_selfplay_batch_size_produces_samples_and_trains(tmp_path):
    config = _tiny_config(games_per_iteration=4, selfplay_batch_size=2, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1
    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/failed_games"] == 0
    assert record["self_play/samples_generated"] > 0


def test_run_training_loop_with_selfplay_batch_size_and_multiple_workers(tmp_path):
    # games_per_iteration=4, batch_size=2 -> 2 groups of 2, sharded across 2
    # worker processes: batching and process-parallelism must compose.
    config = _tiny_config(games_per_iteration=4, selfplay_batch_size=2, workers=2, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1


def test_run_training_loop_with_selfplay_batch_size_one_matches_default_behavior(tmp_path):
    # selfplay_batch_size=1 is the explicit form of the default - both should
    # go through the exact same unbatched path (run_self_play_iteration), so
    # this should behave identically to a config without the field set.
    config = _tiny_config(games_per_iteration=2, selfplay_batch_size=1, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1


# --- run_training_loop: self-play vs a fixed heuristic opponent -------------


def test_run_training_loop_vs_heuristic_produces_samples_and_trains(tmp_path):
    config = _tiny_config(games_per_iteration=2, selfplay_opponent="heuristic", iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1
    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/failed_games"] == 0
    assert record["self_play/samples_generated"] > 0


def test_run_training_loop_vs_heuristic_alternates_net_seat(tmp_path, monkeypatch):
    seen_net_seats = []
    original = loop_module.generate_episode_vs_bot

    def spy(initial_state, evaluate, opponent_choose_action, net_seat, **kwargs):
        seen_net_seats.append(net_seat)
        return original(initial_state, evaluate, opponent_choose_action, net_seat, **kwargs)

    monkeypatch.setattr(loop_module, "generate_episode_vs_bot", spy)
    config = _tiny_config(games_per_iteration=4, selfplay_opponent="heuristic", iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    assert seen_net_seats == [0, 1, 0, 1]


def test_run_training_loop_vs_heuristic_with_multiple_workers_produces_samples(tmp_path):
    config = _tiny_config(games_per_iteration=2, selfplay_opponent="heuristic", workers=2, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1


def test_run_training_loop_vs_random_produces_samples_and_trains(tmp_path):
    config = _tiny_config(games_per_iteration=2, selfplay_opponent="random", iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1
    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/failed_games"] == 0
    assert record["self_play/samples_generated"] > 0


def test_run_training_loop_batched_selfplay_survives_a_whole_group_failing(tmp_path, monkeypatch):
    # generate_episodes_batch failing takes down its entire group (see
    # _play_batch_of_games's docstring) - unlike the per-game path, so the
    # failed-game count should reflect every seed in the failed group, not
    # just one.
    def always_fails(*args, **kwargs):
        raise AssertionError("simulated hidden_info bookkeeping bug")

    monkeypatch.setattr(loop_module, "generate_episodes_batch", always_fails)
    config = _tiny_config(games_per_iteration=4, selfplay_batch_size=2, iterations=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1
    assert final_state.total_train_steps == 0
    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/failed_games"] == 4
    assert record["self_play/samples_generated"] == 0


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
    config = _tiny_config(games_per_iteration=3, iterations=1, selfplay_batch_size=1)

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
    config = _tiny_config(games_per_iteration=4, batch_size=2, iterations=1, selfplay_batch_size=1)

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
    config = _tiny_config(games_per_iteration=2, iterations=1, selfplay_batch_size=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1
    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/failed_games"] == 2


def test_run_training_loop_writes_failure_tracebacks_to_the_log_dir(tmp_path, monkeypatch):
    def always_fails(*args, **kwargs):
        raise RuntimeError("simulated max_steps timeout")

    monkeypatch.setattr(loop_module, "generate_episode", always_fails)
    config = _tiny_config(games_per_iteration=2, iterations=1, selfplay_batch_size=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    failure_log = tmp_path / "logs" / "self_play_failures.log"
    assert failure_log.exists()
    contents = failure_log.read_text()
    # one full traceback per failed game, not just the one-line summary
    assert contents.count("Traceback (most recent call last):") == 2
    assert "simulated max_steps timeout" in contents
