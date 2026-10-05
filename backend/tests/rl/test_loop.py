import copy
import json
import pickle

import numpy as np
import pytest
import torch

import skyjo.rl.loop as loop_module
from skyjo.domain.engine import new_match
from skyjo.rl.action_space import ACTION_SPACE_SIZE
from skyjo.rl.checkpoint import load_checkpoint, save_checkpoint
from skyjo.rl.evaluator import HeuristicEvalResult
from skyjo.rl.loop import LoopState, TrainingConfig, run_training_loop, sample_pool_checkpoint
from skyjo.rl.match import MatchEvalResult
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
        # Low num_simulations above has no pressure to reveal new information,
        # so a round/game can run indefinitely without these - re-enable the
        # (now default-disabled) safety valves explicitly for fast, bounded tests.
        "round_max_steps": 200,
        "max_rounds": 10,
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
        {"opponent_pool_prob": -0.1},
        {"opponent_pool_prob": 1.1},
        {"opponent_pool_window": 0},
        {"opponent_pool_num_simulations": -1},
        {"eval_checkpoint_num_simulations": -1},
    ],
)
def test_training_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        _tiny_config(**overrides)


def test_training_config_rejects_opponent_pool_prob_with_non_two_player():
    with pytest.raises(ValueError):
        _tiny_config(opponent_pool_prob=0.5, min_players=2, max_players=3)


def test_training_config_accepts_zero_opponent_pool_prob_with_non_two_player():
    config = _tiny_config(opponent_pool_prob=0.0, min_players=2, max_players=3)
    assert config.opponent_pool_prob == 0.0


def test_training_config_rejects_eval_checkpoint_path_with_non_two_player():
    with pytest.raises(ValueError):
        _tiny_config(eval_checkpoint_path="some/checkpoint.pt", min_players=2, max_players=3)


def test_training_config_rejects_unknown_selfplay_opponent():
    with pytest.raises(ValueError):
        _tiny_config(selfplay_opponent="bogus")


def test_training_config_rejects_heuristic_opponent_with_non_two_player():
    with pytest.raises(ValueError):
        _tiny_config(selfplay_opponent="heuristic", min_players=2, max_players=3)


def test_training_config_rejects_negative_eval_workers():
    with pytest.raises(ValueError):
        _tiny_config(eval_workers=-1)


def test_training_config_rejects_eval_batch_size_below_one():
    with pytest.raises(ValueError):
        _tiny_config(eval_batch_size=0)


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
        iterations=1, eval_every=1, eval_games=7, eval_num_simulations=3, round_max_steps=11, max_rounds=2,
        eval_workers=4, eval_batch_size=8,
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
    # eval_workers/eval_batch_size are a separate knob from self-play's own
    # workers/selfplay_batch_size - passed through, not silently defaulted.
    assert kwargs["workers"] == 4
    assert kwargs["eval_batch_size"] == 8

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


# --- run_training_loop: eval gating -----------------------------------------


def test_training_config_gate_on_eval_requires_eval_every():
    with pytest.raises(ValueError):
        _tiny_config(gate_on_eval=True)


def test_training_config_rejects_negative_gate_tolerance():
    with pytest.raises(ValueError):
        _tiny_config(gate_on_eval=True, eval_every=1, gate_tolerance=-0.1)


def test_run_training_loop_gate_accepts_improving_evals(tmp_path, monkeypatch):
    win_rates = iter([0.5, 0.6, 0.7])
    monkeypatch.setattr(
        loop_module,
        "evaluate_vs_heuristic",
        lambda *a, **k: HeuristicEvalResult(games_played=1, win_rate=next(win_rates), avg_rank=0.0, avg_points=0.0),
    )
    config = _tiny_config(iterations=3, eval_every=1, gate_on_eval=True)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    lines = [json.loads(line) for line in (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()]
    gate_records = [line for line in lines if "eval/gate_accepted" in line]
    assert [r["eval/gate_accepted"] for r in gate_records] == [1.0, 1.0, 1.0]
    assert gate_records[-1]["eval/gate_best_win_rate"] == pytest.approx(0.7)


def test_run_training_loop_gate_rejects_a_regressing_eval(tmp_path, monkeypatch):
    win_rates = iter([0.7, 0.3])  # iter 1 accepted (beats the 0.0 initial floor); iter 2 regresses
    monkeypatch.setattr(
        loop_module,
        "evaluate_vs_heuristic",
        lambda *a, **k: HeuristicEvalResult(games_played=1, win_rate=next(win_rates), avg_rank=0.0, avg_points=0.0),
    )
    config = _tiny_config(iterations=2, eval_every=1, gate_on_eval=True)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    lines = [json.loads(line) for line in (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()]
    gate_records = [line for line in lines if "eval/gate_accepted" in line]
    assert [r["eval/gate_accepted"] for r in gate_records] == [1.0, 0.0]
    # best_win_rate stays at the accepted iteration's value, not the rejected one
    assert gate_records[-1]["eval/gate_best_win_rate"] == pytest.approx(0.7)


def test_run_training_loop_gate_rejection_rolls_back_weights(tmp_path, monkeypatch):
    # Two separate run_training_loop calls, resuming the same net/optimizer -
    # mirrors how a real run would gate a later regression against an
    # already-accepted checkpoint from an earlier call.
    net = AlphaZeroNet(trunk_dim=8, num_residual_blocks=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    monkeypatch.setattr(
        loop_module,
        "evaluate_vs_heuristic",
        lambda *a, **k: HeuristicEvalResult(games_played=1, win_rate=0.7, avg_rank=0.0, avg_points=0.0),
    )
    config_accept = _tiny_config(iterations=1, eval_every=1, gate_on_eval=True, network_kwargs={"trunk_dim": 8, "num_residual_blocks": 1})
    with MetricsLogger(tmp_path / "logs1") as metrics:
        run_training_loop(config_accept, metrics, net=net, optimizer=optimizer)
    accepted_snapshot = copy.deepcopy(net.state_dict())

    monkeypatch.setattr(
        loop_module,
        "evaluate_vs_heuristic",
        lambda *a, **k: HeuristicEvalResult(games_played=1, win_rate=0.2, avg_rank=0.0, avg_points=0.0),
    )
    config_reject = _tiny_config(
        iterations=2, eval_every=1, gate_on_eval=True, gate_initial_best_win_rate=0.7,
        network_kwargs={"trunk_dim": 8, "num_residual_blocks": 1},
    )
    with MetricsLogger(tmp_path / "logs2") as metrics:
        run_training_loop(config_reject, metrics, net=net, optimizer=optimizer, start_state=LoopState(iteration=1, total_train_steps=1))

    for key, accepted_tensor in accepted_snapshot.items():
        assert torch.equal(net.state_dict()[key], accepted_tensor)


def test_run_training_loop_gate_tolerance_accepts_a_small_regression(tmp_path, monkeypatch):
    win_rates = iter([0.7, 0.65])  # within gate_tolerance=0.1 of the 0.7 floor
    monkeypatch.setattr(
        loop_module,
        "evaluate_vs_heuristic",
        lambda *a, **k: HeuristicEvalResult(games_played=1, win_rate=next(win_rates), avg_rank=0.0, avg_points=0.0),
    )
    config = _tiny_config(iterations=2, eval_every=1, gate_on_eval=True, gate_tolerance=0.1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    lines = [json.loads(line) for line in (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()]
    gate_records = [line for line in lines if "eval/gate_accepted" in line]
    assert [r["eval/gate_accepted"] for r in gate_records] == [1.0, 1.0]


def test_run_training_loop_gate_initial_best_win_rate_protects_a_known_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(
        loop_module,
        "evaluate_vs_heuristic",
        lambda *a, **k: HeuristicEvalResult(games_played=1, win_rate=0.5, avg_rank=0.0, avg_points=0.0),
    )
    config = _tiny_config(iterations=1, eval_every=1, gate_on_eval=True, gate_initial_best_win_rate=0.9)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    lines = [json.loads(line) for line in (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()]
    gate_record = next(line for line in lines if "eval/gate_accepted" in line)
    assert gate_record["eval/gate_accepted"] == 0.0
    assert gate_record["eval/gate_best_win_rate"] == pytest.approx(0.9)


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


# --- sample_pool_checkpoint --------------------------------------------------


def test_sample_pool_checkpoint_returns_none_when_the_directory_has_no_checkpoints(tmp_path):
    assert sample_pool_checkpoint(str(tmp_path), window=10, rng=np.random.default_rng(0)) is None


def test_sample_pool_checkpoint_only_draws_from_the_most_recent_window(tmp_path):
    net = AlphaZeroNet(trunk_dim=8, num_residual_blocks=1)
    for iteration in range(1, 6):
        save_checkpoint(tmp_path / f"checkpoint_{iteration:06d}.pt", net, None, iteration=iteration, total_train_steps=0)
    # A non-matching file (latest.pt) must never be picked.
    save_checkpoint(tmp_path / "latest.pt", net, None, iteration=5, total_train_steps=0)

    rng = np.random.default_rng(0)
    seen = {sample_pool_checkpoint(str(tmp_path), window=2, rng=rng) for _ in range(30)}

    assert seen == {str(tmp_path / "checkpoint_000004.pt"), str(tmp_path / "checkpoint_000005.pt")}


# --- run_training_loop: opponent pool self-play ------------------------------


def test_run_training_loop_opponent_pool_prob_zero_never_samples_the_pool(tmp_path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("sample_pool_checkpoint should never be called when opponent_pool_prob is 0.0")

    monkeypatch.setattr(loop_module, "sample_pool_checkpoint", fail_if_called)
    config = _tiny_config(checkpoint_dir=str(tmp_path / "checkpoints"), iterations=2)

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 2


def test_run_training_loop_falls_back_to_self_when_no_pool_checkpoint_exists_yet(tmp_path, monkeypatch):
    # opponent_pool_prob=1.0 always wants the pool, but a fresh run's
    # checkpoint_dir starts empty - the very first iteration has nothing to
    # sample yet, so it must fall back to ordinary self-play rather than error.
    calls = []
    monkeypatch.setattr(loop_module, "generate_episodes_batch_vs_decider", lambda *a, **k: calls.append(1))
    config = _tiny_config(
        checkpoint_dir=str(tmp_path / "checkpoints"), opponent_pool_prob=1.0, games_per_iteration=1, iterations=1,
    )

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert calls == []
    assert final_state.iteration == 1


def test_run_training_loop_uses_the_pool_opponent_once_a_checkpoint_exists(tmp_path):
    net_kwargs = {"trunk_dim": 8, "num_residual_blocks": 1}
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    save_checkpoint(
        checkpoint_dir / "checkpoint_000001.pt", AlphaZeroNet(**net_kwargs), None, iteration=1, total_train_steps=0,
    )
    config = _tiny_config(
        network_kwargs=net_kwargs, checkpoint_dir=str(checkpoint_dir), opponent_pool_prob=1.0,
        opponent_pool_window=5, games_per_iteration=2, iterations=1,
    )

    with MetricsLogger(tmp_path / "logs") as metrics:
        _, final_state = run_training_loop(config, metrics)

    assert final_state.iteration == 1
    record = json.loads((tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["self_play/failed_games"] == 0
    assert record["self_play/samples_generated"] > 0


# --- run_training_loop: fixed-checkpoint eval --------------------------------


def test_run_training_loop_does_not_evaluate_vs_checkpoint_by_default(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(loop_module, "evaluate_vs_decider", lambda *a, **k: calls.append(1))
    config = _tiny_config(iterations=1, eval_every=1)

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    assert calls == []


def test_run_training_loop_logs_eval_metrics_vs_checkpoint_when_configured(tmp_path, monkeypatch):
    calls = []

    def fake_eval(net, checkpoint_path, num_games, **kwargs):
        calls.append((checkpoint_path, num_games, kwargs))
        return MatchEvalResult(games_played=num_games, win_rate=0.4, avg_rank=0.6, avg_points=30.0)

    monkeypatch.setattr(loop_module, "evaluate_vs_decider", fake_eval)
    config = _tiny_config(iterations=1, eval_every=1, eval_games=5, eval_checkpoint_path="some/checkpoint.pt")

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    assert len(calls) == 1
    checkpoint_path, num_games, kwargs = calls[0]
    assert checkpoint_path == "some/checkpoint.pt"
    assert num_games == 5
    # eval_checkpoint_num_simulations wasn't set, so this must fall back to
    # eval_num_simulations for both sides rather than some other default.
    assert kwargs["num_simulations"] == config.eval_num_simulations
    assert kwargs["opponent_num_simulations"] == config.eval_num_simulations

    lines = [json.loads(line) for line in (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()]
    eval_records = [line for line in lines if "eval/win_rate_vs_checkpoint" in line]
    assert len(eval_records) == 1
    assert eval_records[0]["eval/win_rate_vs_checkpoint"] == 0.4
    assert eval_records[0]["eval/avg_rank_vs_checkpoint"] == 0.6
    assert eval_records[0]["eval/avg_points_vs_checkpoint"] == 30.0
    # win_rate_vs_heuristic must still be logged alongside it, unaffected.
    assert "eval/win_rate_vs_heuristic" in eval_records[0]


def test_run_training_loop_eval_checkpoint_num_simulations_overrides_eval_num_simulations(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        loop_module,
        "evaluate_vs_decider",
        lambda net, checkpoint_path, num_games, **kwargs: calls.append(kwargs)
        or MatchEvalResult(games_played=num_games, win_rate=0.5, avg_rank=0.5, avg_points=50.0),
    )
    config = _tiny_config(
        iterations=1, eval_every=1, eval_num_simulations=20, eval_checkpoint_path="some/checkpoint.pt",
        eval_checkpoint_num_simulations=3,
    )

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    assert len(calls) == 1
    assert calls[0]["num_simulations"] == 3
    assert calls[0]["opponent_num_simulations"] == 3


def test_run_training_loop_gate_on_eval_ignores_the_checkpoint_eval_result(tmp_path, monkeypatch):
    # gate_on_eval must stay anchored to win_rate_vs_heuristic - a poor
    # win_rate_vs_checkpoint (a much stronger fixed opponent, say) should
    # never by itself cause a good heuristic-beating update to be rejected.
    monkeypatch.setattr(
        loop_module,
        "evaluate_vs_heuristic",
        lambda *a, **k: HeuristicEvalResult(games_played=1, win_rate=0.9, avg_rank=0.0, avg_points=0.0),
    )
    monkeypatch.setattr(
        loop_module,
        "evaluate_vs_decider",
        lambda *a, **k: MatchEvalResult(games_played=1, win_rate=0.0, avg_rank=1.0, avg_points=100.0),
    )
    config = _tiny_config(iterations=1, eval_every=1, gate_on_eval=True, eval_checkpoint_path="some/checkpoint.pt")

    with MetricsLogger(tmp_path / "logs") as metrics:
        run_training_loop(config, metrics)

    lines = [json.loads(line) for line in (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()]
    gate_record = next(line for line in lines if "eval/gate_accepted" in line)
    assert gate_record["eval/gate_accepted"] == 1.0


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
    points_y = np.zeros(n_act, dtype=np.float32)
    return [ReplaySample(state=state, n_act=n_act, pi=pi, y=y, points_y=points_y) for _ in range(count)]


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
