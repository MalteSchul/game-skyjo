import pytest

import skyjo.rl.bootstrap as bootstrap_module
from skyjo.rl.bootstrap import generate_heuristic_dataset, load_replay_samples, save_replay_samples

# --- happy path ------------------------------------------------------------


def test_generate_heuristic_dataset_produces_samples_with_no_failures():
    samples, failed_games = generate_heuristic_dataset(5, min_players=2, max_players=2, workers=0, seed=0)

    assert failed_games == 0
    assert len(samples) > 0
    assert all(sample.n_act == 2 for sample in samples)


def test_generate_heuristic_dataset_is_deterministic_for_the_same_seed():
    first, _ = generate_heuristic_dataset(4, min_players=2, max_players=2, workers=0, seed=3)
    second, _ = generate_heuristic_dataset(4, min_players=2, max_players=2, workers=0, seed=3)

    assert [s.n_act for s in first] == [s.n_act for s in second]
    assert [tuple(s.y.tolist()) for s in first] == [tuple(s.y.tolist()) for s in second]


def test_generate_heuristic_dataset_supports_a_player_count_range():
    samples, failed_games = generate_heuristic_dataset(6, min_players=2, max_players=4, workers=0, seed=1)

    assert failed_games == 0
    assert {s.n_act for s in samples} <= {2, 3, 4}


# --- resilience: one bad game must not sink the whole dataset --------------


def test_generate_heuristic_dataset_survives_a_failing_game(monkeypatch):
    calls = {"n": 0}
    original = bootstrap_module.generate_bot_episode

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated stuck game")
        return original(*args, **kwargs)

    monkeypatch.setattr(bootstrap_module, "generate_bot_episode", flaky)

    samples, failed_games = generate_heuristic_dataset(3, min_players=2, max_players=2, workers=0, seed=0)

    assert failed_games == 1
    assert len(samples) > 0


# --- bad path ------------------------------------------------------------


def test_generate_heuristic_dataset_rejects_non_positive_num_games():
    with pytest.raises(ValueError):
        generate_heuristic_dataset(0)


def test_generate_heuristic_dataset_rejects_an_invalid_player_range():
    with pytest.raises(ValueError):
        generate_heuristic_dataset(3, min_players=4, max_players=2)


# --- on_game_done progress/checkpoint hook ----------------------------------


def test_on_game_done_is_called_once_per_game_with_matching_totals():
    calls: list[tuple[int, bool]] = []

    samples, failed_games = generate_heuristic_dataset(
        5,
        min_players=2,
        max_players=2,
        workers=0,
        seed=0,
        on_game_done=lambda episode_samples, failed: calls.append((len(episode_samples), failed)),
    )

    assert len(calls) == 5
    assert sum(n for n, _ in calls) == len(samples)
    assert sum(1 for _, failed in calls if failed) == failed_games


def test_on_game_done_sees_completed_games_even_when_a_later_one_is_interrupted(monkeypatch):
    original = bootstrap_module.generate_bot_episode
    completed = 0

    def flaky(*args, **kwargs):
        nonlocal completed
        if completed == 2:
            raise KeyboardInterrupt("simulated ctrl-c")
        completed += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(bootstrap_module, "generate_bot_episode", flaky)

    seen: list[list] = []
    with pytest.raises(KeyboardInterrupt):
        generate_heuristic_dataset(
            5,
            min_players=2,
            max_players=2,
            workers=0,
            seed=0,
            on_game_done=lambda episode_samples, failed: seen.append(episode_samples),
        )

    # the two games before the interrupt were still handed to the callback,
    # even though the function itself never returned - that's what lets a
    # caller keep what was generated instead of losing it all.
    assert len(seen) == 2
    assert all(episode_samples for episode_samples in seen)


# --- save/load round-trip ---------------------------------------------------


def test_save_and_load_replay_samples_round_trips(tmp_path):
    samples, _ = generate_heuristic_dataset(4, min_players=2, max_players=2, workers=0, seed=2)
    path = tmp_path / "dataset.pkl"

    save_replay_samples(samples, path)
    loaded = load_replay_samples(path)

    assert len(loaded) == len(samples)
    assert [s.n_act for s in loaded] == [s.n_act for s in samples]
    assert [tuple(s.y.tolist()) for s in loaded] == [tuple(s.y.tolist()) for s in samples]
    assert [s.pi.tolist() for s in loaded] == [s.pi.tolist() for s in samples]
    assert [s.state for s in loaded] == [s.state for s in samples]


def test_save_replay_samples_creates_missing_parent_directories(tmp_path):
    samples, _ = generate_heuristic_dataset(2, min_players=2, max_players=2, workers=0, seed=4)
    path = tmp_path / "nested" / "dir" / "dataset.pkl"

    save_replay_samples(samples, path)

    assert path.exists()
    assert len(load_replay_samples(path)) == len(samples)


def test_save_replay_samples_does_not_leave_a_tmp_file_behind(tmp_path):
    samples, _ = generate_heuristic_dataset(2, min_players=2, max_players=2, workers=0, seed=5)
    path = tmp_path / "dataset.pkl"

    save_replay_samples(samples, path)

    assert not path.with_suffix(path.suffix + ".tmp").exists()


# --- bad path ------------------------------------------------------------


def test_save_replay_samples_rejects_an_empty_list(tmp_path):
    with pytest.raises(ValueError):
        save_replay_samples([], tmp_path / "dataset.pkl")


def test_load_replay_samples_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_replay_samples(tmp_path / "does_not_exist.pkl")
