import pytest

import skyjo.rl.bootstrap as bootstrap_module
from skyjo.rl.bootstrap import generate_heuristic_dataset

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
