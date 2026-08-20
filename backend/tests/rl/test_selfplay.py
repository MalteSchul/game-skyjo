import numpy as np
import pytest

from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.domain.engine import new_match
from skyjo.rl.action_space import ACTION_SPACE_SIZE, action_to_index
from skyjo.rl.selfplay import generate_bot_episode, generate_episode, generate_episodes_batch


def _uniform_evaluate(state):
    actions = engine_legal_actions(state)
    priors = {a: 1.0 / len(actions) for a in actions}
    return priors, np.zeros(len(state.boards))


def _batch_evaluate(states):
    return [_uniform_evaluate(s) for s in states]


# --- happy path ------------------------------------------------------------


def test_generate_episode_produces_consistent_samples_for_a_full_match():
    state = new_match(player_count=2, seed=1)

    samples = generate_episode(
        state, _uniform_evaluate, num_simulations=2, rng=np.random.default_rng(0), max_steps=3000
    )

    assert len(samples) > 0
    for sample in samples:
        assert sample.n_act == 2
        assert sample.pi.shape == (ACTION_SPACE_SIZE,)
        assert sample.pi.sum() == pytest.approx(1.0, abs=1e-4)
        assert sample.y.shape == (2,)
    # the final ranks are the same object recorded at every decision step
    assert {tuple(s.y) for s in samples} == {tuple(samples[-1].y)}
    assert sorted(samples[-1].y.tolist()) == [0, 1]  # a strict permutation for 2 players


def test_generate_episode_accepts_a_callable_tau_schedule():
    # A near-zero tau combined with an evaluator that gives no real signal (as
    # here) can make deterministic argmax play cycle indefinitely - that's a
    # property of the degenerate test evaluator, not of generate_episode, so
    # this only exercises that the callable-schedule code path runs; the
    # near-zero-tau behavior itself is unit-tested directly against
    # visit_distribution in test_mcts.py.
    state = new_match(player_count=3, seed=2)
    calls: list[int] = []

    def tau_schedule(step: int) -> float:
        calls.append(step)
        return 1.0

    samples = generate_episode(
        state, _uniform_evaluate, num_simulations=2, tau_schedule=tau_schedule, rng=np.random.default_rng(1),
        max_steps=3000,
    )

    assert all(s.n_act == 3 for s in samples)
    assert sorted(samples[-1].y.tolist()) == [0, 1, 2]
    # one call per recorded decision; step skips ahead across round transitions
    # (which don't call tau_schedule), so calls is increasing but not contiguous.
    assert len(calls) == len(samples)
    assert calls == sorted(set(calls))


# --- bad path ------------------------------------------------------------


def test_generate_episode_raises_if_the_match_does_not_finish_within_max_steps():
    state = new_match(player_count=2, seed=1)

    with pytest.raises(RuntimeError):
        generate_episode(state, _uniform_evaluate, num_simulations=1, rng=np.random.default_rng(0), max_steps=1)


# --- generate_episodes_batch ------------------------------------------------


def test_generate_episodes_batch_matches_generate_episode_played_one_at_a_time():
    # Same property as run_mcts_batch's equivalence test, one level up: playing
    # N games concurrently must produce exactly the samples/outcomes that
    # playing each one alone (with the same seed) would - batching only
    # changes when the evaluator is called, never the result.
    state_a = new_match(player_count=2, seed=1)
    state_b = new_match(player_count=3, seed=2)

    solo_a = generate_episode(state_a, _uniform_evaluate, num_simulations=2, rng=np.random.default_rng(10), max_steps=3000)
    solo_b = generate_episode(state_b, _uniform_evaluate, num_simulations=2, rng=np.random.default_rng(20), max_steps=3000)

    batch_a, batch_b = generate_episodes_batch(
        [state_a, state_b],
        _batch_evaluate,
        num_simulations=2,
        rngs=[np.random.default_rng(10), np.random.default_rng(20)],
        max_steps=3000,
    )

    assert [s.pi.tolist() for s in batch_a] == [s.pi.tolist() for s in solo_a]
    assert [s.y.tolist() for s in batch_a] == [s.y.tolist() for s in solo_a]
    assert [s.pi.tolist() for s in batch_b] == [s.pi.tolist() for s in solo_b]
    assert [s.y.tolist() for s in batch_b] == [s.y.tolist() for s in solo_b]


def test_generate_episodes_batch_handles_games_finishing_at_different_times():
    # player counts differ (2 vs 4), so the games are very unlikely to reach
    # game_over in the same number of decisions - this exercises the "batch
    # shrinks as games finish early" path rather than every game ending on
    # the same round.
    states = [new_match(player_count=2, seed=1), new_match(player_count=4, seed=2)]

    results = generate_episodes_batch(
        states,
        _batch_evaluate,
        num_simulations=2,
        rngs=[np.random.default_rng(0), np.random.default_rng(1)],
        max_steps=3000,
    )

    assert len(results) == 2
    assert len(results[0]) > 0
    assert len(results[1]) > 0
    assert sorted(results[0][-1].y.tolist()) == [0, 1]
    assert sorted(results[1][-1].y.tolist()) == [0, 1, 2, 3]


def test_generate_episodes_batch_with_no_games_returns_an_empty_list():
    assert generate_episodes_batch([], _batch_evaluate, num_simulations=2) == []


# --- generate_episodes_batch: bad path --------------------------------------


def test_generate_episodes_batch_raises_if_a_game_does_not_finish_within_max_steps():
    states = [new_match(player_count=2, seed=1)]

    with pytest.raises(RuntimeError):
        generate_episodes_batch(states, _batch_evaluate, num_simulations=1, rngs=[np.random.default_rng(0)], max_steps=1)


def test_generate_episodes_batch_rejects_mismatched_rngs_length():
    states = [new_match(player_count=2, seed=1), new_match(player_count=2, seed=2)]

    with pytest.raises(ValueError):
        generate_episodes_batch(states, _batch_evaluate, num_simulations=1, rngs=[np.random.default_rng(0)])


# --- generate_bot_episode --------------------------------------------------


def test_generate_bot_episode_produces_one_hot_pi_matching_the_bots_actual_choices():
    state = new_match(player_count=2, seed=1)
    bots = [HeuristicBot(seed=1), HeuristicBot(seed=2)]

    samples = generate_bot_episode(state, [b.choose_action for b in bots], max_steps=3000)

    assert len(samples) > 0
    for sample in samples:
        assert sample.n_act == 2
        assert sample.pi.shape == (ACTION_SPACE_SIZE,)
        # one-hot: exactly one action carries all the probability mass
        assert sample.pi.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.count_nonzero(sample.pi) == 1
    assert sorted(samples[-1].y.tolist()) == [0, 1]


def test_generate_bot_episode_pi_is_always_a_legal_action_for_its_own_state():
    state = new_match(player_count=3, seed=3)
    bots = [HeuristicBot(seed=1), HeuristicBot(seed=2), HeuristicBot(seed=3)]

    samples = generate_bot_episode(state, [b.choose_action for b in bots], max_steps=3000)

    for sample in samples:
        legal_indices = {action_to_index(a) for a in engine_legal_actions(sample.state)}
        chosen_index = int(np.argmax(sample.pi))
        assert chosen_index in legal_indices


def test_generate_bot_episode_is_deterministic_for_the_same_seeded_bots():
    def play(seed: int) -> list[int]:
        state = new_match(player_count=2, seed=seed)
        bots = [HeuristicBot(seed=1), HeuristicBot(seed=2)]
        samples = generate_bot_episode(state, [b.choose_action for b in bots], max_steps=3000)
        return [int(np.argmax(s.pi)) for s in samples]

    assert play(7) == play(7)


# --- bad path ------------------------------------------------------------


def test_generate_bot_episode_raises_if_the_match_does_not_finish_within_max_steps():
    state = new_match(player_count=2, seed=1)
    bots = [HeuristicBot(seed=1), HeuristicBot(seed=2)]

    with pytest.raises(RuntimeError):
        generate_bot_episode(state, [b.choose_action for b in bots], max_steps=1)
