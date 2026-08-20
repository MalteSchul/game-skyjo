import numpy as np
import pytest

from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.domain.engine import legal_actions as engine_legal_actions
from skyjo.domain.engine import new_match
from skyjo.rl.action_space import ACTION_SPACE_SIZE, action_to_index
from skyjo.rl.selfplay import generate_bot_episode, generate_episode


def _uniform_evaluate(state):
    actions = engine_legal_actions(state)
    priors = {a: 1.0 / len(actions) for a in actions}
    return priors, np.zeros(len(state.boards))


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
