import numpy as np
import pytest

from skyjo.domain.engine import new_match
from skyjo.rl.action_space import ACTION_SPACE_SIZE
from skyjo.rl.encoding import encode_state
from skyjo.rl.replay_buffer import ReplayBuffer
from skyjo.rl.selfplay import ReplaySample


def _sample(tag: int) -> ReplaySample:
    state = new_match(player_count=2, seed=tag)
    pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    pi[0] = 1.0
    return ReplaySample(
        state=state,
        n_act=2,
        pi=pi,
        y=np.array([0, 1], dtype=np.int64),
        points_y=np.array([0.2, 0.5], dtype=np.float32),
    )


# --- happy path ------------------------------------------------------------


def test_add_and_sample_batch_round_trips_a_sample():
    buffer = ReplayBuffer(capacity=10)
    sample = _sample(1)

    buffer.add(sample)

    assert len(buffer) == 1
    assert buffer.sample_batch(1, np.random.default_rng(0)) == [sample]


def test_add_episode_adds_every_sample():
    buffer = ReplayBuffer(capacity=10)

    buffer.add_episode([_sample(1), _sample(2), _sample(3)])

    assert len(buffer) == 3


def test_buffer_evicts_oldest_sample_once_over_capacity():
    buffer = ReplayBuffer(capacity=2)
    first, second, third = _sample(1), _sample(2), _sample(3)

    buffer.add(first)
    buffer.add(second)
    buffer.add(third)

    assert len(buffer) == 2
    remaining = buffer.sample_batch(2, np.random.default_rng(0))
    assert first not in remaining
    assert second in remaining and third in remaining


def test_sample_batch_with_encodings_matches_encode_state_in_lockstep():
    buffer = ReplayBuffer(capacity=10)
    first, second = _sample(1), _sample(2)
    buffer.add(first)
    buffer.add(second)

    samples, encodings = buffer.sample_batch_with_encodings(2, np.random.default_rng(0))

    assert len(samples) == len(encodings) == 2
    for sample, encoding in zip(samples, encodings, strict=True):
        expected = encode_state(sample.state)
        np.testing.assert_array_equal(encoding.features, expected.features)
        np.testing.assert_array_equal(encoding.legal_action_mask, expected.legal_action_mask)


def test_sample_batch_with_encodings_drops_evicted_slots_encoding():
    buffer = ReplayBuffer(capacity=2)
    buffer.add(_sample(1))
    buffer.add(_sample(2))
    buffer.add(_sample(3))  # evicts sample(1)'s slot, including its cached encoding

    samples, encodings = buffer.sample_batch_with_encodings(2, np.random.default_rng(0))

    states = {sample.state for sample in samples}
    assert new_match(player_count=2, seed=1) not in states
    for sample, encoding in zip(samples, encodings, strict=True):
        np.testing.assert_array_equal(encoding.features, encode_state(sample.state).features)


# --- bad / sad path ---------------------------------------------------------


def test_replay_buffer_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


def test_sample_batch_rejects_a_batch_size_larger_than_the_buffer():
    buffer = ReplayBuffer(capacity=10)
    buffer.add(_sample(1))

    with pytest.raises(ValueError):
        buffer.sample_batch(2, np.random.default_rng(0))


def test_sample_batch_rejects_non_positive_batch_size():
    buffer = ReplayBuffer(capacity=10)
    buffer.add(_sample(1))

    with pytest.raises(ValueError):
        buffer.sample_batch(0, np.random.default_rng(0))


def test_sample_batch_with_encodings_rejects_a_batch_size_larger_than_the_buffer():
    buffer = ReplayBuffer(capacity=10)
    buffer.add(_sample(1))

    with pytest.raises(ValueError):
        buffer.sample_batch_with_encodings(2, np.random.default_rng(0))
