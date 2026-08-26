"""Fixed-capacity FIFO replay buffer of self-play `ReplaySample`s."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from skyjo.rl.encoding import StateEncoding, encode_state
from skyjo.rl.selfplay import ReplaySample


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer: capacity must be > 0")
        self._capacity = capacity
        self._samples: list[ReplaySample] = []
        # Parallel to `_samples` (same index, same length always) - `encode_state`
        # is pure given a sample's own (immutable) state, so computing it once
        # here and reusing it for every later `sample_batch_with_encodings` draw
        # avoids re-running it on every training step a sample happens to be
        # drawn on. Indexed rather than keyed by state: `GameState` is hashable,
        # but hashing its full `stock`/`discard` tuples on every lookup would
        # itself cost close to what encoding does, and the buffer already tracks
        # each sample's slot for FIFO eviction, so piggy-backing on that index is
        # free. Not stored on `ReplaySample` itself - see that class's docstring
        # for why a sample deliberately keeps only the raw state.
        self._encodings: list[StateEncoding] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> list[ReplaySample]:
        """Current contents in FIFO storage order (not insertion-recency order
        once the buffer has wrapped) - read-only, for persistence only."""
        return list(self._samples)

    def add(self, sample: ReplaySample) -> None:
        encoding = encode_state(sample.state)
        if len(self._samples) < self._capacity:
            self._samples.append(sample)
            self._encodings.append(encoding)
        else:
            self._samples[self._next_index] = sample
            self._encodings[self._next_index] = encoding
        self._next_index = (self._next_index + 1) % self._capacity

    def add_episode(self, samples: Sequence[ReplaySample]) -> None:
        for sample in samples:
            self.add(sample)

    def _sample_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("ReplayBuffer.sample_batch: batch_size must be > 0")
        if batch_size > len(self._samples):
            raise ValueError(
                f"ReplayBuffer.sample_batch: requested {batch_size} but buffer only has "
                f"{len(self._samples)} samples"
            )
        return rng.choice(len(self._samples), size=batch_size, replace=False)

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> list[ReplaySample]:
        indices = self._sample_indices(batch_size, rng)
        return [self._samples[i] for i in indices]

    def sample_batch_with_encodings(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[list[ReplaySample], list[StateEncoding]]:
        """Same sampling as `sample_batch`, plus each sample's cached
        `StateEncoding` (computed once in `add`) in lockstep order - lets a
        caller (`rl.loop.run_training_loop`) skip `train.collate_batch`
        re-running `encode_state` on every draw of an already-seen sample.
        """
        indices = self._sample_indices(batch_size, rng)
        return [self._samples[i] for i in indices], [self._encodings[i] for i in indices]
