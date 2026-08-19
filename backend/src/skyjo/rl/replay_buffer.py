"""Fixed-capacity FIFO replay buffer of self-play `ReplaySample`s."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from skyjo.rl.selfplay import ReplaySample


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer: capacity must be > 0")
        self._capacity = capacity
        self._samples: list[ReplaySample] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: ReplaySample) -> None:
        if len(self._samples) < self._capacity:
            self._samples.append(sample)
        else:
            self._samples[self._next_index] = sample
        self._next_index = (self._next_index + 1) % self._capacity

    def add_episode(self, samples: Sequence[ReplaySample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> list[ReplaySample]:
        if batch_size <= 0:
            raise ValueError("ReplayBuffer.sample_batch: batch_size must be > 0")
        if batch_size > len(self._samples):
            raise ValueError(
                f"ReplayBuffer.sample_batch: requested {batch_size} but buffer only has "
                f"{len(self._samples)} samples"
            )
        indices = rng.choice(len(self._samples), size=batch_size, replace=False)
        return [self._samples[i] for i in indices]
