"""A bot with no strategy: picks uniformly among the legal actions.

Exists to exercise the Bot interface end-to-end (engine, Turn, a full match)
with something simple and reproducible-when-seeded, ahead of any bot that
actually plays Skyjo well.
"""

from __future__ import annotations

import random

from skyjo.bots.base import ProgressReporter
from skyjo.domain.engine import Action
from skyjo.domain.observation import Turn


class RandomBot:
    def __init__(self, seed: int | None = None) -> None:
        if seed is not None and not isinstance(seed, int):
            raise TypeError("RandomBot: seed must be an int if provided")
        self._random = random.Random(seed)

    def choose_action(self, turn: Turn, *, report_progress: ProgressReporter | None = None) -> Action:
        return self._random.choice(turn.legal_actions)
