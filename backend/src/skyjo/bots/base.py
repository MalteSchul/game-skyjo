"""Bot interface: anything that can play a turn from a Turn view.

A Bot only ever sees a Turn, never GameState - the same redacted information
a human player has. That keeps a bot's decision from accidentally depending
on information it shouldn't have, and lets any Bot slot into the match store
or a future Gym-style training loop unmodified.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from skyjo.domain.engine import Action
from skyjo.domain.observation import Turn

# Called with a fraction in [0, 1] while a slow bot (e.g. MCTS) is still
# deciding, so the API can surface live progress to a polling client.
ProgressReporter = Callable[[float], None]


class Bot(Protocol):
    def choose_action(self, turn: Turn, *, report_progress: ProgressReporter | None = None) -> Action: ...
