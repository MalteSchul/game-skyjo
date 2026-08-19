"""A bot with RandomBot's strategy, except it takes real wall-clock time to
"decide" and reports its progress along the way.

Exists so the API's autoplay "thinking" status and progress polling
(api.matches.AUTOPLAY_GRACE_SECONDS, the frontend's THINKING_POLL_INTERVAL_MS
loop) has something slow enough to actually exercise it end-to-end, visibly,
in a browser - RandomBot always resolves well inside the grace period, so the
"thinking" UI is otherwise untestable outside of unit tests that inject a
bot directly.
"""

from __future__ import annotations

import random
import time

from skyjo.bots.base import ProgressReporter
from skyjo.domain.engine import Action
from skyjo.domain.observation import Turn

DEFAULT_THINK_SECONDS = 2.0
_PROGRESS_STEPS = 20


class ThinkingBot:
    def __init__(self, seed: int | None = None, think_seconds: float = DEFAULT_THINK_SECONDS) -> None:
        if seed is not None and not isinstance(seed, int):
            raise TypeError("ThinkingBot: seed must be an int if provided")
        if think_seconds < 0:
            raise ValueError("ThinkingBot: think_seconds must be >= 0")
        self._random = random.Random(seed)
        self._think_seconds = think_seconds

    def choose_action(self, turn: Turn, *, report_progress: ProgressReporter | None = None) -> Action:
        action = self._random.choice(turn.legal_actions)
        step_seconds = self._think_seconds / _PROGRESS_STEPS
        for step in range(1, _PROGRESS_STEPS + 1):
            time.sleep(step_seconds)
            if report_progress is not None:
                report_progress(step / _PROGRESS_STEPS)
        return action
