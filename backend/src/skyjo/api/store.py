"""In-memory match storage.

Good enough for a single dev/API process; a restart loses all matches. Swap for a
real store (Redis, DB) if the API needs to survive restarts or run multi-process.
"""

from __future__ import annotations

import threading
from uuid import uuid4

from skyjo.domain.engine import GameState


class MatchNotFoundError(Exception):
    pass


class MatchStore:
    def __init__(self) -> None:
        self._matches: dict[str, GameState] = {}
        self._lock = threading.Lock()

    def create(self, state: GameState) -> str:
        match_id = uuid4().hex
        with self._lock:
            self._matches[match_id] = state
        return match_id

    def get(self, match_id: str) -> GameState:
        with self._lock:
            state = self._matches.get(match_id)
        if state is None:
            raise MatchNotFoundError(match_id)
        return state

    def update(self, match_id: str, state: GameState) -> None:
        with self._lock:
            if match_id not in self._matches:
                raise MatchNotFoundError(match_id)
            self._matches[match_id] = state
