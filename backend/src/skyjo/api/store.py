"""In-memory match storage.

Good enough for a single dev/API process; a restart loses all matches. Swap for a
real store (Redis, DB) if the API needs to survive restarts or run multi-process.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from uuid import uuid4

from skyjo.domain.engine import GameState


class MatchNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class MatchRecord:
    state: GameState
    # Display-only seat labels; not part of the rules engine's GameState.
    player_names: tuple[str, ...]


class MatchStore:
    def __init__(self) -> None:
        self._matches: dict[str, MatchRecord] = {}
        self._lock = threading.Lock()

    def create(self, state: GameState, player_names: tuple[str, ...]) -> str:
        match_id = uuid4().hex
        with self._lock:
            self._matches[match_id] = MatchRecord(state=state, player_names=player_names)
        return match_id

    def get(self, match_id: str) -> MatchRecord:
        with self._lock:
            record = self._matches.get(match_id)
        if record is None:
            raise MatchNotFoundError(match_id)
        return record

    def update(self, match_id: str, state: GameState) -> None:
        with self._lock:
            record = self._matches.get(match_id)
            if record is None:
                raise MatchNotFoundError(match_id)
            self._matches[match_id] = replace(record, state=state)
