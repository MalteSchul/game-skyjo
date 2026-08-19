"""Maps an API-facing player type string to a live Bot instance.

Kept separate from the API layer (schemas.py) so the set of valid player
types is validated once, at the wire boundary, while this module only has to
trust that it's being handed one of those already-validated strings.
"""

from __future__ import annotations

from skyjo.bots.base import Bot
from skyjo.bots.random_bot import RandomBot


def create_bot(player_type: str, seed: int | None = None) -> Bot | None:
    """Returns None for "human" - no bot controls that seat."""
    if player_type == "human":
        return None
    if player_type == "random_bot":
        return RandomBot(seed=seed)
    raise ValueError(f"create_bot: unknown player_type {player_type!r}")
