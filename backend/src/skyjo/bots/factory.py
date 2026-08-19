"""Maps an API-facing player type string to a live Bot instance.

Kept separate from the API layer (schemas.py) so the set of valid player
types is validated once, at the wire boundary, while this module only has to
trust that it's being handed one of those already-validated strings.
"""

from __future__ import annotations

from skyjo.bots.base import Bot
from skyjo.bots.mcts_bot import MctsBot, default_evaluator, default_num_simulations
from skyjo.bots.random_bot import RandomBot
from skyjo.bots.thinking_bot import ThinkingBot


def create_bot(player_type: str, seed: int | None = None) -> Bot | None:
    """Returns None for "human" - no bot controls that seat."""
    if player_type == "human":
        return None
    if player_type == "random_bot":
        return RandomBot(seed=seed)
    if player_type == "thinking_bot":
        return ThinkingBot(seed=seed)
    if player_type == "mcts_bot":
        return MctsBot(evaluate=default_evaluator(), num_simulations=default_num_simulations(), seed=seed)
    raise ValueError(f"create_bot: unknown player_type {player_type!r}")
