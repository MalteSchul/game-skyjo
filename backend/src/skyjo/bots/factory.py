"""Maps an API-facing player type string to a live Bot instance.

Kept separate from the API layer (schemas.py) so the set of valid player
types is validated once, at the wire boundary, while this module only has to
trust that it's being handed one of those already-validated strings.
"""

from __future__ import annotations

from skyjo.bots.base import Bot
from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.bots.mcts_bot import (
    MctsBot,
    default_evaluator,
    default_num_simulations,
    evaluator_for_model,
)
from skyjo.bots.random_bot import RandomBot
from skyjo.bots.thinking_bot import ThinkingBot


def create_bot(
    player_type: str,
    seed: int | None = None,
    mcts_model: str | None = None,
    num_simulations: int | None = None,
) -> Bot | None:
    """Returns None for "human" - no bot controls that seat. `mcts_model` (a
    name from `mcts_bot.list_available_models`) and `num_simulations` only
    apply to "mcts_bot" - both are ignored for every other player_type."""
    if player_type == "human":
        return None
    if player_type == "random_bot":
        return RandomBot(seed=seed)
    if player_type == "thinking_bot":
        return ThinkingBot(seed=seed)
    if player_type == "heuristic_bot":
        return HeuristicBot(seed=seed)
    if player_type == "mcts_bot":
        evaluate = evaluator_for_model(mcts_model) if mcts_model is not None else default_evaluator()
        sims = num_simulations if num_simulations is not None else default_num_simulations()
        return MctsBot(evaluate=evaluate, num_simulations=sims, seed=seed)
    raise ValueError(f"create_bot: unknown player_type {player_type!r}")
