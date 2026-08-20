"""Generates a warm-start `ReplaySample` dataset from bot-vs-bot games (e.g.
`HeuristicBot`) instead of MCTS self-play - see `scripts/bootstrap_heuristic.py`.
No search or network evaluation is needed, so games are cheap and reliably
finish, giving real win/loss outcomes (and an imitation-learned policy
target) to seed training on before switching to `skyjo.rl.loop`'s real,
MCTS-driven self-play loop.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from skyjo.bots.heuristic_bot import HeuristicBot
from skyjo.domain.engine import MAX_PLAYERS, MIN_PLAYERS, new_match
from skyjo.rl.selfplay import DEFAULT_MAX_STEPS, ReplaySample, generate_bot_episode


@dataclass(frozen=True)
class _BotGameJob:
    game_seed: int
    player_count: int
    max_steps: int


def _play_one_bot_game(job: _BotGameJob) -> list[ReplaySample]:
    bot_seed_rng = np.random.default_rng(job.game_seed)
    bots = [HeuristicBot(seed=int(s)) for s in bot_seed_rng.integers(0, 2**31 - 1, size=job.player_count)]
    state = new_match(player_count=job.player_count, seed=job.game_seed)
    try:
        return generate_bot_episode(state, [b.choose_action for b in bots], max_steps=job.max_steps)
    except RuntimeError as exc:
        print(f"_play_one_bot_game: seed={job.game_seed} failed, skipping: {exc}")
        return []


def generate_heuristic_dataset(
    num_games: int,
    *,
    min_players: int = MIN_PLAYERS,
    max_players: int = MIN_PLAYERS,
    max_steps: int = DEFAULT_MAX_STEPS,
    workers: int = 0,
    seed: int = 0,
) -> tuple[list[ReplaySample], int]:
    """Returns (samples, failed_game_count) from `num_games` HeuristicBot-vs-
    HeuristicBot games (each seat gets its own seed, for tie-break variety).

    `workers <= 1` runs serially in-process - no subprocess/pickling, which
    is what keeps this fast and deterministic to unit-test; `workers > 1`
    spawns a plain `ProcessPoolExecutor` (no shared state needed across
    workers, unlike MCTS self-play's network broadcast in `rl.loop`).
    """
    if num_games <= 0:
        raise ValueError("generate_heuristic_dataset: num_games must be > 0")
    if not (MIN_PLAYERS <= min_players <= max_players <= MAX_PLAYERS):
        raise ValueError(
            "generate_heuristic_dataset: require MIN_PLAYERS <= min_players <= max_players <= MAX_PLAYERS"
        )

    rng = np.random.default_rng(seed)
    player_counts = rng.integers(min_players, max_players + 1, size=num_games)
    game_seeds = rng.integers(0, 2**31 - 1, size=num_games)
    jobs = [
        _BotGameJob(game_seed=int(game_seed), player_count=int(player_count), max_steps=max_steps)
        for game_seed, player_count in zip(game_seeds, player_counts, strict=True)
    ]

    if workers <= 1:
        results = [_play_one_bot_game(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_play_one_bot_game, jobs))

    samples: list[ReplaySample] = []
    failed_games = 0
    for episode_samples in results:
        if episode_samples:
            samples.extend(episode_samples)
        else:
            failed_games += 1
    return samples, failed_games
