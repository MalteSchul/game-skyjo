"""Generates a warm-start `ReplaySample` dataset from bot-vs-bot games (e.g.
`HeuristicBot`) instead of MCTS self-play - see `scripts/bootstrap_heuristic.py`.
No search or network evaluation is needed, so games are cheap and reliably
finish, giving real win/loss outcomes (and an imitation-learned policy
target) to seed training on before switching to `skyjo.rl.loop`'s real,
MCTS-driven self-play loop.

`save_replay_samples`/`load_replay_samples` cache that dataset on disk: since
`ReplaySample` holds the raw `GameState` rather than an encoded feature
vector (encoding happens later, in `collate_batch`), a cached dataset stays
valid even across changes to `skyjo.rl.encoding` - only regenerating it
(replaying `HeuristicBot` games) is expensive, not re-encoding it.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

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
    on_game_done: Callable[[list[ReplaySample], bool], None] | None = None,
) -> tuple[list[ReplaySample], int]:
    """Returns (samples, failed_game_count) from `num_games` HeuristicBot-vs-
    HeuristicBot games (each seat gets its own seed, for tie-break variety).

    `workers <= 1` runs serially in-process - no subprocess/pickling, which
    is what keeps this fast and deterministic to unit-test; `workers > 1`
    spawns a plain `ProcessPoolExecutor` (no shared state needed across
    workers, unlike MCTS self-play's network broadcast in `rl.loop`).

    `on_game_done`, if given, is called once per completed game (in
    completion order) with that game's samples (empty if the game failed)
    and whether it failed. Games are consumed one at a time rather than
    materialized as a whole batch first, so a caller can use this to report
    progress and checkpoint partial results to disk as they arrive: if this
    function is interrupted (e.g. `KeyboardInterrupt`) partway through, it
    never returns, but every already-completed game was already handed to
    `on_game_done` - that's the caller's route to keeping what was generated
    before the interruption instead of losing all of it.
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

    samples: list[ReplaySample] = []
    failed_games = 0

    def _consume(episode_samples: list[ReplaySample]) -> None:
        nonlocal failed_games
        if episode_samples:
            samples.extend(episode_samples)
        else:
            failed_games += 1
        if on_game_done is not None:
            on_game_done(episode_samples, not episode_samples)

    if workers <= 1:
        for job in jobs:
            _consume(_play_one_bot_game(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for episode_samples in pool.map(_play_one_bot_game, jobs):
                _consume(episode_samples)

    return samples, failed_games


def save_replay_samples(samples: Sequence[ReplaySample], path: str | Path) -> None:
    """Pickles `samples` to `path` via temp file + atomic rename (mirroring
    `checkpoint.save_checkpoint`), so a crash mid-write can't corrupt a
    dataset that took a real generation run to produce.
    """
    if not samples:
        raise ValueError("save_replay_samples: samples must be non-empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(list(samples), f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def load_replay_samples(path: str | Path) -> list[ReplaySample]:
    with Path(path).open("rb") as f:
        samples = pickle.load(f)
    if not samples:
        raise ValueError(f"load_replay_samples: {path} contains no samples")
    return samples


def subsample_replay_samples(
    samples: Sequence[ReplaySample], max_samples: int, rng: np.random.Generator
) -> list[ReplaySample]:
    """Randomly subsamples down to `max_samples`, or returns `samples`
    unchanged if it's already at or below that count - see
    `scripts/bootstrap_heuristic.py`'s `--max-samples`: a dataset generated
    (or loaded) for reuse across experiments can be far larger than any one
    bootstrap training run needs resident in memory at once, since a training
    run of `train_steps * batch_size` draws rarely needs the full set to be
    statistically well-covered.
    """
    if max_samples <= 0:
        raise ValueError("subsample_replay_samples: max_samples must be > 0")
    if len(samples) <= max_samples:
        return list(samples)
    indices = rng.choice(len(samples), size=max_samples, replace=False)
    return [samples[i] for i in indices]
