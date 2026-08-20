"""The outer AlphaZero loop: self-play -> replay buffer -> train steps ->
checkpoint -> repeat. `scripts/train_mcts.py` is a thin CLI front-end over
`run_training_loop`; the loop itself lives here so it's directly unit-testable
with tiny configs (no subprocess/CLI needed).

No arena/gating yet: every iteration's freshly-trained network becomes the
next iteration's self-play network unconditionally, rather than the more
traditional "only promote if it beats the previous best in an arena match"
gate. Good enough to bootstrap a first checkpoint; add gating later if
training turns out to regress rather than monotonically improve.

Self-play parallelism (`config.workers > 1`) uses a fresh `ProcessPoolExecutor`
per iteration rather than one long-lived pool: the network changes every
iteration, and re-spawning is far cheaper than the self-play search itself,
so it's not worth the complexity of pushing weight updates into long-lived
workers. `workers <= 1` runs self-play serially in-process instead - no
subprocess/pickling involved, which is what keeps the unit tests below fast
and deterministic.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from skyjo.domain.engine import MAX_PLAYERS, MIN_PLAYERS, new_match
from skyjo.rl.checkpoint import save_checkpoint
from skyjo.rl.evaluator import make_network_evaluator
from skyjo.rl.mcts import (
    DEFAULT_C_PUCT,
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DIRICHLET_EPSILON,
    EvaluateFn,
)
from skyjo.rl.metrics import MetricsLogger
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.replay_buffer import ReplayBuffer
from skyjo.rl.selfplay import DEFAULT_MAX_STEPS, ReplaySample, generate_episode
from skyjo.rl.train import DEFAULT_L2_COEF, DEFAULT_LAMBDA_RANK, collate_batch, training_step


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int
    games_per_iteration: int
    num_simulations: int = 200
    min_players: int = MIN_PLAYERS
    max_players: int = MIN_PLAYERS
    tau_moves: int = 15
    tau_high: float = 1.0
    # Deliberately not 0.0: `visit_distribution`'s tau<=1e-3 path breaks ties
    # deterministically (first-max, no randomization), and low simulation
    # counts or an early undertrained network both produce lots of exactly-
    # tied visit counts - `MctsBot` avoids the same trap with an explicit
    # random tie-break (see bots/mcts_bot.py), but self-play here samples
    # from `visit_distribution` instead, so a small residual randomness is
    # what keeps a game from looping on the same tied action pair forever.
    tau_low: float = 0.05
    c_puct: float = DEFAULT_C_PUCT
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON
    buffer_capacity: int = 200_000
    batch_size: int = 256
    train_steps_per_iteration: int = 100
    min_buffer_size: int | None = None  # None = require at least one full batch (batch_size)
    lr: float = 1e-3
    lambda_rank: float = DEFAULT_LAMBDA_RANK
    l2_coef: float = DEFAULT_L2_COEF
    network_kwargs: dict[str, Any] = field(default_factory=dict)
    max_steps_per_episode: int = DEFAULT_MAX_STEPS
    workers: int = 0
    seed: int = 0
    checkpoint_dir: str | None = None
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("TrainingConfig: iterations must be > 0")
        if self.games_per_iteration <= 0:
            raise ValueError("TrainingConfig: games_per_iteration must be > 0")
        if not (MIN_PLAYERS <= self.min_players <= self.max_players <= MAX_PLAYERS):
            raise ValueError(
                "TrainingConfig: require MIN_PLAYERS <= min_players <= max_players <= MAX_PLAYERS "
                f"(got min_players={self.min_players}, max_players={self.max_players})"
            )
        if self.num_simulations < 0:
            raise ValueError("TrainingConfig: num_simulations must be >= 0")
        if self.buffer_capacity <= 0:
            raise ValueError("TrainingConfig: buffer_capacity must be > 0")
        if self.batch_size <= 0:
            raise ValueError("TrainingConfig: batch_size must be > 0")
        if self.train_steps_per_iteration < 0:
            raise ValueError("TrainingConfig: train_steps_per_iteration must be >= 0")
        if self.max_steps_per_episode <= 0:
            raise ValueError("TrainingConfig: max_steps_per_episode must be > 0")
        if self.workers < 0:
            raise ValueError("TrainingConfig: workers must be >= 0")
        if self.checkpoint_every <= 0:
            raise ValueError("TrainingConfig: checkpoint_every must be > 0")


@dataclass
class LoopState:
    iteration: int = 0
    total_train_steps: int = 0


def make_tau_schedule(tau_moves: int, high: float = 1.0, low: float = 0.0) -> Callable[[int], float]:
    """Explore (tau=`high`) for the first `tau_moves` decisions of an episode,
    then switch to near-greedy (tau=`low`) - the standard AlphaZero anneal so
    early-game training data stays diverse while later moves target the
    search's actual best line.
    """
    if tau_moves < 0:
        raise ValueError("make_tau_schedule: tau_moves must be >= 0")

    def schedule(step: int) -> float:
        return high if step < tau_moves else low

    return schedule


@dataclass(frozen=True)
class _SelfPlayJob:
    seed: int
    player_count: int
    num_simulations: int
    tau_moves: int
    tau_high: float
    tau_low: float
    c_puct: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    max_steps: int
    failure_log_path: str


# Set by `_init_worker`, either in a subprocess (parallel self-play) or
# in-process (serial self-play, `config.workers <= 1`) - see module docstring.
_worker_evaluate: EvaluateFn | None = None


def _init_worker(state_dict: dict[str, torch.Tensor], network_kwargs: dict[str, Any]) -> None:
    global _worker_evaluate
    net = AlphaZeroNet(**network_kwargs)
    net.load_state_dict(state_dict)
    _worker_evaluate = make_network_evaluator(net)


def _log_failure(job: _SelfPlayJob, exc: Exception) -> None:
    """Appends a full traceback to `job.failure_log_path` and prints a
    one-line summary. The file (not just stdout) is what makes a crash
    diagnosable after the fact - a `ProcessPoolExecutor` worker's stdout has
    nowhere durable to go, so a failure that isn't logged here leaves no
    trace once its console scrolls away.

    Concurrent workers may append to this file at nearly the same time; each
    append is a single `write` call (open/write/close, not a kept-open
    handle) which is atomic in practice for a traceback-sized write, and an
    occasional interleaving would only garble one diagnostic entry, never
    the training run itself.
    """
    summary = f"_play_one_game: seed={job.seed} player_count={job.player_count} failed, skipping: {exc}"
    print(summary)
    with open(job.failure_log_path, "a", encoding="utf-8") as f:
        f.write(f"--- {datetime.now(UTC).isoformat()} {summary}\n")
        f.write(traceback.format_exc())
        f.write("\n")


def _play_one_game(job: _SelfPlayJob) -> list[ReplaySample]:
    """Returns `[]` (never raises) if this one game fails - a still-unresolved,
    rare `hidden_info` bookkeeping bug (see rl/hidden_info.py), a game that
    runs past `max_steps`, or any other bug in the self-play path can
    otherwise take down an entire iteration's worth of sibling games (or, in
    the `workers>1` pool, the whole training process) over one unlucky
    sample. A long unattended run should degrade to "one game's data is
    missing" rather than crash outright; `run_training_loop` surfaces how
    often this happens via the `self_play/failed_games` metric, and
    `_log_failure` writes the full traceback to `job.failure_log_path` so a
    root-cause fix later has something to confirm against instead of just an
    aggregate count.

    Catches `Exception` broadly rather than the two types actually expected
    (`AssertionError` from `hidden_info`'s conservation check, `RuntimeError`
    from hitting `max_steps`) - a narrower catch only protects against
    failure modes already anticipated, and the whole point of this guard is
    surviving the ones that aren't. `BaseException` subclasses like
    `KeyboardInterrupt` are deliberately left uncaught.
    """
    if _worker_evaluate is None:
        raise RuntimeError("_play_one_game: worker was not initialized via _init_worker")
    rng = np.random.default_rng(job.seed)
    initial_state = new_match(player_count=job.player_count, seed=job.seed)
    tau_schedule = make_tau_schedule(job.tau_moves, job.tau_high, job.tau_low)
    try:
        return generate_episode(
            initial_state,
            _worker_evaluate,
            num_simulations=job.num_simulations,
            tau_schedule=tau_schedule,
            c_puct=job.c_puct,
            dirichlet_alpha=job.dirichlet_alpha,
            dirichlet_epsilon=job.dirichlet_epsilon,
            rng=rng,
            max_steps=job.max_steps,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately blind, see docstring above
        _log_failure(job, exc)
        return []


def _build_jobs(config: TrainingConfig, rng: np.random.Generator, failure_log_path: str) -> list[_SelfPlayJob]:
    player_counts = rng.integers(config.min_players, config.max_players + 1, size=config.games_per_iteration)
    seeds = rng.integers(0, 2**31 - 1, size=config.games_per_iteration)
    return [
        _SelfPlayJob(
            seed=int(seed),
            player_count=int(player_count),
            num_simulations=config.num_simulations,
            tau_moves=config.tau_moves,
            tau_high=config.tau_high,
            tau_low=config.tau_low,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            max_steps=config.max_steps_per_episode,
            failure_log_path=failure_log_path,
        )
        for seed, player_count in zip(seeds, player_counts, strict=True)
    ]


def run_self_play_iteration(
    net: AlphaZeroNet, config: TrainingConfig, jobs: list[_SelfPlayJob]
) -> tuple[list[ReplaySample], int]:
    """Returns (samples, failed_game_count). A real completed Skyjo game
    always yields multiple decision points, so an empty per-job result
    unambiguously means `_play_one_game` caught a failure for that job, not
    a legitimately tiny game.
    """
    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}

    if config.workers <= 1:
        _init_worker(state_dict, config.network_kwargs)
        episode_results = [_play_one_game(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers, initializer=_init_worker, initargs=(state_dict, config.network_kwargs)
        ) as pool:
            episode_results = list(pool.map(_play_one_game, jobs))

    samples: list[ReplaySample] = []
    failed_games = 0
    for episode_samples in episode_results:
        if episode_samples:
            samples.extend(episode_samples)
        else:
            failed_games += 1
    return samples, failed_games


def run_training_loop(
    config: TrainingConfig,
    metrics: MetricsLogger,
    *,
    net: AlphaZeroNet | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    start_state: LoopState | None = None,
) -> tuple[AlphaZeroNet, LoopState]:
    net = net if net is not None else AlphaZeroNet(**config.network_kwargs)
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(net.parameters(), lr=config.lr)
    state = start_state if start_state is not None else LoopState()
    buffer = ReplayBuffer(config.buffer_capacity)
    # Not restored from a checkpoint on resume, so a resumed run's self-play
    # samples diverge from an equivalent from-scratch run - acceptable, this
    # is exploration noise, not something training correctness depends on.
    rng = np.random.default_rng(config.seed + state.iteration)
    min_buffer_size = config.min_buffer_size if config.min_buffer_size is not None else config.batch_size
    failure_log_path = str(metrics.log_dir / "self_play_failures.log")

    for _ in range(state.iteration, config.iterations):
        step = state.iteration

        self_play_start = time.monotonic()
        jobs = _build_jobs(config, rng, failure_log_path)
        samples, failed_games = run_self_play_iteration(net, config, jobs)
        for sample in samples:
            buffer.add(sample)
        successful_games = max(config.games_per_iteration - failed_games, 1)
        metrics.log(
            step,
            {
                "self_play/samples_generated": len(samples),
                "self_play/avg_moves_per_game": len(samples) / successful_games,
                "self_play/failed_games": failed_games,
                "self_play/seconds": time.monotonic() - self_play_start,
                "buffer/size": len(buffer),
            },
        )

        train_start = time.monotonic()
        step_metrics: list[dict[str, float]] = []
        if len(buffer) >= min_buffer_size:
            for _ in range(config.train_steps_per_iteration):
                batch_samples = buffer.sample_batch(config.batch_size, rng)
                batch = collate_batch(batch_samples)
                step_metrics.append(
                    training_step(net, optimizer, batch, lambda_rank=config.lambda_rank, l2_coef=config.l2_coef)
                )
                state.total_train_steps += 1

        train_log = {"train/steps_this_iteration": float(len(step_metrics)), "train/seconds": time.monotonic() - train_start}
        if step_metrics:
            train_log.update(
                {f"train/{k}": float(np.mean([m[k] for m in step_metrics])) for k in step_metrics[0]}
            )
        metrics.log(step, train_log)

        state.iteration += 1

        if config.checkpoint_dir is not None and state.iteration % config.checkpoint_every == 0:
            extra = {"config": asdict(config)}
            save_checkpoint(
                Path(config.checkpoint_dir) / f"checkpoint_{state.iteration:06d}.pt",
                net,
                optimizer,
                iteration=state.iteration,
                total_train_steps=state.total_train_steps,
                extra=extra,
            )
            save_checkpoint(
                Path(config.checkpoint_dir) / "latest.pt",
                net,
                optimizer,
                iteration=state.iteration,
                total_train_steps=state.total_train_steps,
                extra=extra,
            )

    return net, state
