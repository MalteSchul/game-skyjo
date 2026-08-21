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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from skyjo.domain.engine import MAX_PLAYERS, MIN_PLAYERS, new_match
from skyjo.rl.checkpoint import save_checkpoint
from skyjo.rl.evaluator import (
    evaluate_vs_heuristic,
    make_batch_network_evaluator,
    make_network_evaluator,
)
from skyjo.rl.mcts import (
    DEFAULT_C_PUCT,
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DIRICHLET_EPSILON,
    BatchEvaluateFn,
    EvaluateFn,
)
from skyjo.rl.metrics import MetricsLogger
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.replay_buffer import ReplayBuffer
from skyjo.rl.selfplay import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_MAX_STEPS,
    DEFAULT_ROUND_MAX_STEPS,
    ReplaySample,
    generate_episode,
    generate_episodes_batch,
)
from skyjo.rl.train import DEFAULT_L2_COEF, DEFAULT_LAMBDA_RANK, collate_batch, training_step


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int
    games_per_iteration: int
    num_simulations: int = 200
    min_players: int = MIN_PLAYERS
    max_players: int = MIN_PLAYERS
    # One fixed tau for every decision in an episode - no annealing schedule.
    # Not 0.0 by default: `visit_distribution`'s tau<=1e-3 path breaks ties
    # deterministically (first-max, no randomization), and low simulation
    # counts or an early undertrained network both produce lots of exactly-
    # tied visit counts - `MctsBot` avoids the same trap with an explicit
    # random tie-break (see bots/mcts_bot.py), but self-play here samples
    # from `visit_distribution` instead, so a small residual randomness is
    # what keeps a game from looping on the same tied action pair forever.
    tau: float = 0.3
    c_puct: float = DEFAULT_C_PUCT
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON
    buffer_capacity: int = 200_000
    batch_size: int = 256
    train_steps_per_iteration: int = 200
    min_buffer_size: int | None = None  # None = require at least one full batch (batch_size)
    lr: float = 1e-3
    lambda_rank: float = DEFAULT_LAMBDA_RANK
    l2_coef: float = DEFAULT_L2_COEF
    network_kwargs: dict[str, Any] = field(default_factory=dict)
    max_steps_per_episode: int = DEFAULT_MAX_STEPS
    # Safety valves against a stuck round/game - see selfplay.py's
    # DEFAULT_ROUND_MAX_STEPS/DEFAULT_MAX_ROUNDS docstrings for why a round
    # can legitimately never end on its own. max_steps_per_episode above is
    # now expected to be near-unreachable rather than the routine failure
    # path it used to be - keep it comfortably above round_max_steps *
    # max_rounds (worst case) or every game fails before these valves get a
    # chance to work (see benchmark_selfplay_batching.py's --max-steps help
    # for a concrete example of this exact trap).
    round_max_steps: int = DEFAULT_ROUND_MAX_STEPS
    max_rounds: int = DEFAULT_MAX_ROUNDS
    workers: int = 0
    # 1 (default) = today's behavior: each job plays one game at a time,
    # evaluating its MCTS leaves one state per network call. >1 groups
    # `games_per_iteration` into jobs of this many games each, played
    # concurrently via `generate_episodes_batch` so every decision round's
    # leaf evaluations across the whole group become one batched network
    # call instead of one call per state - see `rl.mcts.run_mcts_batch`.
    # `workers` still shards jobs (now groups) across processes the same way.
    selfplay_batch_size: int = 1
    seed: int = 0
    checkpoint_dir: str | None = None
    checkpoint_every: int = 1
    # None (default) = no periodic eval. Otherwise, every `eval_every`
    # iterations, play `eval_games` games of this iteration's net vs
    # HeuristicBot (see `evaluator.evaluate_vs_heuristic`) and log
    # win_rate/avg_rank/avg_points under the "eval/" prefix - a cheap "are we
    # still improving" signal alongside the self_play/train scalars in the
    # same MetricsLogger run, not a rigorous benchmark (`eval_num_simulations`
    # defaults far below production self-play's `num_simulations` on purpose).
    # Always the same seed, so every eval call plays the identical game set -
    # comparable across iterations instead of noise from a different sample.
    eval_every: int | None = None
    eval_games: int = 20
    eval_num_simulations: int = 20

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
        if self.round_max_steps <= 0:
            raise ValueError("TrainingConfig: round_max_steps must be > 0")
        if self.max_rounds <= 0:
            raise ValueError("TrainingConfig: max_rounds must be > 0")
        if self.workers < 0:
            raise ValueError("TrainingConfig: workers must be >= 0")
        if self.selfplay_batch_size < 1:
            raise ValueError("TrainingConfig: selfplay_batch_size must be >= 1")
        if self.checkpoint_every <= 0:
            raise ValueError("TrainingConfig: checkpoint_every must be > 0")
        if self.eval_every is not None and self.eval_every <= 0:
            raise ValueError("TrainingConfig: eval_every must be > 0 if given")
        if self.eval_games <= 0:
            raise ValueError("TrainingConfig: eval_games must be > 0")
        if self.eval_num_simulations < 0:
            raise ValueError("TrainingConfig: eval_num_simulations must be >= 0")


@dataclass
class LoopState:
    iteration: int = 0
    total_train_steps: int = 0


@dataclass(frozen=True)
class _SelfPlayJob:
    seed: int
    player_count: int
    num_simulations: int
    tau: float
    c_puct: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    max_steps: int
    round_max_steps: int
    max_rounds: int
    failure_log_path: str


# Set by `_init_worker`, either in a subprocess (parallel self-play) or
# in-process (serial self-play, `config.workers <= 1`) - see module docstring.
_worker_evaluate: EvaluateFn | None = None
_worker_evaluate_batch: BatchEvaluateFn | None = None


def _init_worker(state_dict: dict[str, torch.Tensor], network_kwargs: dict[str, Any]) -> None:
    global _worker_evaluate, _worker_evaluate_batch
    net = AlphaZeroNet(**network_kwargs)
    net.load_state_dict(state_dict)
    # Building both is cheap (they just close over the same net) and keeps
    # worker init identical regardless of which self-play path is used.
    _worker_evaluate = make_network_evaluator(net)
    _worker_evaluate_batch = make_batch_network_evaluator(net)


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


def _play_one_game(job: _SelfPlayJob) -> tuple[list[ReplaySample], tuple[int, tuple[int, ...]] | None]:
    """Returns `([], None)` (never raises) if this one game fails - a still-unresolved,
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
    round_stats: list[tuple[int, tuple[int, ...]]] = []
    try:
        samples = generate_episode(
            initial_state,
            _worker_evaluate,
            num_simulations=job.num_simulations,
            tau_schedule=job.tau,
            c_puct=job.c_puct,
            dirichlet_alpha=job.dirichlet_alpha,
            dirichlet_epsilon=job.dirichlet_epsilon,
            rng=rng,
            max_steps=job.max_steps,
            round_max_steps=job.round_max_steps,
            max_rounds=job.max_rounds,
            round_stats_sink=round_stats,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately blind, see docstring above
        _log_failure(job, exc)
        return [], None
    return samples, round_stats[0] if round_stats else None


@dataclass(frozen=True)
class _SelfPlayBatchJob:
    seeds: tuple[int, ...]
    player_counts: tuple[int, ...]
    num_simulations: int
    tau: float
    c_puct: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    max_steps: int
    round_max_steps: int
    max_rounds: int
    failure_log_path: str


def _log_batch_failure(job: _SelfPlayBatchJob, exc: Exception) -> None:
    summary = f"_play_batch_of_games: seeds={list(job.seeds)} failed, skipping the whole group: {exc}"
    print(summary)
    with open(job.failure_log_path, "a", encoding="utf-8") as f:
        f.write(f"--- {datetime.now(UTC).isoformat()} {summary}\n")
        f.write(traceback.format_exc())
        f.write("\n")


def _play_batch_of_games(job: _SelfPlayBatchJob) -> tuple[list[ReplaySample], list[tuple[int, tuple[int, ...]]]]:
    """Plays `len(job.seeds)` games concurrently via `generate_episodes_batch`,
    batching every decision round's MCTS leaf evaluations across the whole
    group into one network call (see `rl.mcts.run_mcts_batch`).

    Unlike `_play_one_game`, a failure here takes down the *entire* group,
    not just one game - `generate_episodes_batch` doesn't isolate its games
    from each other's exceptions the way per-game `_play_one_game` does.
    Acceptable for measuring throughput; would need per-game isolation before
    this path is trusted for a long, unattended production run.
    """
    if _worker_evaluate_batch is None:
        raise RuntimeError("_play_batch_of_games: worker was not initialized via _init_worker")
    initial_states = [
        new_match(player_count=player_count, seed=seed)
        for seed, player_count in zip(job.seeds, job.player_counts, strict=True)
    ]
    rngs = [np.random.default_rng(seed) for seed in job.seeds]
    round_stats: list[tuple[int, tuple[int, ...]]] = []
    try:
        results = generate_episodes_batch(
            initial_states,
            _worker_evaluate_batch,
            num_simulations=job.num_simulations,
            tau_schedule=job.tau,
            c_puct=job.c_puct,
            dirichlet_alpha=job.dirichlet_alpha,
            dirichlet_epsilon=job.dirichlet_epsilon,
            rngs=rngs,
            max_steps=job.max_steps,
            round_max_steps=job.round_max_steps,
            max_rounds=job.max_rounds,
            round_stats_sink=round_stats,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately blind, see docstring above
        _log_batch_failure(job, exc)
        return [], []
    return [sample for game_samples in results for sample in game_samples], round_stats


def _build_batch_jobs(
    config: TrainingConfig, rng: np.random.Generator, failure_log_path: str
) -> list[_SelfPlayBatchJob]:
    player_counts = rng.integers(config.min_players, config.max_players + 1, size=config.games_per_iteration)
    seeds = rng.integers(0, 2**31 - 1, size=config.games_per_iteration)
    batch_size = config.selfplay_batch_size
    return [
        _SelfPlayBatchJob(
            seeds=tuple(int(s) for s in seeds[start : start + batch_size]),
            player_counts=tuple(int(p) for p in player_counts[start : start + batch_size]),
            num_simulations=config.num_simulations,
            tau=config.tau,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            max_steps=config.max_steps_per_episode,
            round_max_steps=config.round_max_steps,
            max_rounds=config.max_rounds,
            failure_log_path=failure_log_path,
        )
        for start in range(0, config.games_per_iteration, batch_size)
    ]


def run_self_play_iteration_batched(
    net: AlphaZeroNet, config: TrainingConfig, jobs: list[_SelfPlayBatchJob]
) -> tuple[list[ReplaySample], int, list[tuple[int, tuple[int, ...]]]]:
    """Batched sibling of `run_self_play_iteration`: each job plays a whole
    group of games concurrently (`_play_batch_of_games`) instead of one game
    at a time. `workers` still shards jobs (now groups, not single games)
    across processes exactly as before.

    Returns (samples, failed_game_count, round_stats) - see
    `run_self_play_iteration`'s docstring for what `round_stats` is.
    """
    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}

    if config.workers <= 1:
        _init_worker(state_dict, config.network_kwargs)
        group_results = [_play_batch_of_games(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers, initializer=_init_worker, initargs=(state_dict, config.network_kwargs)
        ) as pool:
            group_results = list(pool.map(_play_batch_of_games, jobs))

    samples: list[ReplaySample] = []
    failed_games = 0
    round_stats: list[tuple[int, tuple[int, ...]]] = []
    for job, (group_samples, group_round_stats) in zip(jobs, group_results, strict=True):
        if group_samples:
            samples.extend(group_samples)
            round_stats.extend(group_round_stats)
        else:
            failed_games += len(job.seeds)
    return samples, failed_games, round_stats


def _build_jobs(config: TrainingConfig, rng: np.random.Generator, failure_log_path: str) -> list[_SelfPlayJob]:
    player_counts = rng.integers(config.min_players, config.max_players + 1, size=config.games_per_iteration)
    seeds = rng.integers(0, 2**31 - 1, size=config.games_per_iteration)
    return [
        _SelfPlayJob(
            seed=int(seed),
            player_count=int(player_count),
            num_simulations=config.num_simulations,
            tau=config.tau,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            max_steps=config.max_steps_per_episode,
            round_max_steps=config.round_max_steps,
            max_rounds=config.max_rounds,
            failure_log_path=failure_log_path,
        )
        for seed, player_count in zip(seeds, player_counts, strict=True)
    ]


def run_self_play_iteration(
    net: AlphaZeroNet, config: TrainingConfig, jobs: list[_SelfPlayJob]
) -> tuple[list[ReplaySample], int, list[tuple[int, tuple[int, ...]]]]:
    """Returns (samples, failed_game_count, round_stats). A real completed
    Skyjo game always yields multiple decision points, so an empty per-job
    result unambiguously means `_play_one_game` caught a failure for that
    job, not a legitimately tiny game.

    `round_stats` has one `(round_count, final total_scores)` entry per
    successful game - `run_training_loop` reduces it into an
    `avg_points_per_round` health metric; kept as raw per-game data here so
    this function doesn't need to know how that reduction is done.
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
    round_stats: list[tuple[int, tuple[int, ...]]] = []
    for episode_samples, game_round_stats in episode_results:
        if episode_samples:
            samples.extend(episode_samples)
            if game_round_stats is not None:
                round_stats.append(game_round_stats)
        else:
            failed_games += 1
    return samples, failed_games, round_stats


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
        if config.selfplay_batch_size <= 1:
            jobs = _build_jobs(config, rng, failure_log_path)
            samples, failed_games, round_stats = run_self_play_iteration(net, config, jobs)
        else:
            batch_jobs = _build_batch_jobs(config, rng, failure_log_path)
            samples, failed_games, round_stats = run_self_play_iteration_batched(net, config, batch_jobs)
        for sample in samples:
            buffer.add(sample)
        successful_games = max(config.games_per_iteration - failed_games, 1)
        total_rounds = sum(round_count for round_count, _ in round_stats)
        avg_points_per_round = (
            sum(float(np.mean(points)) for _, points in round_stats) / total_rounds if total_rounds > 0 else 0.0
        )
        metrics.log(
            step,
            {
                "self_play/samples_generated": len(samples),
                "self_play/avg_moves_per_game": len(samples) / successful_games,
                "self_play/avg_points_per_round": avg_points_per_round,
                "self_play/failed_games": failed_games,
                "self_play/seconds": time.monotonic() - self_play_start,
                "buffer/size": len(buffer),
            },
        )

        train_start = time.monotonic()
        step_metrics: list[dict[str, float]] = []
        if len(buffer) >= min_buffer_size:
            for _ in range(config.train_steps_per_iteration):
                batch_samples, batch_encodings = buffer.sample_batch_with_encodings(config.batch_size, rng)
                batch = collate_batch(batch_samples, encodings=batch_encodings)
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

        if config.eval_every is not None and state.iteration % config.eval_every == 0:
            eval_start = time.monotonic()
            eval_result = evaluate_vs_heuristic(
                net,
                config.eval_games,
                num_simulations=config.eval_num_simulations,
                c_puct=config.c_puct,
                max_steps=config.max_steps_per_episode,
                round_max_steps=config.round_max_steps,
                max_rounds=config.max_rounds,
                seed=config.seed,
            )
            metrics.log(
                step,
                {
                    "eval/win_rate_vs_heuristic": eval_result.win_rate,
                    "eval/avg_rank_vs_heuristic": eval_result.avg_rank,
                    "eval/avg_points_vs_heuristic": eval_result.avg_points,
                    "eval/seconds": time.monotonic() - eval_start,
                },
            )

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
