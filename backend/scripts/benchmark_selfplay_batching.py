"""Benchmarks self-play wall-clock time across (workers, selfplay_batch_size)
scenarios, using the same job-building/iteration-running code the real
training loop calls (`skyjo.rl.loop`) - not a synthetic proxy.

Every scenario plays the *same* games (same seeds/player-counts, derived from
a fresh `np.random.default_rng(seed)` per scenario) so wall-clock differences
reflect the self-play mechanism, not random variation in game length.

Usage:
  uv run python scripts/benchmark_selfplay_batching.py \
      --games 8 --num-simulations 25 --workers 0,4 --batch-sizes 1,4,8
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from skyjo.rl.loop import (
    TrainingConfig,
    _build_batch_jobs,
    _build_jobs,
    run_self_play_iteration,
    run_self_play_iteration_batched,
)
from skyjo.rl.network import AlphaZeroNet


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=8, help="games_per_iteration for every scenario")
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--player-count", type=int, default=2)
    parser.add_argument("--trunk-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--workers", default="0,4", help="comma-separated worker counts to try")
    parser.add_argument("--batch-sizes", default="1,4,8", help="comma-separated selfplay_batch_size values to try")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="cap per game (default DEFAULT_MAX_STEPS=5000 is far too loose for a throughput "
        "benchmark - a rare tie-locked game at low num_simulations should fail fast, not burn "
        "thousands of seconds before this script reports anything)",
    )
    parser.add_argument(
        "--out",
        default="scripts/output/benchmark_selfplay_batching.csv",
        help="results are appended here as each scenario finishes (flushed immediately), so the "
        "file has every completed scenario's row even if the run is interrupted or a later "
        "scenario hangs",
    )
    return parser.parse_args()


def _run_scenario(net: AlphaZeroNet, config: TrainingConfig, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if config.selfplay_batch_size <= 1:
        jobs = _build_jobs(config, rng, failure_log_path="benchmark_failures.log")
        samples, failed = run_self_play_iteration(net, config, jobs)
    else:
        jobs = _build_batch_jobs(config, rng, failure_log_path="benchmark_failures.log")
        samples, failed = run_self_play_iteration_batched(net, config, jobs)
    elapsed = time.perf_counter() - t0
    return {
        "seconds": elapsed,
        "samples": len(samples),
        "failed_games": failed,
        "samples_per_sec": len(samples) / elapsed if elapsed > 0 else float("nan"),
        "games_per_sec": config.games_per_iteration / elapsed if elapsed > 0 else float("nan"),
    }


def main() -> None:
    args = _parse_args()
    worker_values = [int(w) for w in args.workers.split(",")]
    batch_values = [int(b) for b in args.batch_sizes.split(",")]
    network_kwargs = {"trunk_dim": args.trunk_dim, "num_residual_blocks": args.residual_blocks}
    # Pinned so results are reproducible run-to-run - an unseeded random init can land on a
    # network whose priors happen to be unusually tie-prone, which (at low num_simulations)
    # triggers self-play's known tie-lock failure mode far more than a typical init would,
    # swamping the throughput comparison with failed-game noise.
    torch.manual_seed(args.seed)
    net = AlphaZeroNet(**network_kwargs)

    print(
        f"games={args.games} num_simulations={args.num_simulations} player_count={args.player_count} "
        f"max_steps={args.max_steps} network={network_kwargs}\nwriting results to {args.out} as each "
        f"scenario finishes\n"
    )
    fieldnames = ["workers", "batch_size", "seconds", "games_per_sec", "samples_per_sec", "samples", "failed_games"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"{'workers':>7} {'batch_size':>10} {'seconds':>9} {'games/s':>9} {'samples/s':>10} {'samples':>8} {'failed':>7}"
    print(header)
    print("-" * len(header))

    results = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        for workers in worker_values:
            for batch_size in batch_values:
                config = TrainingConfig(
                    iterations=1,
                    games_per_iteration=args.games,
                    num_simulations=args.num_simulations,
                    min_players=args.player_count,
                    max_players=args.player_count,
                    network_kwargs=network_kwargs,
                    workers=workers,
                    selfplay_batch_size=batch_size,
                    seed=args.seed,
                    max_steps_per_episode=args.max_steps,
                )
                result = _run_scenario(net, config, seed=args.seed)
                row = {"workers": workers, "batch_size": batch_size, **result}
                results.append(row)
                print(
                    f"{workers:>7} {batch_size:>10} {result['seconds']:>9.2f} "
                    f"{result['games_per_sec']:>9.3f} {result['samples_per_sec']:>10.1f} "
                    f"{result['samples']:>8} {result['failed_games']:>7}"
                )
                # Written and flushed immediately (not batched at the end) so the file holds
                # every completed scenario's row even if the run is killed or a later scenario
                # takes too long - the whole point of streaming rather than collect-then-print.
                writer.writerow(row)
                f.flush()

    baseline = next((r for r in results if r["workers"] == 0 and r["batch_size"] == 1), None)
    if baseline is not None:
        print(f"\nspeedup vs. workers=0/batch_size=1 baseline ({baseline['seconds']:.2f}s):")
        for r in results:
            speedup = baseline["seconds"] / r["seconds"] if r["seconds"] > 0 else float("nan")
            print(f"  workers={r['workers']:<3} batch_size={r['batch_size']:<3} -> {speedup:.2f}x")


if __name__ == "__main__":
    main()
