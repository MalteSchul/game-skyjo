"""Bootstraps AlphaZeroNet from HeuristicBot-vs-HeuristicBot games instead of
(much slower, and initially directionless) MCTS self-play. The dataset
generation logic lives in `skyjo.rl.bootstrap.generate_heuristic_dataset`
(unit-tested there); this script is just an argparse front-end that then
runs plain supervised training steps on the result and saves a checkpoint.

HeuristicBot needs no search or network evaluation, so games are cheap and
reliably finish - a fast, reliable source of real win/loss outcomes (and an
imitation-learned policy target) to seed the network on before switching to
real self-play, without first having to tune MCTS-specific knobs
(num_simulations, max_steps, tau) against an as-yet-untrained network.

Usage:
  uv run python scripts/bootstrap_heuristic.py --target-samples 500000 --workers 6 \
      --train-steps 3000 --checkpoint-out scripts/output/checkpoints/bootstrap/latest.pt \
      --log-dir scripts/output/runs/bootstrap --samples-out scripts/output/datasets/bootstrap.pkl

Then hand off to real MCTS self-play:
  uv run python scripts/train_mcts.py --resume scripts/output/checkpoints/bootstrap/latest.pt \
      --checkpoint-dir scripts/output/checkpoints/run1 --log-dir scripts/output/runs/run1 [...]

Re-training on a previously generated dataset (e.g. to try a different
--train-steps/--lr without re-playing thousands of HeuristicBot games) skips
generation entirely:
  uv run python scripts/bootstrap_heuristic.py --load-samples scripts/output/datasets/bootstrap.pkl \
      --train-steps 5000 --checkpoint-out scripts/output/checkpoints/bootstrap2/latest.pt \
      --log-dir scripts/output/runs/bootstrap2

Generation prints a progress line every --progress-interval-seconds. If
--samples-out is set, it also checkpoints whatever samples exist so far to
that path every --checkpoint-interval-seconds, and once more immediately on
Ctrl+C - so interrupting a long `--target-samples 5000000` run still leaves a
usable (if smaller) dataset on disk instead of losing everything back to
nothing.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from skyjo.rl.bootstrap import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    generate_heuristic_dataset,
    load_replay_samples,
    save_replay_samples,
    subsample_replay_samples,
)
from skyjo.rl.checkpoint import save_checkpoint
from skyjo.rl.metrics import MetricsLogger
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import DEFAULT_MAX_STEPS, ReplaySample
from skyjo.rl.train import collate_batch, training_step


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=2)
    parser.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--l2-coef", type=float, default=1e-4)
    parser.add_argument("--trunk-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-out", default="scripts/output/checkpoints/bootstrap/latest.pt")
    parser.add_argument("--log-dir", default="scripts/output/runs/bootstrap")
    parser.add_argument(
        "--games-per-batch",
        type=int,
        default=None,
        help="games played per round before checking whether --target-samples has been reached "
        "(passed through to generate_heuristic_dataset's games_per_batch; default resolves from "
        "--workers there).",
    )
    parser.add_argument(
        "--load-samples",
        default=None,
        help="Path to a dataset previously written by --samples-out; if given, skips game "
        "generation entirely (--target-samples/--min-players/--max-players/"
        "--max-steps-per-episode/--workers/--games-per-batch/--seed are ignored).",
    )
    parser.add_argument(
        "--samples-out",
        default=None,
        help="Path to pickle the generated (or loaded) samples to, for reuse across training runs "
        "without replaying games. Also checkpointed periodically during generation (see "
        "--checkpoint-interval-seconds) so an interrupted run doesn't lose its progress.",
    )
    parser.add_argument("--progress-interval-seconds", type=float, default=2.0)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=30.0)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="stop right after generating/loading (and, if --samples-out is set, saving) the dataset - "
        "for when this script is only being used to produce a dataset, not a checkpoint.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500_000,
        help="cap on samples held in memory for training, applied by randomly subsampling right after "
        "generating/loading (--samples-out, if set, still saves the full un-subsampled set). A dataset "
        "worth caching for reuse across experiments (e.g. --target-samples 3000000) can be far bigger "
        "than any one bootstrap run needs resident at once - default sized for an 8GB machine, raise it "
        "if you have the RAM to spare.",
    )
    return parser.parse_args()


def _generate_with_progress(args: argparse.Namespace) -> tuple[list[ReplaySample], int, int, float]:
    """Runs `generate_heuristic_dataset`, printing progress and (if
    `--samples-out` is set) periodically checkpointing samples to disk as
    games complete. On Ctrl+C, saves once more immediately and exits instead
    of propagating the interrupt - `generate_heuristic_dataset` itself never
    returns on interrupt, so this is the only place partial progress is kept.
    """
    samples: list[ReplaySample] = []
    failed_games = 0
    games_done = 0
    start = time.monotonic()
    last_progress = start
    last_checkpoint = start

    def on_game_done(episode_samples: list[ReplaySample], failed: bool) -> None:
        nonlocal failed_games, games_done, last_progress, last_checkpoint
        games_done += 1
        samples.extend(episode_samples)
        if failed:
            failed_games += 1

        now = time.monotonic()
        if now - last_progress >= args.progress_interval_seconds or len(samples) >= args.target_samples:
            elapsed = now - start
            rate = games_done / elapsed if elapsed > 0 else 0.0
            print(
                f"[games={games_done}] samples={len(samples)}/{args.target_samples} failed={failed_games} "
                f"elapsed={elapsed:.1f}s ({rate:.1f} games/s)"
            )
            last_progress = now
        if args.samples_out is not None and now - last_checkpoint >= args.checkpoint_interval_seconds:
            save_replay_samples(samples, args.samples_out)
            print(f"checkpointed {len(samples)} samples to {args.samples_out}")
            last_checkpoint = now

    try:
        samples, failed_games, games_done = generate_heuristic_dataset(
            args.target_samples,
            min_players=args.min_players,
            max_players=args.max_players,
            max_steps=args.max_steps_per_episode,
            workers=args.workers,
            seed=args.seed,
            games_per_batch=args.games_per_batch,
            on_game_done=on_game_done,
        )
    except KeyboardInterrupt:
        print(f"\ninterrupted after {games_done} games - keeping {len(samples)} samples generated so far")
        if args.samples_out is not None:
            save_replay_samples(samples, args.samples_out)
            print(f"saved {len(samples)} samples to {args.samples_out}")
        raise SystemExit(0) from None

    return samples, failed_games, games_done, time.monotonic() - start


def main() -> None:
    args = _parse_args()
    network_kwargs = {"trunk_dim": args.trunk_dim, "num_residual_blocks": args.residual_blocks}

    if args.load_samples is not None:
        start = time.monotonic()
        samples = load_replay_samples(args.load_samples)
        print(f"loaded {len(samples)} samples from {args.load_samples} in {time.monotonic() - start:.1f}s")
    else:
        samples, failed_games, games_played, elapsed = _generate_with_progress(args)
        print(
            f"generated {len(samples)} samples from {games_played - failed_games}/{games_played} games "
            f"({failed_games} failed) in {elapsed:.1f}s"
        )

    if args.samples_out is not None:
        save_replay_samples(samples, args.samples_out)
        print(f"saved {len(samples)} samples to {args.samples_out}")

    if args.skip_training:
        return

    rng = np.random.default_rng(args.seed)
    if len(samples) > args.max_samples:
        before = len(samples)
        samples = subsample_replay_samples(samples, args.max_samples, rng)
        print(f"subsampled {before} -> {len(samples)} samples (--max-samples={args.max_samples})")

    if len(samples) < args.batch_size:
        raise SystemExit(f"only {len(samples)} samples generated, need at least --batch-size={args.batch_size}")

    net = AlphaZeroNet(**network_kwargs)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Sampling directly from `samples` rather than through a `ReplayBuffer`: that class's
    # `add()` eagerly encodes and caches every sample up front, which only pays off when a
    # sample gets redrawn many times over its buffer lifetime (the self-play loop's case).
    # Here `collate_batch` re-encodes on every draw anyway (no encodings passed in), so an
    # eager cache sized to the whole dataset would just be wasted memory - at 3M+ samples,
    # multiple GB of it.
    with MetricsLogger(args.log_dir) as metrics:
        for step in range(args.train_steps):
            indices = rng.choice(len(samples), size=args.batch_size, replace=False)
            batch = collate_batch([samples[i] for i in indices])
            step_metrics = training_step(net, optimizer, batch, lambda_rank=args.lambda_rank, l2_coef=args.l2_coef)
            if step % args.log_every == 0 or step == args.train_steps - 1:
                metrics.log(step, step_metrics, prefix="train/")

        save_checkpoint(
            args.checkpoint_out,
            net,
            optimizer,
            iteration=0,
            total_train_steps=args.train_steps,
            extra={"source": "heuristic_bootstrap", "target_samples": args.target_samples, "samples": len(samples)},
        )
    print(f"saved bootstrap checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
