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
  uv run python scripts/bootstrap_heuristic.py --games 1000 --workers 6 \
      --train-steps 3000 --checkpoint-out scripts/output/checkpoints/bootstrap/latest.pt \
      --log-dir scripts/output/runs/bootstrap

Then hand off to real MCTS self-play:
  uv run python scripts/train_mcts.py --resume scripts/output/checkpoints/bootstrap/latest.pt \
      --checkpoint-dir scripts/output/checkpoints/run1 --log-dir scripts/output/runs/run1 [...]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from skyjo.rl.bootstrap import generate_heuristic_dataset
from skyjo.rl.checkpoint import save_checkpoint
from skyjo.rl.metrics import MetricsLogger
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.replay_buffer import ReplayBuffer
from skyjo.rl.selfplay import DEFAULT_MAX_STEPS
from skyjo.rl.train import collate_batch, training_step


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=1000)
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    network_kwargs = {"trunk_dim": args.trunk_dim, "num_residual_blocks": args.residual_blocks}

    start = time.monotonic()
    samples, failed_games = generate_heuristic_dataset(
        args.games,
        min_players=args.min_players,
        max_players=args.max_players,
        max_steps=args.max_steps_per_episode,
        workers=args.workers,
        seed=args.seed,
    )
    print(
        f"generated {len(samples)} samples from {args.games - failed_games}/{args.games} games "
        f"({failed_games} failed) in {time.monotonic() - start:.1f}s"
    )
    if len(samples) < args.batch_size:
        raise SystemExit(f"only {len(samples)} samples generated, need at least --batch-size={args.batch_size}")

    buffer = ReplayBuffer(capacity=len(samples))
    buffer.add_episode(samples)

    net = AlphaZeroNet(**network_kwargs)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    with MetricsLogger(args.log_dir) as metrics:
        for step in range(args.train_steps):
            batch = collate_batch(buffer.sample_batch(args.batch_size, rng))
            step_metrics = training_step(net, optimizer, batch, lambda_rank=args.lambda_rank, l2_coef=args.l2_coef)
            if step % args.log_every == 0 or step == args.train_steps - 1:
                metrics.log(step, step_metrics, prefix="train/")

        save_checkpoint(
            args.checkpoint_out,
            net,
            optimizer,
            iteration=0,
            total_train_steps=args.train_steps,
            extra={"source": "heuristic_bootstrap", "games": args.games, "samples": len(samples)},
        )
    print(f"saved bootstrap checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
