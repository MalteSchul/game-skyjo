"""Trains the mcts_bot's AlphaZeroNet via self-play. The outer loop
(self-play -> replay buffer -> train steps -> checkpoint) lives in
`skyjo.rl.loop.run_training_loop` and is unit-tested there with tiny configs;
this script is just an argparse front-end plus checkpoint resume/save wiring
for a real, long-running run.

Usage:
  uv run python scripts/train_mcts.py --iterations 200 --games-per-iteration 20 \
      --num-simulations 200 --workers 4 --log-dir scripts/output/runs/run1 \
      --checkpoint-dir scripts/output/checkpoints/run1

Resume a stopped/crashed run from its latest checkpoint:
  uv run python scripts/train_mcts.py --resume scripts/output/checkpoints/run1/latest.pt \
      --log-dir scripts/output/runs/run1 --checkpoint-dir scripts/output/checkpoints/run1 \
      [... same other flags as the original run ...]

Watch live metrics while a run is in progress:
  uv run tensorboard --logdir scripts/output/runs/run1
  (or tail -f scripts/output/runs/run1/metrics.jsonl - written even without tensorboard)

Point a live mcts_bot at the result (see skyjo.bots.mcts_bot):
  SKYJO_MCTS_CHECKPOINT_PATH=scripts/output/checkpoints/run1/latest.pt uv run uvicorn skyjo.api:app
"""

from __future__ import annotations

import argparse

import torch

from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.loop import LoopState, TrainingConfig, run_training_loop
from skyjo.rl.metrics import MetricsLogger
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import DEFAULT_MAX_STEPS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iteration", type=int, default=20)
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=2)
    parser.add_argument("--tau-moves", type=int, default=15, help="decisions per episode kept at tau=1 before annealing to greedy")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps-per-iteration", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--l2-coef", type=float, default=1e-4)
    parser.add_argument("--trunk-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="cap on decision points per self-play game before it's abandoned as failed. "
        "Lower this to fail a stuck/looping game faster rather than burn wall-clock on it "
        "before the resilience skip kicks in.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 or 1 = self-play runs serially in-process; >1 spawns a process pool of this size",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="scripts/output/checkpoints/default")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="iterations between checkpoint writes")
    parser.add_argument("--resume", help="path to a checkpoint (e.g. .../latest.pt) to resume from")
    parser.add_argument("--log-dir", default="scripts/output/runs/default")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    network_kwargs = {"trunk_dim": args.trunk_dim, "num_residual_blocks": args.residual_blocks}

    config = TrainingConfig(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        num_simulations=args.num_simulations,
        min_players=args.min_players,
        max_players=args.max_players,
        tau_moves=args.tau_moves,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        max_steps_per_episode=args.max_steps_per_episode,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        train_steps_per_iteration=args.train_steps_per_iteration,
        lr=args.lr,
        lambda_rank=args.lambda_rank,
        l2_coef=args.l2_coef,
        network_kwargs=network_kwargs,
        workers=args.workers,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
    )

    net = AlphaZeroNet(**network_kwargs)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    start_state = LoopState()

    if args.resume:
        loaded = load_checkpoint(args.resume, net, optimizer)
        start_state = LoopState(iteration=loaded.iteration, total_train_steps=loaded.total_train_steps)
        print(f"resumed from {args.resume} at iteration {loaded.iteration} ({loaded.total_train_steps} train steps so far)")

    with MetricsLogger(args.log_dir) as metrics:
        print(
            f"training for {config.iterations - start_state.iteration} more iteration(s) "
            f"(starting at {start_state.iteration}); tensorboard={'on' if metrics.tensorboard_available else 'off'}"
        )
        _, final_state = run_training_loop(config, metrics, net=net, optimizer=optimizer, start_state=start_state)
        print(f"finished at iteration {final_state.iteration} ({final_state.total_train_steps} total train steps)")


if __name__ == "__main__":
    main()
