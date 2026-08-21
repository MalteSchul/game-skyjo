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

Self-play throughput (--workers / --selfplay-batch-size):
  Batching MCTS leaf evaluations (--selfplay-batch-size > 1) amortizes the
  network's per-call dispatch overhead, which dominates self-play wall time at
  batch=1 - see rl.mcts.run_mcts_batch. Both default to None here and are
  resolved from the local machine (os.cpu_count()) and --games-per-iteration
  if not passed explicitly, rather than a fixed number baked into the script -
  the right (workers, batch_size) pair depends on your hardware and how many
  games you self-play per iteration, and a batch size much bigger than
  games_per_iteration/workers just leaves a worker idle. Prefer a few workers
  with a large batch each over many workers with a small one (a small batch
  loses more from being diluted than it gains from an extra process) - see
  the benchmark in scripts/benchmark_selfplay_batching.py if you want to
  profile your own hardware instead of trusting the default heuristic.
  Caveat: a batch_size > 1 group currently fails as a whole if any one game
  in it does (unlike the batch_size=1 path, which isolates failures per
  game) - fine for a monitored run, worth knowing for a long unattended one.
"""

from __future__ import annotations

import argparse
import os

import torch

from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.loop import LoopState, TrainingConfig, run_training_loop
from skyjo.rl.metrics import MetricsLogger
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import DEFAULT_MAX_ROUNDS, DEFAULT_MAX_STEPS, DEFAULT_ROUND_MAX_STEPS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--games-per-iteration", type=int, default=32)
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.3, help="fixed tau for every self-play decision - not 0.0 by default, see TrainingConfig.tau's docstring")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps-per-iteration", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--l2-coef", type=float, default=1e-4)
    parser.add_argument("--trunk-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="cap on decision points per self-play game before it's abandoned as failed. Keep "
        "comfortably above --round-max-steps * --max-rounds (worst case) or every game fails "
        "before those safety valves get a chance to work.",
    )
    parser.add_argument(
        "--round-max-steps",
        type=int,
        default=DEFAULT_ROUND_MAX_STEPS,
        help="a round running this long without closing naturally gets force-closed instead of "
        "letting the whole game fail - see selfplay.DEFAULT_ROUND_MAX_STEPS's docstring for why a "
        "round can legitimately never end on its own (an untrained/low-search policy has no "
        "pressure to reveal new information).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help="after this many rounds close (naturally or forced) without reaching target_score, the "
        "game just ends there on whatever total_scores currently stand.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="0 or 1 = self-play runs serially in-process; >1 spawns a process pool of this size. "
        "Default (unset): max(1, os.cpu_count() // 2) - see the module docstring.",
    )
    parser.add_argument(
        "--selfplay-batch-size",
        type=int,
        default=None,
        help="1 (unbatched, today's per-game behavior) or >1 to batch that many games' MCTS leaf "
        "evaluations together per network call - see rl.mcts.run_mcts_batch. Default (unset): "
        "max(1, games_per_iteration // workers) once --workers is resolved, so it's never a "
        "bigger batch than games_per_iteration/workers can actually fill. See the module "
        "docstring for the resilience caveat.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="scripts/output/checkpoints/default")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="iterations between checkpoint writes")
    parser.add_argument("--resume", help="path to a checkpoint (e.g. .../latest.pt) to resume from")
    parser.add_argument("--log-dir", default="scripts/output/runs/default")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="iterations between heuristic-eval checks (default: unset, no periodic eval). Logged under "
        "the 'eval/' prefix in the same run as train/self_play metrics - see rl.evaluator.evaluate_vs_heuristic.",
    )
    parser.add_argument("--eval-games", type=int, default=20, help="games per heuristic-eval check")
    parser.add_argument(
        "--eval-num-simulations",
        type=int,
        default=20,
        help="MCTS simulations per move during heuristic-eval - deliberately smaller than "
        "--num-simulations, a cheap diagnostic rather than a rigorous benchmark",
    )
    return parser.parse_args()


def _resolve_selfplay_concurrency(args: argparse.Namespace) -> tuple[int, int]:
    """Both --workers and --selfplay-batch-size default to None (not resolved
    at argparse time) so the batch size can be derived from *this run's* own
    --workers/--games-per-iteration rather than a number baked into the
    script - see the module docstring's "Self-play throughput" section.
    """
    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 4) // 2)
    if args.selfplay_batch_size is not None:
        selfplay_batch_size = args.selfplay_batch_size
    else:
        # workers=0/1 both mean "no process pool" (see --workers' help) - divide by the
        # effective parallelism (>= 1), not the literal possibly-zero worker count.
        selfplay_batch_size = max(1, args.games_per_iteration // max(workers, 1))
    return workers, selfplay_batch_size


def main() -> None:
    args = _parse_args()
    network_kwargs = {"trunk_dim": args.trunk_dim, "num_residual_blocks": args.residual_blocks}
    workers, selfplay_batch_size = _resolve_selfplay_concurrency(args)

    config = TrainingConfig(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        num_simulations=args.num_simulations,
        min_players=args.min_players,
        max_players=args.max_players,
        tau=args.tau,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        max_steps_per_episode=args.max_steps_per_episode,
        round_max_steps=args.round_max_steps,
        max_rounds=args.max_rounds,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        train_steps_per_iteration=args.train_steps_per_iteration,
        lr=args.lr,
        lambda_rank=args.lambda_rank,
        l2_coef=args.l2_coef,
        network_kwargs=network_kwargs,
        workers=workers,
        selfplay_batch_size=selfplay_batch_size,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        eval_num_simulations=args.eval_num_simulations,
    )
    print(f"self-play concurrency: workers={workers} selfplay_batch_size={selfplay_batch_size}")

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
