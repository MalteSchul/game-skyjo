"""Runs the full heuristic-bootstrap -> self-play pipeline as one command:

  1. generate a HeuristicBot-vs-HeuristicBot dataset and save it to disk
  2. train a bootstrap checkpoint on that dataset (supervised, no search)
  3. sanity-check the bootstrap checkpoint against HeuristicBot - a hard gate:
     if it doesn't clear --min-win-rate, the pipeline stops before spending
     the much more expensive self-play compute on what's likely a broken
     upstream result (self-play won't fix a bootstrap that never learned to
     beat a heuristic at all)
  4. kick off the self-play + learn loop, resumed from the bootstrap
     checkpoint, with periodic heuristic-eval logged into the same
     tensorboard run as the training/self-play scalars

Stages 1/2/4 shell out to the script that already owns that stage
(bootstrap_heuristic.py / train_mcts.py) rather than reimplementing their
logic - this is just consistent path-threading between them, so progress
printing, checkpoint-on-interrupt, and --resume all keep working exactly as
they do standalone, defined in exactly one place. Stage 3 calls
`evaluator.evaluate_vs_heuristic` directly instead of shelling out to
tournament.py: the gate needs a structured win_rate number to check against
--min-win-rate, not a printed table to parse.

"Watch it in tensorboard" is deliberately not automated - it's an
interactive step, not something to launch as a subprocess.

Usage:
  uv run python scripts/run_bootstrap_pipeline.py --run-name exp1 --target-samples 500000 \
      --bootstrap-train-steps 5000 --iterations 200 --games-per-iteration 32 --eval-every 5

Every stage's output goes to a predictable path under --output-root/--run-name,
so rerunning the same --run-name skips dataset generation and bootstrap
training if their output already exists (--force to redo them anyway), and
resumes the self-play loop from wherever it left off rather than restarting
from the bootstrap checkpoint every time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from skyjo.rl.bootstrap import DEFAULT_BOOTSTRAP_SAMPLES
from skyjo.rl.checkpoint import load_checkpoint
from skyjo.rl.evaluator import evaluate_vs_heuristic
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import DEFAULT_MAX_ROUNDS, DEFAULT_MAX_STEPS, DEFAULT_ROUND_MAX_STEPS

_STAGES = ["dataset", "bootstrap", "sanity", "selfplay"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="scripts/output/pipeline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trunk-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--max-steps-per-episode", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--stop-after", choices=_STAGES, default="selfplay")
    parser.add_argument("--force", action="store_true", help="redo the dataset/bootstrap stages even if their output already exists")

    dataset = parser.add_argument_group("stage 1: dataset")
    dataset.add_argument("--target-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    dataset.add_argument("--dataset-workers", type=int, default=0, help="parallelism barely helps at this game size - see bootstrap_heuristic.py's benchmark notes")
    dataset.add_argument(
        "--dataset-checkpoint-interval-seconds",
        type=float,
        default=120.0,
        help="bootstrap_heuristic.py's own default (30s) rewrites the *entire* growing dataset to disk "
        "every interval - fine for a small run, but at a large --target-samples that file gets large "
        "enough (approaching a GB) that repeated full rewrites can end up dominating wall time over just "
        "letting generation run and saving occasionally. Raised here to bound that to at most a couple of "
        "rewrites for a ~500k-sample run rather than one every 30s; raise further (or pass a very large "
        "value) if you don't need interrupt-safety at all for a run you're actively supervising.",
    )

    bootstrap = parser.add_argument_group("stage 2: bootstrap training")
    bootstrap.add_argument("--bootstrap-train-steps", type=int, default=5000)
    bootstrap.add_argument("--bootstrap-batch-size", type=int, default=256)
    bootstrap.add_argument("--bootstrap-lr", type=float, default=1e-3)
    bootstrap.add_argument(
        "--bootstrap-max-samples",
        type=int,
        default=500_000,
        help="passed through to bootstrap_heuristic.py's --max-samples - see its help text",
    )

    sanity = parser.add_argument_group("stage 3: sanity check")
    sanity.add_argument("--sanity-games", type=int, default=40)
    sanity.add_argument("--sanity-num-simulations", type=int, default=20)
    sanity.add_argument("--min-win-rate", type=float, default=0.5, help="the bootstrap checkpoint must clear this win rate vs HeuristicBot or the pipeline stops here")
    sanity.add_argument(
        "--sanity-workers",
        type=int,
        default=0,
        help="passed through to evaluate_vs_heuristic's workers - shards games across a process pool; "
        "requires --sanity-batch-size > 1 to have any effect (see evaluate_vs_heuristic's docstring)",
    )
    sanity.add_argument(
        "--sanity-batch-size",
        type=int,
        default=1,
        help="passed through to evaluate_vs_heuristic's eval_batch_size - batches this many games' MCTS "
        "leaf evaluations into one network call, at the cost of no tree reuse across a game's own turns",
    )

    selfplay = parser.add_argument_group("stage 4: self-play loop")
    selfplay.add_argument("--iterations", type=int, default=200)
    selfplay.add_argument("--games-per-iteration", type=int, default=32)
    selfplay.add_argument("--num-simulations", type=int, default=200)
    selfplay.add_argument("--tau", type=float, default=0.3, help="fixed tau for every self-play decision, no annealing")
    selfplay.add_argument(
        "--round-max-steps",
        type=int,
        default=DEFAULT_ROUND_MAX_STEPS,
        help="force-close a round that runs this long without closing naturally - see selfplay.py",
    )
    selfplay.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help="end the game after this many rounds close without reaching target_score - see selfplay.py",
    )
    selfplay.add_argument("--selfplay-workers", type=int, default=None, help="passed through to train_mcts.py's --workers; default (unset) resolves from CPU count there")
    selfplay.add_argument(
        "--selfplay-batch-size",
        type=int,
        default=None,
        help="passed through to train_mcts.py's --selfplay-batch-size; default (unset) resolves from "
        "--selfplay-workers/--games-per-iteration there - see its module docstring's 'Self-play throughput' section",
    )
    selfplay.add_argument("--checkpoint-every", type=int, default=1)
    selfplay.add_argument("--eval-every", type=int, default=5)
    selfplay.add_argument("--eval-games", type=int, default=20)
    selfplay.add_argument("--eval-num-simulations", type=int, default=20)

    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def _run_dataset_stage(args: argparse.Namespace, dataset_path: Path) -> None:
    if dataset_path.exists() and not args.force:
        print(f"[1/4] dataset already exists at {dataset_path}, skipping (--force to redo)")
        return
    _run(
        [
            sys.executable,
            "scripts/bootstrap_heuristic.py",
            "--target-samples",
            str(args.target_samples),
            "--workers",
            str(args.dataset_workers),
            "--max-steps-per-episode",
            str(args.max_steps_per_episode),
            "--seed",
            str(args.seed),
            "--samples-out",
            str(dataset_path),
            "--checkpoint-interval-seconds",
            str(args.dataset_checkpoint_interval_seconds),
            "--skip-training",
        ]
    )


def _run_bootstrap_stage(args: argparse.Namespace, dataset_path: Path, checkpoint_path: Path, log_dir: Path) -> None:
    if checkpoint_path.exists() and not args.force:
        print(f"[2/4] bootstrap checkpoint already exists at {checkpoint_path}, skipping (--force to redo)")
        return
    _run(
        [
            sys.executable,
            "scripts/bootstrap_heuristic.py",
            "--load-samples",
            str(dataset_path),
            "--train-steps",
            str(args.bootstrap_train_steps),
            "--batch-size",
            str(args.bootstrap_batch_size),
            "--lr",
            str(args.bootstrap_lr),
            "--trunk-dim",
            str(args.trunk_dim),
            "--residual-blocks",
            str(args.residual_blocks),
            "--seed",
            str(args.seed),
            "--checkpoint-out",
            str(checkpoint_path),
            "--log-dir",
            str(log_dir),
            "--max-samples",
            str(args.bootstrap_max_samples),
        ]
    )


def _run_sanity_stage(args: argparse.Namespace, checkpoint_path: Path) -> float:
    network_kwargs = {"trunk_dim": args.trunk_dim, "num_residual_blocks": args.residual_blocks}
    net = AlphaZeroNet(**network_kwargs)
    load_checkpoint(checkpoint_path, net)
    result = evaluate_vs_heuristic(
        net,
        args.sanity_games,
        num_simulations=args.sanity_num_simulations,
        max_steps=args.max_steps_per_episode,
        round_max_steps=args.round_max_steps,
        max_rounds=args.max_rounds,
        seed=args.seed,
        workers=args.sanity_workers,
        eval_batch_size=args.sanity_batch_size,
        network_kwargs=network_kwargs,
    )
    print(
        f"[3/4] sanity check vs HeuristicBot: {result.games_played} games, "
        f"win_rate={result.win_rate:.1%} avg_rank={result.avg_rank:.3f} avg_points={result.avg_points:.1f}"
    )
    return result.win_rate


def _run_selfplay_stage(args: argparse.Namespace, checkpoint_path: Path, checkpoint_dir: Path, log_dir: Path) -> None:
    latest = checkpoint_dir / "latest.pt"
    resume_from = latest if latest.exists() else checkpoint_path
    print(f"[4/4] self-play loop resuming from {resume_from}")

    cmd = [
        sys.executable,
        "scripts/train_mcts.py",
        "--iterations",
        str(args.iterations),
        "--games-per-iteration",
        str(args.games_per_iteration),
        "--num-simulations",
        str(args.num_simulations),
        "--tau",
        str(args.tau),
        "--trunk-dim",
        str(args.trunk_dim),
        "--residual-blocks",
        str(args.residual_blocks),
        "--max-steps-per-episode",
        str(args.max_steps_per_episode),
        "--round-max-steps",
        str(args.round_max_steps),
        "--max-rounds",
        str(args.max_rounds),
        "--seed",
        str(args.seed),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--log-dir",
        str(log_dir),
        "--resume",
        str(resume_from),
        "--eval-every",
        str(args.eval_every),
        "--eval-games",
        str(args.eval_games),
        "--eval-num-simulations",
        str(args.eval_num_simulations),
    ]
    if args.selfplay_workers is not None:
        cmd += ["--workers", str(args.selfplay_workers)]
    if args.selfplay_batch_size is not None:
        cmd += ["--selfplay-batch-size", str(args.selfplay_batch_size)]
    _run(cmd)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.output_root) / args.run_name
    dataset_path = run_dir / "dataset.pkl"
    bootstrap_checkpoint_path = run_dir / "bootstrap_checkpoint.pt"
    bootstrap_log_dir = run_dir / "bootstrap_runs"
    selfplay_checkpoint_dir = run_dir / "selfplay_checkpoints"
    selfplay_log_dir = run_dir / "selfplay_runs"

    print(f"[1/4] generating/loading heuristic dataset at {dataset_path}")
    _run_dataset_stage(args, dataset_path)
    if args.stop_after == "dataset":
        return

    print(f"[2/4] bootstrap training -> {bootstrap_checkpoint_path}")
    _run_bootstrap_stage(args, dataset_path, bootstrap_checkpoint_path, bootstrap_log_dir)
    if args.stop_after == "bootstrap":
        return

    win_rate = _run_sanity_stage(args, bootstrap_checkpoint_path)
    if win_rate < args.min_win_rate:
        raise SystemExit(
            f"sanity check failed: win_rate={win_rate:.1%} < --min-win-rate={args.min_win_rate:.1%} - "
            "stopping before the self-play stage. Something upstream (encoding, bootstrap hyperparameters, "
            "dataset quality) likely needs fixing before self-play compute is worth spending."
        )
    if args.stop_after == "sanity":
        return

    _run_selfplay_stage(args, bootstrap_checkpoint_path, selfplay_checkpoint_dir, selfplay_log_dir)
    print(f"\ndone. watch progress with: uv run tensorboard --logdir {selfplay_log_dir}")


if __name__ == "__main__":
    main()
