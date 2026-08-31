"""One-shot burst-training test: take a checkpoint and its paired replay
buffer, and train it for `--multiplier`x its own train_steps_per_iteration
(default 20x) against that SAME FROZEN buffer - no new self-play in between.
Answers "does grinding more gradient steps against a static buffer snapshot
help or hurt", isolated from any self-play-trajectory divergence: unlike
training more per iteration in the real loop, buffer contents never change
here, so any effect is purely about gradient-step count vs. this one
snapshot.

Saves a checkpoint at each x-multiple in --save-at (default 1, 5, 10, then
every 5 up to --multiplier) rather than only at the end, so the result is a
curve - the only way to see an overfitting signature (quality rising then
falling) rather than a single before/after comparison.

Loss composition, lr, batch_size, and network architecture are read from the
checkpoint's own extra["config"], matching what it was actually trained
with - see lr_range_test.py's identical convention. Read-only against the
input checkpoint and buffer; nothing is ever written back to them.

Usage:
    uv run python scripts/overtrain_buffer_test.py \
        --checkpoint scripts/output/pipeline/exp1/rl_checkpoints_selfplay_v7/checkpoint_000462.pt \
        --buffer scripts/output/pipeline/exp1/rl_checkpoints_selfplay_v7/buffer_latest.pkl \
        --multiplier 20 \
        --out-dir scripts/output/overtrain_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyjo.rl.bootstrap import load_replay_samples
from skyjo.rl.checkpoint import save_checkpoint
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.train import collate_batch, training_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument(
        "--multiplier", type=int, default=20,
        help="train for this many multiples of the checkpoint's own train_steps_per_iteration",
    )
    parser.add_argument("--out-dir", required=True, help="where to write checkpoint_<x>x.pt files")
    parser.add_argument(
        "--save-at", type=int, nargs="*", default=None,
        help="which x-multiples to checkpoint (default: 1, 5, 10, then every 5 up to --multiplier)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["extra"]["config"]
    print(f"checkpoint: iteration={payload['iteration']} total_train_steps={payload['total_train_steps']}")
    print(f"cfg: lr={cfg['lr']} batch_size={cfg['batch_size']} train_steps_per_iteration={cfg['train_steps_per_iteration']}")

    net = AlphaZeroNet(**cfg["network_kwargs"])
    net.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    steps_per_x = cfg["train_steps_per_iteration"]
    total_steps = steps_per_x * args.multiplier
    save_at_x = sorted(set(args.save_at)) if args.save_at else sorted(
        {1, 5, 10, *range(5, args.multiplier + 1, 5), args.multiplier}
    )
    save_at_x = [x for x in save_at_x if 1 <= x <= args.multiplier]
    save_at_steps = {x * steps_per_x: x for x in save_at_x}

    print(f"loading replay buffer from {args.buffer} ...")
    samples = load_replay_samples(args.buffer)
    print(f"buffer has {len(samples)} samples")
    print(f"running {total_steps} steps ({args.multiplier}x {steps_per_x}), checkpointing at x={save_at_x}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recent_losses: list[float] = []
    for step in range(1, total_steps + 1):
        indices = rng.choice(len(samples), size=cfg["batch_size"], replace=False)
        batch = collate_batch([samples[j] for j in indices])
        metrics = training_step(
            net, optimizer, batch,
            lambda_rank=cfg["lambda_rank"], lambda_points=cfg["lambda_points"], l2_coef=cfg["l2_coef"],
        )
        recent_losses.append(metrics["total_loss"])

        if step in save_at_steps:
            x = save_at_steps[step]
            out_path = out_dir / f"checkpoint_{x}x.pt"
            save_checkpoint(
                out_path, net, optimizer,
                iteration=payload["iteration"],
                total_train_steps=payload["total_train_steps"] + step,
                extra={
                    "config": cfg,
                    "overtrain_test": {
                        "source_checkpoint": str(args.checkpoint),
                        "source_buffer": str(args.buffer),
                        "multiplier_x": x,
                        "extra_steps": step,
                    },
                },
            )
            window = recent_losses[-steps_per_x:]
            avg_loss = sum(window) / len(window)
            print(
                f"step {step:>5} (x={x:>2}): saved {out_path.name} | "
                f"avg_loss(last {len(window)})={avg_loss:.4f} "
                f"grad_norm={metrics['grad_norm']:.3f} update_ratio={metrics['update_ratio']:.5f}"
            )

    print("done.")


if __name__ == "__main__":
    main()
