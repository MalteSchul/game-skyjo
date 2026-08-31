"""One-shot LR range test (Leslie Smith / fastai style): sweep the learning
rate log-uniformly over a few hundred training steps on a throwaway copy of
a checkpoint's network, training against the existing replay buffer, and
report where the loss stops descending and starts climbing.

Read-only against the checkpoint and buffer files - a fresh net/optimizer is
built in memory and nothing is ever written back to them. Loss composition
(lambda_rank/lambda_points/l2_coef) and network architecture are read from
the checkpoint's own saved `extra["config"]` so the sweep matches what that
checkpoint was actually trained with, not today's CLI defaults.

Usage:
    uv run python scripts/lr_range_test.py \
        --checkpoint scripts/output/pipeline/exp1/rl_checkpoints_selfplay_v7/checkpoint_000130.pt \
        --buffer scripts/output/pipeline/exp1/rl_checkpoints_selfplay_v7/buffer_latest.pkl
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyjo.rl.bootstrap import load_replay_samples
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.train import collate_batch, training_step

_SPARK_CHARS = ".-:=+*#%@"


def _sparkline(values: list[float]) -> str:
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    return "".join(_SPARK_CHARS[min(len(_SPARK_CHARS) - 1, int((v - lo) / span * len(_SPARK_CHARS)))] for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-max", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=None, help="default: the checkpoint's own batch_size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema-beta", type=float, default=0.98)
    parser.add_argument("--diverge-factor", type=float, default=4.0, help="stop early once smoothed loss exceeds this multiple of its running minimum")
    parser.add_argument("--out-csv", default="scripts/output/lr_range_test.csv")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["extra"]["config"]
    print(f"checkpoint: iteration={payload['iteration']} total_train_steps={payload['total_train_steps']}")
    print(f"training lr was: {cfg['lr']}")

    net = AlphaZeroNet(**cfg["network_kwargs"])
    net.load_state_dict(payload["model_state_dict"])

    batch_size = args.batch_size or cfg["batch_size"]
    lambda_rank = cfg["lambda_rank"]
    lambda_points = cfg["lambda_points"]
    l2_coef = cfg["l2_coef"]

    print(f"loading replay buffer from {args.buffer} ...")
    samples = load_replay_samples(args.buffer)
    print(f"buffer has {len(samples)} samples, sampling batch_size={batch_size}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_min)

    lrs = np.geomspace(args.lr_min, args.lr_max, args.steps)
    records: list[tuple[float, float, float]] = []  # (lr, raw_loss, smoothed_loss)
    ema = 0.0  # zero-started EMA, bias-corrected below (standard fastai/Leslie-Smith LR-finder smoothing)
    best_smoothed = float("inf")

    for i, lr in enumerate(lrs):
        for group in optimizer.param_groups:
            group["lr"] = float(lr)

        indices = rng.choice(len(samples), size=batch_size, replace=False)
        batch = collate_batch([samples[j] for j in indices])
        metrics = training_step(net, optimizer, batch, lambda_rank=lambda_rank, lambda_points=lambda_points, l2_coef=l2_coef)
        loss = metrics["total_loss"]

        ema = args.ema_beta * ema + (1 - args.ema_beta) * loss
        smoothed = ema / (1 - args.ema_beta ** (i + 1))
        records.append((float(lr), loss, smoothed))
        best_smoothed = min(best_smoothed, smoothed)

        if i > 10 and smoothed > args.diverge_factor * best_smoothed:
            print(f"step {i}: smoothed loss {smoothed:.4f} > {args.diverge_factor}x its running min {best_smoothed:.4f} - stopping early (diverged)")
            break

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lr", "loss", "smoothed_loss"])
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {out_path}")

    smoothed_series = [r[2] for r in records]
    lr_series = [r[0] for r in records]
    min_idx = int(np.argmin(smoothed_series))
    min_lr, min_loss = lr_series[min_idx], smoothed_series[min_idx]

    knee_idx = None
    for i in range(min_idx, len(smoothed_series)):
        if smoothed_series[i] > 1.1 * min_loss:
            knee_idx = i
            break
    knee_lr = lr_series[knee_idx] if knee_idx is not None else lr_series[-1]

    print()
    print(f"loss sparkline (lr {lr_series[0]:.1e} -> {lr_series[-1]:.1e}, {len(lr_series)} steps):")
    print(_sparkline(smoothed_series))
    print()
    print(f"minimum smoothed loss {min_loss:.4f} at lr={min_lr:.2e}")
    print(f"knee (loss rises >10% above minimum) at lr={knee_lr:.2e}")
    print(f"suggested ceiling: ~{knee_lr:.2e}  |  suggested safe operating LR: ~{knee_lr / 10:.2e}")
    print(f"current training lr {cfg['lr']:.2e} is {'BELOW' if cfg['lr'] < knee_lr / 10 else ('within 10x of the ceiling' if cfg['lr'] < knee_lr else 'AT/ABOVE the ceiling')}")


if __name__ == "__main__":
    main()
