"""Save/load a training run's full resumable state: net weights, optimizer
state, and loop progress (iteration / total_train_steps), as one file.

Writes go through a temp file + atomic rename so a crash or kill mid-`torch.save`
(a real risk on a long-running training box) can never leave `latest.pt`
truncated/corrupt for the next resume to load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from skyjo.rl.network import AlphaZeroNet


def save_checkpoint(
    path: str | Path,
    net: AlphaZeroNet,
    optimizer: torch.optim.Optimizer | None,
    *,
    iteration: int,
    total_train_steps: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "iteration": iteration,
        "total_train_steps": total_train_steps,
        "extra": extra or {},
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


@dataclass(frozen=True)
class LoadedCheckpoint:
    iteration: int
    total_train_steps: int
    extra: dict[str, Any]


def load_checkpoint(
    path: str | Path,
    net: AlphaZeroNet,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    net.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return LoadedCheckpoint(
        iteration=payload["iteration"],
        total_train_steps=payload["total_train_steps"],
        extra=payload.get("extra", {}),
    )
