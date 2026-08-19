"""Live + durable metrics for a long-running self-play/train loop.

Every call to `log` writes one JSON line to `metrics.jsonl` under `log_dir`
(durable, greppable, needs no extra dependency) and, since `tensorboard` is a
dev dependency, also feeds a `SummaryWriter` so a run in progress can be
watched live:

    uv run tensorboard --logdir scripts/output/runs/<name>

The tensorboard import is still soft (falls back to JSONL-only with a one-time
warning) so a missing/broken tensorboard install degrades the run's own
metrics rather than crashing it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Self

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - exercised only without the dev dependency group
    SummaryWriter = None  # type: ignore[assignment, misc]


class MetricsLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._log_dir / "metrics.jsonl"
        self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")  # noqa: SIM115 - kept open for the logger's lifetime
        self._writer = SummaryWriter(log_dir=str(self._log_dir)) if SummaryWriter is not None else None
        if self._writer is None:
            print(
                f"MetricsLogger: tensorboard not installed - writing {self._jsonl_path} only "
                "(uv sync --group dev should already provide it)"
            )

    @property
    def tensorboard_available(self) -> bool:
        return self._writer is not None

    def log(self, step: int, metrics: dict[str, float], *, prefix: str = "") -> None:
        if not metrics:
            raise ValueError("MetricsLogger.log: metrics must be non-empty")
        record: dict[str, Any] = {"step": step, "time": time.time()}
        record.update({f"{prefix}{k}": v for k, v in metrics.items()})
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(f"{prefix}{k}", v, global_step=step)
            self._writer.flush()

    def close(self) -> None:
        self._jsonl_file.close()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
