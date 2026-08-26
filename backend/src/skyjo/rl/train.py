"""Batch collation, joint loss, and a training step over `ReplaySample`s.

Policy loss is cross-entropy against the soft MCTS visit distribution `pi`
(not a single argmax label - that's what makes it "categorical cross-entropy
between pi and predicted action probabilities" per the spec, rather than a
one-hot policy target). Rank loss is masked so rows/players beyond a
sample's own `N_act` contribute nothing, matching the rank head's own
active-row masking in `network.py`. Points loss (MSE against
`points_target`, see `TrainingBatch`) is masked the same way and weighted
far below rank loss by default (`DEFAULT_LAMBDA_POINTS`) - it's an
auxiliary signal for the value head's magnitude, not the primary target
MCTS backs up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from skyjo.rl.encoding import N_MAX_PLAYERS, StateEncoding, encode_state
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import ReplaySample

DEFAULT_LAMBDA_RANK = 1.0
DEFAULT_LAMBDA_POINTS = 0.5
DEFAULT_L2_COEF = 1e-4
_LOG_EPS = 1e-8


@dataclass(frozen=True)
class TrainingBatch:
    features: torch.Tensor  # (B, INPUT_DIM)
    legal_action_mask: torch.Tensor  # (B, ACTION_SPACE_SIZE) bool
    active_count: torch.Tensor  # (B,) long
    pi_target: torch.Tensor  # (B, ACTION_SPACE_SIZE)
    y: torch.Tensor  # (B, N_MAX_PLAYERS) long, canonical slot order (see encoding.rotation_perm); padding beyond n_act is masked out, not read
    points_target: torch.Tensor  # (B, N_MAX_PLAYERS) float, canonical slot order, same padding convention as y


def collate_batch(
    samples: Sequence[ReplaySample],
    device: str | torch.device = "cpu",
    *,
    encodings: Sequence[StateEncoding] | None = None,
) -> TrainingBatch:
    """`encodings`, if given, must be each sample's own `StateEncoding` in the
    same order (e.g. from `ReplayBuffer.sample_batch_with_encodings`) - skips
    re-running `encode_state` here for samples already encoded once when they
    were added to the buffer. Recomputed from `s.state` when omitted, so any
    caller with only raw `ReplaySample`s (a freshly loaded bootstrap dataset,
    the existing tests) keeps working unchanged.
    """
    if not samples:
        raise ValueError("collate_batch: samples must be non-empty")
    if encodings is None:
        encodings = [encode_state(s.state) for s in samples]
    elif len(encodings) != len(samples):
        raise ValueError(
            f"collate_batch: got {len(encodings)} encodings for {len(samples)} samples"
        )

    features = torch.from_numpy(np.stack([e.features for e in encodings])).to(device)
    legal_action_mask = torch.from_numpy(np.stack([e.legal_action_mask for e in encodings])).to(device)
    active_count = torch.tensor([s.n_act for s in samples], dtype=torch.long, device=device)
    pi_target = torch.from_numpy(np.stack([s.pi for s in samples])).to(device)

    y = torch.zeros(len(samples), N_MAX_PLAYERS, dtype=torch.long, device=device)
    points_target = torch.zeros(len(samples), N_MAX_PLAYERS, dtype=torch.float32, device=device)
    for i, (sample, encoding) in enumerate(zip(samples, encodings, strict=True)):
        n = sample.n_act
        y[i, :n] = torch.from_numpy(sample.y[encoding.perm[:n]])  # absolute order -> canonical slot order
        points_target[i, :n] = torch.from_numpy(sample.points_y[encoding.perm[:n]])

    return TrainingBatch(
        features=features,
        legal_action_mask=legal_action_mask,
        active_count=active_count,
        pi_target=pi_target,
        y=y,
        points_target=points_target,
    )


def compute_loss(
    net: AlphaZeroNet,
    batch: TrainingBatch,
    *,
    lambda_rank: float = DEFAULT_LAMBDA_RANK,
    lambda_points: float = DEFAULT_LAMBDA_POINTS,
    l2_coef: float = DEFAULT_L2_COEF,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_probs, rank_probs, _utility, points_pred = net(
        batch.features, batch.legal_action_mask, batch.active_count
    )

    policy_loss = -(batch.pi_target * torch.log(policy_probs + _LOG_EPS)).sum(dim=-1).mean()

    row_active = torch.arange(N_MAX_PLAYERS, device=batch.features.device).unsqueeze(
        0
    ) < batch.active_count.unsqueeze(1)
    target_probs = rank_probs.gather(-1, batch.y.unsqueeze(-1)).squeeze(-1)  # (B, N_MAX)
    row_cross_entropy = -torch.log(target_probs + _LOG_EPS) * row_active.float()
    rank_loss = (row_cross_entropy.sum(dim=-1) / batch.active_count.float()).mean()

    row_squared_error = (points_pred - batch.points_target) ** 2 * row_active.float()
    points_loss = (row_squared_error.sum(dim=-1) / batch.active_count.float()).mean()

    l2_term = sum((p**2).sum() for p in net.parameters())

    total_loss = policy_loss + lambda_rank * rank_loss + lambda_points * points_loss + l2_coef * l2_term

    metrics = {
        "policy_loss": float(policy_loss.detach()),
        "rank_loss": float(rank_loss.detach()),
        "points_loss": float(points_loss.detach()),
        "l2_term": float(l2_term.detach()),
        "total_loss": float(total_loss.detach()),
    }
    return total_loss, metrics


def training_step(
    net: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    *,
    lambda_rank: float = DEFAULT_LAMBDA_RANK,
    lambda_points: float = DEFAULT_LAMBDA_POINTS,
    l2_coef: float = DEFAULT_L2_COEF,
) -> dict[str, float]:
    net.train()
    optimizer.zero_grad()
    loss, metrics = compute_loss(net, batch, lambda_rank=lambda_rank, lambda_points=lambda_points, l2_coef=l2_coef)
    loss.backward()
    optimizer.step()
    return metrics
