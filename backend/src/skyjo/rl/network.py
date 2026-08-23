"""Shared-trunk network: masked policy head + NxN rank-distribution head.

Both heads always operate on the full N_MAX_PLAYERS=8 shape; "inactive"
players (index >= N_act) are handled by masking rather than by slicing to a
variably-shaped tensor, so a single batch can freely mix samples from
matches with different player counts. Masked positions are fed a finite
sentinel (dtype min) rather than -inf before softmax, since softmax over a
row that is *entirely* -inf (a fully inactive rank row) would otherwise
divide 0/0 into NaN; finite-min keeps every row's softmax well-defined and
still drives masked probabilities to (numerically) zero.
"""

from __future__ import annotations

import torch
from torch import nn

from skyjo.rl.action_space import ACTION_SPACE_SIZE
from skyjo.rl.encoding import INPUT_DIM, N_MAX_PLAYERS

DEFAULT_TRUNK_DIM = 256
DEFAULT_RESIDUAL_BLOCKS = 4


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = torch.relu(self.norm1(self.fc1(x)))
        h = self.norm2(self.fc2(h))
        return torch.relu(residual + h)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim, restricted to positions where mask is True.

    Assumes every row has at least one True entry (always holds for a legal
    action mask - `legal_actions` is never empty at a decision node).
    """
    sentinel = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask, logits, torch.full_like(logits, sentinel))
    return torch.softmax(masked_logits, dim=-1)


def active_rank_mask(active_count: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(B, N_MAX, N_MAX) bool mask, True at (player, rank) pairs with both < active_count[b]."""
    idx = torch.arange(N_MAX_PLAYERS, device=device)
    active = idx.unsqueeze(0) < active_count.unsqueeze(1)  # (B, N_MAX)
    return active.unsqueeze(2) & active.unsqueeze(1)  # (B, N_MAX, N_MAX)


def rank_utility_weights(active_count: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(B, N_MAX) payoff-by-rank vector w[r] = 1 - 2r/(N_act-1) for r < N_act, else 0."""
    idx = torch.arange(N_MAX_PLAYERS, device=device, dtype=torch.float32)
    denom = (active_count.float() - 1.0).clamp(min=1.0).unsqueeze(1)  # N_act >= 2 always
    w = 1.0 - 2.0 * idx.unsqueeze(0) / denom
    active = idx.unsqueeze(0) < active_count.float().unsqueeze(1)
    return w * active.float()


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        trunk_dim: int = DEFAULT_TRUNK_DIM,
        num_residual_blocks: int = DEFAULT_RESIDUAL_BLOCKS,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, trunk_dim)
        self.input_norm = nn.LayerNorm(trunk_dim)
        self.blocks = nn.ModuleList([ResidualBlock(trunk_dim) for _ in range(num_residual_blocks)])
        self.policy_head = nn.Linear(trunk_dim, ACTION_SPACE_SIZE)
        self.rank_head = nn.Linear(trunk_dim, N_MAX_PLAYERS * N_MAX_PLAYERS)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_norm(self.input_proj(x)))
        for block in self.blocks:
            h = block(h)
        return h

    def forward(
        self,
        features: torch.Tensor,
        legal_action_mask: torch.Tensor,
        active_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (policy_probs (B, |A|), rank_probs (B, 8, 8), utility (B, 8)).

        `rank_probs[b, i, r]` is P(the player at canonical slot i finishes at
        rank r | state b) - `i` indexes canonical (rotated) player order, not
        absolute seat id; see `encoding.rotation_perm`. Zero for any i or
        r >= active_count[b]. `utility[b, i]` is likewise canonical-slot-
        indexed and zero for i >= active_count[b].
        """
        h = self.trunk(features)

        policy_logits = self.policy_head(h)
        policy_probs = _masked_softmax(policy_logits, legal_action_mask)

        rank_logits = self.rank_head(h).view(-1, N_MAX_PLAYERS, N_MAX_PLAYERS)
        mask = active_rank_mask(active_count, features.device)
        rank_probs = _masked_softmax(rank_logits, mask)
        row_active = torch.arange(N_MAX_PLAYERS, device=features.device).unsqueeze(0) < active_count.unsqueeze(1)
        rank_probs = rank_probs * row_active.unsqueeze(2).float()

        w = rank_utility_weights(active_count, features.device)  # (B, N_MAX)
        utility = torch.einsum("bir,br->bi", rank_probs, w)

        return policy_probs, rank_probs, utility
