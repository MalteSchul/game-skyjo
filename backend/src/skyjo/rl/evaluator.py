"""Bridges `AlphaZeroNet` to MCTS's `EvaluateFn` signature: one `GameState`
in, `(priors over legal actions, utility vector)` out.
"""

from __future__ import annotations

import numpy as np
import torch

from skyjo.domain.engine import Action, GameState
from skyjo.rl.action_space import ACTION_SPACE_SIZE, index_to_action
from skyjo.rl.encoding import encode_state
from skyjo.rl.mcts import EvaluateFn
from skyjo.rl.network import AlphaZeroNet


def make_network_evaluator(net: AlphaZeroNet, device: str | torch.device = "cpu") -> EvaluateFn:
    net.to(device)
    net.eval()

    def evaluate(state: GameState) -> tuple[dict[Action, float], np.ndarray]:
        encoding = encode_state(state)
        with torch.no_grad():
            features = torch.from_numpy(encoding.features).unsqueeze(0).to(device)
            mask = torch.from_numpy(encoding.legal_action_mask).unsqueeze(0).to(device)
            active_count = torch.tensor([encoding.active_count], dtype=torch.long, device=device)
            policy_probs, _rank_probs, utility = net(features, mask, active_count)

        policy_probs = policy_probs[0].cpu().numpy()
        priors = {
            index_to_action(i): float(policy_probs[i])
            for i in range(ACTION_SPACE_SIZE)
            if encoding.legal_action_mask[i]
        }
        value = utility[0, : encoding.active_count].cpu().numpy()
        return priors, value

    return evaluate
