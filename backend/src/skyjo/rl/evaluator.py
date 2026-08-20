"""Bridges `AlphaZeroNet` to MCTS's `EvaluateFn` signature: one `GameState`
in, `(priors over legal actions, utility vector)` out.
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence

import numpy as np
import torch

from skyjo.domain.engine import Action, GameState
from skyjo.rl.action_space import ACTION_SPACE_SIZE, index_to_action
from skyjo.rl.encoding import encode_state
from skyjo.rl.mcts import BatchEvaluateFn, EvaluateFn, LeafResult
from skyjo.rl.network import AlphaZeroNet


def make_network_evaluator(
    net: AlphaZeroNet,
    device: str | torch.device = "cpu",
    *,
    rank_probs_sink: MutableMapping[GameState, np.ndarray] | None = None,
) -> EvaluateFn:
    """`rank_probs_sink`, if given, is filled in as a side effect of every
    `evaluate` call: `sink[state]` becomes that state's `rank_probs[i, r]` =
    P(player i finishes at rank r), the same forward pass this function
    already runs to get `value` (`value` is in fact just `rank_probs`
    reduced by a fixed linear weighting - see `AlphaZeroNet.forward`'s
    docstring - so this is strictly more information from work already
    being done, not an extra network call).

    Exists only for diagnostics (`scripts/dump_mcts_tree.py --network`) - training/self-play
    never pass it, so `EvaluateFn`'s contract for every other caller is
    untouched, and the `.cpu().numpy()` conversion for `rank_probs` doesn't
    even run when no sink is given.
    """
    net.to(device)
    net.eval()

    def evaluate(state: GameState) -> tuple[dict[Action, float], np.ndarray]:
        encoding = encode_state(state)
        with torch.no_grad():
            features = torch.from_numpy(encoding.features).unsqueeze(0).to(device)
            mask = torch.from_numpy(encoding.legal_action_mask).unsqueeze(0).to(device)
            active_count = torch.tensor([encoding.active_count], dtype=torch.long, device=device)
            policy_probs, rank_probs, utility = net(features, mask, active_count)

        policy_probs = policy_probs[0].cpu().numpy()
        priors = {
            index_to_action(i): float(policy_probs[i])
            for i in range(ACTION_SPACE_SIZE)
            if encoding.legal_action_mask[i]
        }
        value = utility[0, : encoding.active_count].cpu().numpy()
        if rank_probs_sink is not None:
            n = encoding.active_count
            rank_probs_sink[state] = rank_probs[0, :n, :n].cpu().numpy()
        return priors, value

    return evaluate


def make_batch_network_evaluator(
    net: AlphaZeroNet,
    device: str | torch.device = "cpu",
) -> BatchEvaluateFn:
    """Batched sibling of `make_network_evaluator`: encodes every given state
    and runs exactly one forward pass over all of them, regardless of how
    many there are - a batch-of-N call is far cheaper per-state than N
    batch-of-1 calls, which is the entire reason this exists (see
    `rl.mcts.run_mcts_batch`, the intended caller). Returns results in the
    same order as `states`; `states=[]` short-circuits without touching the
    network at all.
    """
    net.to(device)
    net.eval()

    def evaluate_batch(states: Sequence[GameState]) -> list[LeafResult]:
        if not states:
            return []

        encodings = [encode_state(state) for state in states]
        with torch.no_grad():
            features = torch.from_numpy(np.stack([e.features for e in encodings])).to(device)
            mask = torch.from_numpy(np.stack([e.legal_action_mask for e in encodings])).to(device)
            active_count = torch.tensor(
                [e.active_count for e in encodings], dtype=torch.long, device=device
            )
            policy_probs, _rank_probs, utility = net(features, mask, active_count)

        policy_probs = policy_probs.cpu().numpy()
        utility = utility.cpu().numpy()

        results: list[LeafResult] = []
        for i, encoding in enumerate(encodings):
            priors = {
                index_to_action(j): float(policy_probs[i, j])
                for j in range(ACTION_SPACE_SIZE)
                if encoding.legal_action_mask[j]
            }
            results.append((priors, utility[i, : encoding.active_count]))
        return results

    return evaluate_batch
