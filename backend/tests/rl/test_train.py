from dataclasses import replace

import numpy as np
import pytest
import torch

from skyjo.domain.engine import new_match
from skyjo.rl.action_space import ACTION_SPACE_SIZE
from skyjo.rl.encoding import encode_state
from skyjo.rl.network import AlphaZeroNet
from skyjo.rl.selfplay import ReplaySample
from skyjo.rl.train import collate_batch, compute_loss, training_step


def _sample(seed: int, n_act: int, y: list[int], *, current_player: int = 0) -> ReplaySample:
    state = new_match(player_count=n_act, seed=seed)
    state = replace(state, current_player=current_player)
    pi = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    pi[: n_act + 1] = 1.0 / (n_act + 1)  # arbitrary valid distribution
    return ReplaySample(state=state, n_act=n_act, pi=pi, y=np.array(y, dtype=np.int64))


def _tiny_net() -> AlphaZeroNet:
    torch.manual_seed(0)
    return AlphaZeroNet(trunk_dim=16, num_residual_blocks=1)


# --- happy path ------------------------------------------------------------


def test_collate_batch_produces_correctly_shaped_tensors():
    samples = [_sample(1, 2, [0, 1]), _sample(2, 5, [3, 1, 0, 4, 2])]

    batch = collate_batch(samples)

    assert batch.features.shape[0] == 2
    assert batch.legal_action_mask.shape == (2, ACTION_SPACE_SIZE)
    assert batch.active_count.tolist() == [2, 5]
    assert batch.pi_target.shape == (2, ACTION_SPACE_SIZE)
    assert batch.y.shape == (2, 8)
    assert batch.y[0, 2:].tolist() == [0, 0, 0, 0, 0, 0]  # padding beyond n_act=2


def test_collate_batch_with_precomputed_encodings_matches_recomputing_them():
    samples = [_sample(1, 2, [0, 1]), _sample(2, 5, [3, 1, 0, 4, 2])]
    encodings = [encode_state(s.state) for s in samples]

    recomputed = collate_batch(samples)
    precomputed = collate_batch(samples, encodings=encodings)

    torch.testing.assert_close(precomputed.features, recomputed.features)
    torch.testing.assert_close(precomputed.legal_action_mask, recomputed.legal_action_mask)


def test_compute_loss_returns_finite_positive_losses_with_expected_keys():
    net = _tiny_net()
    batch = collate_batch([_sample(1, 2, [0, 1]), _sample(2, 3, [1, 0, 2])])

    total_loss, metrics = compute_loss(net, batch)

    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0
    assert set(metrics) == {"policy_loss", "rank_loss", "l2_term", "total_loss"}
    assert all(np.isfinite(v) for v in metrics.values())


def test_training_step_runs_and_updates_parameters():
    net = _tiny_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    batch = collate_batch([_sample(1, 2, [0, 1]), _sample(2, 4, [2, 0, 3, 1])])
    before = [p.clone() for p in net.parameters()]

    training_step(net, optimizer, batch)

    after = list(net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_rank_loss_ignores_padding_beyond_n_act():
    net = _tiny_net()
    real = collate_batch([_sample(1, 2, [0, 1])])
    corrupted = collate_batch([_sample(1, 2, [0, 1])])
    corrupted.y[0, 2:] = 7  # garbage in the padded, inactive region

    _, real_metrics = compute_loss(net, real)
    _, corrupted_metrics = compute_loss(net, corrupted)

    assert real_metrics["rank_loss"] == pytest.approx(corrupted_metrics["rank_loss"])


# --- y is rotated into each sample's own canonical slot order --------------


def test_collate_batch_rotates_y_through_the_samples_own_permutation():
    # sample.y is stored in absolute player order; collate_batch must permute
    # it into the same canonical (current-player-at-slot-0) order the
    # encoding uses, or the network trains against mismatched inputs/targets.
    sample = _sample(1, 4, [10, 20, 30, 40], current_player=1)
    encoding = encode_state(sample.state)
    assert encoding.perm[:4].tolist() == [1, 2, 3, 0]  # sanity: a genuine rotation, not identity

    batch = collate_batch([sample], encodings=[encoding])

    # canonical slot s holds absolute player perm[s], so y[s] must be
    # sample.y[perm[s]].
    assert batch.y[0].tolist()[:4] == [20, 30, 40, 10]


def test_collate_batch_rotates_y_independently_per_row_in_a_mixed_batch():
    sample_a = _sample(1, 3, [1, 2, 3], current_player=2)
    sample_b = _sample(2, 3, [4, 5, 6], current_player=0)

    batch = collate_batch([sample_a, sample_b])

    # current_player=2, n_act=3 -> perm = [2, 0, 1] -> y = [3, 1, 2]
    assert batch.y[0].tolist()[:3] == [3, 1, 2]
    # current_player=0 is the identity permutation -> y unchanged
    assert batch.y[1].tolist()[:3] == [4, 5, 6]


# --- bad path ------------------------------------------------------------


def test_collate_batch_rejects_an_empty_sample_list():
    with pytest.raises(ValueError):
        collate_batch([])


def test_collate_batch_rejects_mismatched_encodings_length():
    samples = [_sample(1, 2, [0, 1]), _sample(2, 5, [3, 1, 0, 4, 2])]

    with pytest.raises(ValueError):
        collate_batch(samples, encodings=[encode_state(samples[0].state)])
