import torch

from skyjo.rl.action_space import ACTION_SPACE_SIZE
from skyjo.rl.encoding import INPUT_DIM, N_MAX_PLAYERS
from skyjo.rl.network import AlphaZeroNet, _masked_softmax, active_rank_mask, rank_utility_weights


def _tiny_net() -> AlphaZeroNet:
    return AlphaZeroNet(trunk_dim=16, num_residual_blocks=1)


def _random_batch(batch_size: int, active_counts: list[int]):
    torch.manual_seed(0)
    features = torch.randn(batch_size, INPUT_DIM)
    mask = torch.zeros(batch_size, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, :3] = True  # first 3 actions always "legal" for these synthetic inputs
    active_count = torch.tensor(active_counts, dtype=torch.long)
    return features, mask, active_count


# --- forward pass across the full N_act range, incl. a mixed batch -----------


def test_forward_pass_works_for_every_active_count_from_2_to_8_in_one_batch():
    net = _tiny_net()
    features, mask, active_count = _random_batch(7, [2, 3, 4, 5, 6, 7, 8])

    policy_probs, rank_probs, utility = net(features, mask, active_count)

    assert policy_probs.shape == (7, ACTION_SPACE_SIZE)
    assert rank_probs.shape == (7, N_MAX_PLAYERS, N_MAX_PLAYERS)
    assert utility.shape == (7, N_MAX_PLAYERS)
    assert torch.isfinite(policy_probs).all()
    assert torch.isfinite(rank_probs).all()
    assert torch.isfinite(utility).all()


def test_policy_probs_are_zero_off_the_mask_and_sum_to_one_on_it():
    net = _tiny_net()
    features, mask, active_count = _random_batch(4, [2, 4, 6, 8])

    policy_probs, _, _ = net(features, mask, active_count)

    assert torch.allclose(policy_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.all(policy_probs[~mask] < 1e-6)


def test_rank_probs_active_rows_sum_to_one_and_inactive_rows_are_exactly_zero():
    net = _tiny_net()
    features, mask, active_count = _random_batch(3, [2, 5, 8])

    _, rank_probs, _ = net(features, mask, active_count)

    for b, n_act in enumerate([2, 5, 8]):
        active_rows = rank_probs[b, :n_act, :]
        assert torch.allclose(active_rows.sum(dim=-1), torch.ones(n_act), atol=1e-5)
        if n_act < N_MAX_PLAYERS:
            inactive_rows = rank_probs[b, n_act:, :]
            assert torch.all(inactive_rows == 0.0)


def test_utility_is_zero_for_players_beyond_active_count():
    net = _tiny_net()
    features, mask, active_count = _random_batch(2, [3, 8])

    _, _, utility = net(features, mask, active_count)

    assert torch.all(utility[0, 3:] == 0.0)


# --- gradient isolation: verification criterion 3 -----------------------------


def test_inactive_rank_rows_and_columns_receive_zero_gradient():
    # Checks this at the raw-logits level (not just net.parameters()), since
    # that's the only place "row/column" has a meaning to zero out - a
    # Linear layer's weight gradient is already summed across all 64 output
    # positions and wouldn't show a per-cell zero even if this property broke.
    net = _tiny_net()
    features, _mask, active_count = _random_batch(3, [2, 5, 8])

    rank_logits = net.rank_head(net.trunk(features)).view(-1, N_MAX_PLAYERS, N_MAX_PLAYERS)
    rank_logits.retain_grad()
    active = active_rank_mask(active_count, features.device)
    probs = _masked_softmax(rank_logits, active)
    row_active = torch.arange(N_MAX_PLAYERS).unsqueeze(0) < active_count.unsqueeze(1)
    probs = probs * row_active.unsqueeze(2).float()
    probs.sum().backward()

    grad = rank_logits.grad
    for b, n_act in enumerate([2, 5, 8]):
        stray = grad[b].clone()
        stray[:n_act, :n_act] = 0.0  # zero out the active block; everything else must already be zero
        assert torch.all(stray == 0.0), f"sample {b} (n_act={n_act}) leaked gradient outside its active block"


# --- helper functions used inside the head -------------------------------------


def test_active_rank_mask_is_true_only_within_the_active_square():
    mask = active_rank_mask(torch.tensor([3]), torch.device("cpu"))[0]

    assert mask[:3, :3].all()
    assert not mask[3:, :].any()
    assert not mask[:, 3:].any()


def test_rank_utility_weights_span_plus_one_to_minus_one_and_are_zero_past_n_act():
    w = rank_utility_weights(torch.tensor([5]), torch.device("cpu"))[0]

    assert w[0].item() == 1.0  # best rank -> best payoff
    assert w[4].item() == -1.0  # worst active rank -> worst payoff
    assert torch.all(w[5:] == 0.0)
