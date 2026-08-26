import pytest
import torch

from skyjo.rl.checkpoint import load_checkpoint, save_checkpoint
from skyjo.rl.network import AlphaZeroNet


def _tiny_net() -> AlphaZeroNet:
    return AlphaZeroNet(trunk_dim=16, num_residual_blocks=1)


# --- happy path ------------------------------------------------------------


def test_save_then_load_round_trips_weights_and_progress(tmp_path):
    torch.manual_seed(0)
    net = _tiny_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    path = tmp_path / "nested" / "checkpoint.pt"

    save_checkpoint(path, net, optimizer, iteration=7, total_train_steps=700, extra={"note": "hi"})

    loaded_net = _tiny_net()
    loaded_optimizer = torch.optim.Adam(loaded_net.parameters(), lr=1e-2)
    result = load_checkpoint(path, loaded_net, loaded_optimizer)

    assert result.iteration == 7
    assert result.total_train_steps == 700
    assert result.extra == {"note": "hi"}
    for original, restored in zip(net.parameters(), loaded_net.parameters(), strict=True):
        assert torch.equal(original, restored)


def test_load_checkpoint_without_optimizer_arg_still_restores_net(tmp_path):
    net = _tiny_net()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, net, None, iteration=1, total_train_steps=1)

    loaded_net = _tiny_net()
    result = load_checkpoint(path, loaded_net)

    assert result.iteration == 1
    for original, restored in zip(net.parameters(), loaded_net.parameters(), strict=True):
        assert torch.equal(original, restored)


# --- bad path ----------------------------------------------------------------


def test_load_checkpoint_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pt", _tiny_net())
