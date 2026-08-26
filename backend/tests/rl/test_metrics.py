import json

import pytest

from skyjo.rl.metrics import MetricsLogger

# --- happy path --------------------------------------------------------------


def test_log_appends_one_jsonl_line_per_call(tmp_path):
    with MetricsLogger(tmp_path) as metrics:
        metrics.log(0, {"loss": 1.5, "acc": 0.1})
        metrics.log(1, {"loss": 1.2, "acc": 0.2})

    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["step"] == 0
    assert first["loss"] == 1.5
    assert first["acc"] == 0.1
    assert "time" in first


def test_log_applies_prefix_to_metric_keys(tmp_path):
    with MetricsLogger(tmp_path) as metrics:
        metrics.log(0, {"loss": 1.0}, prefix="train/")

    record = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert record["train/loss"] == 1.0
    assert "loss" not in record


def test_tensorboard_available_reflects_installed_dependency(tmp_path):
    # tensorboard is a dev dependency of this package (see pyproject.toml),
    # so it should be importable and feeding a live SummaryWriter here.
    with MetricsLogger(tmp_path) as metrics:
        assert metrics.tensorboard_available is True
    assert any((tmp_path).glob("events.out.tfevents.*"))


# --- bad path ------------------------------------------------------------


def test_log_rejects_empty_metrics_dict(tmp_path):
    with MetricsLogger(tmp_path) as metrics, pytest.raises(ValueError):
        metrics.log(0, {})


def test_logging_after_close_raises(tmp_path):
    metrics = MetricsLogger(tmp_path)
    metrics.close()

    with pytest.raises(ValueError):
        metrics.log(0, {"loss": 1.0})
