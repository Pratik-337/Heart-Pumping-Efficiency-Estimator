"""Memory/speed probe functions on the CPU with a tiny model (peaks are None without CUDA)."""

import pytest
import torch
from torch import nn

from echoef.config import load_config
from echoef.train import probe
from echoef.train.trainer import make_optimizer

CPU = torch.device("cpu")


class TinyRegressor(nn.Module):
    """(B, 3, T, H, W) -> (B,): mean-pool, then a linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(2, 3, 4))).squeeze(-1)


@pytest.fixture
def cfg():
    return load_config("fs8", ["train.precision=fp32"])


def _batches(n: int, b: int = 2, T: int = 8):
    for _ in range(n):
        yield {"clip": torch.randn(b, 3, T, 8, 8), "ef": torch.rand(b) * 80}


def test_memory_trial_on_cpu(cfg):
    trial = probe.memory_trial(lambda ckpt: TinyRegressor(), cfg, 2, CPU, False)
    assert trial.fits and trial.peak_alloc_mb is None and trial.micro_batch == 2


def test_memory_trial_reports_oom(cfg):
    def too_big(ckpt):
        raise torch.OutOfMemoryError("CUDA out of memory")

    trial = probe.memory_trial(too_big, cfg, 8, CPU, True)
    assert not trial.fits and trial.activation_checkpointing


def test_largest_micro_batch_stops_at_first_failure(cfg, monkeypatch):
    calls = []

    def fake_trial(make_model, cfg, micro_batch, device, ckpt):
        calls.append(micro_batch)
        return probe.MemoryTrial(micro_batch, ckpt, micro_batch <= 16, None, None)

    monkeypatch.setattr(probe, "memory_trial", fake_trial)
    trials = probe.largest_micro_batch(None, cfg, CPU, False, [32, 12, 16, 24, 48])
    assert calls == [12, 16, 24]
    assert [t.fits for t in trials] == [True, True, False]


def test_time_training_and_eval(cfg):
    model = TinyRegressor()
    opt = make_optimizer(model, cfg)
    train = probe.time_training(model, opt, _batches(14), cfg, CPU, n_steps=10, warmup_steps=4)
    assert set(train) == {"train_step_ms", "train_step_ms_median", "train_step_ms_max"}
    assert 0 < train["train_step_ms_median"] <= train["train_step_ms_max"]
    ev = probe.time_eval(model, _batches(7), cfg, CPU, n_steps=5, warmup_steps=2)
    assert ev["eval_step_ms"] > 0
    with pytest.raises(ValueError):
        probe.time_training(model, opt, _batches(20), cfg, CPU, n_steps=5, warmup_steps=4)


def test_epoch_seconds():
    # 7,460 training videos -> 933 micro-batches of 8; 1,288 validation -> 161 batches
    assert probe.epoch_seconds(7460, 1288, 8, 1.0, 0.5) == pytest.approx(933 + 80.5)
