"""Training-step pieces on the CPU: loss, gradient accumulation, optimizer, autocast."""

import pytest
import torch
from torch import nn

from echoef.config import load_config
from echoef.train import trainer as tr


class TinyRegressor(nn.Module):
    """(B, 3, T, H, W) -> (B,): mean-pool, then a linear layer. No BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(2, 3, 4))).squeeze(-1)


def _grads(model: nn.Module) -> list[torch.Tensor]:
    return [p.grad.clone() for p in model.parameters()]


def test_accumulated_micro_steps_equal_one_big_batch():
    torch.manual_seed(0)
    model = TinyRegressor()
    clip, ef = torch.randn(8, 3, 4, 6, 6), torch.rand(8) * 80
    tr.train_micro_step(model, clip, ef, grad_accum=1, precision="fp32")
    full = _grads(model)
    model.zero_grad()
    for half in (slice(0, 4), slice(4, 8)):
        tr.train_micro_step(model, clip[half], ef[half], grad_accum=2, precision="fp32")
    for a, b in zip(full, _grads(model), strict=True):
        torch.testing.assert_close(a, b)


def test_micro_step_returns_mse():
    model = TinyRegressor()
    clip, ef = torch.randn(4, 3, 2, 5, 5), torch.tensor([10.0, 20.0, 30.0, 40.0])
    with torch.no_grad():
        expected = ((model(clip) - ef) ** 2).mean()
    loss = tr.train_micro_step(model, clip, ef, grad_accum=2, precision="fp32")
    torch.testing.assert_close(loss, expected)
    assert not loss.requires_grad


def test_bf16_autocast_runs_on_cpu():
    model = TinyRegressor()
    pred = tr.predict_batch(model, torch.randn(2, 3, 2, 5, 5), "bf16")
    assert pred.dtype == torch.float32 and pred.shape == (2,)
    with pytest.raises(ValueError):
        tr.autocast(torch.device("cpu"), "fp16")


def test_optimizer_from_config():
    cfg = load_config("fs8")
    opt = tr.make_optimizer(TinyRegressor(), cfg)
    assert isinstance(opt, torch.optim.AdamW)
    group = opt.param_groups[0]
    assert (group["lr"], group["weight_decay"]) == (cfg.train.lr, cfg.train.weight_decay)
    with pytest.raises(ValueError):
        tr.make_optimizer(TinyRegressor(), load_config("fs8", ["train.optimizer=sgd"]))
    with pytest.raises(ValueError):
        tr.check_loss(load_config("fs8", ["train.loss=l1"]))
