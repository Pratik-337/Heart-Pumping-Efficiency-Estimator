"""The training run on CPU with a tiny model: outputs, schedule, resume, guards (synthetic data)."""

import json

import pandas as pd
import pytest
import torch
from conftest import IDENTITY
from torch import nn

from echoef.config import load_config
from echoef.contracts import PREDICTION_COLUMNS
from echoef.train import trainer as tr

CPU = torch.device("cpu")
OVERRIDES = ["train.epochs=3", "train.micro_batch=1", "train.grad_accum=2", "train.precision=fp32"]


class TinyNet(nn.Module):
    """(B, 3, T, H, W) -> (B,): 1x1x1 conv, mean-pool, linear head with the given bias."""

    def __init__(self, bias: float) -> None:
        super().__init__()
        self.conv = nn.Conv3d(3, 2, 1)
        self.fc = nn.Linear(2, 1)
        nn.init.constant_(self.fc.bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).mean(dim=(2, 3, 4))).squeeze(-1)


def _train(processed, run, overrides=(), make_model=TinyNet):
    cfg = load_config("fs8", OVERRIDES + list(overrides))
    return tr.train_run(cfg, run, processed_root=processed, stats=IDENTITY, device=CPU,
                        workers=0, make_model=make_model)  # fmt: skip


def test_run_folder_has_every_contract_file(processed, tmp_path):
    run = tmp_path / "fs8" / "seed0"
    metrics = _train(processed, run)
    for name in ["config.yaml", "meta.json", "log.csv", "best.pt", "last.pt", "metrics.json",
                 "DONE", "predictions_val_M1.csv", "predictions_val_M2.csv"]:  # fmt: skip
        assert (run / name).exists(), name
    assert (run / "tb").is_dir()
    log = pd.read_csv(run / "log.csv")
    assert list(log.columns[:5]) == tr.LOG_COLUMNS and log["epoch"].tolist() == [0, 1, 2]
    preds = pd.read_csv(run / "predictions_val_M2.csv")
    assert list(preds.columns) == PREDICTION_COLUMNS and len(preds) == 2
    best = torch.load(run / "best.pt", weights_only=True)
    assert best["epoch"] == metrics["best_epoch"] == log["val_mae_m1"].idxmin()
    assert best["config"]["T"] == 8 and best["config"]["config_id"] == "fs8"
    assert json.loads((run / "meta.json").read_text())["finished"]
    assert _train(processed, run) == metrics  # DONE: nothing is trained again


def test_resume_after_a_crash_gives_the_same_model(processed, tmp_path, monkeypatch):
    straight = tmp_path / "straight"
    _train(processed, straight)

    resumed = tmp_path / "resumed"
    calls = {"n": 0}
    real_predict = tr.predict_split

    def crash_in_second_validation(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(tr, "predict_split", crash_in_second_validation)
    with pytest.raises(RuntimeError, match="simulated"):
        _train(processed, resumed)
    assert torch.load(resumed / "last.pt", weights_only=True)["epoch"] == 0
    assert not (resumed / "DONE").exists()
    monkeypatch.setattr(tr, "predict_split", real_predict)
    _train(processed, resumed)

    assert pd.read_csv(resumed / "log.csv")["epoch"].tolist() == [0, 1, 2]
    a = torch.load(straight / "last.pt", weights_only=True)["state_dict"]
    b = torch.load(resumed / "last.pt", weights_only=True)["state_dict"]
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_changed_config_is_refused(processed, tmp_path):
    run = tmp_path / "run"
    _train(processed, run, ["train.epochs=1"])
    (run / "DONE").unlink()
    with pytest.raises(FileExistsError):
        _train(processed, run, ["train.epochs=2"])


def test_nan_loss_stops_the_run(processed, tmp_path):
    class NaNNet(TinyNet):
        def forward(self, x):
            return super().forward(x) * float("nan")

    with pytest.raises(FloatingPointError):
        _train(processed, tmp_path / "run", make_model=NaNNet)


def test_lr_schedule_warms_up_then_decays_to_zero():
    f = [tr.lr_factor(s, warmup_steps=4, total_steps=20) for s in range(21)]
    assert f[:4] == [0.25, 0.5, 0.75, 1.0]
    assert f[4] == 1.0 and f[20] == pytest.approx(0.0, abs=1e-12)
    assert all(a >= b for a, b in zip(f[4:], f[5:], strict=False))
    assert f[12] == pytest.approx(0.5)  # half-way through the cosine
