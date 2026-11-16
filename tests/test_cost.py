"""Cost metrics on the CPU (task A7)."""

import json

import pytest
import torch
from torch import nn

from echoef.eval import cost
from echoef.models.factory import build_model


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x.mean(dim=(2, 3, 4))).squeeze(-1)


def test_flops_match_torchvision_doubled_and_scale_with_t():
    model = build_model(pretrained=False)
    assert cost.count_params(model) == 31_300_638
    g8 = cost.gflops_per_clip(model, 8)
    assert g8 == pytest.approx(40.52, abs=0.01)  # torchvision's 40.52 G multiply-adds at T=16
    assert cost.gflops_per_clip(model, 16) == pytest.approx(2 * g8, rel=1e-3)


def test_mean_m2_clips():
    # W = 64: 173 frames -> 4 clips, 64 frames -> 1, 40 frames -> 1 (padded)
    assert cost.mean_m2_clips([173, 64, 40], 32, 2) == pytest.approx(2.0)


def test_latency_and_memory_on_cpu():
    ms = cost.latency_ms(Tiny(), 8, 5, torch.device("cpu"), "fp32", chunk=2, warmup=1, runs=3)
    assert ms > 0
    assert cost.peak_infer_mb(Tiny(), 8, torch.device("cpu"), "fp32") is None


def test_epoch_time_from_metrics(tmp_path):
    assert cost.epoch_time_s(tmp_path) is None
    (tmp_path / "metrics.json").write_text(json.dumps({"epoch_seconds_mean": 97.5}))
    assert cost.epoch_time_s(tmp_path) == 97.5


def test_epoch_time_only_from_this_gpu(tmp_path):
    """D-23/D-31: a run trained on another GPU never sets the epoch time; the probe does."""
    gpu, other = "NVIDIA GeForce RTX 4060 Laptop GPU", "NVIDIA GeForce RTX 4050 Laptop GPU"

    def run(config_id, seed, gpu_name, seconds, done=True):
        path = tmp_path / "runs" / config_id / f"seed{seed}"
        path.mkdir(parents=True)
        (path / "meta.json").write_text(json.dumps({"gpu_name": gpu_name}))
        (path / "metrics.json").write_text(json.dumps({"epoch_seconds_mean": seconds}))
        if done:
            (path / "DONE").write_text("")

    run("fs8", 0, other, 120.0)
    run("fs8", 1, gpu, 97.0)  # seed 1 trained here: used although seed 0 exists
    run("fs16", 0, other, 250.0)
    run("fs32", 0, gpu, 380.0, done=False)  # unfinished: ignored
    probe = tmp_path / "probe.csv"
    probe.write_text(
        "gpu_name,config_id,T,epoch_s_est\n"
        f"{gpu},fs16,16,196.0\n{other},fs32,32,500.0\n{gpu},fs32,32,386.0\n"
    )
    runs = tmp_path / "runs"
    assert cost.config_epoch_time(runs, "fs8", 8, gpu, probe) == (97.0, "log fs8/seed1")
    seconds, source = cost.config_epoch_time(runs, "fs16", 16, gpu, probe)
    assert seconds == 196.0 and source.startswith("probe T=16")
    assert cost.config_epoch_time(runs, "fs32", 32, gpu, probe)[0] == 386.0
    assert cost.config_epoch_time(runs, "fw16", 16, gpu, None) == (None, "none")
