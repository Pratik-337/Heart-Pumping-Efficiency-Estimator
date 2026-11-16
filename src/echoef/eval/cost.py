"""Cost metrics of a config (task A7; ``scripts/benchmark_cost.py``).

Spec, "Cost metrics (fixed protocol)":

- parameters: trainable parameter count;
- GFLOPs per clip: ``torch.utils.flop_counter.FlopCounterMode`` on one clip, 1 multiply-add =
  2 FLOPs (counted on the CPU; FLOPs do not depend on the device); per video = mean M2 clips x
  per clip;
- latency: batch of one video, bf16, 10 warm-up runs, median of 50, CUDA events after
  ``torch.cuda.synchronize()``. M1 = one clip; M2 = a video with the mean number of M2 clips,
  its clips run in chunks of ``train.micro_batch`` as ``predict.predict_video`` does;
- peak VRAM: ``torch.cuda.max_memory_allocated`` for M1 inference and for one training step;
- epoch time: mean epoch wall-clock of the config's seed-0 run on this GPU (``metrics.json``).
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from echoef.contracts import FRAME_SIZE
from echoef.data.sampling import clip_starts
from echoef.train.steps import predict_batch

WARMUP, RUNS = 10, 50


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gflops_per_clip(model: nn.Module, T: int) -> float:
    """Forward GFLOPs of one (1, 3, T, 112, 112) clip, counted on the CPU."""
    model = model.to("cpu").eval()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1, 3, T, FRAME_SIZE, FRAME_SIZE))
    return counter.get_total_flops() / 1e9


def mean_m2_clips(n_frames: list[int], T: int, period: int) -> float:
    """Mean number of M2 clips per video over the given video lengths."""
    return float(np.mean([len(clip_starts(int(n), T, period, "M2")) for n in n_frames]))


def _clips(n: int, T: int, device: torch.device) -> torch.Tensor:
    return torch.randn(n, 3, T, FRAME_SIZE, FRAME_SIZE, device=device)


def latency_ms(
    model: nn.Module,
    T: int,
    n_clips: int,
    device: torch.device,
    precision: str,
    chunk: int,
    warmup: int = WARMUP,
    runs: int = RUNS,
) -> float:
    """Median wall time (ms) to predict one video of ``n_clips`` clips, input already on device."""
    model = model.to(device).eval()
    clips = _clips(n_clips, T, device)

    def once() -> None:
        for i in range(0, n_clips, chunk):
            predict_batch(model, clips[i : i + chunk], precision)

    for _ in range(warmup):
        once()
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            once()
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            once()
            times.append((time.perf_counter() - t0) * 1000)
    return float(statistics.median(times))


def peak_infer_mb(model: nn.Module, T: int, device: torch.device, precision: str) -> float | None:
    """Peak allocated VRAM (MB) of one M1 prediction (batch 1); None on the CPU."""
    if device.type != "cuda":
        return None
    model = model.to(device).eval()
    clip = _clips(1, T, device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    predict_batch(model, clip, precision)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2**20


def epoch_time_s(run: Path) -> float | None:
    """Mean epoch seconds from a finished run's ``metrics.json``, or None."""
    try:
        return float(json.loads((Path(run) / "metrics.json").read_text())["epoch_seconds_mean"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None
