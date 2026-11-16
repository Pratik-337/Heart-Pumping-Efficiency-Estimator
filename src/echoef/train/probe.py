"""Memory and speed probe for one training config (task A4).

Spec, task A4: the largest micro-batch per T, plus 50 timed training steps per T, to choose the
activation-checkpointing setting and replace the Compute budget's epoch-time estimates.

- ``memory_trial``: two optimizer steps on random clips (so AdamW's state exists) at one
  micro-batch size; peak memory, or "does not fit" on ``torch.OutOfMemoryError``.
- ``time_training`` / ``time_eval``: timed steps on real batches from a DataLoader, the GPU
  synchronised after each optimizer step (training) or batch (evaluation).

Every function also runs on the CPU (peaks are then None), so it is unit-tested without a GPU.
"""

import gc
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from echoef.contracts import FRAME_SIZE
from echoef.train.trainer import make_optimizer, predict_batch, train_micro_step

MB = 2**20


@dataclass(frozen=True)
class MemoryTrial:
    micro_batch: int
    activation_checkpointing: bool
    fits: bool
    peak_alloc_mb: float | None  # torch.cuda.max_memory_allocated; None on the CPU or on OOM
    peak_reserved_mb: float | None  # torch.cuda.max_memory_reserved


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> tuple[float | None, float | None]:
    """(peak allocated, peak reserved) in MB since the last reset; (None, None) on the CPU."""
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.max_memory_allocated(device) / MB,
        torch.cuda.max_memory_reserved(device) / MB,
    )


def memory_trial(
    make_model: Callable[[bool], nn.Module],
    cfg: DictConfig,
    micro_batch: int,
    device: torch.device,
    activation_checkpointing: bool,
) -> MemoryTrial:
    """Does a training step at ``micro_batch`` fit? Fresh model and optimizer each time.

    Runs ``2 x train.grad_accum`` micro-steps on random (micro_batch, 3, T, 112, 112) clips
    with ``train.precision``, stepping AdamW after every ``train.grad_accum`` of them.

    Args:
        make_model: ``activation_checkpointing -> model`` on the CPU.
    """
    grad_accum, precision = int(cfg.train.grad_accum), str(cfg.train.precision)
    _release(device)
    model = optimizer = clip = ef = None
    try:
        model = make_model(activation_checkpointing).to(device).train()
        optimizer = make_optimizer(model, cfg)
        for i in range(2 * grad_accum):
            clip = torch.randn(micro_batch, 3, int(cfg.T), FRAME_SIZE, FRAME_SIZE, device=device)
            ef = torch.full((micro_batch,), 55.0, device=device)
            train_micro_step(model, clip, ef, grad_accum, precision)
            if (i + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        _sync(device)
        return MemoryTrial(micro_batch, activation_checkpointing, True, *peak_memory_mb(device))
    except torch.OutOfMemoryError:
        return MemoryTrial(micro_batch, activation_checkpointing, False, None, None)
    finally:
        model = optimizer = clip = ef = None
        _release(device)


def largest_micro_batch(
    make_model: Callable[[bool], nn.Module],
    cfg: DictConfig,
    device: torch.device,
    activation_checkpointing: bool,
    candidates: Iterable[int],
) -> list[MemoryTrial]:
    """``memory_trial`` for increasing micro-batch sizes, stopping after the first that fails."""
    trials = []
    for micro_batch in sorted(candidates):
        trials.append(memory_trial(make_model, cfg, micro_batch, device, activation_checkpointing))
        if not trials[-1].fits:
            break
    return trials


def _to(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["clip"].to(device, non_blocking=True), batch["ef"].to(device, non_blocking=True)


def _summary(seconds: list[float], prefix: str) -> dict[str, float]:
    ms = np.asarray(seconds) * 1000
    return {
        f"{prefix}_ms": float(ms.mean()),
        f"{prefix}_ms_median": float(np.median(ms)),
        f"{prefix}_ms_max": float(ms.max()),
    }


def time_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[dict],
    cfg: DictConfig,
    device: torch.device,
    n_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    """Wall-clock time per training micro-step on real batches, data loading included.

    Each optimizer step (``train.grad_accum`` micro-steps) is timed after a GPU sync and
    divided by ``grad_accum``; the first ``warmup_steps`` micro-steps (cuDNN autotuning,
    worker start-up) are not timed.

    Returns:
        ``train_step_ms`` (mean), ``train_step_ms_median``, ``train_step_ms_max``.
    """
    grad_accum, precision = int(cfg.train.grad_accum), str(cfg.train.precision)
    if n_steps % grad_accum or warmup_steps % grad_accum:
        raise ValueError(f"n_steps and warmup_steps must be multiples of grad_accum {grad_accum}")
    model.train()
    batches = iter(batches)
    times: list[float] = []
    _sync(device)
    t0 = time.perf_counter()
    for i in range(warmup_steps + n_steps):
        clip, ef = _to(next(batches), device)
        train_micro_step(model, clip, ef, grad_accum, precision)
        if (i + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _sync(device)
            t1 = time.perf_counter()
            if i >= warmup_steps:
                times.append((t1 - t0) / grad_accum)
            t0 = t1
    return _summary(times, "train_step")


def time_eval(
    model: nn.Module,
    batches: Iterable[dict],
    cfg: DictConfig,
    device: torch.device,
    n_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    """Wall-clock time per evaluation batch (no gradients), data loading included.

    Returns:
        ``eval_step_ms`` (mean), ``eval_step_ms_median``, ``eval_step_ms_max``.
    """
    model.eval()
    batches = iter(batches)
    times: list[float] = []
    _sync(device)
    t0 = time.perf_counter()
    for i in range(warmup_steps + n_steps):
        clip, _ = _to(next(batches), device)
        predict_batch(model, clip, str(cfg.train.precision))
        _sync(device)
        t1 = time.perf_counter()
        if i >= warmup_steps:
            times.append(t1 - t0)
        t0 = t1
    return _summary(times, "eval_step")


def epoch_seconds(
    n_train: int, n_val: int, micro_batch: int, train_step_s: float, eval_step_s: float
) -> float:
    """One epoch: one clip per training video, then M1 validation on every validation video."""
    return (
        math.ceil(n_train / micro_batch) * train_step_s
        + math.ceil(n_val / micro_batch) * eval_step_s
    )
