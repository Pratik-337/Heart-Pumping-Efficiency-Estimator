"""Training and inference step pieces shared by the probe (A4), trainer (A6) and predict (A5).

Spec, "Model and training design": MSE loss, AdamW, bf16 autocast, TF32 for the remaining FP32
maths, micro-batch 8 with gradient accumulation 2. The probe times exactly these functions, so
its epoch-time estimates describe the real training step.
"""

from contextlib import AbstractContextManager, nullcontext

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

PRECISIONS = ("bf16", "fp32")


def configure_backends(cfg: DictConfig, cudnn_benchmark: bool = True) -> None:
    """TF32 as configured; cuDNN autotuning on or off.

    cuDNN is never forced deterministic (spec, Seeds). Autotuning (``cudnn_benchmark``) picks
    the fastest convolution algorithms but may use more workspace memory (A4 probe).
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.train.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.train.tf32)
    torch.backends.cudnn.benchmark = cudnn_benchmark


def autocast(device: torch.device, precision: str) -> AbstractContextManager:
    """bf16 autocast (``train.precision: bf16``) or plain FP32 (``fp32``, e.g. CPU tests)."""
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    if precision == "fp32":
        return nullcontext()
    raise ValueError(f"precision must be one of {PRECISIONS}; got {precision!r}")


def make_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """AdamW over all parameters with ``train.lr`` and ``train.weight_decay`` (D-05, D-18)."""
    if cfg.train.optimizer != "adamw":
        raise ValueError(f"train.optimizer {cfg.train.optimizer!r}: only 'adamw' is implemented")
    on_cuda = next(model.parameters()).is_cuda
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        fused=on_cuda,  # one kernel per step on the GPU; same maths
    )


def check_loss(cfg: DictConfig) -> None:
    if cfg.train.loss != "mse":
        raise ValueError(f"train.loss {cfg.train.loss!r}: only 'mse' is implemented")


def train_micro_step(
    model: nn.Module, clip: torch.Tensor, ef: torch.Tensor, grad_accum: int, precision: str
) -> torch.Tensor:
    """Forward and backward of one micro-batch; gradients add up until the optimizer steps.

    The loss is divided by ``grad_accum``, so ``grad_accum`` micro-batches of size b give the
    gradient of the mean loss over all of them (for layers without BatchNorm, exactly).

    Args:
        clip: (B, 3, T, H, W) normalised clips on the model's device.
        ef: (B,) reference EF in %.

    Returns:
        The micro-batch's MSE loss (detached, FP32).
    """
    with autocast(clip.device, precision):
        pred = model(clip)
    loss = F.mse_loss(pred.float(), ef.float())
    (loss / grad_accum).backward()
    return loss.detach()


@torch.no_grad()
def predict_batch(model: nn.Module, clip: torch.Tensor, precision: str) -> torch.Tensor:
    """(B, 3, T, H, W) -> (B,) predicted EF in FP32, without gradients."""
    with autocast(clip.device, precision):
        return model(clip).float()
