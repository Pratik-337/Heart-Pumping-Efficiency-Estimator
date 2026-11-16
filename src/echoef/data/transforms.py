"""Clip transforms (task A3): normalise, grey to 3 channels, zero-pad-and-crop augmentation.

The order follows EchoNet's loader (echonet/datasets/echo.py): normalise first, pad afterwards.
Every zero added later (frames past the video's end, the 12-pixel augmentation border) is
therefore the training-set mean grey level, not black (spec, "Short videos").
"""

import numpy as np
import torch
import torch.nn.functional as F


def normalise(frames: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Grey levels to ``(x - mean) / std``.

    Args:
        frames: uint8 array (..., H, W) of grey levels 0-255.
        mean, std: training-set statistics from ``stats.json``, in grey levels.

    Returns:
        float32 array of the same shape.
    """
    if not std > 0:
        raise ValueError(f"std must be > 0; got {std}")
    return (frames.astype(np.float32) - np.float32(mean)) / np.float32(std)


def denormalise(clip: torch.Tensor | np.ndarray, mean: float, std: float) -> np.ndarray:
    """Inverse of ``normalise`` for display: uint8 array of the same shape, clipped to 0-255."""
    values = clip.detach().cpu().numpy() if isinstance(clip, torch.Tensor) else np.asarray(clip)
    return np.clip(np.rint(values.astype(np.float64) * std + mean), 0, 255).astype(np.uint8)


def to_channels(clip: torch.Tensor, channels: int = 3) -> torch.Tensor:
    """Copy a grey clip into every input channel: (T, H, W) -> (channels, T, H, W).

    The Kinetics-pretrained stem expects 3 channels (spec, "Model and training design", Input).
    """
    if clip.ndim != 3:
        raise ValueError(f"expected a (T, H, W) clip; got shape {tuple(clip.shape)}")
    return clip.unsqueeze(0).repeat(channels, 1, 1, 1)


def pad_crop(
    clip: torch.Tensor, pad: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Zero-pad height and width by ``pad`` on every side, then crop back at a random offset.

    One offset for the whole clip (all frames), drawn uniformly from ``0 .. 2 x pad`` on each
    axis, so the image moves by up to ``pad`` pixels in any direction: a small probe shift
    (EchoNet's augmentation, ``data.augment.pad_crop`` in configs/base.yaml).

    Args:
        clip: float tensor (..., H, W), already normalised.
        pad: border width in pixels; 0 returns ``clip`` unchanged.

    Returns:
        Tensor of the same shape as ``clip``.
    """
    if pad < 0:
        raise ValueError(f"pad must be >= 0; got {pad}")
    if pad == 0:
        return clip
    h, w = clip.shape[-2:]
    i, j = torch.randint(2 * pad + 1, (2,), generator=generator).tolist()
    padded = F.pad(clip, (pad, pad, pad, pad))  # zeros on the last two dimensions
    return padded[..., i : i + h, j : j + w].contiguous()
