"""Attribution methods for the EF regressor (task B6). Spec, "Explainability design".

Every function takes one normalised clip (3, T, 112, 112) and a model mapping (B, 3, T, H, W) to
(B,) EF, and returns a NumPy array. Attributions say what moves the model's output, not what
causes the heart's function.

- ``occlusion_curve`` (T,): |change in EF| when one frame is replaced by the linear
  interpolation of its neighbours (not a black frame, which is out of distribution). T + 1
  forward passes.
- ``integrated_gradients`` (T, 112, 112): Captum IG summed over the 3 (identical) channels;
  baseline ``"zero"`` = the all-zero normalised clip (mean grey), ``"blur"`` = the clip blurred
  in space (sensitivity check). 32 steps, raised to 64 when Captum's convergence delta is large.
- ``gradcam3d`` (T, 112, 112): Captum ``LayerGradCam`` on ``layer4`` with
  ``relu_attributions=False``, then the absolute value (a regression output has no class, both
  directions matter), trilinearly upsampled. ``layer4`` has T/8 time steps, so Grad-CAM is used
  for spatial maps only.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam
from torch import nn

from echoef.train.steps import autocast

IG_STEPS, IG_STEPS_MAX = 32, 64
# Convergence delta above this share of |f(x) - f(baseline)| triggers the 64-step rerun.
IG_DELTA_TOL = 0.05
BLUR_KERNEL, BLUR_SIGMA = 21, 5.0  # blurred-video baseline: removes structure, keeps brightness


def _forward(model: nn.Module, precision: str):
    def f(x: torch.Tensor) -> torch.Tensor:
        with autocast(x.device, precision):
            return model(x).float()

    return f


def interpolate_frames(clip: torch.Tensor, remove: np.ndarray) -> torch.Tensor:
    """Replace the frames flagged in ``remove`` by linear interpolation of the nearest kept frames.

    Args:
        clip: (C, T, H, W).
        remove: bool (T,). A removed frame at an end takes the nearest kept frame; if every frame
            is removed the clip becomes the zero (mean grey) clip.

    Returns:
        A new (C, T, H, W) tensor.
    """
    remove = np.asarray(remove, dtype=bool)
    out = clip.clone()
    kept = np.flatnonzero(~remove)
    if len(kept) == 0:
        return torch.zeros_like(clip)
    for t in np.flatnonzero(remove):
        left, right = kept[kept < t], kept[kept > t]
        if len(left) and len(right):
            a, b = left[-1], right[0]
            w = (t - a) / (b - a)
            out[:, t] = (1 - w) * clip[:, a] + w * clip[:, b]
        else:
            out[:, t] = clip[:, (left[-1] if len(left) else right[0])]
    return out


@torch.no_grad()
def predict_variants(
    model: nn.Module, clips: torch.Tensor, precision: str, batch: int
) -> np.ndarray:
    """EF for a stack of clips (N, C, T, H, W) in batches; float64 array (N,)."""
    f = _forward(model, precision)
    return np.concatenate(
        [f(clips[i : i + batch]).double().cpu().numpy() for i in range(0, len(clips), batch)]
    )


def occlusion_curve(
    model: nn.Module, clip: torch.Tensor, precision: str = "bf16", batch: int = 8
) -> np.ndarray:
    """|EF(clip) - EF(clip with frame t interpolated)| for every frame t. Contract function.

    Args:
        clip: (3, T, 112, 112) on the model's device.

    Returns:
        float64 array (T,).
    """
    model.eval()
    t = clip.shape[1]
    variants = [clip]
    for k in range(t):
        mask = np.zeros(t, dtype=bool)
        mask[k] = True
        variants.append(interpolate_frames(clip, mask))
    preds = predict_variants(model, torch.stack(variants), precision, batch)
    return np.abs(preds[1:] - preds[0])


def blurred(clip: torch.Tensor) -> torch.Tensor:
    """Each frame blurred with a Gaussian (sigma 5 px): the IG sensitivity baseline."""
    c, t, h, w = clip.shape
    frames = clip.permute(1, 0, 2, 3)  # (T, C, H, W)
    return TF.gaussian_blur(frames, [BLUR_KERNEL, BLUR_KERNEL], [BLUR_SIGMA, BLUR_SIGMA]).permute(
        1, 0, 2, 3
    )


def integrated_gradients(
    model: nn.Module,
    clip: torch.Tensor,
    baseline: str = "zero",
    steps: int = IG_STEPS,
    precision: str = "bf16",
    internal_batch_size: int = 4,
    max_steps: int = IG_STEPS_MAX,
) -> np.ndarray:
    """Integrated Gradients per pixel and frame, summed over channels. Contract function.

    If the convergence delta exceeds ``IG_DELTA_TOL`` x |f(x) - f(baseline)| at ``steps``, the
    attribution is recomputed with ``IG_STEPS_MAX`` steps (spec).

    Returns:
        float64 array (T, 112, 112), signed.
    """
    model.eval()
    if baseline == "zero":
        base = torch.zeros_like(clip)
    elif baseline == "blur":
        base = blurred(clip)
    else:
        raise ValueError(f"baseline must be 'zero' or 'blur'; got {baseline!r}")
    ig = IntegratedGradients(_forward(model, precision))
    x, b = clip.unsqueeze(0), base.unsqueeze(0)
    gap = abs(float(predict_variants(model, torch.stack([clip, base]), precision, 2) @ [1, -1]))
    for n in sorted({steps, max(steps, max_steps)}):
        attr, delta = ig.attribute(
            x,
            baselines=b,
            n_steps=n,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
        )
        if abs(float(delta)) <= IG_DELTA_TOL * max(gap, 1e-6):
            break
    return attr[0].sum(dim=0).double().detach().cpu().numpy()


def gradcam3d(model: nn.Module, clip: torch.Tensor, precision: str = "bf16") -> np.ndarray:
    """|3D Grad-CAM| of ``layer4``, upsampled to the clip. Contract function.

    Returns:
        float64 array (T, 112, 112), non-negative.
    """
    model.eval()
    cam = LayerGradCam(_forward(model, precision), model.layer4)
    attr = cam.attribute(clip.unsqueeze(0), relu_attributions=False)  # (1, 1, T/8, 7, 7)
    up = LayerAttribution.interpolate(attr.float(), tuple(clip.shape[1:]), "trilinear")
    return up[0, 0].abs().double().detach().cpu().numpy()


def frame_importance(attribution: np.ndarray) -> np.ndarray:
    """(T, H, W) map -> (T,) importance: summed |attribution| per frame; (T,) passes through."""
    a = np.abs(np.asarray(attribution, dtype=np.float64))
    return a if a.ndim == 1 else a.reshape(len(a), -1).sum(axis=1)
