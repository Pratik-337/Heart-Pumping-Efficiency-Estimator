"""Numeric tests of attributions (task B6). Spec, "Explainability design", "Numeric tests".

1. ``hit_rate``: share of the 2 most important frames within +-1 sampled frame of ED or ES,
   with the chance rate of a random ranking (permutation test across videos: ``hit_test``).
2. ``lv_energy_ratio``: on the ED and ES frames, share of |attribution| inside the traced LV
   divided by the LV's share of the frame (> 1 = concentrated on the LV).
3. ``deletion_aopc``: delete frames from most to least important (interpolated replacement),
   up to T/2, and average |change in EF| over the steps; compared with random deletion orders.
4. ``spearman``: rank correlation of an attribution map before and after weight randomisation
   (model-randomisation sanity check, Adebayo et al. 2018).
"""

import copy

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from echoef.xai.attribution import interpolate_frames, predict_variants

TOP_K = 2
N_RANDOM_ORDERS = 10
N_PERMUTATIONS = 1000


def clip_positions(frame: int, start: int, period: int) -> float:
    """Position of raw frame ``frame`` on the clip's sampled-frame axis (may be fractional)."""
    return (frame - start) / period


def hit_mask(T: int, ed: int, es: int, start: int, period: int) -> np.ndarray:
    """bool (T,): sampled frames within +-1 sampled frame of ED or ES."""
    k = np.arange(T)
    near = [np.abs(k - clip_positions(f, start, period)) <= 1 for f in (ed, es)]
    return near[0] | near[1]


def hit_rate(importance: np.ndarray, mask: np.ndarray, k: int = TOP_K) -> tuple[float, float]:
    """(share of the top-k frames that are hits, chance share of hits under a random ranking)."""
    top = np.argsort(-np.asarray(importance), kind="stable")[:k]
    return float(mask[top].mean()), float(mask.mean())


def hit_test(
    hit_masks: list[np.ndarray],
    hits: np.ndarray,
    seed: int = 0,
    n_perm: int = N_PERMUTATIONS,
    k: int = TOP_K,
) -> dict[str, float]:
    """Permutation test: is the mean hit rate above that of random top-k frame choices?

    Returns:
        ``{"hit_rate", "chance_rate", "p"}``; p = share of random draws with a mean hit rate at
        least the observed one (with the +1 correction).
    """
    rng = np.random.default_rng(seed)
    observed = float(np.mean(hits))
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.mean([m[rng.permutation(len(m))[:k]].mean() for m in hit_masks])
    chance = float(np.mean([m.mean() for m in hit_masks]))
    return {
        "hit_rate": observed,
        "chance_rate": chance,
        "p": float((1 + np.sum(null >= observed)) / (1 + n_perm)),
    }


def lv_energy_ratio(attribution: np.ndarray, mask: np.ndarray) -> float:
    """(share of |attribution| inside ``mask``) / (mask area / frame area) for one frame.

    Args:
        attribution: (H, W); mask: bool (H, W).
    """
    a = np.abs(np.asarray(attribution, dtype=np.float64))
    total, area = a.sum(), mask.mean()
    if total == 0 or area == 0:
        return float("nan")
    return float((a[mask].sum() / total) / area)


def deletion_aopc(
    model: nn.Module,
    clip: torch.Tensor,
    order: np.ndarray,
    precision: str = "bf16",
    batch: int = 8,
    max_frac: float = 0.5,
) -> float:
    """Mean |EF change| when deleting frames in ``order`` one by one, up to ``max_frac`` x T.

    Args:
        clip: (3, T, H, W); order: frame indices, most important first.
    """
    t = clip.shape[1]
    steps = max(1, int(t * max_frac))
    removed = np.zeros(t, dtype=bool)
    variants = [clip]
    for frame in order[:steps]:
        removed[frame] = True
        variants.append(interpolate_frames(clip, removed))
    preds = predict_variants(model, torch.stack(variants), precision, batch)
    return float(np.mean(np.abs(preds[1:] - preds[0])))


def random_aopc(
    model: nn.Module,
    clip: torch.Tensor,
    precision: str = "bf16",
    batch: int = 8,
    n_orders: int = N_RANDOM_ORDERS,
    seed: int = 0,
) -> float:
    """Mean deletion AOPC over ``n_orders`` random deletion orders (the baseline of test 3)."""
    rng = np.random.default_rng(seed)
    t = clip.shape[1]
    return float(
        np.mean(
            [
                deletion_aopc(model, clip, rng.permutation(t), precision, batch)
                for _ in range(n_orders)
            ]
        )
    )


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two attribution maps (flattened |values|)."""
    x, y = np.abs(np.ravel(a)), np.abs(np.ravel(b))
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


RANDOMISATION_STAGES = ("fc", "layer4", "layer3", "layer2", "layer1", "stem")


def randomise_cascade(model: nn.Module, upto: str, seed: int = 0) -> nn.Module:
    """Copy of ``model`` with weights re-initialised from ``fc`` back to ``upto`` (cumulative)."""
    if upto not in RANDOMISATION_STAGES:
        raise ValueError(f"upto must be one of {RANDOMISATION_STAGES}")
    out = copy.deepcopy(model)
    torch.manual_seed(seed)
    for name in RANDOMISATION_STAGES[: RANDOMISATION_STAGES.index(upto) + 1]:
        for module in getattr(out, name).modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()
    return out.eval()
