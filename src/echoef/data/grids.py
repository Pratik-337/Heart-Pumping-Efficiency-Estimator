"""Batch image grids and numeric batch checks, for checking the loader (task A3).

The grids show echo frames, so they stay on A's laptop: never committed, never shared
(CLAUDE.md, Hard rules). The numeric checks carry no image content and can be logged anywhere.
"""

import numpy as np
import torch
from matplotlib.figure import Figure

from echoef.data.sampling import clip_indices
from echoef.data.transforms import denormalise


def _past_end(batch: dict, period: int) -> torch.Tensor:
    """bool (B, T): which frames of each clip lie past its video's end (zero padding)."""
    t = batch["clip"].shape[2]
    idx = torch.stack([torch.from_numpy(clip_indices(int(s), t, period)) for s in batch["start"]])
    return idx >= batch["n_frames"].reshape(-1, 1)


def batch_checks(batch: dict, period: int) -> dict:
    """Numbers that expose loader bugs: shape, channel copies, value range, padding.

    Args:
        batch: a collated ``EchoClipDataset`` batch; ``clip`` is (B, 3, T, H, W).

    Returns:
        dict with ``shape``, ``dtype``, ``channels_identical``, ``mean`` and ``std`` over the
        frames inside the videos (near 0 and 1 for normalised clips, a little lower with the
        augmentation border), ``min``, ``max``, ``clips_padded``, ``frames_padded``,
        ``padding_is_zero`` (every frame past the end is 0) and ``ef_range``.
    """
    clip = batch["clip"]
    past_end = _past_end(batch, period)
    inside = clip[:, 0][~past_end]  # (n_frames_inside, H, W)
    ef = batch["ef"]
    return {
        "shape": tuple(clip.shape),
        "dtype": str(clip.dtype),
        "channels_identical": bool(
            torch.equal(clip[:, 0], clip[:, 1]) and torch.equal(clip[:, 0], clip[:, 2])
        ),
        "mean": float(inside.mean()) if inside.numel() else float("nan"),
        "std": float(inside.std()) if inside.numel() > 1 else float("nan"),
        "min": float(clip.min()),
        "max": float(clip.max()),
        "clips_padded": int(batch["padded"].sum()),
        "frames_padded": int(past_end.sum()),
        "padding_is_zero": bool((clip[:, 0][past_end] == 0).all()),
        "ef_range": (float(ef.min()), float(ef.max())),
    }


def batch_grid(
    batch: dict, stats: dict[str, float], period: int, max_frames: int = 16, title: str = ""
) -> Figure:
    """One row per clip, up to ``max_frames`` evenly spaced frames per row, in grey levels.

    Each tile is titled with its raw frame index (``pad`` if past the video's end); each row is
    labelled with the file name, EF, clip start and video length.

    Args:
        batch: a collated ``EchoClipDataset`` batch; ``clip`` is (B, 3, T, H, W).
    """
    clips = batch["clip"]
    b, _, t = clips.shape[:3]
    cols = np.unique(np.linspace(0, t - 1, min(t, max_frames)).round().astype(int))
    grey = denormalise(clips[:, 0], stats["mean"], stats["std"])  # (B, T, H, W) uint8
    past_end = _past_end(batch, period)
    fig = Figure(figsize=(0.9 * len(cols) + 1.2, 1.0 * b + 0.6), layout="constrained")
    axes = fig.subplots(b, len(cols), squeeze=False)
    for r in range(b):
        start = int(batch["start"][r])
        for c, k in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(grey[r, k], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{start + period * k}{' pad' if past_end[r, k] else ''}", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(
            f"{batch['file_name'][r]}\nEF {float(batch['ef'][r]):.1f}\n"
            f"start {start} / {int(batch['n_frames'][r])}",
            fontsize=6,
            rotation=0,
            ha="right",
            va="center",
        )
    if title:
        fig.suptitle(title, fontsize=9)
    return fig
