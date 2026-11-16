"""Batch grid and numeric batch checks on a synthetic batch (CPU only)."""

import pytest
import torch

from echoef.data.grids import batch_checks, batch_grid
from echoef.data.transforms import to_channels

STATS = {"mean": 30.0, "std": 50.0}


def _batch(starts=(0, 10), n_frames=(40, 14), T=8, period=2) -> dict:
    """Clips whose frames past the video's end are zero, as the loader makes them."""
    clips = []
    for s, n in zip(starts, n_frames, strict=True):
        grey = torch.randn(T, 6, 6)
        idx = s + period * torch.arange(T)
        grey[idx >= n] = 0
        clips.append(to_channels(grey))
    return {
        "clip": torch.stack(clips),
        "ef": torch.tensor([60.0, 35.0]),
        "file_name": ["a", "b"],
        "start": torch.tensor(starts),
        "n_frames": torch.tensor(n_frames),
        "padded": torch.tensor(
            [s + period * (T - 1) >= n for s, n in zip(starts, n_frames, strict=True)]
        ),
    }


def test_batch_checks():
    checks = batch_checks(_batch(), period=2)
    assert checks["shape"] == (2, 3, 8, 6, 6)
    assert checks["channels_identical"] and checks["padding_is_zero"]
    assert checks["clips_padded"] == 1
    assert checks["frames_padded"] == 6  # clip b reads 10, 12, 14, ...: 6 frames >= 14
    assert checks["ef_range"] == (35.0, 60.0)
    assert checks["mean"] == pytest.approx(0.0, abs=0.3)


def test_batch_checks_spots_unequal_channels():
    batch = _batch()
    batch["clip"][0, 2] += 1
    assert not batch_checks(batch, period=2)["channels_identical"]


def test_batch_grid_layout():
    fig = batch_grid(_batch(), STATS, period=2, max_frames=4, title="t")
    axes = fig.get_axes()
    assert len(axes) == 2 * 4
    titles = [ax.get_title() for ax in axes]
    assert titles[:4] == ["0", "4", "10", "14"]  # frames 0, 2, 5, 7 of clip a
    assert titles[4:] == ["10", "14 pad", "20 pad", "24 pad"]
