"""Shared test helpers: a tiny synthetic processed folder (no real data, no GPU).

Every pixel of frame k holds the grey level k, so with ``IDENTITY`` stats a clip's values are
the raw frame indices it read.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from echoef.contracts import INDEX_COLUMNS, TRACINGS_COLUMNS
from echoef.data.layout import INDEX_FILE, STATS_FILE, TRACINGS_FILE, video_path

IDENTITY = {"mean": 0.0, "std": 1.0}
# name: (split, n_frames, EF, has tracings)
SMALL_VIDEOS = {
    "tr_a": ("TRAIN", 120, 60.0, True),
    "tr_b": ("TRAIN", 40, 30.0, True),
    "va_a": ("VAL", 173, 45.0, True),
    "va_b": ("VAL", 28, 70.0, True),  # T = 8, period 4 reads frame 28: padded (D-28)
    "te_a": ("TEST", 100, 52.0, True),
}


def make_processed(root: Path, videos: dict = SMALL_VIDEOS) -> Path:
    """Write videos/*.npy, index.parquet, tracings.parquet and identity stats.json under root."""
    video_path(root, "x").parent.mkdir(parents=True, exist_ok=True)
    index, tracings = [], []
    for name, (split, n, ef, traced) in videos.items():
        if n:
            frames = np.broadcast_to(np.arange(n, dtype=np.uint8)[:, None, None], (n, 112, 112))
            np.save(video_path(root, name), np.ascontiguousarray(frames))
        index.append((name, split, ef, 100.0, 100.0 - ef, 50, n, -1, -1, False))
        if traced:
            tracings.append((name, 0, 10.0, 10.0, 20.0, 20.0))
    pd.DataFrame(index, columns=INDEX_COLUMNS).to_parquet(root / INDEX_FILE, index=False)
    pd.DataFrame(tracings, columns=TRACINGS_COLUMNS).to_parquet(root / TRACINGS_FILE, index=False)
    (root / STATS_FILE).write_text(json.dumps(IDENTITY))
    return root


@pytest.fixture
def processed(tmp_path) -> Path:
    return make_processed(tmp_path / "processed")


class MeanModel(nn.Module):
    """(B, 3, T, H, W) -> (B,): the clip's mean value, i.e. the mean frame index it read."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))  # so .to(device) and .parameters() work

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0].mean(dim=(1, 2, 3)) + self.dummy
