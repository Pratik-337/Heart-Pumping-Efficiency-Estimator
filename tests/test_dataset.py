"""Clip loading and EchoClipDataset on a tiny synthetic processed folder (CPU only).

Every pixel of frame k holds the grey level k, so with ``mean = 0, std = 1`` a clip's values
are the raw frame indices it read.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from echoef.config import load_config
from echoef.contracts import INDEX_COLUMNS, TRACINGS_COLUMNS
from echoef.data import dataset as ds
from echoef.data.layout import INDEX_FILE, STATS_FILE, TRACINGS_FILE, video_path
from echoef.data.sampling import clip_starts
from echoef.utils import seed_worker

IDENTITY = {"mean": 0.0, "std": 1.0}
# name: (split, n_frames in the index, EF, has tracings); n_frames 0 = not cached
VIDEOS = {
    "tr_long": ("TRAIN", 200, 60.0, True),
    "tr_short": ("TRAIN", 20, 35.0, True),
    "tr_untraced": ("TRAIN", 50, 50.0, False),  # D-27: left out
    "tr_uncached": ("TRAIN", 0, 55.0, True),  # e.g. a wrong frame size: left out
    "va_a": ("VAL", 173, 45.0, True),
    "va_b": ("VAL", 90, 70.0, True),
    "te_a": ("TEST", 120, 52.0, True),
}


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "processed"
    video_path(root, "x").parent.mkdir(parents=True)
    index, tracings = [], []
    for name, (split, n, ef, traced) in VIDEOS.items():
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


def test_load_videos_keeps_cached_traced_videos_of_the_split(root):
    assert ds.load_videos(root, "TRAIN")["FileName"].tolist() == ["tr_long", "tr_short"]
    assert ds.load_videos(root, "VAL")["FileName"].tolist() == ["va_a", "va_b"]
    assert list(ds.load_videos(root, "TEST").columns) == INDEX_COLUMNS
    with pytest.raises(ValueError):
        ds.load_videos(root, "train")


def test_load_stats(root):
    assert ds.load_stats(root) == IDENTITY


def test_load_clip_reads_strided_frames_and_pads_with_zeros(root):
    clip = ds.load_clip("tr_long", 8, 2, 10, root, IDENTITY)
    assert clip.shape == (3, 8, 112, 112) and clip.dtype == torch.float32
    assert torch.equal(clip[0], clip[1]) and torch.equal(clip[0], clip[2])
    assert clip[0, :, 0, 0].tolist() == [10, 12, 14, 16, 18, 20, 22, 24]

    short = ds.load_clip("tr_short", 8, 4, 4, root, IDENTITY)  # reads 4, 8, ..., 32 of 20 frames
    assert short[0, :, 5, 5].tolist() == [4, 8, 12, 16, 0, 0, 0, 0]
    assert (short[:, 4:] == 0).all()


def test_load_clip_normalises(root):
    stats = {"mean": 10.0, "std": 4.0}
    clip = ds.load_clip("tr_long", 4, 1, 10, root, stats)
    np.testing.assert_allclose(clip[0, :, 0, 0].numpy(), [0, 0.25, 0.5, 0.75])
    padded = ds.load_clip("tr_short", 4, 8, 0, root, stats)  # frames 0, 8, 16, 24 of 20
    assert padded[0, 3].abs().max() == 0  # padding is the mean grey, not black


def _dataset(root, split, mode, **kw):
    return ds.EchoClipDataset(ds.load_videos(root, split), root, IDENTITY, 8, 4, mode, **kw)


def test_m1_and_m2_items(root):
    m1 = _dataset(root, "VAL", "M1")
    assert len(m1) == 2
    item = m1[0]
    assert item["file_name"] == "va_a" and item["start"] == clip_starts(173, 8, 4, "M1")[0]
    assert item["ef"].dtype == torch.float32 and item["ef"].item() == 45.0
    assert item["clip"][0, :, 0, 0].tolist() == [item["start"] + 4 * k for k in range(8)]

    m2 = _dataset(root, "VAL", "M2")
    starts = [m2[i]["start"] for i in range(len(m2))]
    expected = clip_starts(173, 8, 4, "M2") + clip_starts(90, 8, 4, "M2")
    assert starts == expected and len(m2) == len(expected)


def test_padded_flag(root):
    short = _dataset(root, "TRAIN", "M1")[1]  # 20 frames, window 32: reads frames up to 28
    assert short["file_name"] == "tr_short" and short["padded"]
    assert not _dataset(root, "TRAIN", "M1")[0]["padded"]


def test_train_items_are_random_augmented_and_reproducible(root):
    data = _dataset(root, "TRAIN", "train", pad_crop=12)
    torch.manual_seed(0)
    first = [data[0] for _ in range(30)]
    starts = {x["start"] for x in first}
    assert len(starts) > 5 and all(0 <= x <= 200 - 32 for x in starts)
    assert any((x["clip"][0, 0] == 0).any() for x in first)  # some crops show the zero border
    torch.manual_seed(0)
    again = [data[0] for _ in range(30)]
    assert all(torch.equal(a["clip"], b["clip"]) for a, b in zip(first, again, strict=True))

    fixed = _dataset(root, "TRAIN", "train", random_start=False)
    assert fixed[0]["start"] == clip_starts(200, 8, 4, "M1")[0]


# Fork-based workers are the spec's choice (Tech stack, OS row); Python 3.12 warns about forking
# a process that has started threads (torch's thread pool), which PyTorch's workers handle.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_dataloader_batches_are_reproducible_with_workers(root):
    data = _dataset(root, "TRAIN", "train", pad_crop=12)

    def starts() -> list[int]:
        loader = DataLoader(
            data,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(1),
        )
        out = []
        for _ in range(3):  # epochs
            for batch in loader:
                assert batch["clip"].shape == (2, 3, 8, 112, 112)
                assert batch["ef"].shape == (2,) and len(batch["file_name"]) == 2
                out += batch["start"].tolist()
        return out

    assert starts() == starts()


def test_pad_crop_only_in_train_mode(root):
    with pytest.raises(ValueError):
        _dataset(root, "VAL", "M1", pad_crop=12)
    with pytest.raises(ValueError):
        _dataset(root, "VAL", "M3")


def test_cache_length_mismatch_is_caught(root):
    videos = ds.load_videos(root, "VAL").assign(n_frames=[170, 90])
    with pytest.raises(ValueError, match="cache has 173"):
        ds.EchoClipDataset(videos, root, IDENTITY, 8, 2, "M1")[0]


def test_dataset_from_config(root):
    cfg = load_config("fs8")
    train = ds.dataset_from_config(cfg, "TRAIN", "train", root, IDENTITY)
    assert (train.T, train.period, train.pad_crop, train.random_start) == (8, 2, 12, True)
    val = ds.dataset_from_config(cfg, "VAL", "M2", root, IDENTITY)
    assert val.pad_crop == 0 and len(val) == len(clip_starts(173, 8, 2, "M2")) + len(
        clip_starts(90, 8, 2, "M2")
    )
    with pytest.raises(ValueError, match="hflip"):
        ds.dataset_from_config(
            load_config("fs8", ["data.augment.hflip=true"]), "TRAIN", "train", root, IDENTITY
        )
