"""EchoNet clips from the memory-mapped frame cache (task A3).

``load_clip`` is the contract function B calls (spec, "Functions A provides").
``EchoClipDataset`` serves:

- ``"train"``: one clip per video and epoch, random start and pad-and-crop augmentation;
- ``"M1"``: one centred clip per video;
- ``"M2"``: every M2 clip of every video, one item each; the caller averages per ``file_name``.

Which videos: every video of the split that was cached and has tracings. The 6 untraced videos
are left out of training and evaluation, as in EchoNet's loader (D-27). Videos whose tracings are
irregular (``has_trace`` False, D-26) stay in: their EF labels are fine.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from echoef.contracts import FRAME_SIZE, MODES, SPLITS
from echoef.data.layout import INDEX_FILE, STATS_FILE, TRACINGS_FILE, video_path
from echoef.data.sampling import (
    M2_OVERLAP,
    clip_indices,
    clip_starts,
    is_padded,
    m1_start,
    random_start,
)
from echoef.data.transforms import normalise, pad_crop, to_channels

log = logging.getLogger(__name__)

TRAIN_MODE = "train"
DATASET_MODES = (TRAIN_MODE, *MODES)
IN_CHANNELS = 3


def load_stats(processed_root: Path) -> dict[str, float]:
    """Training-set pixel mean and SD in grey levels: ``{"mean": float, "std": float}``."""
    stats = json.loads((Path(processed_root) / STATS_FILE).read_text())
    return {"mean": float(stats["mean"]), "std": float(stats["std"])}


def load_videos(processed_root: Path, split: str) -> pd.DataFrame:
    """Index rows (``contracts.INDEX_COLUMNS``) of the videos of ``split`` that are used.

    Used = cached (``n_frames > 0``) and listed in ``tracings.parquet`` (D-27). Row order is
    ``index.parquet``'s (FileList.csv's) order.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
    root = Path(processed_root)
    index = pd.read_parquet(root / INDEX_FILE)
    traced = set(pd.read_parquet(root / TRACINGS_FILE, columns=["FileName"])["FileName"])
    in_split = index["Split"] == split
    used = in_split & (index["n_frames"] > 0) & index["FileName"].isin(traced)
    log.info("%s: %d of %d videos used (cached, with tracings)", split, used.sum(), in_split.sum())
    return index[used].reset_index(drop=True)


def clip_from_frames(
    frames: np.ndarray, T: int, period: int, start: int, mean: float, std: float
) -> np.ndarray:
    """Read and normalise one clip from a video's frames.

    Args:
        frames: uint8 array (F, H, W), e.g. a memory-mapped cache file; only the frames the clip
            needs are read.

    Returns:
        float32 array (T, H, W); frames at or past the video's end are 0 (the mean grey).
    """
    n = len(frames)
    idx = clip_indices(start, T, period)
    n_valid = int((idx < n).sum())
    clip = np.zeros((T, *frames.shape[1:]), dtype=np.float32)
    if n_valid:
        stop = int(idx[n_valid - 1]) + 1
        clip[:n_valid] = normalise(np.asarray(frames[start:stop:period]), mean, std)
    return clip


def load_clip(
    file_name: str,
    T: int,
    period: int,
    start: int,
    processed_root: Path,
    stats: dict[str, float],
) -> torch.Tensor:
    """One normalised clip from the frame cache. Contract function.

    Returns:
        float32 tensor (3, T, 112, 112); frames at or past the video's end are zeros.
    """
    frames = np.load(video_path(processed_root, file_name), mmap_mode="r")
    clip = clip_from_frames(frames, T, period, start, stats["mean"], stats["std"])
    return to_channels(torch.from_numpy(clip), IN_CHANNELS)


class EchoClipDataset(Dataset):
    """Clips of a set of videos for training or for M1/M2 evaluation.

    Each item is a dict:

    - ``clip``: float32 tensor (3, T, 112, 112), normalised;
    - ``ef``: float32 tensor (), reference EF in %;
    - ``file_name`` (str), ``start`` (int, first raw frame), ``n_frames`` (int, video length);
    - ``padded`` (bool): the clip holds zero frames past the video's end.

    The default collate turns a batch into ``clip`` (B, 3, T, 112, 112), ``ef`` (B,), lists of
    file names, and int64/bool tensors (B,) for the rest.
    """

    def __init__(
        self,
        videos: pd.DataFrame,
        processed_root: Path,
        stats: dict[str, float],
        T: int,
        period: int,
        mode: str,
        *,
        random_start: bool = True,
        pad_crop: int = 0,
        m2_overlap: float = M2_OVERLAP,
    ) -> None:
        """
        Args:
            videos: index rows, e.g. from ``load_videos``; needs FileName, EF and n_frames.
            mode: ``"train"``, ``"M1"`` or ``"M2"``.
            random_start: train mode only; False uses the M1 start.
            pad_crop: train mode only; augmentation border in pixels (0 = none).
            m2_overlap: M2 mode only; overlap between neighbouring clips.
        """
        if mode not in DATASET_MODES:
            raise ValueError(f"mode must be one of {DATASET_MODES}; got {mode!r}")
        if mode != TRAIN_MODE and pad_crop:
            raise ValueError("pad_crop augmentation is for train mode only")
        self.processed_root = Path(processed_root)
        self.stats = stats
        self.T, self.period, self.mode = T, period, mode
        self.random_start, self.pad_crop = random_start, pad_crop
        # Plain lists: cheap to read from forked DataLoader workers.
        self.file_names = videos["FileName"].astype(str).tolist()
        self.efs = videos["EF"].astype(float).tolist()
        self.n_frames = videos["n_frames"].astype(int).tolist()
        if any(n < 1 for n in self.n_frames):
            raise ValueError("every video needs n_frames >= 1 (a cached video)")
        if mode == TRAIN_MODE:
            self.items = [(v, -1) for v in range(len(self.file_names))]  # start drawn per item
        else:
            self.items = [
                (v, s)
                for v, n in enumerate(self.n_frames)
                for s in clip_starts(n, T, period, mode, m2_overlap)
            ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        v, start = self.items[i]
        n = self.n_frames[v]
        if self.mode == TRAIN_MODE:
            start = (
                random_start(n, self.T, self.period)
                if self.random_start
                else m1_start(n, self.T, self.period)
            )
        frames = np.load(video_path(self.processed_root, self.file_names[v]), mmap_mode="r")
        if len(frames) != n:
            raise ValueError(f"{self.file_names[v]}: cache has {len(frames)} frames, index {n}")
        clip = torch.from_numpy(
            clip_from_frames(
                frames, self.T, self.period, start, self.stats["mean"], self.stats["std"]
            )
        )
        if self.mode == TRAIN_MODE:
            clip = pad_crop(clip, self.pad_crop)
        return {
            "clip": to_channels(clip, IN_CHANNELS),
            "ef": torch.tensor(self.efs[v], dtype=torch.float32),
            "file_name": self.file_names[v],
            "start": start,
            "n_frames": n,
            "padded": is_padded(start, self.T, self.period, n),
        }


def check_data_config(cfg: DictConfig) -> None:
    """Refuse ``data`` settings this loader does not implement, instead of ignoring them."""
    data = cfg.data
    if data.frame_size != FRAME_SIZE:
        raise ValueError(f"data.frame_size {data.frame_size}: the cache is {FRAME_SIZE} x 112")
    if data.in_channels != IN_CHANNELS:
        raise ValueError(f"data.in_channels {data.in_channels}: the model takes {IN_CHANNELS}")
    if data.short_video_padding != "zeros":
        raise ValueError(f"data.short_video_padding {data.short_video_padding!r}: only 'zeros'")
    if data.augment.hflip:
        raise ValueError("data.augment.hflip is not supported: it moves the LV to the wrong side")


def dataset_from_config(
    cfg: DictConfig,
    split: str,
    mode: str,
    processed_root: Path,
    stats: dict[str, float],
) -> EchoClipDataset:
    """The dataset for one split and mode, with T, period, augmentation and overlap from ``cfg``.

    Augmentation (random start, pad-and-crop) applies in train mode only.
    """
    check_data_config(cfg)
    train = mode == TRAIN_MODE
    return EchoClipDataset(
        load_videos(processed_root, split),
        processed_root,
        stats,
        int(cfg.T),
        int(cfg.period),
        mode,
        random_start=train and bool(cfg.data.augment.random_start),
        pad_crop=int(cfg.data.augment.pad_crop) if train else 0,
        m2_overlap=float(cfg.eval.m2_overlap),
    )
