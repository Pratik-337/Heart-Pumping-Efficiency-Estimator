"""Inference: M1/M2 predictions per video and per split, seed ensembles (task A5).

Spec, "Inference modes": M1 scores one centred clip per video; M2 averages the predictions of
clips every W / 2 frames. Predictions are written in the contract format
(``contracts.PREDICTION_COLUMNS``), one row per video; ``padded`` is 1 when any clip of the video
reads past its end (D-28).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from echoef.contracts import MODES, PREDICTION_COLUMNS, SPLITS
from echoef.data.dataset import EchoClipDataset, load_clip
from echoef.data.layout import video_path
from echoef.data.sampling import clip_starts
from echoef.train.steps import predict_batch

log = logging.getLogger(__name__)

ENSEMBLE_SEED = "ens"  # seed label of the seed ensemble in all_runs.csv (spec, "Files on disk")


def predict_clips(
    model: nn.Module, clips: torch.Tensor, device: torch.device, precision: str, batch_size: int
) -> torch.Tensor:
    """Predictions for a stack of clips, in batches.

    Args:
        clips: (N, 3, T, 112, 112) normalised clips on the CPU.

    Returns:
        (N,) predicted EF in %, FP32, on the CPU.
    """
    model.eval()
    out = [
        predict_batch(model, clips[i : i + batch_size].to(device), precision).cpu()
        for i in range(0, len(clips), batch_size)
    ]
    return torch.cat(out)


def predict_video(
    model: nn.Module,
    file_name: str,
    T: int,
    period: int,
    mode: str,
    processed_root: Path,
    stats: dict[str, float],
    device: torch.device | str,
    precision: str = "bf16",
    batch_size: int = 8,
) -> tuple[float, int]:
    """EF of one cached video: the mean over its M1 or M2 clips. Contract function.

    Returns:
        ``(ef_pred, n_clips)``.
    """
    device = torch.device(device)
    n_frames = len(np.load(video_path(processed_root, file_name), mmap_mode="r"))
    starts = clip_starts(n_frames, T, period, mode)
    clips = torch.stack([load_clip(file_name, T, period, s, processed_root, stats) for s in starts])
    preds = predict_clips(model, clips, device, precision, batch_size)
    return float(preds.mean()), len(starts)


def predict_split(
    model: nn.Module,
    videos: pd.DataFrame,
    split: str,
    mode: str,
    *,
    config_id: str,
    seed: int,
    T: int,
    period: int,
    processed_root: Path,
    stats: dict[str, float],
    device: torch.device,
    precision: str,
    batch_size: int,
    workers: int = 0,
    m2_overlap: float | None = None,
) -> pd.DataFrame:
    """Predictions for every video in ``videos``, one row each, in ``PREDICTION_COLUMNS``.

    All clips of all videos go through the model in batches (a video's M2 clips may span
    batches); a video's prediction is the mean over its clips. Row order follows ``videos``.
    """
    if split not in SPLITS or mode not in MODES:
        raise ValueError(f"split {split!r} / mode {mode!r}: expected one of {SPLITS} / {MODES}")
    extra = {} if m2_overlap is None else {"m2_overlap": m2_overlap}
    data = EchoClipDataset(videos, processed_root, stats, T, period, mode, **extra)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    model.eval()
    names, preds, padded = [], [], []
    for batch in loader:
        preds.append(predict_batch(model, batch["clip"].to(device, non_blocking=True), precision))
        names += batch["file_name"]
        padded.append(batch["padded"])
    clips = pd.DataFrame(
        {
            "file_name": names,
            "ef_pred": torch.cat(preds).cpu().numpy().astype(np.float64),
            "padded": torch.cat(padded).numpy(),
        }
    )
    per_video = clips.groupby("file_name", sort=False).agg(
        ef_pred=("ef_pred", "mean"), n_clips=("ef_pred", "size"), padded=("padded", "any")
    )
    out = pd.DataFrame({"file_name": videos["FileName"].astype(str).to_numpy()})
    out["split"], out["mode"], out["config_id"], out["seed"] = split, mode, config_id, seed
    out["T"], out["period"] = T, period
    out["ef_true"] = videos["EF"].astype(float).to_numpy()
    out = out.join(per_video, on="file_name")
    out["padded"] = out["padded"].astype(int)
    if out["ef_pred"].isna().any():
        raise RuntimeError("some videos got no prediction")
    return out[PREDICTION_COLUMNS]


def mae(predictions: pd.DataFrame) -> float:
    """Mean absolute error in EF points over the rows of a predictions table."""
    return float((predictions["ef_pred"] - predictions["ef_true"]).abs().mean())


def ensemble(predictions: list[pd.DataFrame]) -> pd.DataFrame:
    """Seed ensemble of one config: per video, the mean ``ef_pred`` over the given runs.

    All tables must hold the same videos, split, mode and config; ``seed`` becomes ``"ens"`` and
    ``padded`` is 1 if it is 1 in any run.
    """
    if not predictions:
        raise ValueError("no predictions to ensemble")
    keys = ["file_name", "split", "mode", "config_id", "T", "period", "ef_true", "n_clips"]
    stacked = pd.concat(predictions, ignore_index=True)
    for col in ("split", "mode", "config_id"):
        if stacked[col].nunique() != 1:
            raise ValueError(f"cannot ensemble across different {col} values")
    counts = stacked.groupby("file_name").size()
    if (counts != len(predictions)).any():
        raise ValueError("the runs do not cover the same videos")
    out = stacked.groupby(keys, sort=False, as_index=False).agg(
        ef_pred=("ef_pred", "mean"), padded=("padded", "max")
    )
    out["seed"] = ENSEMBLE_SEED
    return out[PREDICTION_COLUMNS]
