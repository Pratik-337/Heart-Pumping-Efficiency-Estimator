"""M1/M2 prediction per video and per split, ensembles (CPU, synthetic data)."""

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import IDENTITY, MeanModel

from echoef.contracts import PREDICTION_COLUMNS
from echoef.data.dataset import load_videos
from echoef.data.sampling import clip_indices, clip_starts
from echoef.eval import predict as p

CPU = torch.device("cpu")
T, PERIOD = 8, 4  # window 32


def _expected(n_frames: int, mode: str) -> tuple[float, int]:
    """Mean over clips of the mean frame index read (frames past the end count as 0)."""
    starts = clip_starts(n_frames, T, PERIOD, mode)
    means = [np.where((i := clip_indices(s, T, PERIOD)) < n_frames, i, 0).mean() for s in starts]
    return float(np.mean(means)), len(starts)


def _predict(processed, mode, **kw):
    videos = load_videos(processed, "VAL")
    return p.predict_split(
        MeanModel(), videos, "VAL", mode, config_id="fs16", seed=1, T=T, period=PERIOD,
        processed_root=processed, stats=IDENTITY, device=CPU, precision="fp32",
        batch_size=3, **kw,
    )  # fmt: skip


@pytest.mark.parametrize("mode", ["M1", "M2"])
def test_predict_split_averages_clips_per_video(processed, mode):
    out = _predict(processed, mode)
    assert list(out.columns) == PREDICTION_COLUMNS
    assert out["file_name"].tolist() == ["va_a", "va_b"]
    for row, n in zip(out.itertuples(), (173, 28), strict=True):
        ef, n_clips = _expected(n, mode)
        assert row.ef_pred == pytest.approx(ef, abs=1e-4) and row.n_clips == n_clips
    assert out["padded"].tolist() == [0, 1]  # 28 frames: the clip reads frame 28
    assert (out["split"] == "VAL").all() and (out["mode"] == mode).all()
    assert (out["config_id"] == "fs16").all() and (out["seed"] == 1).all()
    assert out["ef_true"].tolist() == [45.0, 70.0]


def test_predict_video_matches_predict_split(processed):
    row = _predict(processed, "M2").iloc[0]
    ef, n_clips = p.predict_video(MeanModel(), "va_a", T, PERIOD, "M2", processed, IDENTITY, "cpu",
                                  precision="fp32", batch_size=2)  # fmt: skip
    assert ef == pytest.approx(row.ef_pred, abs=1e-4) and n_clips == row.n_clips


def test_mae():
    df = pd.DataFrame({"ef_pred": [50.0, 60.0], "ef_true": [55.0, 58.0]})
    assert p.mae(df) == pytest.approx(3.5)


def test_ensemble(processed):
    a = _predict(processed, "M1")
    b = a.assign(seed=2, ef_pred=a["ef_pred"] + 2.0)
    ens = p.ensemble([a, b])
    assert list(ens.columns) == PREDICTION_COLUMNS
    assert (ens["seed"] == p.ENSEMBLE_SEED).all()
    np.testing.assert_allclose(ens["ef_pred"], a["ef_pred"] + 1.0)
    with pytest.raises(ValueError):
        p.ensemble([a, b.iloc[:1]])
    with pytest.raises(ValueError):
        p.ensemble([a, b.assign(mode="M2")])
