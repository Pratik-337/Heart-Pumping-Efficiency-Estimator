"""Demo app on the CPU: video reading, seed-ensemble EF, reports and the UI (task B8).

Uses fake checkpoints (random weights, T = 8) and synthetic videos; no real data.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import inference as inf  # noqa: E402

from echoef.data.layout import INDEX_FILE  # noqa: E402
from echoef.data.sampling import clip_starts  # noqa: E402
from echoef.models.factory import build_model, save_checkpoint  # noqa: E402

STATS = {"mean": 30.0, "std": 40.0}


def _avi(path: Path, n: int, h: int, w: int, colour: bool = False) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        r = int(min(h, w) / 6 * (1.5 + np.sin(2 * np.pi * i / 20)))
        color = (40, 200, 60) if colour else (180, 180, 180)
        cv2.circle(frame, (w // 2, h // 2), r, color, -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    paths = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        run = root / "fs8" / f"seed{seed}"
        run.mkdir(parents=True)
        config = {"config_id": "fs8", "seed": seed, "T": 8, "period": 2,
                  "train": {"precision": "bf16", "micro_batch": 8}}  # fmt: skip
        save_checkpoint(run / "best.pt", build_model(False, 55.0), config, 0, 5.0)
        paths.append(run)
    return paths


def test_read_video_crops_and_warns(tmp_path):
    frames, warnings = inf.read_video(_avi(tmp_path / "wide.avi", 30, 96, 160))
    assert frames.shape == (30, 112, 112) and frames.dtype == np.uint8
    assert any("out of distribution" in w for w in warnings)
    frames, warnings = inf.read_video(_avi(tmp_path / "ok.avi", 30, 112, 112))
    assert frames.shape == (30, 112, 112) and warnings == []
    _, warnings = inf.read_video(_avi(tmp_path / "colour.avi", 30, 112, 112, colour=True))
    assert any("colour" in w for w in warnings)
    (tmp_path / "junk.avi").write_bytes(b"not a video")
    with pytest.raises(ValueError):
        inf.read_video(tmp_path / "junk.avi")


def test_ensemble_prediction(runs, tmp_path):
    ens = inf.load_ensemble(runs, "cpu")
    assert (ens.config_id, ens.T, ens.period, ens.seeds) == ("fs8", 8, 2, [0, 1])
    assert ens.precision == "fp32"  # bf16 only on the GPU
    frames, _ = inf.read_video(_avi(tmp_path / "v.avi", 40, 112, 112))
    out = inf.predict(ens, frames, STATS)
    assert out["n_clips"] == len(clip_starts(40, 8, 2, "M2"))
    assert out["ef_min"] <= out["ef"] <= out["ef_max"] and len(out["per_seed"]) == 2
    text, notes = inf.report(ens, out, [], None, len(frames))
    assert "EF band" in text and inf.NO_REFERENCE in text and "Seed range" in text
    assert notes == "No warnings."
    assert len(inf.importance_figure(ens, frames, STATS).axes) == 1
    short, _ = inf.read_video(_avi(tmp_path / "s.avi", 10, 112, 112))
    out = inf.predict(ens, short, STATS)
    assert out["padded"] and "padded" in inf.report(ens, out, [], None, len(short))[1]


def test_ef_band_and_reference(tmp_path):
    assert inf.ef_band(35).startswith("EF band: reduced")
    assert "mildly" in inf.ef_band(45) and "preserved" in inf.ef_band(60)
    pd.DataFrame({"FileName": ["0XABC", "0XDEF"], "Split": ["TEST", "TRAIN"],
                  "EF": [41.5, 60.0]}).to_parquet(tmp_path / INDEX_FILE)  # fmt: skip
    assert inf.reference_ef("0XABC.avi", tmp_path) == 41.5
    assert inf.reference_ef("0XDEF.avi", tmp_path) is None  # not a test video
    assert inf.reference_ef("mine.mp4", None) is None


def test_ui_builds_and_runner_handles_bad_input(runs, tmp_path):
    import app as app_module

    ens = inf.load_ensemble(runs[:1], "cpu")
    run = app_module.make_runner(ens, STATS, None)
    assert run(None, False)[0] == "Upload a video first."
    (tmp_path / "bad.mp4").write_bytes(b"\x00" * 100)
    assert "could not be decoded" in run(str(tmp_path / "bad.mp4"), False)[0]
    text, _, fig, overlay = run(str(_avi(tmp_path / "v.avi", 30, 112, 112)), False)
    assert text.startswith("## EF") and fig is not None and overlay is None
    assert app_module.build_ui(run, []) is not None


def test_ig_overlay_strip(runs, monkeypatch):
    ens = inf.load_ensemble(runs[:1], "cpu")
    monkeypatch.setattr(inf, "integrated_gradients", lambda *a, **k: np.random.rand(8, 112, 112))
    strip = inf.ig_overlay(ens, np.zeros((40, 112, 112), np.uint8), STATS)
    assert strip.shape == (112, 8 * 112, 3) and strip.dtype == np.uint8
