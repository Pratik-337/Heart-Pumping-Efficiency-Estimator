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
    text, _, fig, strip, animation = run(str(_avi(tmp_path / "v.avi", 30, 112, 112)), False)
    assert text.startswith("## EF") and fig is not None and strip is None and animation is None
    assert app_module.build_ui(run, []) is not None


def test_ig_overlay_strip(runs, monkeypatch):
    ens = inf.load_ensemble(runs[:1], "cpu")
    monkeypatch.setattr(inf, "integrated_gradients", lambda *a, **k: np.random.rand(8, 112, 112))
    strip = inf.ig_overlay(ens, np.zeros((40, 112, 112), np.uint8), STATS)
    assert strip.shape == (112, 8 * 112, 3) and strip.dtype == np.uint8


def test_ig_strip_and_animation_share_one_computation(runs, monkeypatch, tmp_path):
    import gradio as gr
    from PIL import Image

    import app as app_module

    ens = inf.load_ensemble(runs[:1], "cpu")
    calls = []

    def fake_ig(*args, **kwargs):
        calls.append(1)
        return np.random.rand(8, 112, 112)

    monkeypatch.setattr(inf, "integrated_gradients", fake_ig)
    run = app_module.make_runner(ens, STATS, None, tmp_path)
    _, _, _, strip, animation = run(str(_avi(tmp_path / "v.avi", 40, 112, 112)), True)
    assert len(calls) == 1 and strip.shape == (112, 8 * 112, 3)
    gif = Image.open(animation)
    assert gif.n_frames == 8 and gif.size == (2 * 336, 336)  # every frame, input | overlay
    assert gif.info["duration"] == 40  # period 2 at 50 fps
    assert Path(animation).parent == tmp_path
    shown = gr.Image(type="filepath").postprocess(animation)  # served as a file: stays animated
    assert shown is not None and str(shown.path).endswith(".gif")


def test_video_info_and_ed_es_marks(runs, tmp_path):
    pd.DataFrame({"FileName": ["0XT", "0XV", "0XI"], "Split": ["TEST", "VAL", "TEST"],
                  "EF": [41.5, 60.0, 30.0], "ed_frame": [10, 12, -1], "es_frame": [24, 30, -1],
                  "has_trace": [True, True, False]}).to_parquet(tmp_path / INDEX_FILE)  # fmt: skip
    assert inf.video_info("0XT.avi", tmp_path) == {"reference": 41.5, "ed": 10, "es": 24}
    assert inf.video_info("0XV.avi", tmp_path) == {"reference": None, "ed": 12, "es": 30}
    assert inf.video_info("0XI.avi", tmp_path) == {"reference": 30.0, "ed": None, "es": None}
    assert inf.video_info("mine.mp4", tmp_path)["reference"] is None
    ens = inf.load_ensemble(runs[:1], "cpu")
    frames = np.zeros((40, 112, 112), np.uint8)  # M1 clip: frames 12..26 at period 2
    fig = inf.importance_figure(ens, frames, STATS, ed=14, es=99)
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert labels == ["ED"]  # ES lies outside the clip's window: not drawn


def test_demo_runs_follow_the_validation_choice(tmp_path):
    runs_root, results = tmp_path / "runs", tmp_path / "results"
    for seed, done in ((0, True), (1, True), (2, False)):
        (runs_root / "fw16" / f"seed{seed}").mkdir(parents=True)
        if done:
            (runs_root / "fw16" / f"seed{seed}" / "DONE").touch()
    with pytest.raises(FileNotFoundError):
        inf.demo_runs(results, runs_root)
    (results / "tables").mkdir(parents=True)
    pd.DataFrame({"config_id": ["fs8", "fw16"], "chosen": [False, True]}).to_csv(
        results / "tables" / "demo_choice.csv", index=False
    )
    assert inf.demo_runs(results, runs_root) == [
        runs_root / "fw16" / "seed0",
        runs_root / "fw16" / "seed1",
    ]
