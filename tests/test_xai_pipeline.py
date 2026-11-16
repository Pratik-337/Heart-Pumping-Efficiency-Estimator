"""XAI sample, per-video rows, summary, sanity check and figures (CPU, synthetic data)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from echoef.contracts import INDEX_COLUMNS, TRACINGS_COLUMNS, XAI_COLUMNS
from echoef.data.layout import INDEX_FILE, STATS_FILE, TRACINGS_FILE, video_path
from echoef.models.factory import build_model
from echoef.xai import pipeline as pl
from echoef.xai.render import example_panel, summary_chart

STATS = {"mean": 30.0, "std": 40.0}
CPU = torch.device("cpu")


def _rect_rows(x0, x1, y0, h):
    return [(20.0, y0 - 5, 20.0, y0 + h + 5)] + [(x0, y0 + k, x1, y0 + k) for k in range(h + 1)]


@pytest.fixture
def processed(tmp_path):
    root = tmp_path / "processed"
    video_path(root, "x").parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    index, tracings = [], []
    # name: (EF, ED frame, ES frame); FS-8 window is 16 frames, so "far" cannot fit ED and ES
    for name, (ef, ed, es) in {"v_low": (30.0, 10, 18), "v_mid": (45.0, 20, 30),
                               "v_high": (60.0, 5, 12), "far": (65.0, 3, 40)}.items():  # fmt: skip
        np.save(video_path(root, name), rng.integers(0, 255, (60, 112, 112), dtype=np.uint8))
        index.append((name, "VAL", ef, 100.0, 100.0 - ef, 50, 60, ed, es, True))
        for frame, half in ((ed, 30), (es, 20)):
            tracings += [(name, frame, *r) for r in _rect_rows(56 - half, 56 + half, 30, 40)]
    pd.DataFrame(index, columns=INDEX_COLUMNS).to_parquet(root / INDEX_FILE, index=False)
    pd.DataFrame(tracings, columns=TRACINGS_COLUMNS).to_parquet(root / TRACINGS_FILE, index=False)
    (root / STATS_FILE).write_text(json.dumps(STATS))
    return root


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(pretrained=False, head_bias=55.0).eval()


def test_draw_sample_is_stratified_and_reproducible():
    videos = pd.DataFrame({"FileName": [f"v{i}" for i in range(300)], "Split": "TEST",
                           "EF": np.linspace(10, 80, 300), "n_frames": 100, "ed_frame": 1,
                           "es_frame": 5, "has_trace": True})  # fmt: skip
    s = pl.draw_sample(videos, 100, seed=3)
    assert s["band"].value_counts().to_dict() == {"EF >= 50": 50, "EF <= 40": 35, "EF 41-49": 15}
    assert s.equals(pl.draw_sample(videos, 100, seed=3))


def test_explain_video_rows_and_skip(processed, model):
    videos = pd.read_parquet(processed / INDEX_FILE).set_index("FileName", drop=False)
    methods = ("occlusion", "gradcam")
    rows, example = pl.explain_video(model, videos.loc["v_low"], "fs8", 8, 2, processed, STATS,
                                     CPU, "fp32", methods)  # fmt: skip
    df = pl.metrics_frame(rows)
    assert list(df.columns[: len(XAI_COLUMNS)]) == XAI_COLUMNS and len(df) == 2
    occ = df.set_index("method").loc["occlusion"]
    assert 0 <= occ["hit"] <= 1 and occ["random_aopc"] >= 0 and np.isnan(occ["lv_energy_ratio"])
    cam = df.set_index("method").loc["gradcam"]
    assert cam["lv_energy_ratio"] > 0 and np.isnan(cam["hit"])
    assert example["clip"].shape == (3, 8, 112, 112)
    rows, example = pl.explain_video(model, videos.loc["far"], "fs8", 8, 2, processed, STATS,
                                     CPU, "fp32", methods)  # fmt: skip
    assert example is None and all(r["skipped"] == 1 for r in rows)


def test_summary_sanity_and_figures(processed, model):
    videos = pd.read_parquet(processed / INDEX_FILE).set_index("FileName", drop=False)
    rows = []
    for name in ("v_low", "v_mid", "v_high", "far"):
        r, _ = pl.explain_video(model, videos.loc[name], "fs8", 8, 2, processed, STATS, CPU,
                                "fp32", ("occlusion", "gradcam"))  # fmt: skip
        rows += r
    summary = pl.summarise(pl.metrics_frame(rows)).set_index("method")
    assert summary.loc["occlusion", "n"] == 3 and summary.loc["occlusion", "n_skipped"] == 1
    assert 0 < summary.loc["occlusion", "hit_p"] <= 1
    clip = torch.randn(3, 8, 112, 112)
    sanity = pl.sanity_rows(model, {"a": clip}, "fs8", "fp32", ("occlusion", "gradcam"))
    assert len(sanity) == 6 * 2 and sanity["spearman"].between(-1, 1).all()
    table = pl.sanity_summary(sanity)
    assert len(table) == 2 * 6 and (table["n"] == 1).all()  # 2 methods x 6 stages, 1 video
    assert table["stage"].tolist()[:6] == ["fc", "layer4", "layer3", "layer2", "layer1", "stem"]
    fig = summary_chart(summary.reset_index().assign(config_id="fs8"), sanity)
    assert len(fig.axes) >= 4
    ex = {"clip": clip, "ig": np.random.rand(8, 112, 112), "occlusion": np.random.rand(8),
          "ed_pos": 2.0, "es_pos": 6.0, "label": "t"}  # fmt: skip
    assert len(example_panel([ex, ex], STATS, "t").axes) == 6
