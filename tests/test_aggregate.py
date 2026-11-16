"""Aggregation of run predictions into all_runs.csv and tables (task B4; synthetic runs)."""

import json

import numpy as np
import pandas as pd
import pytest

from echoef.contracts import ALL_RUNS_COLUMNS, CONFIGS, PREDICTION_COLUMNS, predictions_path
from echoef.eval import aggregate as ag

N = 60


def _write_run(root, config_id, seed, noise, done=True, splits=("VAL",), gpu="GPU-X"):
    run = root / config_id / f"seed{seed}"
    run.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash((config_id, seed))) % 2**32)
    y = rng.uniform(20, 75, N)
    base = config_id.removeprefix("b0-")
    _, t, period = CONFIGS[base]
    for split in splits:
        for mode in ("M1", "M2"):
            df = pd.DataFrame({"file_name": [f"v{i}" for i in range(N)], "split": split,
                               "mode": mode, "config_id": config_id, "seed": seed, "T": t,
                               "period": period, "ef_true": y,
                               "ef_pred": y + rng.normal(0, noise, N), "n_clips": 1,
                               "padded": (np.arange(N) < 5).astype(int)})  # fmt: skip
            df[PREDICTION_COLUMNS].to_csv(predictions_path(run, split, mode), index=False)
    (run / "meta.json").write_text(json.dumps({"gpu_name": gpu}))
    if done:
        (run / "DONE").touch()


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    _write_run(root, "fs8", 0, 6.0)
    _write_run(root, "fs8", 1, 6.0)
    _write_run(root, "fs16", 0, 3.0)
    _write_run(root, "fs32", 0, 3.0, done=False)  # still training: ignored
    _write_run(root, "b0-fs8", 0, 8.0)
    return root


def test_all_runs_rows_ensembles_and_tier0(runs, tmp_path):
    table = ag.aggregate(runs, tmp_path / "results")
    saved = pd.read_csv(tmp_path / "results" / "all_runs.csv")
    assert list(saved.columns) == ALL_RUNS_COLUMNS + ag.EXTRA_COLUMNS
    assert set(table["config_id"]) == {"fs8", "fs16", "b0-fs8"}  # fs32 not DONE
    assert len(table) == (3 + 1 + 1) * 2  # 4 runs + 1 ensemble, x 2 modes
    ens = table[(table["config_id"] == "fs8") & (table["seed"] == "ens")]
    assert len(ens) == 2 and (ens["n_seeds"] == 2).all()
    b0 = table[table["config_id"] == "b0-fs8"].iloc[0]
    assert (b0["regime"], b0["T"], b0["period"]) == ("FS", 8, 2)
    assert (table["gpu_name"] == "GPU-X").all()


def test_tables(runs, tmp_path):
    ag.aggregate(runs, tmp_path / "results")
    tables = tmp_path / "results" / "tables"
    wil = pd.read_csv(tables / "wilcoxon_holm_val.csv")  # no TEST predictions yet: VAL, named so
    assert wil["comparison"].tolist() == ["fs8 vs fs16"]
    assert wil.loc[0, "median_diff"] > 0 and wil.loc[0, "significant"]  # fs8 is noisier
    seeds = pd.read_csv(tables / "seed_summary.csv")
    fs8 = seeds[(seeds["config_id"] == "fs8") & (seeds["mode"] == "M1")].iloc[0]
    assert fs8["n_seeds"] == 2 and fs8["mae_sd"] > 0
    unpadded = pd.read_csv(tables / "unpadded.csv")
    assert (unpadded["n_padded"] == 5).all() and unpadded["mae_unpadded"].notna().all()


def test_cost_columns_are_joined_on_config(runs, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame([{"config_id": "fs8", "T": 8, "period": 2, "params": 1, "gflops_clip": 2.0,
                   "mean_clips_m2": 20.0, "gflops_video_m2": 40.0, "latency_ms_m1": 5.0,
                   "latency_ms_m2": 50.0, "peak_vram_mb_infer": 300.0,
                   "peak_vram_mb_train": 1500.0, "micro_batch": 8, "epoch_time_s": 97.0,
                   "gpu_name": "RTX"}]).to_csv(results / "cost.csv", index=False)  # fmt: skip
    table = ag.aggregate(runs, results)
    assert (table.loc[table["config_id"].isin(["fs8", "b0-fs8"]), "gflops_clip"] == 2.0).all()
    assert table.loc[table["config_id"] == "fs16", "gflops_clip"].isna().all()


def test_no_predictions_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ag.aggregate(tmp_path / "empty", tmp_path / "results")


def test_hardware_control_table(runs, tmp_path):
    _write_run(runs / "_hwcontrol_rtx4050", "fs8", 0, 6.5, gpu="RTX 4050")
    table = ag.aggregate(runs, tmp_path / "results")
    hw = pd.read_csv(tmp_path / "results" / "tables" / "hardware_control.csv")
    assert len(hw) == 2 and (hw["machine"] == "rtx4050").all()
    assert (hw["gpu_name"] == "RTX 4050").all() and (hw["main_gpu_name"] == "GPU-X").all()
    assert hw["seed_gap"].notna().all()  # fs8 has a second seed on the main GPU
    assert (
        len(table[(table["config_id"] == "fs8") & (table["seed"] == 0)]) == 2
    )  # not counted twice
