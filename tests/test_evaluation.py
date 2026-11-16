"""Run discovery, the TEST-split lock and prediction files (CPU, synthetic data)."""

import subprocess

import pandas as pd
import pytest
import torch
from conftest import IDENTITY, MeanModel

from echoef.contracts import PREDICTION_COLUMNS
from echoef.eval import evaluation as ev


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo, check=True, capture_output=True,
    )  # fmt: skip


def test_test_split_needs_confirmation_and_the_frozen_tag(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "x")
    with pytest.raises(PermissionError, match="confirm"):
        ev.require_test_unlocked(False, tmp_path)
    with pytest.raises(PermissionError, match="models-frozen"):
        ev.require_test_unlocked(True, tmp_path)
    _git(tmp_path, "tag", "models-frozen")
    ev.require_test_unlocked(True, tmp_path)


def test_finished_runs_needs_done_marker(tmp_path):
    for name, done in [("fs8/seed0", True), ("fs8/seed1", False), ("fw64/seed0", True),
                       ("b0-fs8/seed0", True)]:  # fmt: skip
        (tmp_path / name).mkdir(parents=True)
        if done:
            (tmp_path / name / ev.DONE_FILE).touch()
    assert ev.finished_runs(tmp_path) == [tmp_path / "fs8/seed0", tmp_path / "fw64/seed0"]


def test_tier0_names():
    assert ev.tier0_run_id("fs32") == "b0-fs32"
    assert ev.tier0_run(ev.Path("runs"), "fw8") == ev.Path("runs/b0-fw8/seed0")
    with pytest.raises(ValueError):
        ev.tier0_run_id("b0")


def test_evaluate_model_writes_contract_csvs(processed, tmp_path):
    run = tmp_path / "runs" / "b0-fs16" / "seed0"
    maes = ev.evaluate_model(
        MeanModel(), run, "VAL", ["M1", "M2"], config_id="b0-fs16", seed=0, T=16, period=2,
        processed_root=processed, stats=IDENTITY, device=torch.device("cpu"), precision="fp32",
        batch_size=4, workers=0, m2_overlap=0.5,
    )  # fmt: skip
    for mode in ("M1", "M2"):
        df = pd.read_csv(run / f"predictions_val_{mode}.csv")
        assert list(df.columns) == PREDICTION_COLUMNS and len(df) == 2
        assert maes[mode] == pytest.approx((df["ef_pred"] - df["ef_true"]).abs().mean())
