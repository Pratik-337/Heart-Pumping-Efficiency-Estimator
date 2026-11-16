"""Evaluate runs on a split and write contract prediction CSVs (task A5; used by evaluate.py).

Two kinds of run folder under ``runs_root``:

- trained runs ``<config_id>/seed<k>/`` with a ``DONE`` marker and ``best.pt``;
- Tier 0 runs ``b0-<config_id>/seed0/``: EchoNet's released weights (B0) evaluated at that
  config's (T, period) without training (D-19). The folder holds ``config.yaml``,
  ``meta.json`` and the predictions.

The TEST split is locked: it needs ``confirm_test`` and the git tag ``models-frozen`` (CLAUDE.md,
Hard rules; spec, "Test-set lock").
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from echoef.config import REPO_ROOT
from echoef.contracts import CONFIGS, TIER0_PREFIX, predictions_path, run_dir
from echoef.data.dataset import load_videos
from echoef.eval.predict import mae, predict_split
from echoef.train.trainer import DONE_FILE
from echoef.utils.gpu import gpu_name
from echoef.utils.meta import git_tag_exists, run_meta

log = logging.getLogger(__name__)

MODELS_FROZEN_TAG = "models-frozen"
TIER0_SEED = 0  # B0 is one fixed model; its runs are stored as seed 0


def require_test_unlocked(confirm_test: bool, repo: Path = REPO_ROOT) -> None:
    """Refuse the TEST split unless confirmed and the ``models-frozen`` tag exists (Gate C).

    Raises:
        PermissionError: either condition is missing.
    """
    if not confirm_test:
        raise PermissionError("the TEST split needs --confirm-test")
    if not git_tag_exists(MODELS_FROZEN_TAG, repo):
        raise PermissionError(f"the TEST split needs the git tag {MODELS_FROZEN_TAG!r} (Gate C)")


def tier0_run_id(config_id: str) -> str:
    """Run ID of B0 evaluated at a sweep config's setting, e.g. ``b0-fs32``."""
    if config_id not in CONFIGS:
        raise ValueError(f"unknown config_id {config_id!r}")
    return f"{TIER0_PREFIX}{config_id}"


def finished_runs(runs_root: Path) -> list[Path]:
    """Trained run folders with a ``DONE`` marker, sorted by config then seed."""
    runs = []
    for config_id in CONFIGS:
        for run in sorted((Path(runs_root) / config_id).glob("seed*")):
            if (run / DONE_FILE).exists():
                runs.append(run)
    return runs


def write_tier0_record(run: Path, cfg: DictConfig, started: str, device: torch.device) -> None:
    """``config.yaml`` and ``meta.json`` of a Tier 0 run folder."""
    run.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run / "config.yaml")
    meta = run_meta(
        OmegaConf.to_container(cfg, resolve=True),
        TIER0_SEED,
        gpu_name(device) if device.type == "cuda" else "cpu",
        started,
        datetime.now(UTC).isoformat(timespec="seconds"),
        task="A5 Tier 0",
    )
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def evaluate_model(
    model: nn.Module,
    run: Path,
    split: str,
    modes: list[str],
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
    workers: int,
    m2_overlap: float,
) -> dict[str, float]:
    """Predict every used video of ``split`` in each mode; write ``predictions_<split>_<mode>.csv``.

    Returns:
        ``{mode: MAE}`` (also logged), for quick checks such as Gate A.
    """
    videos = load_videos(processed_root, split)
    run.mkdir(parents=True, exist_ok=True)
    maes = {}
    for mode in modes:
        preds = predict_split(
            model, videos, split, mode, config_id=config_id, seed=seed, T=T, period=period,
            processed_root=processed_root, stats=stats, device=device, precision=precision,
            batch_size=batch_size, workers=workers, m2_overlap=m2_overlap,
        )  # fmt: skip
        path = predictions_path(run, split, mode)
        preds.to_csv(path, index=False)
        maes[mode] = mae(preds)
        log.info(
            "%s seed %s %s %s: n=%d, MAE %.3f, mean clips %.2f -> %s",
            config_id, seed, split, mode, len(preds), maes[mode], preds["n_clips"].mean(), path,
        )  # fmt: skip
    return maes


def tier0_run(runs_root: Path, config_id: str) -> Path:
    """Folder of B0 evaluated at ``config_id``'s setting: ``runs/b0-<config_id>/seed0``."""
    return run_dir(Path(runs_root), tier0_run_id(config_id), TIER0_SEED)
