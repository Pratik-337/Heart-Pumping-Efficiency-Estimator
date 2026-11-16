"""Collect every run's predictions into results tables (task B4; ``scripts/aggregate_results.py``).

Sources under ``runs_root``: trained runs ``<config_id>/seed<k>/`` with ``DONE``, and Tier 0 runs
``b0-<config_id>/seed0/`` (D-19). Outputs (spec, "Files on disk" and "Pre-specified analysis"):

- ``all_runs.csv``: ``contracts.ALL_RUNS_COLUMNS`` plus ``gpu_name`` and ``n_seeds``, one row per
  run, split and mode, plus seed-ensemble rows (``seed`` = "ens") for configs with 2+ seeds;
  the cost columns of ``cost.csv`` are joined on ``config_id`` when that file exists;
- ``tables/wilcoxon_holm.csv``: the 9 pre-specified paired comparisons;
- ``tables/seed_summary.csv``: mean and SD of MAE across seeds;
- ``tables/unpadded.csv``: MAE on the videos whose clips are not zero-padded (D-09, D-28).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from echoef.contracts import (
    ALL_RUNS_COLUMNS,
    CONFIGS,
    COST_COLUMNS,
    MODES,
    SPLITS,
    TIER0_PREFIX,
    predictions_path,
)
from echoef.eval.metrics import compute_metrics
from echoef.eval.predict import ENSEMBLE_SEED, ensemble
from echoef.eval.stats import PAIR_COLUMNS, wilcoxon_holm

log = logging.getLogger(__name__)

DONE_FILE = "DONE"
EXTRA_COLUMNS = ["gpu_name", "n_seeds"]
# Pre-specified family (spec, "Hypothesis tests"): neighbouring T within FS, within FW (FW-32 is
# the shared FS-32 run), and FS vs FW at T = 8, 16, 64. Pairs are (A, B); diff = err_A - err_B.
COMPARISONS = [
    ("fs8", "fs16"), ("fs16", "fs32"), ("fs32", "fs64"),
    ("fw8", "fw16"), ("fw16", "fs32"), ("fs32", "fw64"),
    ("fs8", "fw8"), ("fs16", "fw16"), ("fs64", "fw64"),
]  # fmt: skip


@dataclass(frozen=True)
class Run:
    config_id: str  # e.g. "fs8" or "b0-fs8"
    base: str  # the sweep config whose (regime, T, period) it uses
    seed: int
    path: Path
    tier0: bool


def discover_runs(runs_root: Path) -> list[Run]:
    """Finished trained runs (``DONE``) and Tier 0 runs, in contract config order."""
    runs = []
    for base in CONFIGS:
        for path in sorted((Path(runs_root) / base).glob("seed*")):
            if (path / DONE_FILE).exists():
                runs.append(Run(base, base, int(path.name.removeprefix("seed")), path, False))
    for base in CONFIGS:
        path = Path(runs_root) / f"{TIER0_PREFIX}{base}" / "seed0"
        if path.is_dir():
            runs.append(Run(f"{TIER0_PREFIX}{base}", base, 0, path, True))
    return runs


def _gpu_name(run: Path) -> str:
    try:
        return str(json.loads((run / "meta.json").read_text()).get("gpu_name", ""))
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def _row(config_id: str, base: str, seed, split: str, mode: str, preds: pd.DataFrame) -> dict:
    regime, t, period = CONFIGS[base]
    head = {"config_id": config_id, "regime": regime, "T": t, "period": period, "seed": seed,
            "split": split, "mode": mode}  # fmt: skip
    return head | compute_metrics(preds)


def collect_predictions(runs: list[Run]) -> dict[tuple, pd.DataFrame]:
    """``{(config_id, seed, split, mode): predictions}`` for every prediction file found."""
    found = {}
    for run in runs:
        for split in SPLITS:
            for mode in MODES:
                path = predictions_path(run.path, split, mode)
                if path.exists():
                    found[(run.config_id, run.seed, split, mode)] = pd.read_csv(path)
    return found


def all_runs_table(runs: list[Run], preds: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    """One metrics row per run, split and mode, plus seed-ensemble rows."""
    by_id = {(r.config_id, r.seed): r for r in runs}
    rows = []
    for (config_id, seed, split, mode), df in preds.items():
        run = by_id[(config_id, seed)]
        rows.append(_row(config_id, run.base, seed, split, mode, df)
                    | {"gpu_name": _gpu_name(run.path), "n_seeds": 1})  # fmt: skip
    keys = {(c, sp, m) for (c, _, sp, m) in preds}
    for config_id, split, mode in sorted(keys):
        seeds = sorted(s for (c, s, sp, m) in preds if (c, sp, m) == (config_id, split, mode))
        if config_id.startswith(TIER0_PREFIX) or len(seeds) < 2:
            continue
        ens = ensemble([preds[(config_id, s, split, mode)] for s in seeds])
        gpus = sorted({_gpu_name(by_id[(config_id, s)].path) for s in seeds})
        rows.append(_row(config_id, config_id, ENSEMBLE_SEED, split, mode, ens)
                    | {"gpu_name": "+".join(gpus), "n_seeds": len(seeds)})  # fmt: skip
    table = pd.DataFrame(rows, columns=ALL_RUNS_COLUMNS + EXTRA_COLUMNS)
    order = {c: i for i, c in enumerate(list(CONFIGS) + [TIER0_PREFIX + c for c in CONFIGS])}
    table["_o"] = table["config_id"].map(order)
    table["_s"] = table["seed"].map(lambda s: 99 if s == ENSEMBLE_SEED else int(s))
    return (
        table.sort_values(["_o", "_s", "split", "mode"]).drop(columns=["_o", "_s"])
        .reset_index(drop=True)
    )  # fmt: skip


def join_cost(table: pd.DataFrame, cost_csv: Path) -> pd.DataFrame:
    """Add the ``cost.csv`` columns (spec: all_runs.csv = metrics + cost, joined on config_id).

    Tier 0 rows take the cost of the config whose setting they use.
    """
    if not Path(cost_csv).exists():
        return table
    cost = pd.read_csv(cost_csv)[COST_COLUMNS].drop(columns=["T", "period"])
    base = table["config_id"].str.removeprefix(TIER0_PREFIX)
    joined = table.assign(_base=base).merge(
        cost.rename(columns={"config_id": "_base", "gpu_name": "cost_gpu_name"}),
        on="_base", how="left",
    )  # fmt: skip
    return joined.drop(columns="_base")


def per_video_errors(
    preds: dict[tuple, pd.DataFrame], config_id: str, split: str, mode: str
) -> pd.Series | None:
    """Absolute error per video, averaged over the config's seeds (spec: seed-averaged)."""
    frames = [df for (c, _, sp, m), df in preds.items() if (c, sp, m) == (config_id, split, mode)]
    if not frames:
        return None
    errs = [df.set_index("file_name").eval("abs(ef_pred - ef_true)") for df in frames]
    return pd.concat(errs, axis=1).mean(axis=1)


def comparison_pairs(preds: dict[tuple, pd.DataFrame], split: str, mode: str) -> pd.DataFrame:
    """``stats.PAIR_COLUMNS`` rows for every pre-specified comparison whose two configs exist."""
    parts = []
    for a, b in COMPARISONS:
        err_a = per_video_errors(preds, a, split, mode)
        err_b = per_video_errors(preds, b, split, mode)
        if err_a is None or err_b is None:
            continue
        both = pd.concat([err_a.rename("err_a"), err_b.rename("err_b")], axis=1, join="inner")
        parts.append(both.reset_index(names="file_name").assign(comparison=f"{a} vs {b}"))
    if not parts:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.concat(parts, ignore_index=True)[PAIR_COLUMNS]


def seed_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of MAE across the individual seeds of each trained config."""
    single = table[
        (table["seed"] != ENSEMBLE_SEED) & ~table["config_id"].str.startswith(TIER0_PREFIX)
    ]
    grouped = single.groupby(["config_id", "split", "mode"], sort=False)["mae"]
    out = grouped.agg(n_seeds="size", mae_mean="mean", mae_sd="std").reset_index()
    out.loc[out["n_seeds"] < 2, "mae_sd"] = np.nan
    return out


def unpadded_table(preds: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    """MAE on all videos vs on the videos without padded clips (D-09 sensitivity check)."""
    rows = []
    for (config_id, seed, split, mode), df in preds.items():
        err = (df["ef_pred"] - df["ef_true"]).abs()
        keep = df["padded"] == 0
        rows.append({"config_id": config_id, "seed": seed, "split": split, "mode": mode,
                     "n": len(df), "n_padded": int((~keep).sum()), "mae_all": err.mean(),
                     "mae_unpadded": err[keep].mean() if keep.any() else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def aggregate(
    runs_root: Path, results_root: Path, test_split: str = "TEST", mode: str = "M1"
) -> pd.DataFrame:
    """Write ``all_runs.csv`` and the tables; return ``all_runs``.

    The Wilcoxon family uses ``test_split`` and ``mode`` (primary endpoint: TEST, M1, D-10);
    if that split has no predictions yet, VAL is used and the file name says so.
    """
    results_root = Path(results_root)
    runs = discover_runs(runs_root)
    preds = collect_predictions(runs)
    if not preds:
        raise FileNotFoundError(f"no prediction files under {runs_root}")
    table = join_cost(all_runs_table(runs, preds), results_root / "cost.csv")
    tables = results_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_root / "all_runs.csv", index=False)
    seed_summary(table).to_csv(tables / "seed_summary.csv", index=False)
    unpadded_table(preds).to_csv(tables / "unpadded.csv", index=False)
    split = test_split if any(k[2] == test_split for k in preds) else "VAL"
    trained = {k: v for k, v in preds.items() if not k[0].startswith(TIER0_PREFIX)}
    pairs = comparison_pairs(trained, split, mode)
    name = "wilcoxon_holm.csv" if split == test_split else f"wilcoxon_holm_{split.lower()}.csv"
    if len(pairs):
        wilcoxon_holm(pairs).assign(split=split, mode=mode).to_csv(tables / name, index=False)
    log.info("%d runs, %d prediction files -> %s (%d rows); Wilcoxon on %s %s, %d comparisons",
             len(runs), len(preds), results_root / "all_runs.csv", len(table), split, mode,
             pairs["comparison"].nunique() if len(pairs) else 0)  # fmt: skip
    return table
