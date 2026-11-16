"""Task B4: report figures from results/all_runs.csv -> results/figures/*.pdf (3.5 in wide).

    uv run python scripts/make_figures.py [--results-root S] [--split test] [--png]

Uses the primary endpoint's split (TEST, M1) when its rows exist, else VAL. The Bland-Altman plot
reads per-video predictions of --ba-config (default: lowest validation MAE in M1) from runs_root.
No figure shows echo frames, so all of them may be committed. CPU only.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from echoef.config import load_paths
from echoef.contracts import CONFIGS, predictions_path, run_dir
from echoef.eval.figures import bland_altman, mae_by_band, mae_vs_t, pareto
from echoef.eval.tables import config_label
from echoef.utils import seed_everything, setup_logging

log = logging.getLogger("make_figures")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draw the report figures (B4).")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default=None)
    parser.add_argument("--ba-config", choices=list(CONFIGS), default=None)
    parser.add_argument("--png", action="store_true", help="also write PNG previews")
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    seed_everything(0)
    roots = load_paths(args.paths) if not (args.results_root and args.runs_root) else {}
    results = args.results_root or roots["results_root"]
    runs_root = args.runs_root or roots["runs_root"]
    table = pd.read_csv(results / "all_runs.csv")
    split = (args.split or ("test" if (table["split"] == "TEST").any() else "val")).upper()
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    figures = {
        "mae_vs_t": mae_vs_t(table, split, "M1"),
        "mae_by_band": mae_by_band(table, split, "M1"),
    }
    if {"gflops_clip", "gflops_video_m2"} <= set(table.columns):
        figures["pareto"] = pareto(table, split)
    else:
        log.warning("no cost columns in all_runs.csv (run benchmark_cost.py): Pareto skipped")
    is_val_m1 = (table["split"] == "VAL") & (table["mode"] == "M1")
    val = table[
        is_val_m1 & (table["seed"].astype(str) == "0") & table["config_id"].isin(list(CONFIGS))
    ]
    ba = args.ba_config or (val.sort_values("mae")["config_id"].iloc[0] if len(val) else None)
    if ba:
        path = predictions_path(run_dir(Path(runs_root), ba, 0), split, "M1")
        if path.exists():
            figures["bland_altman"] = bland_altman(
                pd.read_csv(path), f"{config_label(ba)}, {split.lower()}, M1, seed 0"
            )
    for name, fig in figures.items():
        fig.savefig(out / f"{name}.pdf")
        if args.png:
            fig.savefig(out / f"{name}.png", dpi=200)
        log.info("wrote %s", out / f"{name}.pdf")


if __name__ == "__main__":
    main()
