"""Task A2: aggregated dataset summaries from <processed_root>/index.parquet.

Writes <results_root>/eda/{split_summary, padding_share, edes_fit}.csv: counts and shares per
split, nothing per video (see echoef.data.eda).

    uv run python scripts/eda.py [--paths P] [--config C]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from echoef.config import CONFIGS_DIR, load_paths
from echoef.data.eda import edes_fit, padding_share, split_summary
from echoef.data.preprocess import INDEX_FILE
from echoef.utils import seed_everything, setup_logging

log = logging.getLogger("eda")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregated EDA tables from the index (A2).")
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "preprocess.yaml")
    args = parser.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    seed_everything(cfg.seed)
    roots = load_paths(args.paths)
    out = roots["results_root"] / "eda"
    setup_logging(out / "eda.log")

    index = pd.read_parquet(roots["processed_root"] / INDEX_FILE)
    lo, hi = cfg.ef_band_edges
    tables = {
        "split_summary": split_summary(index, (lo, hi)),
        "padding_share": padding_share(index),
        "edes_fit": edes_fit(index),
    }
    for name, table in tables.items():
        path = out / f"{name}.csv"
        table.to_csv(path, index=False, float_format="%.4f")
        log.info("wrote %s\n%s", path, table.to_string(index=False))


if __name__ == "__main__":
    main()
