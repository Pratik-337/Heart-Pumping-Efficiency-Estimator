"""Task A2: decode EchoNet-Dynamic once into the grayscale frame cache, index, tracings and stats.

Writes <processed_root>/{videos/*.npy, index.parquet, tracings.parquet, stats.json,
qc_report.csv, preprocess_meta.json, preprocess.log}; see echoef.data.preprocess. Reads raw_root
only. Refuses to run over an existing index unless --overwrite is given.

    uv run python scripts/preprocess.py [--paths P] [--config C] [--workers N] [--overwrite]
"""

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from echoef.config import CONFIGS_DIR, load_paths
from echoef.data.preprocess import check_roots, run_preprocess
from echoef.utils import seed_everything, setup_logging

log = logging.getLogger("preprocess")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the EchoNet frame cache and index (A2).")
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "preprocess.yaml")
    parser.add_argument("--workers", type=int, default=None, help="default: from --config")
    parser.add_argument("--overwrite", action="store_true", help="rebuild an existing cache")
    args = parser.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    seed_everything(cfg.seed)
    roots = load_paths(args.paths)
    raw, processed = roots["raw_root"], roots["processed_root"]
    check_roots(raw, processed)  # before the log file is created under processed_root
    setup_logging(processed / "preprocess.log")
    workers = args.workers if args.workers is not None else cfg.workers
    log.info("raw_root %s -> processed_root %s", raw, processed)
    run_preprocess(
        raw, processed, ef_tolerance=cfg.ef_tolerance, workers=workers, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
