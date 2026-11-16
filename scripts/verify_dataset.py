"""Task A2: check the raw EchoNet-Dynamic files without building the cache.

Decodes every listed video, compares frame counts and sizes with FileList.csv, checks the EF
labels and tracings, measures how far R, G and B differ, and writes
<processed_root>/qc_report.csv. Reads raw_root only.

    uv run python scripts/verify_dataset.py [--paths P] [--config C] [--workers N]
"""

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from echoef.config import CONFIGS_DIR, load_paths
from echoef.data.preprocess import check_roots, verify_dataset
from echoef.utils import seed_everything, setup_logging

log = logging.getLogger("verify_dataset")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify the raw EchoNet-Dynamic files (A2).")
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "preprocess.yaml")
    parser.add_argument("--workers", type=int, default=None, help="default: from --config")
    args = parser.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    seed_everything(cfg.seed)
    roots = load_paths(args.paths)
    raw, processed = roots["raw_root"], roots["processed_root"]
    check_roots(raw, processed)  # before the log file is created under processed_root
    setup_logging(processed / "verify_dataset.log")
    workers = args.workers if args.workers is not None else cfg.workers
    log.info("raw_root %s -> processed_root %s", raw, processed)
    verify_dataset(raw, processed, ef_tolerance=cfg.ef_tolerance, workers=workers)


if __name__ == "__main__":
    main()
