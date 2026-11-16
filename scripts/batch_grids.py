"""Task A3: one training and one validation batch per config, as image grids plus numeric checks.

Writes <out>/<config_id>_train.png and <config_id>_val_m1.png, default
<processed_root>/inspect/batch_grids/, outside the repo: the grids show echo frames and are never
committed. The numeric checks (shape, channel copies, mean/SD, padding) go to the log.

    uv run python scripts/batch_grids.py [--configs fs8 fw8 ...] [--paths P] [--out DIR]
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from echoef.config import load_config, load_paths
from echoef.contracts import CONFIGS
from echoef.data.dataset import TRAIN_MODE, dataset_from_config, load_stats
from echoef.data.grids import batch_checks, batch_grid
from echoef.utils import seed_everything, setup_logging

log = logging.getLogger("batch_grids")
BATCHES = (("train", "TRAIN", TRAIN_MODE), ("val_m1", "VAL", "M1"))  # (file suffix, split, mode)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Save batch image grids per config (A3).")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    parser.add_argument("--out", type=Path, default=None, help="default: <processed>/inspect/...")
    parser.add_argument("--max-frames", type=int, default=16, help="frames shown per clip")
    args = parser.parse_args(argv)

    processed = load_paths(args.paths)["processed_root"]
    out = args.out or processed / "inspect" / "batch_grids"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "batch_grids.log")
    stats = load_stats(processed)
    log.info("stats %s; writing to %s", stats, out)

    for config_id in args.configs:
        cfg = load_config(config_id)
        seed_everything(cfg.seed)  # per config, so each grid is reproducible on its own
        for suffix, split, mode in BATCHES:
            data = dataset_from_config(cfg, split, mode, processed, stats)
            loader = DataLoader(
                data,
                batch_size=cfg.train.micro_batch,
                shuffle=True,
                generator=torch.Generator().manual_seed(cfg.seed),
            )
            batch = next(iter(loader))
            log.info("%s %s: %s", config_id, suffix, batch_checks(batch, cfg.period))
            title = f"{config_id}  T={cfg.T}  period={cfg.period}  {split} {mode}"
            fig = batch_grid(batch, stats, cfg.period, args.max_frames, title)
            path = out / f"{config_id}_{suffix}.png"
            fig.savefig(path, dpi=150)
            log.info("wrote %s", path)


if __name__ == "__main__":
    main()
