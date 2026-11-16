"""Task A6: train one (config, seed) run into runs/<config_id>/seed<k>/ (resumes after a crash).

    uv run python scripts/train.py --config fs8 --seed 0
    uv run python scripts/train.py --config fs8 --set train.epochs=1 --out runs/_smoke/fs8

Overrides (--set) change the frozen recipe, so they require --out: a modified run can never land
in an official run folder. Normally scripts/run_sweep.py calls this script.
"""

import argparse
import logging
from pathlib import Path

import torch

from echoef.config import load_config, load_paths
from echoef.contracts import CONFIGS, run_dir
from echoef.data.dataset import load_stats
from echoef.train.trainer import train_run
from echoef.utils import seed_everything, setup_logging
from echoef.utils.gpu import cap_to_free_vram, require_gpu_free

log = logging.getLogger("train")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train one config and seed (A6).")
    parser.add_argument("--config", required=True, choices=list(CONFIGS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides")
    parser.add_argument("--out", type=Path, default=None, help="run folder (needed with --set)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes")
    parser.add_argument("--machine", default="", help="assignment.yaml key, kept in meta.json")
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    parser.add_argument("--in-queue", action="store_true", help=argparse.SUPPRESS)  # run_sweep
    args = parser.parse_args(argv)

    if args.set and args.out is None:
        parser.error("--set changes the recipe; give --out so the run is not an official one")
    cfg = load_config(args.config, [f"seed={args.seed}", *args.set])
    roots = load_paths(args.paths)
    run = args.out or run_dir(roots["runs_root"], cfg.config_id, cfg.seed)
    device = torch.device(args.device)
    run.mkdir(parents=True, exist_ok=True)
    setup_logging(run / "train.log")
    seed_everything(cfg.seed)
    if device.type == "cuda":
        if not args.in_queue:
            require_gpu_free(roots["runs_root"])
        cap_to_free_vram(device)
    log.info("training %s seed %d into %s (overrides %s)", cfg.config_id, cfg.seed, run, args.set)
    train_run(
        cfg,
        run,
        processed_root=roots["processed_root"],
        stats=load_stats(roots["processed_root"]),
        device=device,
        workers=args.workers,
        machine=args.machine,
    )


if __name__ == "__main__":
    main()
