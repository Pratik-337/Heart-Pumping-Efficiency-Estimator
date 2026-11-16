"""Task A6: run this laptop's training queue from configs/assignment.yaml (resumable).

    tmux new -s sweep
    uv run python scripts/run_sweep.py --machine rtx4060

Holds the GPU lock (no other GPU job starts meanwhile), trains only with the recipe tagged
``config-frozen``, re-reads the queue before every run, skips DONE runs, retries a failed run
once and then moves on. Each run is a separate ``scripts/train.py`` process, so a crash never
takes down the queue.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from echoef.config import REPO_ROOT, load_assignment, load_paths
from echoef.train.sweep import require_frozen_commit, run_queue
from echoef.utils import setup_logging
from echoef.utils.gpu import queue_lock

log = logging.getLogger("run_sweep")
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run this laptop's training queue (A6).")
    parser.add_argument("--machine", required=True, help="key in configs/assignment.yaml")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes")
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    parser.add_argument("--dry-run", action="store_true", help="list the queue and stop")
    args = parser.parse_args(argv)

    roots = load_paths(args.paths)
    runs_root = roots["runs_root"]
    setup_logging(runs_root / f"sweep_{args.machine}.log")
    if args.machine not in load_assignment():
        parser.error(
            f"--machine {args.machine!r} not in assignment.yaml: {sorted(load_assignment())}"
        )

    def get_queue() -> list[tuple[str, int]]:  # re-read before every run (runs may move laptops)
        return load_assignment()[args.machine]

    queue = get_queue()
    commit = require_frozen_commit()
    log.info("machine %s, commit %s, %d runs in the queue", args.machine, commit, len(queue))
    if args.dry_run:
        for config_id, seed in queue:
            log.info("  %s seed %d", config_id, seed)
        return

    def train_one(config_id: str, seed: int) -> int:
        cmd = [sys.executable, str(TRAIN_SCRIPT), "--config", config_id, "--seed", str(seed),
               "--machine", args.machine, "--workers", str(args.workers), "--in-queue"]  # fmt: skip
        if args.paths:
            cmd += ["--paths", str(args.paths)]
        return subprocess.run(cmd, cwd=REPO_ROOT).returncode

    with queue_lock(runs_root, args.machine):
        run_queue(get_queue, runs_root, train_one)


if __name__ == "__main__":
    main()
