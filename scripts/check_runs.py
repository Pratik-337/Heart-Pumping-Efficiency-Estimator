"""Gate C check (task A10): every planned run is finished and trained with the frozen recipe.

    uv run python scripts/check_runs.py

Reads configs/assignment.yaml (every laptop's queue) and the hardware-control runs under
runs_root, and checks each run folder: DONE, every epoch logged, best.pt, validation predictions
in M1 and M2, and a training recipe equal to the tag config-frozen at the commit in meta.json.
Exit code 1 when any run has a problem: do not tag models-frozen then. Read-only, CPU only.
"""

import argparse
import logging
from pathlib import Path

from echoef.config import load_assignment, load_config, load_paths
from echoef.train.sweep import check_runs
from echoef.utils import seed_everything, setup_logging

log = logging.getLogger("check_runs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate C: are all runs finished and frozen?")
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    seed_everything(0)
    runs_root = load_paths(args.paths)["runs_root"]
    epochs = int(load_config("fs8").train.epochs)  # base.yaml: the same for every config
    table = check_runs(runs_root, load_assignment(), epochs)
    for line in table.to_string(index=False).splitlines():
        log.info("%s", line)
    bad = table[table["problems"] != "ok"]
    log.info("%d runs, %d with problems: %s", len(table), len(bad),
             "do NOT tag models-frozen yet" if len(bad) else "ready for Gate C")  # fmt: skip
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
