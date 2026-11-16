"""Task B4: every run's predictions -> results/all_runs.csv and results/tables/*.csv.

    uv run python scripts/aggregate_results.py [--runs-root R] [--results-root S]

Roots default to configs/paths.yaml. Runs only on the CPU; safe while the training queue runs.
"""

import argparse
import logging
from pathlib import Path

from echoef.config import load_paths
from echoef.eval.aggregate import aggregate
from echoef.utils import seed_everything, setup_logging

log = logging.getLogger("aggregate_results")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate predictions into results (B4).")
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    seed_everything(0)
    roots = load_paths(args.paths) if not (args.runs_root and args.results_root) else {}
    runs_root = args.runs_root or roots["runs_root"]
    results_root = args.results_root or roots["results_root"]
    aggregate(runs_root, results_root)


if __name__ == "__main__":
    main()
