"""Tasks B4/A10: LaTeX tables and number macros for the report from results/*.csv.

    uv run python scripts/make_tables.py [--results-root S]

Writes S/numbers.tex and S/tables/*.tex (see echoef.eval.tables). CPU only.
"""

import argparse
import logging
from pathlib import Path

from echoef.config import load_paths
from echoef.eval.tables import write_tables
from echoef.utils import seed_everything, setup_logging

log = logging.getLogger("make_tables")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report tables and number macros (B4/A10).")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    seed_everything(0)
    results = args.results_root or load_paths(args.paths)["results_root"]
    for path in write_tables(results):
        log.info("wrote %s", path)


if __name__ == "__main__":
    main()
