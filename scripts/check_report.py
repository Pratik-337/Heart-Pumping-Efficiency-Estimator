"""Check that every generated number, table and figure the report uses exists (no LaTeX needed).

    uv run python scripts/check_report.py [--results-root S] [--allow-missing-figures]

Numbers are written ``\\res{name}`` in docs/report/main.tex (or as raw macros starting with
``eda``, ``mae`` etc.); each must be defined in S/numbers.tex or S/dataset.tex, both written by
make_tables.py. ``\\inputtable{name}`` needs S/tables/name.tex and ``\\resfig{name}{...}`` needs
S/figures/name.pdf. Exit code 1 lists what is missing. The XAI example panel shows echo frames,
lives in the git-ignored results/xai/ and is uploaded by hand, so it is not checked.
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREFIXES = ("eda", "mae", "rmse", "r2", "bias", "auroc", "gflops", "latency", "peak")


def check(tex: str, results: Path, figures: bool = True) -> dict[str, list[str]]:
    """Missing macros, tables and figures of ``tex`` against the files in ``results``."""
    used = set(re.findall(r"\\res\{([A-Za-z]+)\}", tex))
    used |= {m for m in re.findall(r"\\([A-Za-z]+)", tex) if m.startswith(PREFIXES)}
    defined = set()
    for path in (results / "numbers.tex", results / "dataset.tex"):
        if path.exists():
            defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", path.read_text()))
    tables = set(re.findall(r"\\inputtable\{([\w-]+)\}", tex))
    figs = set(re.findall(r"\\resfig\{([\w-]+)\}", tex)) if figures else set()
    return {
        "macros": sorted(used - defined),
        "tables": sorted(t for t in tables if not (results / "tables" / f"{t}.tex").exists()),
        "figures": sorted(f for f in figs if not (results / "figures" / f"{f}.pdf").exists()),
        "used": sorted(used),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Are all report numbers, tables, figures there?")
    parser.add_argument("--results-root", type=Path, default=REPO / "results")
    parser.add_argument("--tex", type=Path, default=REPO / "docs" / "report" / "main.tex")
    parser.add_argument("--allow-missing-figures", action="store_true")
    args = parser.parse_args(argv)
    missing = check(args.tex.read_text(), args.results_root, not args.allow_missing_figures)
    print(
        f"{len(missing['used'])} generated macros used, {len(missing['macros'])} undefined: "
        f"{missing['macros']}"
    )
    print(f"missing tables: {missing['tables']}; missing figures: {missing['figures']}")
    return 1 if missing["macros"] or missing["tables"] or missing["figures"] else 0


if __name__ == "__main__":
    sys.exit(main())
