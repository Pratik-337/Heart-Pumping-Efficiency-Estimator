"""Task B10: one zip of the report for Overleaf (no LaTeX needed here).

    uv run python scripts/bundle_report.py [--with-xai-examples] [--out PATH]

The zip holds docs/report/main.tex and refs.bib, and a results/ folder next to them with the
generated numbers.tex, dataset.tex, tables/*.tex and figures/*.pdf; main.tex finds results/ there
first. --with-xai-examples adds results/xai/fs32/examples.pdf, the panel with echo frames
(CLAUDE.md: never committed; ask your guide whether frames may appear in the report). The zip is
written to results/xai/ (git-ignored) by default, so it is never committed either.
"""

import argparse
import logging
import zipfile
from pathlib import Path

from echoef.config import load_paths
from echoef.utils import seed_everything, setup_logging

log = logging.getLogger("bundle_report")
REPORT = Path(__file__).resolve().parents[1] / "docs" / "report"
XAI_PANEL = Path("xai") / "fs32" / "examples.pdf"


def bundle(results: Path, out: Path, with_xai: bool = False, report: Path = REPORT) -> list[str]:
    """Write the zip ``out``; return the names inside it."""
    results = Path(results)
    files = {"main.tex": report / "main.tex", "refs.bib": report / "refs.bib"}
    for name in ("numbers.tex", "dataset.tex"):
        if (results / name).exists():
            files[f"results/{name}"] = results / name
    for pattern in ("tables/*.tex", "figures/*.pdf"):
        for path in sorted(results.glob(pattern)):
            files[f"results/{path.relative_to(results)}"] = path
    if with_xai and (results / XAI_PANEL).exists():
        files[f"results/{XAI_PANEL}"] = results / XAI_PANEL
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in files.items():
            zf.write(path, name)
    return list(files)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Zip the report for Overleaf (B10).")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="default: results/xai/report.zip")
    parser.add_argument("--with-xai-examples", action="store_true", help="adds echo frames")
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    seed_everything(0)
    results = args.results_root or load_paths(args.paths)["results_root"]
    out = args.out or results / "xai" / "report.zip"
    names = bundle(results, out, args.with_xai_examples)
    frames = "with the XAI panel" if f"results/{XAI_PANEL}" in names else "no echo frames"
    log.info("wrote %s with %d files (%s)", out, len(names), frames)


if __name__ == "__main__":
    main()
