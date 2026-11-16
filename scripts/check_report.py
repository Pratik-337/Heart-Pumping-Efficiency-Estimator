"""Check that every macro the report uses is defined by the generated files (no LaTeX needed).

    uv run python scripts/check_report.py

Exit code 1 lists undefined macros that start with ``eda``, ``mae`` etc. (generated names).
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GENERATED = [REPO / "results" / "numbers.tex", REPO / "results" / "dataset.tex"]
PREFIXES = ("eda", "mae", "rmse", "r2", "bias", "auroc", "gflops", "latency", "peak")


def main() -> int:
    tex = (REPO / "docs" / "report" / "main.tex").read_text()
    used = {m for m in re.findall(r"\\([A-Za-z]+)", tex) if m.startswith(PREFIXES)}
    defined = set()
    for path in GENERATED:
        if path.exists():
            defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", path.read_text()))
    missing = sorted(used - defined)
    print(f"{len(used)} generated macros used, {len(missing)} undefined: {missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
