"""Check that every generated number, table and figure the report uses exists (no LaTeX needed).

    uv run python scripts/check_report.py [--results-root S] [--allow-missing-figures]

Numbers are written ``\\res{name}`` in docs/report/main.tex (or as raw macros starting with
``eda``, ``mae`` etc.); each must be defined in S/numbers.tex or S/dataset.tex, both written by
make_tables.py. ``\\inputtable{name}`` needs S/tables/name.tex and ``\\resfig{name}{...}`` needs
S/figures/name.pdf. The XAI example panel shows echo frames, lives in the git-ignored
results/xai/ and is uploaded by hand, so it is not checked.

Sentences that interpret results carry a condition in a comment line above them, e.g.

    % CLAIM: wilSigFsEightFwEightTest = "significant"
    % CLAIM: |wilMedFsThirtyTwoFsSixtyFourTest| < 1

Each side is a macro name, ``|name|`` (absolute value), a number or a quoted string; operators
are = != < > <= >= and ``contains``. A claim the numbers do not support is listed with its line,
so the sentence below it gets rewritten. Exit code 1 if anything is missing or a claim fails.
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREFIXES = ("eda", "mae", "rmse", "r2", "bias", "auroc", "gflops", "latency", "peak")
CLAIM = re.compile(r"^\s*%\s*CLAIM:\s*(.+?)\s*(<=|>=|!=|=|<|>|contains)\s*(.+?)\s*$")


def _number(text: str) -> float | None:
    """A macro value as a number: "$-$1.20" -> -1.2, "$<$0.001" -> just below 0.001."""
    text = text.replace("$-$", "-").replace(",", "")
    below = text.startswith("$<$")
    try:
        value = float(text.removeprefix("$<$"))
    except ValueError:
        return None
    return value * 0.999 if below else value


def _side(token: str, values: dict[str, str]) -> tuple[object, str | None]:
    """(value, missing macro name or None) of one side of a claim."""
    token = token.strip()
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1], None
    absolute = token.startswith("|") and token.endswith("|")
    name = token.strip("|")
    if re.fullmatch(r"[A-Za-z]+", name):
        if name not in values:
            return None, name
        raw = values[name]
        num = _number(raw)
        return (
            abs(num) if absolute and num is not None else (num if num is not None else raw)
        ), None
    return float(token), None


def check_claims(tex: str, values: dict[str, str]) -> list[str]:
    """Claims in ``tex`` that the macro ``values`` do not support (with their line number)."""
    failed = []
    for lineno, line in enumerate(tex.splitlines(), 1):
        match = CLAIM.match(line)
        if not match:
            continue
        (left, lmiss), op, (right, rmiss) = (
            _side(match.group(1), values),
            match.group(2),
            _side(match.group(3), values),
        )
        if lmiss or rmiss:
            failed.append(f"line {lineno}: {line.strip()} (undefined {lmiss or rmiss})")
            continue
        if op == "contains":
            ok = str(right) in str(left)
        elif op in ("=", "!="):
            ok = (left == right) == (op == "=")
        else:
            ok = {"<": left < right, ">": left > right, "<=": left <= right,
                  ">=": left >= right}[op]  # fmt: skip
        if not ok:
            failed.append(f"line {lineno}: {line.strip()} (values {left!r}, {right!r})")
    return failed


def strip_comments(tex: str) -> str:
    """``tex`` without comments (``%`` to the end of the line, except ``\\%``)."""
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())


def structure(tex: str, bib: str | None = None) -> list[str]:
    """LaTeX slips a compiler would catch, found without one: unbalanced braces, unmatched
    environments, references without a label, citations missing from the bibliography."""
    body, problems, depth, stack = strip_comments(tex), [], 0, []
    for lineno, line in enumerate(body.splitlines(), 1):
        depth += len(re.findall(r"(?<!\\)\{", line)) - len(re.findall(r"(?<!\\)\}", line))
        if depth < 0:
            problems.append(f"line {lineno}: closing brace without an opening one")
            depth = 0
        for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", line):
            if kind == "begin":
                stack.append((env, lineno))
            elif not stack or stack[-1][0] != env:
                problems.append(f"line {lineno}: \\end{{{env}}} does not close "
                                f"{stack[-1] if stack else 'anything'}")  # fmt: skip
            else:
                stack.pop()
    if depth:
        problems.append(f"{depth} unclosed brace(s)")
    problems += [f"\\begin{{{env}}} on line {n} never ends" for env, n in stack]
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    problems += [f"\\ref{{{r}}} has no \\label" for r in
                 sorted(set(re.findall(r"\\ref\{([^}]+)\}", body)) - labels)]  # fmt: skip
    if bib is not None:
        keys = set(re.findall(r"@\w+\{([^,]+),", bib))
        cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", body)
                 for k in group.split(",")}  # fmt: skip
        problems += [f"\\cite{{{k}}} not in refs.bib" for k in sorted(cited - keys)]
    return problems


def check(tex: str, results: Path, figures: bool = True, bib: str | None = None) -> dict:
    """Missing macros, tables and figures of ``tex`` against the files in ``results``, failed
    claims and LaTeX structure problems."""
    body = strip_comments(tex)
    used = set(re.findall(r"\\res\{([A-Za-z]+)\}", body))
    used |= {m for m in re.findall(r"\\([A-Za-z]+)", body) if m.startswith(PREFIXES)}
    values: dict[str, str] = {}
    for path in (results / "numbers.tex", results / "dataset.tex"):
        if path.exists():
            values |= dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", path.read_text()))
    defined = set(values)
    tables = set(re.findall(r"\\inputtable\{([\w-]+)\}", body))
    figs = set(re.findall(r"\\resfig\{([\w-]+)\}", body)) if figures else set()
    return {
        "macros": sorted(used - defined),
        "tables": sorted(t for t in tables if not (results / "tables" / f"{t}.tex").exists()),
        "figures": sorted(f for f in figs if not (results / "figures" / f"{f}.pdf").exists()),
        "claims": check_claims(tex, values),
        "structure": structure(tex, bib),
        "used": sorted(used),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Are all report numbers, tables, figures there?")
    parser.add_argument("--results-root", type=Path, default=REPO / "results")
    parser.add_argument("--tex", type=Path, default=REPO / "docs" / "report" / "main.tex")
    parser.add_argument("--allow-missing-figures", action="store_true")
    args = parser.parse_args(argv)
    bib = args.tex.parent / "refs.bib"
    missing = check(args.tex.read_text(), args.results_root, not args.allow_missing_figures,
                    bib.read_text() if bib.exists() else None)  # fmt: skip
    print(
        f"{len(missing['used'])} generated macros used, {len(missing['macros'])} undefined: "
        f"{missing['macros']}"
    )
    print(f"missing tables: {missing['tables']}; missing figures: {missing['figures']}")
    print(f"{len(missing['claims'])} claims not supported by the numbers:")
    for claim in missing["claims"]:
        print(f"  {claim}")
    print(f"LaTeX structure problems: {missing['structure']}")
    bad = any(missing[k] for k in ("macros", "tables", "figures", "claims", "structure"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
