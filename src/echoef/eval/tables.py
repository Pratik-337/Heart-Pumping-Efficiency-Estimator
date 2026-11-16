"""LaTeX tables and number macros for the report, from the results files (task B4/A10).

CLAUDE.md: no result number is typed into the report. ``docs/report/main.tex`` only \\input's
what this module writes:

- ``results/numbers.tex``: one ``\\newcommand`` per number, e.g. ``\\maeFsThirtyTwoTestMOne``
  (TeX macro names cannot hold digits, so digits are spelled out);
- ``results/tables/table_configs.tex`` (Table I, the design), ``table_main.tex`` (Table II),
  ``table_xai.tex`` (Table III) when their inputs exist.
"""

from pathlib import Path

import pandas as pd

from echoef.contracts import CONFIGS, TIER0_PREFIX

DIGITS = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}
NUMBERS = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour"}
FPS = 50  # nominal frame rate for the "seconds" column (spec, Experiment design)


def _word(part) -> str:
    """One macro-name piece: "fs32" -> "FsThirtyTwo", "b0-fs8" -> "BzeroFsEight", "M1" -> "MOne"."""
    part, prefix = str(part), ""
    if part.startswith(TIER0_PREFIX):
        prefix, part = "Bzero", part[len(TIER0_PREFIX) :]
    if part in CONFIGS:
        return prefix + part[:2].capitalize() + NUMBERS[CONFIGS[part][1]]
    w = "".join(ch for ch in part.lower() if ch.isalnum())
    return prefix + "".join(DIGITS.get(ch, ch) for ch in w[:1].upper() + w[1:])


def macro_name(*parts) -> str:
    """``("mae", "fs32", "TEST", "M1")`` -> ``maeFsThirtyTwoTestMOne`` (TeX names: letters only)."""
    name = "".join(_word(p) for p in parts if str(p))
    return name[:1].lower() + name[1:]


def _fmt(value, digits: int = 2) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def numbers_tex(all_runs: pd.DataFrame, cost: pd.DataFrame | None) -> str:
    """``\\newcommand`` lines for every run-level metric of seed 0 and of the ensembles."""
    lines = ["% Written by scripts/make_tables.py from results/all_runs.csv. Do not edit."]
    keep = all_runs[all_runs["seed"].astype(str).isin(["0", "ens"])]
    metrics = ["mae", "mae_ci_lo", "mae_ci_hi", "rmse", "r2", "bias", "auroc_lt50", "auroc_le40"]
    for row in keep.itertuples(index=False):
        seed = "" if str(row.seed) == "0" else "ens"
        for m in metrics:
            name = macro_name(m, row.config_id, row.split, row.mode, seed)
            digits = 3 if m.startswith(("r2", "auroc")) else 2
            lines.append(f"\\newcommand{{\\{name}}}{{{_fmt(getattr(row, m), digits)}}}")
    if cost is not None:
        for row in cost.itertuples(index=False):
            for m, digits in (
                ("gflops_clip", 1),
                ("gflops_video_m2", 1),
                ("latency_ms_m1", 1),
                ("latency_ms_m2", 1),
                ("peak_vram_mb_train", 0),
            ):
                name = macro_name(m, row.config_id)
                lines.append(f"\\newcommand{{\\{name}}}{{{_fmt(getattr(row, m), digits)}}}")
    return "\n".join(lines) + "\n"


def table_configs() -> str:
    """Table I: the 7 configurations (design values, not results)."""
    rows = []
    for config_id, (regime, t, period) in CONFIGS.items():
        window = t * period
        rows.append(
            f"{config_id.upper()[:2]}-{t} & {regime} & {t} & {period} & {window} & "
            f"{window / FPS:.2f} \\\\"
        )
    return (
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Config & Regime & $T$ & $p$ & Window & Seconds \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_main(all_runs: pd.DataFrame, split: str, mode: str = "M1") -> str:
    """Table II: per trained config, MAE [95% CI], RMSE, R^2, AUROC (EF < 50) and cost, seed 0."""
    rows = all_runs[
        (all_runs["split"] == split)
        & (all_runs["mode"] == mode)
        & (all_runs["seed"].astype(str) == "0")
    ].set_index("config_id")
    has_cost = "gflops_video_m2" in rows.columns
    out = []
    for config_id in CONFIGS:
        if config_id not in rows.index:
            continue
        r = rows.loc[config_id]
        cells = [
            f"{config_id.upper()[:2]}-{CONFIGS[config_id][1]}",
            f"{_fmt(r['mae'])} [{_fmt(r['mae_ci_lo'])}, {_fmt(r['mae_ci_hi'])}]",
            _fmt(r["rmse"]),
            _fmt(r["r2"], 3),
            _fmt(r["auroc_lt50"], 3),
        ]
        if has_cost:
            cells += [_fmt(r["gflops_video_m2"], 0), _fmt(r["latency_ms_m1"], 1)]
        out.append(" & ".join(cells) + " \\\\")
    head = "Config & MAE [95\\% CI] & RMSE & $R^2$ & AUROC"
    cols = "lrrrr"
    if has_cost:
        head += " & GFLOPs/video & ms/clip"
        cols += "rr"
    return (
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        + "\n".join(out)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_xai(summary: pd.DataFrame) -> str:
    """Table III: XAI scores per config (occlusion hit rate vs chance, IG LV ratio, AOPC ratio)."""
    occ = summary[summary["method"] == "occlusion"].set_index("config_id")
    ig = summary[summary["method"] == "ig"].set_index("config_id")
    out = []
    for config_id in CONFIGS:
        if config_id not in occ.index:
            continue
        o = occ.loc[config_id]
        lv = ig.loc[config_id, "lv_energy_ratio"] if config_id in ig.index else float("nan")
        out.append(
            f"{config_id.upper()[:2]}-{CONFIGS[config_id][1]} & {int(o['n'])} & "
            f"{_fmt(o.get('hit_rate'))} ({_fmt(o.get('chance_rate'))}) & {_fmt(lv)} & "
            f"{_fmt(o.get('aopc_ratio'))} \\\\"
        )
    return (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Config & $n$ & Hit rate (chance) & LV ratio & AOPC ratio \\\\\n\\midrule\n"
        + "\n".join(out)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def write_tables(results_root: Path) -> list[Path]:
    """Write every table and the macros that its inputs allow; return the written paths."""
    results_root = Path(results_root)
    tables = results_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    written = [tables / "table_configs.tex"]
    written[0].write_text(table_configs())
    all_runs_path = results_root / "all_runs.csv"
    if all_runs_path.exists():
        all_runs = pd.read_csv(all_runs_path)
        cost_path = results_root / "cost.csv"
        cost = pd.read_csv(cost_path) if cost_path.exists() else None
        (results_root / "numbers.tex").write_text(numbers_tex(all_runs, cost))
        split = "TEST" if (all_runs["split"] == "TEST").any() else "VAL"
        (tables / "table_main.tex").write_text(table_main(all_runs, split))
        written += [results_root / "numbers.tex", tables / "table_main.tex"]
    xai_path = tables / "xai_summary.csv"
    if xai_path.exists():
        (tables / "table_xai.tex").write_text(table_xai(pd.read_csv(xai_path)))
        written.append(tables / "table_xai.tex")
    return written
