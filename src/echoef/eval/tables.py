"""LaTeX tables and number macros for the report, from the results files (task B4/A10).

CLAUDE.md: no result number is typed into the report. ``docs/report/main.tex`` only \\input's
what this module writes:

- ``results/numbers.tex``: one ``\\newcommand`` per number, e.g. ``\\maeFsThirtyTwoTestMOne``
  (TeX macro names cannot hold digits, so digits are spelled out). Besides the per-run metrics it
  holds what the Results text needs from the other tables (seed spread, paired tests, hardware
  control, padding check, XAI scores, demo choice) and words derived from the numbers (best
  configuration, Pareto front, "larger"/"smaller"), so the prose stays true for any result;
- ``results/tables/table_configs.tex`` (Table I, the design), ``table_main.tex`` (accuracy),
  ``table_cost.tex``, ``table_wilcoxon.tex`` and ``table_xai.tex`` when their inputs exist.

The report uses the macros through ``\\res{name}``, which prints "??" for a missing name, and
``scripts/check_report.py`` fails the final run if any is missing.
"""

from pathlib import Path

import pandas as pd

from echoef.contracts import CONFIGS, MODES, SPLITS, TIER0_PREFIX
from echoef.eval.figures import _pareto_front

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
    w = "".join(ch for ch in part if ch.isalnum())
    if w.islower() or w.isupper():  # "frames_median", "TEST", "M1"; camelCase keeps its humps
        w = w.lower()
    return prefix + "".join(DIGITS.get(ch, ch) for ch in w[:1].upper() + w[1:])


def macro_name(*parts) -> str:
    """``("mae", "fs32", "TEST", "M1")`` -> ``maeFsThirtyTwoTestMOne`` (TeX names: letters only)."""
    name = "".join(_word(p) for p in parts if str(p))
    return name[:1].lower() + name[1:]


def _fmt(value, digits: int = 2) -> str:
    """Fixed digits; a typographic minus ("$-$") for negatives, "--" for a missing value."""
    if pd.isna(value):
        return "--"
    text = f"{abs(value):.{digits}f}"
    return f"$-${text}" if value < 0 and float(text) != 0 else text


def _cmd(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def _p(value) -> str:
    """A p value for the text: three decimals, or "$<$0.001"."""
    return "--" if pd.isna(value) else ("$<$0.001" if value < 0.001 else f"{value:.3f}")


def config_label(config_id: str) -> str:
    """``"fs8"`` -> ``"FS-8"``; Tier 0 ids (``"b0-fs8"``) get the label of their setting."""
    base = config_id.removeprefix(TIER0_PREFIX)
    return f"{base[:2].upper()}-{CONFIGS[base][1]}"


def _and(items: list[str]) -> str:
    """``["a", "b", "c"]`` -> ``"a, b and c"``."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _single_seeds(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Rows of individual trained seeds (no ensembles, no Tier 0)."""
    seeds = all_runs["seed"].astype(str)
    return all_runs[(seeds != "ens") & ~all_runs["config_id"].str.startswith(TIER0_PREFIX)]


def _seed0(all_runs: pd.DataFrame, split: str, mode: str, tier0: bool = False) -> pd.DataFrame:
    rows = all_runs[
        (all_runs["split"] == split)
        & (all_runs["mode"] == mode)
        & (all_runs["seed"].astype(str) == "0")
    ]
    is_tier0 = rows["config_id"].str.startswith(TIER0_PREFIX)
    return rows[is_tier0 if tier0 else ~is_tier0].set_index("config_id")


def derived_tex(all_runs: pd.DataFrame) -> list[str]:
    """Macros derived from ``all_runs``: seed mean and SD per config, best and worst config,
    Tier 0 range, M2 against M1, and the Pareto front of the cost figure. Words, not only
    numbers, come from the data, so a sentence like "the lowest MAE was reached by ..." is true
    for any result."""
    lines = []
    single = _single_seeds(all_runs)
    present = [
        (sp, m)
        for sp in SPLITS
        for m in MODES
        if ((all_runs["split"] == sp) & (all_runs["mode"] == m)).any()
    ]
    for split, mode in present:
        part = single[(single["split"] == split) & (single["mode"] == mode)]
        if len(part):
            stats = part.groupby("config_id")["mae"].agg(["mean", "std", "size"])
            for config_id, r in stats.iterrows():
                lines.append(_cmd(macro_name("maeMean", config_id, split, mode), _fmt(r["mean"])))
                lines.append(_cmd(macro_name("maeSd", config_id, split, mode), _fmt(r["std"])))
            best = stats["mean"].idxmin()
            lines += [
                _cmd(macro_name("bestMeanConfig", split, mode), config_label(best)),
                _cmd(macro_name("bestMeanMae", split, mode), _fmt(stats["mean"].min())),
                _cmd(macro_name("maxSeedSd", split, mode), _fmt(stats["std"].max())),
                _cmd(macro_name("minSeeds", split, mode), str(int(stats["size"].min()))),
            ]
        for prefix, rows in (
            ("", _seed0(all_runs, split, mode)),
            ("tierZero", _seed0(all_runs, split, mode, tier0=True)),
        ):
            if rows.empty:
                continue
            lo, hi = rows["mae"].idxmin(), rows["mae"].idxmax()
            lines += [
                _cmd(macro_name(prefix, "bestConfig", split, mode), config_label(lo)),
                _cmd(macro_name(prefix, "bestMae", split, mode), _fmt(rows.loc[lo, "mae"])),
                _cmd(macro_name(prefix, "worstConfig", split, mode), config_label(hi)),
                _cmd(macro_name(prefix, "worstMae", split, mode), _fmt(rows.loc[hi, "mae"])),
            ]
    for split in dict.fromkeys(sp for sp, _ in present):
        m1, m2 = _seed0(all_runs, split, "M1"), _seed0(all_runs, split, "M2")
        both = m1.index.intersection(m2.index)
        if len(both):
            gain = (m1.loc[both, "mae"] - m2.loc[both, "mae"]).sort_values()
            lines += [
                _cmd(macro_name("mTwoBetterCount", split), str(int((gain > 0).sum()))),
                _cmd(macro_name("trainedCount", split), str(len(gain))),
                _cmd(macro_name("mTwoGainMax", split), _fmt(gain.iloc[-1])),
                _cmd(macro_name("mTwoGainMaxConfig", split), config_label(gain.index[-1])),
                _cmd(macro_name("mTwoGainMin", split), _fmt(gain.iloc[0])),
                _cmd(macro_name("mTwoGainMinConfig", split), config_label(gain.index[0])),
            ]
        if {"gflops_clip", "gflops_video_m2"} <= set(all_runs.columns) and len(both):
            points = [
                (m1.loc[c, "gflops_clip"], m1.loc[c, "mae"], f"{config_label(c)} (M1)")
                for c in m1.index
            ] + [
                (m2.loc[c, "gflops_video_m2"], m2.loc[c, "mae"], f"{config_label(c)} (M2)")
                for c in m2.index
            ]
            front = [name for _, _, name in _pareto_front(points)]
            if front:
                lines.append(_cmd(macro_name("paretoFront", split), _and(front)))
    return lines


def tables_tex(tables_dir: Path) -> list[str]:
    """Macros from the analysis tables that exist: paired tests, hardware control, padding
    check, XAI scores and sanity check, and the demo choice."""
    tables_dir, lines = Path(tables_dir), []
    for name in ("wilcoxon_holm.csv", "wilcoxon_holm_val.csv"):
        if not (tables_dir / name).exists():
            continue
        wil = pd.read_csv(tables_dir / name)
        split = str(wil["split"].iloc[0]) if "split" in wil and len(wil) else "VAL"
        for r in wil.itertuples(index=False):
            a, b = r.comparison.split(" vs ")
            for key, value in (
                ("Med", _fmt(r.median_diff)),
                ("Lo", _fmt(r.ci_lo)),
                ("Hi", _fmt(r.ci_hi)),
                ("P", _p(r.p_holm)),
                ("Sig", "significant" if r.significant else "not significant"),
                ("Dir", "larger" if r.median_diff > 0 else "smaller"),
            ):
                lines.append(_cmd(macro_name("wil" + key, a, b, split), value))
        lines += [
            _cmd(macro_name("wilSigCount", split), str(int(wil["significant"].sum()))),
            _cmd(macro_name("wilCount", split), str(len(wil))),
        ]
    if (tables_dir / "hardware_control.csv").exists():
        hw = pd.read_csv(tables_dir / "hardware_control.csv")
        for r in hw.itertuples(index=False):
            key = (r.config_id, r.split, r.mode)
            lines += [
                _cmd(macro_name("hwMaeControl", *key), _fmt(r.mae_control)),
                _cmd(macro_name("hwMaeMain", *key), _fmt(r.mae_main)),
                _cmd(macro_name("hwGapAbs", *key), _fmt(abs(r.hardware_gap))),
                _cmd(macro_name("hwSeedGapAbs", *key), _fmt(abs(r.seed_gap))),
            ]
    if (tables_dir / "unpadded.csv").exists():
        unpad = pd.read_csv(tables_dir / "unpadded.csv")
        unpad = unpad[(unpad["seed"].astype(str) == "0") & (unpad["n_padded"] > 0)]
        for r in unpad.itertuples(index=False):
            key = (r.config_id, r.split, r.mode)
            lines += [
                _cmd(macro_name("unpadMae", *key), _fmt(r.mae_unpadded)),
                _cmd(macro_name("unpadMaeAll", *key), _fmt(r.mae_all)),
                _cmd(macro_name("unpadN", *key), str(int(r.n_padded))),
            ]
    for name in ("xai_summary.csv", "xai_summary_val.csv"):
        if (tables_dir / name).exists():
            lines += xai_tex(
                pd.read_csv(tables_dir / name), "" if name == "xai_summary.csv" else "val"
            )
    for name in ("xai_sanity.csv", "xai_sanity_val.csv"):
        if (tables_dir / name).exists():
            sanity, tag = pd.read_csv(tables_dir / name), "" if name == "xai_sanity.csv" else "val"
            for r in sanity.itertuples(index=False):
                lines.append(
                    _cmd(macro_name("sanity", r.method, r.stage, tag), _fmt(r.spearman_mean))
                )
            lines += [
                _cmd(macro_name("sanityConfig", tag), config_label(sanity["config_id"].iloc[0])),
                _cmd(macro_name("sanityN", tag), str(int(sanity["n"].max()))),
            ]
    if (tables_dir / "demo_choice.csv").exists():
        demo = pd.read_csv(tables_dir / "demo_choice.csv")
        chosen = demo[demo["chosen"].astype(bool)].iloc[0]
        lines += [
            _cmd("demoConfig", config_label(chosen["config_id"])),
            _cmd("demoSeeds", str(int(chosen["n_seeds"]))),
            _cmd("demoMaeVal", _fmt(chosen["mae"])),
        ]
    return lines


def xai_tex(summary: pd.DataFrame, tag: str = "") -> list[str]:
    """XAI macros per config and method, and counts over the configs (``tag``: "" or "val")."""
    lines = []
    fields = (
        ("hit_rate", "Hit", 100, 0),
        ("chance_rate", "Chance", 100, 0),
        ("aopc_ratio", "Aopc", 1, 2),
        ("lv_energy_ratio", "Lv", 1, 2),
        ("n", "N", 1, 0),
        ("n_skipped", "Skipped", 1, 0),
    )
    for r in summary.to_dict("records"):
        for col, key, scale, digits in fields:
            if col in r and not pd.isna(r[col]):
                value = _fmt(r[col] * scale, digits)
                lines.append(_cmd(macro_name("xai" + key, r["config_id"], r["method"], tag), value))
        if "hit_p" in r and not pd.isna(r["hit_p"]):
            lines.append(_cmd(macro_name("xaiP", r["config_id"], r["method"], tag), _p(r["hit_p"])))
    occ = summary[summary["method"] == "occlusion"]
    ig = summary[summary["method"] == "ig"]
    lines += [
        _cmd(macro_name("xaiConfigCount", tag), str(len(occ))),
        _cmd(macro_name("xaiHitSigCount", tag), str(int((occ["hit_p"] < 0.05).sum()))),
        _cmd(macro_name("xaiAopcAboveOneCount", tag), str(int((occ["aopc_ratio"] > 1).sum()))),
        _cmd(macro_name("xaiLvAboveOneCount", tag), str(int((ig["lv_energy_ratio"] > 1).sum()))),
    ]
    return lines


def numbers_tex(all_runs: pd.DataFrame, cost: pd.DataFrame | None) -> str:
    """``\\newcommand`` lines for every run-level metric of seed 0 and of the ensembles."""
    lines = ["% Written by scripts/make_tables.py from results/all_runs.csv. Do not edit."]
    keep = all_runs[all_runs["seed"].astype(str).isin(["0", "ens"])]
    metrics = [
        "mae",
        "mae_ci_lo",
        "mae_ci_hi",
        "rmse",
        "r2",
        "bias",
        "loa_lo",
        "loa_hi",
        "auroc_lt50",
        "auroc_le40",
        "mae_le40",
        "mae_41_49",
        "mae_ge50",
        "calib_slope",
    ]
    for row in keep.itertuples(index=False):
        seed = "" if str(row.seed) == "0" else "ens"
        for m in metrics:
            name = macro_name(m, row.config_id, row.split, row.mode, seed)
            digits = 3 if m.startswith(("r2", "auroc")) else 2
            lines.append(_cmd(name, _fmt(getattr(row, m, float("nan")), digits)))
        lines.append(
            _cmd(macro_name("n", row.config_id, row.split, row.mode, seed), f"{int(row.n):,}")
        )
    if cost is not None:
        for row in cost.itertuples(index=False):
            for m, digits in (
                ("gflops_clip", 1),
                ("gflops_video_m2", 1),
                ("mean_clips_m2", 1),
                ("latency_ms_m1", 1),
                ("latency_ms_m2", 1),
                ("peak_vram_mb_infer", 0),
                ("peak_vram_mb_train", 0),
                ("epoch_time_s", 0),
            ):
                name = macro_name(m, row.config_id)
                lines.append(_cmd(name, _fmt(getattr(row, m), digits)))
        lines.append(_cmd("paramsMillions", _fmt(cost["params"].iloc[0] / 1e6, 1)))
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
    """Accuracy table (two columns wide): per trained config, seed 0 with its bootstrap CI, the
    mean and SD over seeds, RMSE, R^2, Bland-Altman bias with limits, both AUROCs, the M2 MAE and
    the Tier 0 MAE at the same setting (B0, no training)."""
    rows = _seed0(all_runs, split, mode)
    m2 = _seed0(all_runs, split, "M2")
    tier0 = _seed0(all_runs, split, mode, tier0=True)
    single = _single_seeds(all_runs)
    single = single[(single["split"] == split) & (single["mode"] == mode)]
    spread = single.groupby("config_id")["mae"].agg(["mean", "std", "size"])
    out = []
    for config_id, (_, t, period) in CONFIGS.items():
        if config_id not in rows.index:
            continue
        r = rows.loc[config_id]
        seeds = (
            f"{_fmt(spread.loc[config_id, 'mean'])} $\\pm$ {_fmt(spread.loc[config_id, 'std'])}"
            if config_id in spread.index and spread.loc[config_id, "size"] > 1
            else "--"
        )
        cells = [
            config_label(config_id),
            str(t),
            str(period),
            f"{_fmt(r['mae'])} [{_fmt(r['mae_ci_lo'])}, {_fmt(r['mae_ci_hi'])}]",
            seeds,
            _fmt(r["rmse"]),
            _fmt(r["r2"], 3),
            f"{_fmt(r['bias'])} [{_fmt(r['loa_lo'])}, {_fmt(r['loa_hi'])}]",
            _fmt(r["auroc_lt50"], 3),
            _fmt(r["auroc_le40"], 3),
            _fmt(m2.loc[config_id, "mae"]) if config_id in m2.index else "--",
            _fmt(tier0.loc[TIER0_PREFIX + config_id, "mae"])
            if TIER0_PREFIX + config_id in tier0.index
            else "--",
        ]
        out.append(" & ".join(cells) + " \\\\")
    head = (
        "Config & $T$ & $p$ & MAE [95\\% CI] & Seeds & RMSE & $R^2$ & Bias [LoA] & "
        "AUROC$_{<50}$ & AUROC$_{\\le 40}$ & MAE M2 & Tier 0"
    )
    return (
        "\\begin{tabular}{lrrlcrrlrrrr}\n\\toprule\n"
        + head
        + " \\\\\n\\midrule\n"
        + "\n".join(out)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_cost(cost: pd.DataFrame) -> str:
    """Cost table (one column wide), one GPU (D-23/D-31): GFLOPs per clip, clips and GFLOPs per
    video in M2, latency per video in M1 and M2, peak training memory and epoch time."""
    cost = cost.set_index("config_id")
    out = []
    for config_id in CONFIGS:
        if config_id not in cost.index:
            continue
        r = cost.loc[config_id]
        cells = [
            config_label(config_id),
            _fmt(r["gflops_clip"], 0),
            _fmt(r["mean_clips_m2"], 1),
            _fmt(r["gflops_video_m2"], 0),
            _fmt(r["latency_ms_m1"], 1),
            _fmt(r["latency_ms_m2"], 1),
            _fmt(r["peak_vram_mb_train"] / 1024, 1),
            _fmt(r["epoch_time_s"] / 60, 1),
        ]
        out.append(" & ".join(cells) + " \\\\")
    head = (
        "Config & \\multicolumn{1}{c}{GFLOPs} & Clips & \\multicolumn{1}{c}{GFLOPs} & "
        "\\multicolumn{2}{c}{ms per video} & GB & min \\\\\n"
        " & clip & M2 & video M2 & M1 & M2 & train & epoch"
    )
    return (
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        + head
        + " \\\\\n\\midrule\n"
        + "\n".join(out)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_wilcoxon(wil: pd.DataFrame) -> str:
    """The 9 pre-specified paired comparisons: median of err_A - err_B with its 95% bootstrap CI,
    Holm-adjusted p, and whether it is significant at 0.05."""
    out = []
    for r in wil.itertuples(index=False):
        a, b = r.comparison.split(" vs ")
        cells = [
            f"{config_label(a)} vs {config_label(b)}",
            f"{_fmt(r.median_diff)} [{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}]",
            _p(r.p_holm),
            "yes" if r.significant else "no",
        ]
        out.append(" & ".join(cells) + " \\\\")
    return (
        "\\begin{tabular}{llrc}\n\\toprule\n"
        "Comparison (A vs B) & Median A$-$B [95\\% CI] & $p_\\mathrm{Holm}$ & Sig. \\\\\n"
        "\\midrule\n" + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n"
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
            f"{config_label(config_id)} & {int(o['n'])} & "
            f"{_fmt(o.get('hit_rate'))} ({_fmt(o.get('chance_rate'))}) & {_fmt(lv)} & "
            f"{_fmt(o.get('aopc_ratio'))} \\\\"
        )
    return (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Config & $n$ & Hit rate (chance) & LV ratio & AOPC ratio \\\\\n\\midrule\n"
        + "\n".join(out)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def eda_tex(eda_dir: Path) -> str:
    """``\\newcommand`` lines for the dataset facts in ``results/eda/*.csv`` (A2 EDA)."""
    lines = ["% Written by scripts/make_tables.py from results/eda/*.csv. Do not edit."]
    split = pd.read_csv(Path(eda_dir) / "split_summary.csv").set_index("split")
    cols = {
        "n": 0,
        "n_has_trace": 0,
        "frames_median": 0,
        "frames_min": 0,
        "frames_max": 0,
        "ef_mean": 1,
        "ef_sd": 1,
        "n_ef_le40": 0,
        "n_ef_41_49": 0,
        "n_ef_ge50": 0,
        "edes_gap_median": 0,
        "fps_median": 0,
    }
    for name, row in split.iterrows():
        for col, digits in cols.items():
            value = f"{row[col]:,.{digits}f}" if digits == 0 else f"{row[col]:.{digits}f}"
            lines.append(f"\\newcommand{{\\{macro_name('eda', col, name)}}}{{{value}}}")
    for csv, col in (("padding_share.csv", "share_padded"), ("edes_fit.csv", "share_skip")):
        table = pd.read_csv(Path(eda_dir) / csv)
        for row in table[table["split"] == "ALL"].itertuples(index=False):
            pct = f"{100 * getattr(row, col):.1f}"
            lines.append(f"\\newcommand{{\\{macro_name('eda', col, row.config_id)}}}{{{pct}}}")
    return "\n".join(lines) + "\n"


SUMMARY_METRICS = (
    "mae",
    "rmse",
    "mse",
    "r2",
    "bias",
    "loa_lo",
    "loa_hi",
    "auroc_lt50",
    "auroc_le40",
)
ENSEMBLE_METRICS = ("mae", "rmse", "mse", "r2")


def metrics_summary(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per config, split and mode: seed mean and SD of each metric in ``all_runs``, and
    the seed ensemble's MAE, RMSE, MSE and R^2 (``ens_*``).

    MSE is derived per run as RMSE^2 (EF points^2) before averaging, so ``mse_mean`` is the mean
    of the seeds' MSEs, not ``rmse_mean`` squared. Tier 0 (one B0 model) has ``n_seeds`` 1, no
    SD and no ensemble. Rows: TEST first, then M1 before M2, trained configs in spec order, then
    Tier 0. Numbers only; ``write_tables`` saves it as ``tables/metrics_summary.csv``.
    """
    runs = all_runs.assign(mse=all_runs["rmse"] ** 2)
    is_ens = runs["seed"].astype(str) == "ens"
    keys = ["config_id", "split", "mode"]
    grouped = runs[~is_ens].groupby(keys)
    out = pd.concat({"n_seeds": grouped.size(), "n_videos": grouped["n"].max()}, axis=1)
    for metric in (m for m in SUMMARY_METRICS if m in runs):
        out[f"{metric}_mean"] = grouped[metric].mean()
        out[f"{metric}_sd"] = grouped[metric].std()
    ens = runs[is_ens].set_index(keys)[[m for m in ENSEMBLE_METRICS if m in runs]]
    out = out.join(ens.add_prefix("ens_")).reset_index()
    rank = {c: i for i, c in enumerate([*CONFIGS, *(TIER0_PREFIX + c for c in CONFIGS)])}
    order = pd.DataFrame(
        {
            "split": out["split"].ne("TEST"),
            "name": out["split"],
            "mode": out["mode"].map({m: i for i, m in enumerate(MODES)}),
            "config": out["config_id"].map(rank),
        }
    ).sort_values(["split", "name", "mode", "config"])
    return out.loc[order.index].reset_index(drop=True)


def _pm(mean, sd, digits: int) -> str:
    """ "mean ± SD", or the mean alone when there is no SD; "–" when the value is missing."""
    if mean is None or pd.isna(mean):
        return "–"
    if sd is None or pd.isna(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def metrics_markdown(summary: pd.DataFrame) -> str:
    """``metrics_summary`` as GitHub Markdown, one table per split and mode (TEST M1, the primary
    endpoint, first); ``write_tables`` saves it as ``tables/metrics_summary.md``."""
    lines = [
        "# Accuracy of every model",
        "",
        "Written by `scripts/make_tables.py` from `results/all_runs.csv`; do not edit by hand. "
        "Errors are in EF percentage points (MSE in points²). Trained configurations: mean ± SD "
        "over seeds; *Ensemble MAE* averages the seeds' predictions first. *B0*: EchoNet's "
        "released 32-frame weights fed each setting without training (Tier 0). M1: one clip per "
        "video (primary endpoint); M2: average over half-overlapping clips. *95% LoA*: "
        "Bland–Altman limits of agreement. *AUROC*: detecting EF < 50 and EF ≤ 40. All columns "
        "per seed and the cost columns are in `results/all_runs.csv`.",
    ]
    head = (
        "| Model | Seeds | Videos | MAE | RMSE | MSE | R² | Bias | 95% LoA "
        "| AUROC EF<50 | AUROC EF≤40 | Ensemble MAE |"
    )
    for (split, mode), part in summary.groupby(["split", "mode"], sort=False):
        name = "single clip" if mode == "M1" else "multi-clip average"
        lines += ["", f"## {split} split, {mode} ({name})", "", head, "|---" * 12 + "|"]
        for r in part.to_dict("records"):
            c = r["config_id"]
            label = config_label(c)
            label = f"B0 at {label}" if c.startswith(TIER0_PREFIX) else label
            lo, hi = r.get("loa_lo_mean"), r.get("loa_hi_mean")
            loa = "–" if lo is None or pd.isna(lo) else f"{lo:.1f} to {hi:.1f}"
            cells = [
                label,
                str(int(r["n_seeds"])),
                str(int(r["n_videos"])),
                _pm(r.get("mae_mean"), r.get("mae_sd"), 2),
                _pm(r.get("rmse_mean"), r.get("rmse_sd"), 2),
                _pm(r.get("mse_mean"), r.get("mse_sd"), 1),
                _pm(r.get("r2_mean"), r.get("r2_sd"), 3),
                _pm(r.get("bias_mean"), r.get("bias_sd"), 2),
                loa,
                _pm(r.get("auroc_lt50_mean"), r.get("auroc_lt50_sd"), 3),
                _pm(r.get("auroc_le40_mean"), r.get("auroc_le40_sd"), 3),
                _pm(r.get("ens_mae"), None, 2),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_tables(results_root: Path) -> list[Path]:
    """Write every table and the macros that its inputs allow; return the written paths."""
    results_root = Path(results_root)
    tables = results_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    written = [tables / "table_configs.tex"]
    written[0].write_text(table_configs())
    if (results_root / "eda" / "split_summary.csv").exists():
        (results_root / "dataset.tex").write_text(eda_tex(results_root / "eda"))
        written.append(results_root / "dataset.tex")
    all_runs_path = results_root / "all_runs.csv"
    if all_runs_path.exists():
        all_runs = pd.read_csv(all_runs_path)
        cost_path = results_root / "cost.csv"
        cost = pd.read_csv(cost_path) if cost_path.exists() else None
        lines = derived_tex(all_runs) + tables_tex(tables)
        (results_root / "numbers.tex").write_text(
            numbers_tex(all_runs, cost) + "\n".join(lines) + "\n"
        )
        split = "TEST" if (all_runs["split"] == "TEST").any() else "VAL"
        (tables / "table_main.tex").write_text(table_main(all_runs, split))
        written += [results_root / "numbers.tex", tables / "table_main.tex"]
        summary = metrics_summary(all_runs)
        summary.to_csv(tables / "metrics_summary.csv", index=False)
        (tables / "metrics_summary.md").write_text(metrics_markdown(summary))
        written += [tables / "metrics_summary.csv", tables / "metrics_summary.md"]
        if cost is not None:
            (tables / "table_cost.tex").write_text(table_cost(cost))
            written.append(tables / "table_cost.tex")
    for name in ("wilcoxon_holm.csv", "wilcoxon_holm_val.csv"):
        if (tables / name).exists():
            (tables / "table_wilcoxon.tex").write_text(table_wilcoxon(pd.read_csv(tables / name)))
            written.append(tables / "table_wilcoxon.tex")
            break
    xai_path = tables / "xai_summary.csv"
    if xai_path.exists():
        (tables / "table_xai.tex").write_text(table_xai(pd.read_csv(xai_path)))
        written.append(tables / "table_xai.tex")
    return written
