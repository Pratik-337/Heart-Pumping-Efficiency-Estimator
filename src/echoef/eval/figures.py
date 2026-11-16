"""Report figures from ``all_runs.csv`` and run predictions (task B4; ``scripts/make_figures.py``).

Spec, "IEEE report plan": vector PDFs one IEEE column wide (3.5 in) with text >= 8 pt. Colours
(validated with the dataviz palette checker on a white page): regime FS = blue, FW = orange
(categorical slots 1-2); EF bands = one-hue ordinal blue ramp. Trained runs have filled markers,
Tier 0 (B0) hollow ones, so identity never rests on colour alone. Text uses ink colours only.
"""

import matplotlib

matplotlib.use("Agg")  # files only; no display needed

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from echoef.contracts import CONFIGS, TIER0_PREFIX  # noqa: E402

WIDTH_IN = 3.5  # one IEEE column
FONT_PT = 8
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#ffffff"
REGIME_COLOR = {"FS": "#2a78d6", "FW": "#eb6834"}
BAND_COLORS = ["#86b6ef", "#3987e5", "#1c5cab"]  # EF <= 40, 41-49, >= 50 (ordinal ramp)
BAND_COLUMNS = [("mae_le40", "EF ≤ 40"), ("mae_41_49", "EF 41–49"), ("mae_ge50", "EF ≥ 50")]
LINE_W, MARKER = 1.5, 5.5  # pt: 2 px line, >= 8 px marker at 96 dpi
# Regime members in T order; FW-32 is the shared FS-32 run.
REGIME_CONFIGS = {"FS": ["fs8", "fs16", "fs32", "fs64"], "FW": ["fw8", "fw16", "fs32", "fw64"]}

STYLE = {
    "font.size": FONT_PT,
    "axes.labelsize": FONT_PT,
    "axes.titlesize": FONT_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
    "pdf.fonttype": 42,  # embed TrueType fonts (IEEE PDF checks)
    "ps.fonttype": 42,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.linewidth": 0.6,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
}


def _figure(height_in: float = 2.4) -> tuple[Figure, object]:
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(WIDTH_IN, height_in), layout="constrained")
        ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)  # recessive, solid
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def _legend_above(ax, ncols: int, handles=None) -> None:
    """Legend in a row above the plot area, so it never covers data."""
    kwargs = {"handles": handles} if handles else {}
    ax.legend(**kwargs, loc="lower left", bbox_to_anchor=(0, 1.01), ncols=ncols, frameon=False,
              borderaxespad=0.0, handlelength=1.4, columnspacing=1.0)  # fmt: skip


def _signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "\u2212")  # typographic minus


def _rows(table: pd.DataFrame, split: str, mode: str, seed) -> pd.DataFrame:
    seeds = table["seed"].astype(str)
    return table[(table["split"] == split) & (table["mode"] == mode) & (seeds == str(seed))]


def mae_vs_t(table: pd.DataFrame, split: str, mode: str, seed=0) -> Figure:
    """MAE (95% bootstrap CI) against T for both regimes, trained and Tier 0 (spec Fig. 2)."""
    rows = _rows(table, split, mode, seed).set_index("config_id")
    tier0 = _rows(table, split, mode, 0).set_index("config_id")
    fig, ax = _figure(2.6)
    handles = []
    with matplotlib.rc_context(STYLE):
        for regime, members in REGIME_CONFIGS.items():
            color = REGIME_COLOR[regime]
            dodge = 0.97 if regime == "FS" else 1.03  # keeps CI bars apart
            sources = [("trained", members, True),
                       ("B0", [TIER0_PREFIX + c for c in members], False)]  # fmt: skip
            for source, ids, filled in sources:
                have = [c for c in ids if c in (rows.index if filled else tier0.index)]
                if not have:
                    continue
                part = (rows if filled else tier0).loc[have]
                t = np.array([CONFIGS[c.removeprefix(TIER0_PREFIX)][1] for c in have]) * dodge
                y = part["mae"].to_numpy()
                err = np.vstack([y - part["mae_ci_lo"], part["mae_ci_hi"] - y])
                style = {"color": color, "linewidth": LINE_W if filled else 1.0, "marker": "o",
                         "markersize": MARKER, "markeredgewidth": 1.2,
                         "markerfacecolor": color if filled else SURFACE,
                         "markeredgecolor": SURFACE if filled else color}  # fmt: skip
                ax.errorbar(t, y, yerr=err, elinewidth=0.8, capsize=0, **style)
                handles.append(Line2D([], [], label=f"{regime}, {source}", **style))
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64], labels=["8", "16", "32", "64"])
        ax.minorticks_off()
        ax.set_xlabel("Frames per clip, T")
        ax.set_ylabel(f"MAE, EF points ({split.lower()}, {mode})")
        _legend_above(ax, 2, handles)
    return fig


def pareto(table: pd.DataFrame, split: str, mode_list=("M1", "M2"), seed=0) -> Figure:
    """MAE against GFLOPs per video, M1 and M2, trained configs (spec Fig. 3).

    Needs the cost columns (``gflops_clip``, ``gflops_video_m2``) joined from ``cost.csv``.
    """
    fig, ax = _figure(2.6)
    markers = {"M1": "o", "M2": "s"}
    points = []
    with matplotlib.rc_context(STYLE):
        for mode in mode_list:
            rows = _rows(table, split, mode, seed)
            rows = rows[~rows["config_id"].str.startswith(TIER0_PREFIX)]
            gflops = rows["gflops_clip"] if mode == "M1" else rows["gflops_video_m2"]
            for regime in ("FS", "FW"):
                pick = rows["regime"].isin([regime, "BOTH"] if regime == "FS" else [regime])
                ax.scatter(gflops[pick], rows.loc[pick, "mae"], s=MARKER**2, marker=markers[mode],
                           color=REGIME_COLOR[regime], edgecolors=SURFACE, linewidths=1.0,
                           label=f"{regime}, {mode}", zorder=3)  # fmt: skip
            points += list(zip(gflops, rows["mae"], rows["config_id"] + " " + mode, strict=True))
        front = _pareto_front(points)
        if len(front) > 1:
            fx, fy, _ = zip(*front, strict=True)
            ax.plot(fx, fy, color=INK_2, linewidth=0.8, zorder=2, label="Pareto front")
        for x, y, name in front:  # M1 labels above-right, M2 below-right: fewer collisions
            dy, va = (3, "bottom") if name.endswith("M1") else (-3, "top")
            ax.annotate(name, (x, y), xytext=(4, dy), textcoords="offset points", va=va,
                        color=INK_2)  # fmt: skip
        ax.set_xscale("log")
        ax.set_xlabel("GFLOPs per video")
        ax.set_ylabel(f"MAE, EF points ({split.lower()})")
        _legend_above(ax, 3)
    return fig


def _pareto_front(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Points no other point beats on both cost and MAE, sorted by cost."""
    pts = sorted(p for p in points if np.isfinite(p[0]) and np.isfinite(p[1]))
    front, best = [], np.inf
    for x, y, name in pts:
        if y < best:
            front.append((x, y, name))
            best = y
    return front


def bland_altman(preds: pd.DataFrame, title: str) -> Figure:
    """Difference (prediction - reference) against their mean, with bias and limits (Fig. 5)."""
    y, y_hat = preds["ef_true"].to_numpy(float), preds["ef_pred"].to_numpy(float)
    diff, mean = y_hat - y, (y_hat + y) / 2
    bias, sd = diff.mean(), diff.std(ddof=1)
    fig, ax = _figure()
    with matplotlib.rc_context(STYLE):
        ax.scatter(mean, diff, s=6, color=REGIME_COLOR["FS"], alpha=0.35, linewidths=0)
        for value, name, color in ((bias, "bias", INK), (bias + 1.96 * sd, "+1.96 SD", INK_2),
                                   (bias - 1.96 * sd, "\u22121.96 SD", INK_2)):  # fmt: skip
            ax.axhline(value, color=color, linewidth=0.9)
            ax.annotate(  # in the right margin, clear of the points
                f"{name}\n{_signed(value)}",
                (1.0, value),
                xycoords=("axes fraction", "data"),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color=color,
                annotation_clip=False,
            )
        ax.set_xlabel("Mean of predicted and reference EF, %")
        ax.set_ylabel("Predicted − reference EF")
        ax.set_title(title, loc="left", color=INK)
    return fig


def mae_by_band(table: pd.DataFrame, split: str, mode: str, seed=0) -> Figure:
    """MAE per reference-EF band for every trained config (grouped bars)."""
    rows = _rows(table, split, mode, seed)
    rows = rows[~rows["config_id"].str.startswith(TIER0_PREFIX)].set_index("config_id")
    order = [c for c in CONFIGS if c in rows.index]
    fig, ax = _figure(2.6)
    width = 0.26
    x = np.arange(len(order))
    with matplotlib.rc_context(STYLE):
        for i, ((col, label), color) in enumerate(zip(BAND_COLUMNS, BAND_COLORS, strict=True)):
            ax.bar(x + (i - 1) * width, rows.loc[order, col], width * 0.88, color=color,
                   label=label, edgecolor="none")  # fmt: skip  # 12% gap = surface gap
        ax.set_xticks(x, labels=[f"{c[:2].upper()}\n{c[2:]}" for c in order])  # "FS" over "8"
        ax.set_ylabel(f"MAE, EF points ({split.lower()}, {mode})")
        _legend_above(ax, 3)
        ax.tick_params(axis="x", length=0)
    return fig
