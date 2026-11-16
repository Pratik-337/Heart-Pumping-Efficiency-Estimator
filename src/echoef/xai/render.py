"""XAI figures (task B6). Spec, "Explainability design", Outputs.

- ``example_panel``: per config, one row per example video: the ED and ES frames with the
  |IG| overlay, and the occlusion curve with ED and ES marked. **Shows echo frames**: written only
  under ``results/xai/`` (git-ignored); it reaches the report by hand (CLAUDE.md, Hard rules).
- ``summary_chart``: the four test scores against T for both regimes. Numbers only, so it may be
  committed with the other figures.

Overlay colour: one hue (palette orange) fading to transparent, never a rainbow map.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from echoef.contracts import CONFIGS  # noqa: E402
from echoef.data.transforms import denormalise  # noqa: E402
from echoef.eval.figures import GRID, INK, INK_2, REGIME_COLOR, STYLE  # noqa: E402

DOUBLE_COLUMN_IN = 7.16  # IEEE two-column width
OVERLAY = LinearSegmentedColormap.from_list(
    "overlay", [(0.92, 0.41, 0.20, 0.0), (0.92, 0.41, 0.20, 0.85)]
)
REGIME_MEMBERS = {"FS": ["fs8", "fs16", "fs32", "fs64"], "FW": ["fw8", "fw16", "fs32", "fw64"]}


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.abs(a)
    top = np.percentile(a, 99.5)
    return np.clip(a / top, 0, 1) if top > 0 else a


def example_panel(examples: list[dict], stats: dict[str, float], title: str) -> Figure:
    """One row per example; each dict has ``clip`` (3, T, H, W) tensor, ``ig`` (T, H, W),
    ``occlusion`` (T,), ``ed_pos``/``es_pos`` (sampled-frame positions), ``label`` (str)."""
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(DOUBLE_COLUMN_IN, 1.75 * len(examples) + 0.3), layout="constrained")
        axes = fig.subplots(len(examples), 3, squeeze=False, width_ratios=[1, 1, 2.2])
        for row, ex in zip(axes, examples, strict=True):
            grey = denormalise(ex["clip"][0], stats["mean"], stats["std"])  # (T, H, W)
            t = grey.shape[0]
            for ax, name in zip(row[:2], ("ED", "ES"), strict=True):
                k = int(np.clip(round(ex[f"{name.lower()}_pos"]), 0, t - 1))
                ax.imshow(grey[k], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                ax.imshow(_norm(ex["ig"][k]), cmap=OVERLAY, vmin=0, vmax=1)
                ax.set_title(f"{name}, frame {k}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            ax = row[2]
            ax.plot(np.arange(t), ex["occlusion"], color=REGIME_COLOR["FS"], linewidth=1.5)
            for name, pos in (("ED", ex["ed_pos"]), ("ES", ex["es_pos"])):
                if 0 <= pos <= t - 1:
                    ax.axvline(pos, color=INK_2, linewidth=0.8)
                    ax.annotate(
                        name,
                        (pos, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(2, -2),
                        textcoords="offset points",
                        va="top",
                        color=INK,
                    )
            ax.set_xlim(-0.5, t - 0.5)
            ax.set_ylabel("|ΔEF| when occluded")
            ax.set_title(ex["label"], loc="left", fontsize=8)
            ax.grid(True, axis="y", color=GRID, linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[-1, 2].set_xlabel("Frame in clip")
        fig.suptitle(title, fontsize=8, x=0.01, ha="left")
    return fig


def summary_chart(summary: pd.DataFrame, sanity: pd.DataFrame | None = None) -> Figure:
    """Four panels: ED/ES hit rate vs chance, LV energy ratio, deletion AOPC / random, and the
    randomisation sanity check (Spearman by stage). ``summary`` has one row per config and
    method with ``hit_rate``, ``chance_rate``, ``lv_energy_ratio``, ``aopc_ratio``."""
    scores = ["hit_rate", "chance_rate", "lv_energy_ratio", "aopc_ratio"]
    summary = summary.reindex(columns=list(dict.fromkeys([*summary.columns, *scores])))
    panels = [
        ("hit_rate", "occlusion", "ED/ES hit rate (occlusion)"),
        ("lv_energy_ratio", "ig", "LV energy ratio (IG)"),
        ("aopc_ratio", "occlusion", "Deletion AOPC ÷ random (occlusion)"),
    ]
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(DOUBLE_COLUMN_IN, 4.2), layout="constrained")
        axes = fig.subplots(2, 2).ravel()
        for ax, (col, method, label) in zip(axes[:3], panels, strict=True):
            part = summary[summary["method"] == method].set_index("config_id")
            for regime, members in REGIME_MEMBERS.items():
                have = [c for c in members if c in part.index]
                if not have:
                    continue
                t = [CONFIGS[c][1] for c in have]
                ax.plot(
                    t,
                    part.loc[have, col],
                    color=REGIME_COLOR[regime],
                    linewidth=1.5,
                    marker="o",
                    markersize=5,
                    markeredgecolor="white",
                    label=regime,
                )
                if col == "hit_rate":
                    ax.plot(
                        t,
                        part.loc[have, "chance_rate"],
                        color=REGIME_COLOR[regime],
                        linewidth=0.8,
                        marker="o",
                        markersize=4,
                        markerfacecolor="white",
                        label=f"{regime} chance",
                    )
            if col in ("lv_energy_ratio", "aopc_ratio"):
                ax.axhline(1.0, color=INK_2, linewidth=0.8)  # = no better than uniform / random
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64], labels=["8", "16", "32", "64"])
            ax.minorticks_off()
            ax.set_xlabel("Frames per clip, T")
            ax.set_title(label, loc="left", fontsize=8)
            ax.grid(True, axis="y", color=GRID, linewidth=0.6)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=8)
        ax = axes[3]
        if sanity is not None and len(sanity):
            stages = list(dict.fromkeys(sanity["stage"]))
            for method, color in (("ig", REGIME_COLOR["FS"]), ("gradcam", REGIME_COLOR["FW"])):
                m = sanity[sanity["method"] == method].groupby("stage", sort=False)["spearman"]
                means = m.mean().reindex(stages)
                ax.plot(
                    range(len(stages)),
                    means,
                    color=color,
                    linewidth=1.5,
                    marker="o",
                    markersize=5,
                    markeredgecolor="white",
                    label=method,
                )
            ax.set_xticks(range(len(stages)), labels=stages)
            ax.axhline(0.0, color=INK_2, linewidth=0.8)
            ax.set_xlabel("Weights randomised from fc down to")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=8)
        ax.set_title("Sanity check: Spearman with original map (FS-32)", loc="left", fontsize=8)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        for a in axes:
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
    return fig
