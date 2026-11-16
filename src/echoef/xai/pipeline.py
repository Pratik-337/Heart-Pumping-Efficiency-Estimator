"""XAI run over a video sample for one trained config (task B7; ``scripts/run_xai.py``).

Spec, "Explainability design": a fixed-seed sample of traced videos stratified by EF band
(35 % EF <= 40, 15 % 41-49, 50 % >= 50); each video gets an ED/ES-centred clip (skipped when
|ED - ES| >= W); four methods (occlusion, IG with zero and blurred baselines, 3D Grad-CAM) are
scored by the numeric tests in ``xai_metrics``. One row per video and method in
``contracts.XAI_COLUMNS`` plus ``EXTRA_COLUMNS``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from echoef.contracts import XAI_COLUMNS
from echoef.data.dataset import load_clip
from echoef.data.sampling import edes_start
from echoef.data.tracings import lv_mask
from echoef.xai.attribution import (
    IG_STEPS,
    frame_importance,
    gradcam3d,
    integrated_gradients,
    occlusion_curve,
    predict_variants,
)
from echoef.xai.xai_metrics import (
    RANDOMISATION_STAGES,
    clip_positions,
    deletion_aopc,
    hit_mask,
    hit_rate,
    hit_test,
    lv_energy_ratio,
    random_aopc,
    randomise_cascade,
    spearman,
)

log = logging.getLogger(__name__)

METHODS = ("occlusion", "ig", "ig_blur", "gradcam")
TEMPORAL = {"occlusion", "ig", "ig_blur"}  # rank frames: hit rate, deletion test
SPATIAL = {"ig", "ig_blur", "gradcam"}  # pixel maps: LV energy ratio
BAND_SHARES = (("EF <= 40", 0.35), ("EF 41-49", 0.15), ("EF >= 50", 0.50))
EXTRA_COLUMNS = ["T", "hit_chance", "ef_true", "ef_pred", "band"]
SANITY_COLUMNS = ["file_name", "config_id", "method", "stage", "spearman"]


def ef_band(ef: float) -> str:
    return "EF <= 40" if ef <= 40 else ("EF 41-49" if ef < 50 else "EF >= 50")


def draw_sample(videos: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """``n`` traced videos stratified by EF band (spec shares), drawn once with ``seed``."""
    traced = videos[videos["has_trace"]].assign(band=lambda d: d["EF"].map(ef_band))
    quotas = [round(n * share) for _, share in BAND_SHARES[:-1]]
    quotas.append(n - sum(quotas))
    parts = []
    for (band, _), quota in zip(BAND_SHARES, quotas, strict=True):
        pool = traced[traced["band"] == band]
        if len(pool) < quota:
            raise ValueError(f"only {len(pool)} traced videos in {band}, need {quota}")
        parts.append(pool.sample(n=quota, random_state=seed))
    cols = ["FileName", "Split", "EF", "n_frames", "ed_frame", "es_frame", "band"]
    return pd.concat(parts, ignore_index=True)[cols]


def _attributions(model, clip, methods, precision, ig_steps) -> dict[str, np.ndarray]:
    maps = {}
    for method in methods:
        if method == "occlusion":
            maps[method] = occlusion_curve(model, clip, precision)
        elif method in ("ig", "ig_blur"):
            base = "zero" if method == "ig" else "blur"
            maps[method] = integrated_gradients(model, clip, base, ig_steps, precision)
        elif method == "gradcam":
            maps[method] = gradcam3d(model, clip, precision)
        else:
            raise ValueError(f"unknown method {method!r}")
    return maps


def explain_video(
    model: nn.Module,
    video: pd.Series,
    config_id: str,
    T: int,
    period: int,
    processed_root: Path,
    stats: dict[str, float],
    device: torch.device,
    precision: str,
    methods: tuple[str, ...] = METHODS,
    ig_steps: int = IG_STEPS,
) -> tuple[list[dict], dict | None]:
    """Metric rows (one per method) for one video, and its example data (None if skipped)."""
    name, ed, es, n = (
        video["FileName"],
        int(video["ed_frame"]),
        int(video["es_frame"]),
        int(video["n_frames"]),
    )
    base = {
        "file_name": name,
        "config_id": config_id,
        "T": T,
        "ef_true": float(video["EF"]),
        "band": ef_band(float(video["EF"])),
    }
    start = edes_start(ed, es, T, period, n)
    if start is None:
        nan = dict.fromkeys(
            ["hit", "lv_energy_ratio", "deletion_aopc", "random_aopc", "hit_chance", "ef_pred"],
            np.nan,
        )
        return [base | nan | {"method": m, "skipped": 1} for m in methods], None
    clip = load_clip(name, T, period, start, processed_root, stats).to(device)
    ef_pred = float(predict_variants(model, clip[None], precision, 1)[0])
    maps = _attributions(model, clip, methods, precision, ig_steps)
    mask = hit_mask(T, ed, es, start, period)
    ed_pos, es_pos = clip_positions(ed, start, period), clip_positions(es, start, period)
    lv = {f: lv_mask(name, f, processed_root) for f in (ed, es)}
    rand = random_aopc(model, clip, precision) if TEMPORAL & set(methods) else np.nan
    rows = []
    for method, attr in maps.items():
        row = base | {
            "method": method,
            "skipped": 0,
            "ef_pred": ef_pred,
            "hit": np.nan,
            "hit_chance": np.nan,
            "lv_energy_ratio": np.nan,
            "deletion_aopc": np.nan,
            "random_aopc": np.nan,
        }
        if method in TEMPORAL:
            importance = frame_importance(attr)
            row["hit"], row["hit_chance"] = hit_rate(importance, mask)
            order = np.argsort(-importance, kind="stable")
            row["deletion_aopc"] = deletion_aopc(model, clip, order, precision)
            row["random_aopc"] = rand
        if method in SPATIAL:
            ratios = []
            for frame, pos in ((ed, ed_pos), (es, es_pos)):
                k = int(np.clip(round(pos), 0, T - 1))  # nearest sampled frame
                ratios.append(lv_energy_ratio(attr[k], lv[frame]))
            row["lv_energy_ratio"] = float(np.nanmean(ratios))
        rows.append(row)
    example = {
        "clip": clip.cpu(),
        "ig": maps.get("ig"),
        "occlusion": maps.get("occlusion"),
        "ed_pos": ed_pos,
        "es_pos": es_pos,
        "ef_true": float(video["EF"]),
        "ef_pred": ef_pred,
        "band": base["band"],
    }
    return rows, example


def metrics_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=XAI_COLUMNS + EXTRA_COLUMNS)


def summarise(metrics: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per config and method: means of the scores, permutation p of the hit rate, AOPC ratio."""
    out = []
    for (config_id, method), g in metrics.groupby(["config_id", "method"], sort=False):
        done = g[g["skipped"] == 0]
        row = {
            "config_id": config_id,
            "method": method,
            "n": len(done),
            "n_skipped": int(g["skipped"].sum()),
        }
        if method in TEMPORAL and len(done):
            t = int(done["T"].iloc[0])
            masks = [
                np.arange(t) < round(c * t) for c in done["hit_chance"]
            ]  # positions irrelevant
            row |= {
                "hit_rate": done["hit"].mean(),
                "chance_rate": done["hit_chance"].mean(),
                "hit_p": hit_test(masks, done["hit"].to_numpy(), seed)["p"],
                "deletion_aopc": done["deletion_aopc"].mean(),
                "random_aopc": done["random_aopc"].mean(),
            }
            row["aopc_ratio"] = row["deletion_aopc"] / row["random_aopc"]
        if method in SPATIAL and len(done):
            row["lv_energy_ratio"] = done["lv_energy_ratio"].mean()
        out.append(row)
    return pd.DataFrame(out)


def sanity_rows(
    model: nn.Module,
    clips: dict[str, torch.Tensor],
    config_id: str,
    precision: str,
    methods: tuple[str, ...] = ("occlusion", "ig", "gradcam"),
    ig_steps: int = IG_STEPS,
) -> pd.DataFrame:
    """Spearman between each method's map on the trained model and on cascading randomisations."""
    original = {
        name: _attributions(model, c, methods, precision, ig_steps) for name, c in clips.items()
    }
    rows = []
    for stage in RANDOMISATION_STAGES:
        random_model = randomise_cascade(model, stage)
        for name, clip in clips.items():
            maps = _attributions(random_model, clip, methods, precision, ig_steps)
            for method in methods:
                rows.append(
                    {
                        "file_name": name,
                        "config_id": config_id,
                        "method": method,
                        "stage": stage,
                        "spearman": spearman(original[name][method], maps[method]),
                    }
                )
        del random_model
    return pd.DataFrame(rows, columns=SANITY_COLUMNS)


def sanity_summary(sanity: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of the Spearman correlation per config, method and randomisation stage (stages
    in cascade order, ``fc`` first), with the number of videos. Numbers only: committable."""
    grouped = sanity.groupby(["config_id", "method", "stage"], sort=False)["spearman"]
    out = grouped.agg(n="count", spearman_mean="mean", spearman_sd="std").reset_index()
    methods = {m: i for i, m in enumerate(dict.fromkeys(sanity["method"]))}
    stages = {s: i for i, s in enumerate(RANDOMISATION_STAGES)}
    order = out["method"].map(methods) * len(stages) + out["stage"].map(stages)
    return out.assign(_o=order).sort_values(["config_id", "_o"]).drop(columns="_o")
