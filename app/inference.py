"""Demo-app logic, testable without a browser (task B8). Spec, "Prototype app".

Request flow: decode the upload with OpenCV (as preprocessing did), convert to grey with the
luma rule (D-25), centre-crop and resize to 112 x 112 if needed (with an out-of-distribution
warning), predict EF in M2 with every seed of the chosen config (seed ensemble), and explain it
with temporal occlusion and, on request, Integrated Gradients. Research prototype only.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from echoef.contracts import FRAME_SIZE
from echoef.data.dataset import clip_from_frames
from echoef.data.layout import INDEX_FILE
from echoef.data.preprocess import decode_video, to_gray
from echoef.data.sampling import clip_starts, is_padded, window
from echoef.data.transforms import denormalise, to_channels
from echoef.eval.predict import predict_clips
from echoef.models.factory import load_checkpoint
from echoef.xai.attribution import integrated_gradients, occlusion_curve

BANNER = "**Research prototype. Not clinically validated. Not for diagnosis.**"
NO_REFERENCE = "No reference EF available, so no error score."
CHUNK = 8  # clips per forward pass (the training micro-batch)


@dataclass
class Ensemble:
    """The seed models of one config; all share T and period."""

    config_id: str
    T: int
    period: int
    precision: str
    seeds: list[int]
    models: list[torch.nn.Module] = field(repr=False)
    device: torch.device = field(default=torch.device("cpu"))


def load_ensemble(runs: list[Path], device: torch.device | str = "cpu") -> Ensemble:
    """``best.pt`` of each run folder (one config, 1-3 seeds) in eval mode on ``device``."""
    device = torch.device(device)
    loaded = [load_checkpoint(Path(r), device) for r in runs]
    configs = {(c["config_id"], c["T"], c["period"]) for _, c in loaded}
    if len(configs) != 1:
        raise ValueError(f"the runs mix configs {sorted(configs)}; give the seeds of one config")
    config_id, t, period = configs.pop()
    precision = str(loaded[0][1]["train"]["precision"]) if device.type == "cuda" else "fp32"
    return Ensemble(
        config_id,
        int(t),
        int(period),
        precision,
        [int(c["seed"]) for _, c in loaded],
        [m for m, _ in loaded],
        device,
    )


def read_video(path: Path) -> tuple[np.ndarray, list[str]]:
    """Decode a video to grey uint8 (F, 112, 112), with warnings for anything unusual.

    Raises:
        ValueError: nothing could be decoded.
    """
    frames = decode_video(Path(path))
    if len(frames) == 0:
        raise ValueError("The file could not be decoded as a video.")
    warnings = []
    gray = to_gray(frames)
    _, h, w = gray.shape
    if (h, w) != (FRAME_SIZE, FRAME_SIZE):
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        square = gray[:, top : top + side, left : left + side]
        gray = np.stack(
            [cv2.resize(f, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA) for f in square]
        )
        warnings.append(
            f"The video is {w}×{h}, not 112×112: it was centre-cropped and resized. It is out of "
            "distribution: the model only saw 112×112 masked apical four-chamber EchoNet videos, "
            "so treat this estimate with extra caution."
        )
    spread = frames.max(axis=-1).astype(int) - frames.min(axis=-1)
    if np.mean(spread > 30) > 0.2:
        warnings.append(
            "The video is in colour (e.g. Doppler); the model was trained on B-mode grey."
        )
    return gray, warnings


def ef_band(ef: float) -> str:
    """EF band of the 2021 universal definition of heart failure; a band, never a diagnosis."""
    if ef <= 40:
        return "EF band: reduced (≤ 40%)"
    if ef < 50:
        return "EF band: mildly reduced (41–49%)"
    return "EF band: preserved (≥ 50%)"


def _clips(frames: np.ndarray, starts: list[int], ens: Ensemble, stats: dict) -> torch.Tensor:
    return torch.stack(
        [
            to_channels(
                torch.from_numpy(
                    clip_from_frames(frames, ens.T, ens.period, s, stats["mean"], stats["std"])
                )
            )
            for s in starts
        ]
    )


def predict(ens: Ensemble, frames: np.ndarray, stats: dict[str, float]) -> dict:
    """M2 seed-ensemble EF of grey frames (F, 112, 112).

    Returns:
        dict with ``ef`` (mean over seeds), ``ef_min``/``ef_max`` (spread over seeds; equal with
        one seed), ``per_seed``, ``n_clips``, ``padded``, ``latency_ms`` (clips to EF, GPU synced).
    """
    t0 = time.perf_counter()
    n = len(frames)
    starts = clip_starts(n, ens.T, ens.period, "M2")
    clips = _clips(frames, starts, ens, stats)
    per_seed = [
        float(predict_clips(m, clips, ens.device, ens.precision, CHUNK).mean()) for m in ens.models
    ]
    if ens.device.type == "cuda":
        torch.cuda.synchronize(ens.device)
    return {
        "ef": float(np.mean(per_seed)),
        "ef_min": float(np.min(per_seed)),
        "ef_max": float(np.max(per_seed)),
        "per_seed": per_seed,
        "n_clips": len(starts),
        "padded": any(is_padded(s, ens.T, ens.period, n) for s in starts),
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }


def _m1_clip(ens: Ensemble, frames: np.ndarray, stats: dict) -> tuple[torch.Tensor, int]:
    start = clip_starts(len(frames), ens.T, ens.period, "M1")[0]
    return _clips(frames, [start], ens, stats)[0].to(ens.device), start


def importance_figure(ens: Ensemble, frames: np.ndarray, stats: dict) -> Figure:
    """Temporal occlusion on the central (M1) clip with the first seed's model."""
    clip, start = _m1_clip(ens, frames, stats)
    curve = occlusion_curve(ens.models[0], clip, ens.precision)
    raw = start + ens.period * np.arange(ens.T)
    fig = Figure(figsize=(6, 2.4), layout="constrained")
    ax = fig.subplots()
    ax.plot(raw, curve, color="#2a78d6", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Frame of the video")
    ax.set_ylabel("|ΔEF| when the frame is occluded")
    ax.set_title("Which frames move the prediction (influence, not causation)", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig


def ig_overlay(ens: Ensemble, frames: np.ndarray, stats: dict, n_show: int = 8) -> np.ndarray:
    """RGB image (uint8): ``n_show`` frames of the central clip with the |IG| map in orange."""
    clip, _ = _m1_clip(ens, frames, stats)
    ig = np.abs(integrated_gradients(ens.models[0], clip, "zero", precision=ens.precision))
    grey = denormalise(clip[0].cpu(), stats["mean"], stats["std"])  # (T, H, W)
    picks = np.unique(np.linspace(0, ens.T - 1, min(n_show, ens.T)).round().astype(int))
    top = np.percentile(ig, 99.5) or 1.0
    tiles = []
    for k in picks:
        g = np.repeat(grey[k][..., None], 3, axis=2).astype(np.float32)
        a = np.clip(ig[k] / top, 0, 1)[..., None] * 0.85
        orange = np.array([235, 104, 52], dtype=np.float32)
        tiles.append(((1 - a) * g + a * orange).astype(np.uint8))
    return np.concatenate(tiles, axis=1)


def reference_ef(file_name: str, processed_root: Path | None) -> float | None:
    """Reference EF if the upload is named like an EchoNet TEST video, else None."""
    if processed_root is None or not (Path(processed_root) / INDEX_FILE).exists():
        return None
    stem = Path(file_name).stem
    index = pd.read_parquet(Path(processed_root) / INDEX_FILE, columns=["FileName", "Split", "EF"])
    hit = index[(index["FileName"] == stem) & (index["Split"] == "TEST")]
    return float(hit["EF"].iloc[0]) if len(hit) else None


def report(
    ens: Ensemble, result: dict, warnings: list[str], reference: float | None, n_frames: int
) -> tuple[str, str]:
    """Markdown for the result panel and for the notes panel."""
    ef = result["ef"]
    lines = [f"## EF {ef:.1f}%"]
    if len(ens.seeds) > 1:
        lines.append(
            f"Seed range {result['ef_min']:.1f}–{result['ef_max']:.1f}% "
            f"({len(ens.seeds)} models; model variance only, so it understates the "
            "true uncertainty)"
        )
    lines.append(ef_band(ef))
    if reference is None:
        lines.append(NO_REFERENCE)
    else:
        lines.append(
            f"Reference EF {reference:.1f}%, absolute error {abs(ef - reference):.1f} points"
        )
    lines.append(
        f"{result['n_clips']} clips of {ens.T} frames (config {ens.config_id.upper()}), "
        f"{result['latency_ms']:.0f} ms"
    )
    notes = list(warnings)
    if n_frames < window(ens.T, ens.period):
        notes.append(
            f"The video has {n_frames} frames, fewer than the model's window of "
            f"{window(ens.T, ens.period)}: padded with mean-grey frames, as in training."
        )
    return "\n\n".join(lines), "\n\n".join(f"⚠ {n}" for n in notes) or "No warnings."
