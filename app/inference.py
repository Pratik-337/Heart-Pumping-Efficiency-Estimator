"""Demo-app logic, testable without a browser (task B8). Spec, "Prototype app".

Request flow: decode the upload with OpenCV (as preprocessing did), convert to grey with the
luma rule (D-25), centre-crop and resize to 112 x 112 if needed (with an out-of-distribution
warning), predict EF in M2 with every seed of the chosen config (seed ensemble), and explain it
with temporal occlusion (ED and ES marked when the file is a traced EchoNet video) and, on
request, Integrated Gradients as a frame strip and as an animation of the whole clip. The default
model is the config chosen on validation (D-13, ``tables/demo_choice.csv``) with all its finished
seeds. Research prototype only.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from PIL import Image

from echoef.contracts import FRAME_SIZE
from echoef.data.dataset import clip_from_frames
from echoef.data.layout import INDEX_FILE
from echoef.data.preprocess import decode_video, to_gray
from echoef.data.sampling import clip_starts, is_padded, window
from echoef.data.transforms import denormalise, to_channels
from echoef.eval.figures import INK_2, REGIME_COLOR
from echoef.eval.predict import predict_clips
from echoef.models.factory import load_checkpoint
from echoef.xai.attribution import integrated_gradients, occlusion_curve

BANNER = "**Research prototype. Not clinically validated. Not for diagnosis.**"
NO_REFERENCE = "No reference EF available, so no error score."
CHUNK = 8  # clips per forward pass (the training micro-batch)
NOMINAL_FPS = 50  # EchoNet's typical frame rate; sets the animation speed
ANIMATION_SCALE = 3  # 112 px frames shown at 336 px
ORANGE = np.array([235, 104, 52], dtype=np.float32)  # palette orange, as in the XAI figures


def demo_runs(results_root: Path, runs_root: Path) -> list[Path]:
    """Run folders of the demo config: the one chosen on validation (D-13), every finished seed.

    Raises:
        FileNotFoundError: no ``tables/demo_choice.csv`` (run ``aggregate_results.py``) or no
            finished run of the chosen config.
    """
    path = Path(results_root) / "tables" / "demo_choice.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run scripts/aggregate_results.py, or --runs")
    choice = pd.read_csv(path)
    config_id = str(choice.loc[choice["chosen"].astype(bool), "config_id"].iloc[0])
    runs = sorted(r.parent for r in (Path(runs_root) / config_id).glob("seed*/DONE"))
    if not runs:
        raise FileNotFoundError(f"no finished run of the demo config {config_id} in {runs_root}")
    return runs


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


def importance_figure(
    ens: Ensemble, frames: np.ndarray, stats: dict, ed: int | None = None, es: int | None = None
) -> Figure:
    """Temporal occlusion on the central (M1) clip with the first seed's model.

    ``ed``/``es``: traced end-diastolic and end-systolic frames of a known EchoNet video, marked
    when they fall inside the clip's window.
    """
    clip, start = _m1_clip(ens, frames, stats)
    curve = occlusion_curve(ens.models[0], clip, ens.precision)
    raw = start + ens.period * np.arange(ens.T)
    fig = Figure(figsize=(6, 2.4), layout="constrained")
    ax = fig.subplots()
    ax.plot(raw, curve, color=REGIME_COLOR["FS"], linewidth=1.5, marker="o", markersize=3)
    for frame, name in ((ed, "ED"), (es, "ES")):
        if frame is not None and raw[0] <= frame <= raw[-1]:
            ax.axvline(frame, color=INK_2, linewidth=0.9, linestyle="--")
            ax.annotate(name, (frame, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                        textcoords="offset points", va="top", color=INK_2)  # fmt: skip
    ax.set_xlabel("Frame of the video")
    ax.set_ylabel("|ΔEF| when the frame is occluded")
    ax.set_title("Which frames move the prediction (influence, not causation)", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig


def ig_maps(ens: Ensemble, frames: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Grey frames (T, H, W) uint8 of the central clip and their |IG| maps (T, H, W), zero
    baseline, first seed's model."""
    clip, _ = _m1_clip(ens, frames, stats)
    ig = np.abs(integrated_gradients(ens.models[0], clip, "zero", precision=ens.precision))
    grey = denormalise(clip[0].cpu(), stats["mean"], stats["std"])
    return np.asarray(grey, dtype=np.uint8), ig


def _tint(grey: np.ndarray, ig: np.ndarray, top: float) -> np.ndarray:
    """One RGB frame: grey with |IG| in orange (alpha up to 0.85)."""
    g = np.repeat(grey[..., None], 3, axis=2).astype(np.float32)
    a = np.clip(ig / top, 0, 1)[..., None] * 0.85
    return ((1 - a) * g + a * ORANGE).astype(np.uint8)


def _top(ig: np.ndarray) -> float:
    return float(np.percentile(ig, 99.5)) or 1.0


def ig_overlay(
    ens: Ensemble, frames: np.ndarray, stats: dict, n_show: int = 8, maps=None
) -> np.ndarray:
    """RGB image (uint8): ``n_show`` frames of the central clip with the |IG| map in orange.

    ``maps``: ``ig_maps`` output, to reuse one IG computation.
    """
    grey, ig = maps if maps is not None else ig_maps(ens, frames, stats)
    picks = np.unique(np.linspace(0, ens.T - 1, min(n_show, ens.T)).round().astype(int))
    top = _top(ig)
    return np.concatenate([_tint(grey[k], ig[k], top) for k in picks], axis=1)


def ig_animation(ens: Ensemble, maps: tuple[np.ndarray, np.ndarray], path: Path) -> Path:
    """Animated GIF of every frame of the central clip: grey on the left, |IG| overlay on the
    right, played at the clip's sampling rate (period / 50 fps per frame). Returns ``path``."""
    grey, ig = maps
    top, size = _top(ig), ANIMATION_SCALE * grey.shape[-1]
    tiles = []
    for k in range(len(grey)):
        plain = np.repeat(grey[k][..., None], 3, axis=2)
        pair = np.concatenate([plain, _tint(grey[k], ig[k], top)], axis=1)
        pair = cv2.resize(pair, (2 * size, size), interpolation=cv2.INTER_LINEAR)
        tiles.append(Image.fromarray(pair))
    ms = max(20, round(1000 * ens.period / NOMINAL_FPS))  # browsers slow delays < 20 ms
    path = Path(path)
    tiles[0].save(path, save_all=True, append_images=tiles[1:], duration=ms, loop=0)
    return path


def video_info(file_name: str, processed_root: Path | None) -> dict:
    """What the index knows about an upload named like an EchoNet video.

    Returns:
        dict with ``reference`` (EF of a TEST video, else None; spec: the error is shown for
        test videos only) and ``ed``/``es`` (traced frames when the video has a regular tracing,
        else None). Empty values for any other file.
    """
    info = {"reference": None, "ed": None, "es": None}
    if processed_root is None or not (Path(processed_root) / INDEX_FILE).exists():
        return info
    index = pd.read_parquet(Path(processed_root) / INDEX_FILE)
    hit = index[index["FileName"] == Path(file_name).stem]
    if hit.empty:
        return info
    row = hit.iloc[0]
    if row["Split"] == "TEST":
        info["reference"] = float(row["EF"])
    traced = bool(row.get("has_trace", False))
    if traced and row.get("ed_frame", -1) >= 0 and row.get("es_frame", -1) >= 0:
        info["ed"], info["es"] = int(row["ed_frame"]), int(row["es_frame"])
    return info


def reference_ef(file_name: str, processed_root: Path | None) -> float | None:
    """Reference EF if the upload is named like an EchoNet TEST video, else None."""
    return video_info(file_name, processed_root)["reference"]


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
