"""Clip sampling: which raw frames a clip reads (task A3).

A clip of ``T`` frames at period ``p`` that starts at raw frame ``s`` reads frames
``s, s + p, ..., s + (T - 1) p``. Its window is ``W = T x p`` raw frames (spec,
"Frame-efficiency experiment design"). Frames at or past the end of the video are zero-padded by
the loader (EchoNet's rule, D-09); ``is_padded`` says whether a clip reads any.

Where clips start (spec, "Inference modes"):

- **M1**: one clip at the video's centre minus ``W / 2``, i.e. ``(F - W) // 2``; 0 if ``F < W``.
- **M2**: clips every ``W x (1 - overlap)`` frames (overlap 0.5: every ``W / 2``) from frame 0;
  ``n = floor((F - W) / stride) + 1`` clips, or one clip at 0 if ``F < W``.
- **Training**: one start per video and epoch, uniform over ``0 .. max(0, F - W)``, so the whole
  window lies inside the video whenever it can.
- **ED/ES-centred** (XAI): the window is centred on the ED-ES span, kept inside the video.
"""

import numpy as np
import torch

from echoef.contracts import MODES

# M2 overlap between neighbouring clips. Must equal eval.m2_overlap in configs/base.yaml (a test
# checks this); the default exists so that B's code can call clip_starts with the contract's
# four arguments.
M2_OVERLAP = 0.5


def window(T: int, period: int) -> int:
    """Raw frames spanned by a clip: ``W = T x period``."""
    if T < 1 or period < 1:
        raise ValueError(f"T and period must be >= 1; got T={T}, period={period}")
    return T * period


def clip_indices(start: int, T: int, period: int) -> np.ndarray:
    """Raw frame indices a clip reads: int64 array (T,) ``start + period * k``, k = 0 .. T - 1."""
    if start < 0:
        raise ValueError(f"start must be >= 0; got {start}")
    window(T, period)  # validates T and period
    return start + period * np.arange(T, dtype=np.int64)


def is_padded(start: int, T: int, period: int, n_frames: int) -> bool:
    """True if the clip reads a frame at or past the video's end, i.e. it holds zero frames."""
    return int(clip_indices(start, T, period)[-1]) >= n_frames


def m1_start(n_frames: int, T: int, period: int) -> int:
    """M1: the start that centres the window in the video (0 if the video is shorter)."""
    _check_n_frames(n_frames)
    return max(0, (n_frames - window(T, period)) // 2)


def m2_starts(n_frames: int, T: int, period: int, overlap: float = M2_OVERLAP) -> list[int]:
    """M2: starts every ``W x (1 - overlap)`` frames; ``[0]`` if the video is shorter than W."""
    _check_n_frames(n_frames)
    w = window(T, period)
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1); got {overlap}")
    stride = round(w * (1 - overlap))
    if stride < 1:
        raise ValueError(f"M2 stride {w} x (1 - {overlap}) rounds to 0 frames")
    if n_frames < w:
        return [0]
    return [k * stride for k in range((n_frames - w) // stride + 1)]


def clip_starts(
    n_frames: int, T: int, period: int, mode: str, overlap: float = M2_OVERLAP
) -> list[int]:
    """Start frames of the clips a video is scored on: one for ``"M1"``, ``n`` for ``"M2"``.

    Contract function (spec, "Functions A provides").
    """
    if mode == "M1":
        return [m1_start(n_frames, T, period)]
    if mode == "M2":
        return m2_starts(n_frames, T, period, overlap)
    raise ValueError(f"mode must be one of {MODES}; got {mode!r}")


def random_start(
    n_frames: int, T: int, period: int, generator: torch.Generator | None = None
) -> int:
    """Training: a start drawn uniformly from ``0 .. max(0, n_frames - W)``.

    Uses torch's random number generator (``generator``, or the global one), which PyTorch seeds
    per DataLoader worker from the loader's own seed, so draws are reproducible with workers.
    """
    _check_n_frames(n_frames)
    last = max(0, n_frames - window(T, period))
    return int(torch.randint(last + 1, (1,), generator=generator).item())


def edes_start(ed: int, es: int, T: int, period: int, n_frames: int) -> int | None:
    """Start of a window that holds both the ED and the ES frame, or None if none can.

    Spec, "ED/ES-centred clips": with ``d = |ED - ES|``, ``start = min(ED, ES) - (W - d) // 2``,
    then clamped to ``0 .. max(0, n_frames - W)`` so the window stays inside the video. Both
    frames then lie in ``[start, start + W)``. Returns None when ``d >= W``. Contract function.

    Raises:
        ValueError: ED or ES is not a frame of the video (e.g. -1 for a video without ED/ES).
    """
    _check_n_frames(n_frames)
    w = window(T, period)
    for name, frame in (("ed", ed), ("es", es)):
        if not 0 <= frame < n_frames:
            raise ValueError(f"{name} = {frame} is not a frame of a {n_frames}-frame video")
    d = abs(ed - es)
    if d >= w:
        return None
    start = min(ed, es) - (w - d) // 2
    return min(max(start, 0), max(0, n_frames - w))


def _check_n_frames(n_frames: int) -> None:
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1; got {n_frames}")
