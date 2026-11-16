"""LV tracings: outline polygon and area (A2), LV mask (A3).

``VolumeTracings.csv`` holds, for each traced frame, rows ``(X1, Y1, X2, Y2)`` in pixel
coordinates. A frame's first row is the LV long axis; the remaining rows are chords across the
ventricle (method of disks). EchoNet's loader builds the LV outline from the chords only: the
``X1, Y1`` end points in file order, then the ``X2, Y2`` end points in reverse order
(echonet/datasets/echo.py, ``LargeTrace``/``SmallTrace``).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from echoef.contracts import FRAME_SIZE
from echoef.data.layout import TRACINGS_FILE

# A chord counts as perpendicular to the long axis when |cos(angle)| is below this (angle > 60
# degrees). Real chords are at 90 degrees (|cos| ~ 0); a second long axis is parallel (~ 1).
MAX_CHORD_COS = 0.5


def is_single_tracing(rows: np.ndarray, max_cos: float = MAX_CHORD_COS) -> bool:
    """True if a traced frame is one regular tracing: a long axis, then >= 2 chords across it.

    About 1% of EchoNet videos have a frame traced several times with the rows interleaved;
    no outline built from those rows is trustworthy, so such videos get no ED/ES (D-26).

    Args:
        rows: array (R, 4) of ``X1, Y1, X2, Y2`` for one frame, in file order.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 4 or rows.shape[0] < 3:
        return False
    vectors = rows[:, 2:4] - rows[:, 0:2]
    lengths = np.linalg.norm(vectors, axis=1)
    if lengths[0] == 0:
        return False
    cos = np.abs(vectors[1:] @ vectors[0]) / np.maximum(lengths[1:] * lengths[0], 1e-12)
    return bool((cos < max_cos).all())


def lv_polygon(rows: np.ndarray) -> np.ndarray:
    """Outline of the LV in one traced frame, built the way EchoNet builds it.

    Args:
        rows: array (R, 4) of ``X1, Y1, X2, Y2`` for one frame, in file order. Row 0 is the long
            axis and is skipped, so R must be at least 3 (the long axis plus 2 chords).

    Returns:
        float64 array (2 * (R - 1), 2) of ``(x, y)`` vertices.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError(f"expected an (R, 4) array of X1, Y1, X2, Y2; got shape {rows.shape}")
    if rows.shape[0] < 3:
        raise ValueError(f"need the long axis plus at least 2 chords; got {rows.shape[0]} rows")
    chords = rows[1:]
    return np.concatenate([chords[:, 0:2], chords[::-1, 2:4]])


def polygon_area(vertices: np.ndarray) -> float:
    """Area enclosed by a polygon (shoelace formula), in square pixels.

    Args:
        vertices: array (N, 2) of ``(x, y)`` vertices in either orientation.
    """
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_mask(
    vertices: np.ndarray, shape: tuple[int, int] = (FRAME_SIZE, FRAME_SIZE)
) -> np.ndarray:
    """Pixels whose centre lies inside a polygon (even-odd rule).

    Pixel (row i, column j) has its centre at ``x = j, y = i``: X is the column and Y the row,
    as in EchoNet's tracings and in ``skimage.draw.polygon``, which EchoNet rasterises with.
    Vertices outside the frame are allowed; the mask is cut to the frame.

    Args:
        vertices: array (N, 2) of ``(x, y)``, N >= 3, in either orientation.
        shape: ``(H, W)`` of the mask.

    Returns:
        bool array ``shape``.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError(f"expected an (N >= 3, 2) array of vertices; got {vertices.shape}")
    h, w = shape
    rows, cols = np.mgrid[0:h, 0:w]
    px, py = cols.reshape(-1, 1), rows.reshape(-1, 1)  # (H*W, 1) pixel centres
    x0, y0 = vertices[:, 0], vertices[:, 1]  # (N,) edge k runs from vertex k to vertex k + 1
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    straddles = (y0 > py) != (y1 > py)  # edge crosses the horizontal line through the pixel
    with np.errstate(divide="ignore", invalid="ignore"):  # horizontal edges never straddle
        x_cross = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    crossings = (straddles & (px < x_cross)).sum(axis=1)  # crossings of a ray towards +x
    return (crossings % 2 == 1).reshape(h, w)


@lru_cache(maxsize=4)
def _traced_frames(tracings_file: str) -> dict[tuple[str, int], np.ndarray]:
    """``(FileName, Frame) -> (R, 4) X1, Y1, X2, Y2`` rows in file order, read once per process."""
    df = pd.read_parquet(tracings_file)
    coords = df[["X1", "Y1", "X2", "Y2"]].to_numpy(dtype=np.float64)
    groups = df.groupby(["FileName", "Frame"], sort=False).indices  # positions keep file order
    return {(str(name), int(frame)): coords[pos] for (name, frame), pos in groups.items()}


def lv_mask(file_name: str, frame: int, processed_root: Path) -> np.ndarray:
    """LV pixels in one traced frame: the inside of EchoNet's outline polygon.

    Contract function (spec, "Functions A provides"). Reads ``<processed_root>/tracings.parquet``
    once per process and caches it.

    Returns:
        bool array (112, 112).

    Raises:
        KeyError: the frame of this video is not traced.
        ValueError: the frame is not one regular tracing (D-26).
    """
    rows = _traced_frames(str(Path(processed_root) / TRACINGS_FILE)).get((file_name, int(frame)))
    if rows is None:
        raise KeyError(f"{file_name} has no tracing at frame {frame}")
    if not is_single_tracing(rows):
        raise ValueError(f"{file_name} frame {frame} is not one regular tracing (D-26)")
    return polygon_mask(lv_polygon(rows))
