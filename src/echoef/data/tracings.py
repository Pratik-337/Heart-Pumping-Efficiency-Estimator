"""LV tracings: outline polygon and area (A2). The LV mask (``lv_mask``) is added in A3.

``VolumeTracings.csv`` holds, for each traced frame, rows ``(X1, Y1, X2, Y2)`` in pixel
coordinates. A frame's first row is the LV long axis; the remaining rows are chords across the
ventricle (method of disks). EchoNet's loader builds the LV outline from the chords only: the
``X1, Y1`` end points in file order, then the ``X2, Y2`` end points in reverse order
(echonet/datasets/echo.py, ``LargeTrace``/``SmallTrace``).
"""

import numpy as np

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
