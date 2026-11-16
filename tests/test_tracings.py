"""LV polygon (EchoNet's construction), its area, and the LV mask."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.path import Path as MplPath

from echoef.contracts import TRACINGS_COLUMNS
from echoef.data.layout import TRACINGS_FILE
from echoef.data.tracings import (
    is_single_tracing,
    lv_mask,
    lv_polygon,
    polygon_area,
    polygon_mask,
)


def _rectangle_rows(x_left=10.0, x_right=30.0, y0=0.0, height=10, long_axis=(20, -5, 20, 15)):
    """Long axis first, then horizontal chords: the outline is a (x_right - x_left) x height box."""
    chords = [(x_left, y0 + k, x_right, y0 + k) for k in range(height + 1)]
    return np.array([long_axis, *chords], dtype=float)


def test_rectangle_area():
    assert polygon_area(lv_polygon(_rectangle_rows())) == pytest.approx(20 * 10)


def test_first_row_is_skipped():
    a = polygon_area(lv_polygon(_rectangle_rows(long_axis=(20, -5, 20, 15))))
    b = polygon_area(lv_polygon(_rectangle_rows(long_axis=(0, 0, 999, 999))))
    assert a == b


def test_vertex_order_is_x1_side_then_x2_side_reversed():
    rows = np.array([[0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=float)
    expected = [[1, 2], [5, 6], [9, 10], [11, 12], [7, 8], [3, 4]]
    np.testing.assert_array_equal(lv_polygon(rows), expected)


def test_triangle_area_either_orientation():
    tri = np.array([[0, 0], [4, 0], [0, 3]], dtype=float)
    assert polygon_area(tri) == pytest.approx(6.0)
    assert polygon_area(tri[::-1]) == pytest.approx(6.0)


@pytest.mark.parametrize("rows", [np.zeros((2, 4)), np.zeros((5, 3))])
def test_bad_input_raises(rows):
    with pytest.raises(ValueError):
        lv_polygon(rows)


def test_regular_tracing_is_single():
    assert is_single_tracing(_rectangle_rows())


def test_frame_traced_twice_is_not_single():
    rows = _rectangle_rows()
    doubled = np.concatenate([rows[:1], rows[:1], rows[1:], rows[1:] + 0.5])  # 2 axes first
    assert not is_single_tracing(doubled)


@pytest.mark.parametrize(
    "rows",
    [
        np.array([[0, 0, 0, 30], [0, 1, 0, 29], [0, 2, 0, 28], [0, 3, 0, 27]], dtype=float),
        np.zeros((2, 4)),
        np.array([[5, 5, 5, 5], [0, 1, 9, 1], [0, 2, 9, 2]], dtype=float),  # zero-length axis
    ],
    ids=["all-rows-parallel", "too-few-rows", "zero-length-axis"],
)
def test_irregular_frames_are_not_single(rows):
    assert not is_single_tracing(rows)


# ----------------------------------------------------------------------------- masks (A3)


def test_polygon_mask_box_selects_pixel_centres_inside():
    box = np.array([[1.5, 1.5], [5.5, 1.5], [5.5, 4.5], [1.5, 4.5]])  # x = columns, y = rows
    mask = polygon_mask(box, (8, 8))
    expected = np.zeros((8, 8), dtype=bool)
    expected[2:5, 2:6] = True  # rows 2-4, columns 2-5
    np.testing.assert_array_equal(mask, expected)


def test_polygon_mask_matches_matplotlib_on_random_polygons():
    rng = np.random.default_rng(0)
    rows, cols = np.mgrid[0:40, 0:40]
    centres = np.column_stack([cols.ravel(), rows.ravel()])
    for _ in range(20):  # star-shaped, so the outline does not cross itself
        angles = np.sort(rng.uniform(0, 2 * np.pi, 15))
        radii = rng.uniform(5, 18, 15)
        poly = np.column_stack([20 + radii * np.cos(angles), 20 + radii * np.sin(angles)])
        expected = MplPath(poly).contains_points(centres).reshape(40, 40)
        np.testing.assert_array_equal(polygon_mask(poly, (40, 40)), expected)


def test_polygon_mask_count_tracks_area_and_is_cut_to_the_frame():
    rows = _rectangle_rows(x_left=-10.2, x_right=30.3, y0=50.25, height=40)
    poly = lv_polygon(rows)
    mask = polygon_mask(poly)
    assert mask.shape == (112, 112)
    assert mask.sum() == 31 * 40  # columns 0-30, rows 51-90: the part inside the frame
    big = lv_polygon(_rectangle_rows(x_left=20.3, x_right=70.6, y0=30.1, height=45))
    assert polygon_mask(big).sum() == pytest.approx(polygon_area(big), rel=0.05)


def _write_tracings(root, frames: dict[tuple[str, int], np.ndarray]) -> None:
    rows = [(name, frame, *r) for (name, frame), coords in frames.items() for r in coords]
    pd.DataFrame(rows, columns=TRACINGS_COLUMNS).to_parquet(root / TRACINGS_FILE, index=False)


def test_lv_mask_reads_tracings_parquet(tmp_path):
    regular = _rectangle_rows(x_left=20.0, x_right=60.0, y0=30.0, height=40)
    doubled = np.concatenate([regular[:1], regular[:1], regular[1:], regular[1:] + 0.5])
    _write_tracings(tmp_path, {("vid", 7): regular, ("vid", 30): doubled})
    mask = lv_mask("vid", 7, tmp_path)
    assert mask.dtype == bool and mask.shape == (112, 112)
    np.testing.assert_array_equal(mask, polygon_mask(lv_polygon(regular)))
    with pytest.raises(KeyError):
        lv_mask("vid", 8, tmp_path)
    with pytest.raises(ValueError, match="D-26"):
        lv_mask("vid", 30, tmp_path)
