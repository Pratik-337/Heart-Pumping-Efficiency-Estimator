"""LV polygon (EchoNet's construction) and its area."""

import numpy as np
import pytest

from echoef.data.tracings import is_single_tracing, lv_polygon, polygon_area


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
