"""Clip sampling: M1/M2 starts, training starts, ED/ES-centred starts, padding (CPU only)."""

import itertools

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from echoef.config import CONFIGS_DIR
from echoef.contracts import CONFIGS
from echoef.data import sampling as s

SETTINGS = [(t, p) for _, t, p in CONFIGS.values()]


def test_window_and_indices():
    assert s.window(32, 2) == 64
    np.testing.assert_array_equal(s.clip_indices(5, 4, 3), [5, 8, 11, 14])
    with pytest.raises(ValueError):
        s.window(0, 2)
    with pytest.raises(ValueError):
        s.clip_indices(-1, 4, 2)


def test_m1_is_centred_or_zero():
    assert s.clip_starts(173, 32, 2, "M1") == [54]  # centre 86 minus W/2 = 32
    assert s.clip_starts(64, 32, 2, "M1") == [0]
    assert s.clip_starts(40, 32, 2, "M1") == [0]  # shorter than the window


def test_m2_worked_examples_from_spec():
    assert s.clip_starts(173, 32, 2, "M2") == [0, 32, 64, 96]  # FS-32: 4 clips
    assert len(s.clip_starts(173, 8, 2, "M2")) == 20  # FS-8: 20 clips
    assert s.clip_starts(100, 64, 2, "M2") == [0]  # F < W: one (padded) clip


@pytest.mark.parametrize(("T", "period"), SETTINGS)
def test_m2_count_formula_and_clips_inside_video(T, period):
    w = T * period
    for n in range(1, 400, 7):
        starts = s.clip_starts(n, T, period, "M2")
        expected = 1 if n < w else (n - w) // (w // 2) + 1
        assert len(starts) == expected
        assert starts[0] == 0
        assert all(b - a == w // 2 for a, b in itertools.pairwise(starts))
        if n >= w:
            assert starts[-1] + w <= n
            assert not any(s.is_padded(x, T, period, n) for x in starts)


def test_bad_mode_and_length_raise():
    with pytest.raises(ValueError):
        s.clip_starts(100, 8, 2, "M3")
    with pytest.raises(ValueError):
        s.clip_starts(0, 8, 2, "M1")


def test_m2_overlap_matches_base_config():
    base = OmegaConf.load(CONFIGS_DIR / "base.yaml")
    assert base.eval.m2_overlap == s.M2_OVERLAP


@pytest.mark.parametrize(
    ("n", "T", "period", "padded"),
    [
        (16, 8, 2, False),  # exactly W
        (15, 8, 2, False),  # F = W - 1: the last read frame is 14, still inside
        (14, 8, 2, True),  # F = W - p: frame 14 is past the end
        (57, 8, 8, False),  # FW-8: frames 0, 8, ..., 56
        (56, 8, 8, True),
    ],
)
def test_is_padded_means_a_frame_past_the_end(n, T, period, padded):
    assert s.is_padded(0, T, period, n) is padded


def test_random_start_covers_the_full_range_reproducibly():
    g = torch.Generator().manual_seed(0)
    draws = [s.random_start(100, 8, 2, g) for _ in range(2000)]  # starts 0 .. 84
    assert min(draws) == 0 and max(draws) == 84
    g2 = torch.Generator().manual_seed(0)
    assert draws[:50] == [s.random_start(100, 8, 2, g2) for _ in range(50)]
    assert s.random_start(10, 8, 2) == 0  # shorter than the window


def test_edes_start_spec_formula():
    # d = 10, W = 64: start = 30 - (64 - 10) // 2 = 3
    assert s.edes_start(40, 30, 32, 2, 200) == 3
    assert s.edes_start(30, 40, 32, 2, 200) == 3  # ED/ES order does not matter
    assert s.edes_start(5, 15, 32, 2, 200) == 0  # clamped at the start
    assert s.edes_start(190, 199, 32, 2, 200) == 136  # clamped so the window ends at 200
    assert s.edes_start(0, 16, 8, 2, 200) is None  # d = W = 16


@pytest.mark.parametrize(("T", "period"), SETTINGS)
def test_edes_start_holds_both_frames(T, period):
    rng = np.random.default_rng(0)
    w = T * period
    for _ in range(500):
        n = int(rng.integers(2, 300))
        ed, es = (int(x) for x in rng.integers(0, n, 2))
        start = s.edes_start(ed, es, T, period, n)
        if abs(ed - es) >= w:
            assert start is None
            continue
        assert 0 <= start <= max(0, n - w)
        assert start <= min(ed, es) and max(ed, es) < start + w


@pytest.mark.parametrize(("ed", "es"), [(-1, -1), (5, 100)])
def test_edes_start_rejects_frames_outside_the_video(ed, es):
    with pytest.raises(ValueError):
        s.edes_start(ed, es, 8, 2, 50)
