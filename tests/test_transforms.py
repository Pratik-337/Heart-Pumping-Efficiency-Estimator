"""Clip transforms: normalisation, grey to 3 channels, pad-and-crop (CPU only)."""

import itertools

import numpy as np
import pytest
import torch

from echoef.data.transforms import denormalise, normalise, pad_crop, to_channels


def test_normalise_and_back():
    frames = np.array([[[0, 50], [100, 255]]], dtype=np.uint8)
    out = normalise(frames, 50.0, 25.0)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[[-2, 0], [2, 8.2]]], rtol=1e-6)
    np.testing.assert_array_equal(denormalise(torch.from_numpy(out), 50.0, 25.0), frames)
    with pytest.raises(ValueError):
        normalise(frames, 0.0, 0.0)


def test_to_channels_copies_grey():
    clip = torch.randn(4, 5, 6)
    out = to_channels(clip)
    assert out.shape == (3, 4, 5, 6)
    for c in range(3):
        assert torch.equal(out[c], clip)
    out[0, 0, 0, 0] = 99.0  # a real copy, not a view of one channel
    assert out[1, 0, 0, 0] != 99.0
    with pytest.raises(ValueError):
        to_channels(torch.zeros(3, 4, 5, 6))


def _shifted(clip: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """``clip`` moved by (dy, dx) pixels with zeros filling the gap."""
    out = torch.zeros_like(clip)
    h, w = clip.shape[-2:]
    out[..., max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = clip[
        ..., max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


def test_pad_crop_is_one_shift_of_up_to_pad_pixels():
    pad = 2
    clip = 1 + torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)  # no zeros
    candidates = list(itertools.product(range(-pad, pad + 1), repeat=2))
    g = torch.Generator().manual_seed(0)
    seen = set()
    for _ in range(300):
        out = pad_crop(clip, pad, g)
        assert out.shape == clip.shape
        matches = [c for c in candidates if torch.equal(out, _shifted(clip, *c))]
        assert len(matches) == 1  # the same shift for every frame
        seen.add(matches[0])
    assert seen == set(candidates)  # every shift in -pad .. pad occurs


def test_pad_crop_zero_pad_is_identity():
    clip = torch.randn(2, 5, 5)
    assert pad_crop(clip, 0) is clip
    with pytest.raises(ValueError):
        pad_crop(clip, -1)
