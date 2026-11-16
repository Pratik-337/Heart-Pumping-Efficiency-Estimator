"""Attribution shapes on a random R(2+1)D-18 at T = 8, CPU (spec, task B6)."""

import numpy as np
import pytest
import torch

from echoef.models.factory import build_model
from echoef.xai import attribution as at

T = 8


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(pretrained=False, head_bias=55.0).eval()


@pytest.fixture(scope="module")
def clip():
    torch.manual_seed(1)
    return torch.randn(3, T, 112, 112)


def test_occlusion_curve(model, clip):
    curve = at.occlusion_curve(model, clip, precision="fp32")
    assert curve.shape == (T,) and (curve >= 0).all()


@pytest.mark.parametrize("baseline", ["zero", "blur"])
def test_integrated_gradients(model, clip, baseline):
    ig = at.integrated_gradients(model, clip, baseline, steps=2, precision="fp32",
                                 internal_batch_size=2, max_steps=2)  # fmt: skip
    assert ig.shape == (T, 112, 112) and np.isfinite(ig).all()


def test_gradcam3d(model, clip):
    cam = at.gradcam3d(model, clip, precision="fp32")
    assert cam.shape == (T, 112, 112) and (cam >= 0).all()


def test_interpolate_frames_is_linear_and_keeps_other_frames():
    clip = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1, 1).repeat(2, 1, 3, 3) * 10
    out = at.interpolate_frames(clip, np.array([0, 1, 1, 0, 0, 1], dtype=bool))
    assert out[0, :, 0, 0].tolist() == [0, 10, 20, 30, 40, 40]  # 1, 2 between 0 and 3; 5 = end
    assert torch.equal(out[:, [0, 3, 4]], clip[:, [0, 3, 4]])
    assert (at.interpolate_frames(clip, np.ones(6, dtype=bool)) == 0).all()


def test_blur_and_frame_importance():
    clip = torch.randn(3, 4, 32, 32)
    b = at.blurred(clip)
    assert b.shape == clip.shape and b.std() < clip.std()
    imp = at.frame_importance(np.ones((4, 5, 5)) * np.arange(4)[:, None, None])
    assert imp.tolist() == [0, 25, 50, 75]
