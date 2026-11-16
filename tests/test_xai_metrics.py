"""Numeric XAI tests on a model that only looks at frame 3 (CPU, task B6)."""

import numpy as np
import pytest
import torch
from torch import nn

from echoef.models.factory import build_model
from echoef.xai import xai_metrics as xm
from echoef.xai.attribution import frame_importance, occlusion_curve


class Frame3(nn.Module):
    """EF = 50 + 10 x mean of frame 3: only frame 3 matters."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        return 50 + self.w * x[:, :, 3].mean(dim=(1, 2, 3))


def _clip(t=8):
    torch.manual_seed(0)
    return torch.randn(3, t, 16, 16) + torch.arange(t).view(1, t, 1, 1)


def test_hit_mask_and_rate():
    mask = xm.hit_mask(T=8, ed=16, es=40, start=10, period=2)  # ED at clip position 3, ES at 15
    assert np.flatnonzero(mask).tolist() == [2, 3, 4]
    importance = np.zeros(8)
    importance[[3, 7]] = [5, 4]
    assert xm.hit_rate(importance, mask) == (0.5, 3 / 8)


def test_hit_test_detects_a_method_that_finds_ed():
    masks = [
        xm.hit_mask(16, 20, 50, 0, 2) for _ in range(30)
    ]  # hits at 9-11 and 24-26 -> 9, 10, 11
    hits = np.ones(30)  # always both top frames on ED/ES
    out = xm.hit_test(masks, hits, n_perm=200)
    assert out["hit_rate"] == 1.0 and out["chance_rate"] == pytest.approx(3 / 16)
    assert out["p"] < 0.01
    assert xm.hit_test(masks, np.full(30, 3 / 16), n_perm=200)["p"] > 0.05


def test_occlusion_finds_the_used_frame():
    curve = occlusion_curve(Frame3(), _clip(), precision="fp32")
    assert int(np.argmax(curve)) == 3 and np.allclose(np.delete(curve, 3), 0)


def test_lv_energy_ratio():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5] = True  # half the frame
    attr = np.zeros((10, 10))
    attr[:5] = 1.0  # all energy inside
    assert xm.lv_energy_ratio(attr, mask) == pytest.approx(2.0)
    assert xm.lv_energy_ratio(np.ones((10, 10)), mask) == pytest.approx(1.0)  # uniform map
    assert np.isnan(xm.lv_energy_ratio(np.zeros((10, 10)), mask))


def test_deletion_beats_random_for_the_right_ranking():
    model, clip = Frame3(), _clip()
    good = np.argsort(-frame_importance(occlusion_curve(model, clip, precision="fp32")))
    aopc = xm.deletion_aopc(model, clip, good, precision="fp32")
    assert aopc > xm.random_aopc(model, clip, precision="fp32")


def test_spearman_and_randomisation_cascade():
    a = np.random.default_rng(0).normal(size=(4, 8, 8))
    assert xm.spearman(a, a) == pytest.approx(1.0)
    torch.manual_seed(0)
    model = build_model(pretrained=False).eval()
    top = xm.randomise_cascade(model, "fc")
    assert not torch.equal(top.fc.weight, model.fc.weight)
    assert torch.equal(top.layer1[0].conv1[0][0].weight, model.layer1[0].conv1[0][0].weight)
    full = xm.randomise_cascade(model, "stem")
    assert not torch.equal(full.stem[0].weight, model.stem[0].weight)
    with pytest.raises(ValueError):
        xm.randomise_cascade(model, "layer9")
