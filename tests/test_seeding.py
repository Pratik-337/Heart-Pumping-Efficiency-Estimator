"""seed_everything makes Python, NumPy and PyTorch draws repeatable."""

import random

import numpy as np
import torch

from echoef.utils import seed_everything


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_same_draws():
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    assert _draw() == first


def test_different_seed_different_draws():
    seed_everything(0)
    first = _draw()
    seed_everything(1)
    assert _draw() != first
