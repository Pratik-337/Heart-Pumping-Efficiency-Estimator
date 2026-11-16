"""Seeding for Python, NumPy, PyTorch and DataLoader workers.

cuDNN is deliberately left non-deterministic: forced determinism slows 3D convolutions
(spec, "Model and training design", Seeds row).

torch is imported inside the functions so that ``import echoef.utils`` stays light: CPU-only
worker processes (e.g. the preprocessing decoders, which re-import the calling script) then do
not load torch.
"""

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch on the CPU and every GPU. Call at the start of each script."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)  # affects child processes
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def seed_worker(worker_id: int) -> None:
    """``DataLoader(worker_init_fn=seed_worker)``: seed NumPy and Python in each worker.

    PyTorch already gives every worker its own torch seed (base seed + worker id); this derives
    the NumPy and Python seeds from it. Pass a seeded ``torch.Generator`` to the DataLoader so the
    base seed itself is reproducible.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
