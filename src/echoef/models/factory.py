"""R(2+1)D-18 EF regressor: Kinetics model, B0 loader, activation checkpointing (task A4).

Spec, "Model and training design": torchvision's ``r2plus1d_18`` with Kinetics-400 weights
(D-03), ``fc`` replaced by ``Linear(512, 1)`` whose bias starts at the training-set mean EF. B0
is EchoNet-Dynamic's released 32-frame model (same architecture, same head), used for Gate A and
Tier 0 only.

``EFRegressor`` subclasses torchvision's ``VideoResNet``, so its parameter names are torchvision's
(``stem``, ``layer1`` .. ``layer4``, ``fc``): EchoNet's weights load without renaming, and B's
Grad-CAM and randomisation tests can address ``model.layer4`` and friends directly.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import R2Plus1D_18_Weights
from torchvision.models.video.resnet import BasicBlock, Conv2Plus1D, R2Plus1dStem, VideoResNet

log = logging.getLogger(__name__)

CHECKPOINT_BEST = "best.pt"
CHECKPOINT_LAST = "last.pt"
FEATURES = 512  # width of layer4 / input of fc in R(2+1)D-18
ECHONET_PREFIX = "module."  # EchoNet saved a DataParallel model
# numpy objects inside EchoNet's checkpoint (optimizer and scheduler state) that
# torch.load(weights_only=True) must be allowed to rebuild. Never use weights_only=False.
_ECHONET_SAFE_GLOBALS = [
    (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),  # name in numpy 1.x pickles
    np._core.multiarray.scalar,  # name in numpy 2.x pickles
    np.dtype,
    type(np.dtype(np.float64)),
]


@contextmanager
def _batchnorm_stats_frozen(module: nn.Module) -> Iterator[None]:
    """Stop BatchNorm layers inside ``module`` from updating their running statistics.

    Used while activation checkpointing recomputes a stage's forward pass, so the running mean
    and variance are updated once per step, exactly as without checkpointing.
    """
    norms = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [(m.momentum, None if m.num_batches_tracked is None else m.num_batches_tracked.clone())
             for m in norms]  # fmt: skip
    for m in norms:
        m.momentum = 0.0  # running = (1 - 0) * running + 0 * batch
    try:
        yield
    finally:
        for m, (momentum, tracked) in zip(norms, saved, strict=True):
            m.momentum = momentum
            if tracked is not None:
                m.num_batches_tracked.copy_(tracked)


def _recompute_contexts(stage: nn.Module) -> tuple:
    return nullcontext(), _batchnorm_stats_frozen(stage)


class EFRegressor(VideoResNet):
    """R(2+1)D-18 with a one-output head: input (B, 3, T, 112, 112) -> output (B,) EF in %.

    Any T >= 1 works (adaptive pooling). With ``activation_checkpointing`` on, each of the 8
    residual blocks of the four stages is recomputed in the backward pass instead of storing
    its activations; gradients and BatchNorm statistics stay the same (spec: "without changing
    the maths"). Per block, not per stage: ``layer1`` (full T, 56 x 56) holds most of the
    activation memory, so recomputing a whole stage at once saved only ~10% at T = 64 (A4).
    """

    def __init__(self, activation_checkpointing: bool = False) -> None:
        super().__init__(BasicBlock, [Conv2Plus1D] * 4, [2, 2, 2, 2], R2Plus1dStem, num_classes=1)
        self.activation_checkpointing = activation_checkpointing

    def _stage(self, stage: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        if not (self.activation_checkpointing and self.training and torch.is_grad_enabled()):
            return stage(x)
        for block in stage:
            x = checkpoint(
                block, x, use_reentrant=False, context_fn=partial(_recompute_contexts, block)
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, T, H, W) normalised clips -> (B,) predicted EF in %."""
        x = self.stem(x)
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = self._stage(stage, x)
        x = self.avgpool(x).flatten(1)
        # The head runs in FP32 even under bf16 autocast: a bf16 output keeps 8 mantissa bits,
        # which rounds EF to 0.25-point steps between 32 and 64 (seen in the A6 smoke run).
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.fc(x.float()).squeeze(-1)


def kinetics_state_dict(progress: bool = False) -> dict[str, torch.Tensor]:
    """torchvision's Kinetics-400 weights for R(2+1)D-18 without the 400-class head.

    Downloaded once to the torch hub cache (``~/.cache/torch``) and hash-checked.
    """
    state = R2Plus1D_18_Weights.KINETICS400_V1.get_state_dict(progress=progress, check_hash=True)
    return {k: v for k, v in state.items() if not k.startswith("fc.")}


def build_model(
    pretrained: bool = True,
    head_bias: float | None = None,
    activation_checkpointing: bool = False,
) -> EFRegressor:
    """The sweep model. Contract function (spec, "Functions A provides").

    Args:
        pretrained: start from Kinetics-400 weights (D-03); False = random initialisation.
        head_bias: initial bias of the ``Linear(512, 1)`` head, normally the training-set mean
            EF (``model.head_bias: train_mean_ef``); None keeps PyTorch's default init.
        activation_checkpointing: recompute each residual block in backward (D-06, A4).

    Returns:
        ``EFRegressor`` mapping (B, 3, T, 112, 112) to (B,), in train mode on the CPU.
    """
    model = EFRegressor(activation_checkpointing)
    if pretrained:
        missing, unexpected = model.load_state_dict(kinetics_state_dict(), strict=False)
        if unexpected or set(missing) != {"fc.weight", "fc.bias"}:
            raise RuntimeError(f"Kinetics weights: missing {missing}, unexpected {unexpected}")
    if head_bias is not None:
        with torch.no_grad():
            model.fc.bias.fill_(float(head_bias))
    return model


def load_b0(weights_file: Path, device: str | torch.device = "cpu") -> EFRegressor:
    """EchoNet-Dynamic's released 32-frame R(2+1)D-18 (B0), in eval mode on ``device``.

    Args:
        weights_file: ``r2plus1d_18_32_2_pretrained.pt`` from the echonet/dynamic v1.0.0 release.
    """
    with torch.serialization.safe_globals(_ECHONET_SAFE_GLOBALS):
        ckpt = torch.load(weights_file, map_location="cpu", weights_only=True)
    state = {k.removeprefix(ECHONET_PREFIX): v for k, v in ckpt["state_dict"].items()}
    model = EFRegressor()
    model.load_state_dict(state, strict=True)
    log.info("loaded B0 from %s (EchoNet epoch %s)", weights_file, ckpt.get("epoch"))
    return model.to(device).eval()


def save_checkpoint(
    path: Path, model: nn.Module, config: dict, epoch: int, val_mae: float, **extra: object
) -> None:
    """Write a run checkpoint in the contract format (spec, "Files on disk", Checkpoint).

    The file is a dict with ``state_dict``, ``config`` (plain dict incl. T, period, config_id,
    seed), ``epoch``, ``val_mae`` plus any ``extra`` entries (e.g. optimizer state in
    ``last.pt``). Written to a temporary file first, so a crash never leaves a half file.
    """
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    payload = {
        "state_dict": model.state_dict(),
        "config": config,
        "epoch": epoch,
        "val_mae": val_mae,
        **extra,
    }
    path = Path(path)
    part = path.with_name(path.name + ".part")
    torch.save(payload, part)
    part.replace(path)


def load_checkpoint(
    run: Path, device: str | torch.device = "cpu", file: str = CHECKPOINT_BEST
) -> tuple[EFRegressor, dict]:
    """A trained model from a run folder. Contract function (spec, "Functions A provides").

    Args:
        run: ``runs/<config_id>/seed<k>/``.
        file: checkpoint file inside ``run``; ``best.pt`` by default.

    Returns:
        ``(model in eval mode on device, config dict)``.
    """
    ckpt = torch.load(Path(run) / file, map_location="cpu", weights_only=True)
    model = EFRegressor()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt["config"]
