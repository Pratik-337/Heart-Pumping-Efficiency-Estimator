"""Model factory on small CPU inputs: head, checkpointing, B0 and run checkpoints (no download)."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from echoef.models import factory as f

# Small clips keep CPU tests fast; the network is fully convolutional plus adaptive pooling.
SMALL = (2, 3, 8, 32, 32)


def _kinetics_cached() -> bool:
    url = f.R2Plus1D_18_Weights.KINETICS400_V1.url
    return (Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[1]).exists()


@pytest.mark.parametrize("T", [1, 4, 8, 16])
def test_output_is_one_ef_per_clip(T):
    model = f.build_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, T, 32, 32))
    assert out.shape == (2,)


def test_head_is_linear_512_to_1_with_bias():
    model = f.build_model(pretrained=False, head_bias=55.6)
    assert (model.fc.in_features, model.fc.out_features) == (512, 1)
    assert model.fc.bias.item() == pytest.approx(55.6)
    params = sum(p.numel() for p in model.parameters())
    assert params == 31_300_638  # torchvision's 31,505,325 minus the 400-class head plus ours


def test_checkpointing_keeps_outputs_gradients_and_batchnorm_stats():
    torch.manual_seed(0)
    plain = f.build_model(pretrained=False).train()
    ckpt = copy.deepcopy(plain)
    ckpt.activation_checkpointing = True
    x = torch.randn(*SMALL)
    outs = []
    for model in (plain, ckpt):
        out = model(x)
        out.square().mean().backward()
        outs.append(out.detach())
    torch.testing.assert_close(outs[0], outs[1])
    for (name, p), q in zip(plain.named_parameters(), ckpt.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, msg=name)
    for (name, a), b in zip(plain.state_dict().items(), ckpt.state_dict().values(), strict=True):
        assert torch.equal(a, b), name  # running mean/var and num_batches_tracked updated once


def test_load_b0_from_echonet_format(tmp_path):
    source = f.build_model(pretrained=False, head_bias=55.6)
    path = tmp_path / "echonet.pt"
    torch.save(
        {
            "state_dict": {f"module.{k}": v for k, v in source.state_dict().items()},
            "epoch": 43,
            "best_loss": np.float64(0.79),  # numpy scalar, as in EchoNet's file
            "scheduler_dict": {"dtype": np.dtype(np.float64)},
        },
        path,
    )
    model = f.load_b0(path)
    assert not model.training
    for k, v in source.state_dict().items():
        assert torch.equal(model.state_dict()[k], v), k


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = f.build_model(pretrained=False, head_bias=50.0)
    config = {"config_id": "fs8", "T": 8, "period": 2, "seed": 0}
    f.save_checkpoint(tmp_path / f.CHECKPOINT_BEST, model, config, epoch=3, val_mae=5.5)
    assert not list(tmp_path.glob("*.part"))
    loaded, cfg = f.load_checkpoint(tmp_path)
    assert cfg == config and not loaded.training
    x = torch.randn(*SMALL)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model.eval()(x))
    raw = torch.load(tmp_path / f.CHECKPOINT_BEST, weights_only=True)
    assert (raw["epoch"], raw["val_mae"]) == (3, 5.5)


@pytest.mark.skipif(not _kinetics_cached(), reason="Kinetics weights not downloaded")
def test_kinetics_weights_load_into_everything_but_the_head():
    model = f.build_model(pretrained=True, head_bias=55.6)
    reference = f.kinetics_state_dict()
    for k, v in reference.items():
        assert torch.equal(model.state_dict()[k], v), k
    assert model.fc.bias.item() == pytest.approx(55.6)
