"""One training run: epochs, checkpoints, logs, resume, final validation (task A6).

The step functions (loss, optimizer, autocast) live in ``train.steps`` and are re-exported here.
"""

import json
import logging
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from echoef.contracts import MODES, predictions_path
from echoef.data.dataset import TRAIN_MODE, dataset_from_config, load_videos
from echoef.eval.predict import mae, predict_split
from echoef.models.factory import (
    CHECKPOINT_BEST,
    CHECKPOINT_LAST,
    build_model,
    save_checkpoint,
)
from echoef.train.steps import (  # re-exported: probe, tests and scripts import them from here
    PRECISIONS,
    autocast,
    check_loss,
    configure_backends,
    make_optimizer,
    predict_batch,
    train_micro_step,
)
from echoef.utils.gpu import gpu_name, nvidia_smi
from echoef.utils.meta import run_meta
from echoef.utils.seeding import seed_worker

log = logging.getLogger(__name__)

# Run-folder files (spec, "Files on disk"); checkpoint names live in models.factory.
CONFIG_FILE, META_FILE, LOG_FILE, METRICS_FILE = (
    "config.yaml",
    "meta.json",
    "log.csv",
    "metrics.json",
)
TB_DIR, DONE_FILE = "tb", "DONE"


__all__ = [
    "PRECISIONS",
    "autocast",
    "check_loss",
    "configure_backends",
    "lr_factor",
    "make_optimizer",
    "predict_batch",
    "train_micro_step",
    "train_run",
]


# ----------------------------------------------------------------------------- training run (A6)

LOG_COLUMNS = ["epoch", "train_loss", "val_mae_m1", "lr", "seconds"]  # contract (spec, Files)
LOG_EXTRA = ["gpu_clock_mhz", "gpu_temp_c"]  # throttling check (spec, Laptop care)
LOG_EVERY = 100  # micro-steps between progress lines


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Learning-rate multiplier at optimizer step ``step`` (0-based).

    Linear warm-up to 1 over ``warmup_steps`` (``train.warmup_epochs``), then cosine decay to 0 at
    ``total_steps`` (``train.epochs``), spec "Model and training design", Optimiser.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _gpu_state(device: torch.device) -> dict[str, str]:
    smi = nvidia_smi() if device.type == "cuda" else None
    return {
        "gpu_clock_mhz": (smi or {}).get("clocks.sm", ""),
        "gpu_temp_c": (smi or {}).get("temperature.gpu", ""),
    }


def _rewrite_log(run: Path, keep_through_epoch: int) -> None:
    """On resume, drop log rows of epochs after the last saved one (they are trained again)."""
    path = run / LOG_FILE
    if path.exists():
        log_df = pd.read_csv(path)
        log_df[log_df["epoch"] <= keep_through_epoch].to_csv(path, index=False)


def train_run(
    cfg: DictConfig,
    run: Path,
    *,
    processed_root: Path,
    stats: dict[str, float],
    device: torch.device,
    workers: int,
    machine: str = "",
    make_model: Callable[[float], nn.Module] | None = None,
) -> dict:
    """Train one (config, seed) run into ``run`` and evaluate its best epoch on validation.

    Writes ``config.yaml``, ``meta.json``, ``log.csv``, ``tb/``, ``best.pt`` (best validation MAE
    in M1), ``last.pt`` (every epoch, with optimizer state), ``predictions_val_{M1,M2}.csv`` from
    ``best.pt``, ``metrics.json`` and finally ``DONE``. A folder with ``last.pt`` but no ``DONE``
    is resumed after its last saved epoch; each epoch reseeds the data order, so a resumed run
    sees the same batches as an uninterrupted one.

    Args:
        make_model: ``head_bias -> model`` (tests); default ``build_model`` as configured.

    Returns:
        The ``metrics.json`` dict.

    Raises:
        FileExistsError: ``run`` holds a ``config.yaml`` that differs from ``cfg``.
        FloatingPointError: the training loss became NaN or infinite.
    """
    run.mkdir(parents=True, exist_ok=True)
    if (run / DONE_FILE).exists():
        log.info("%s is DONE; nothing to do", run)
        return json.loads((run / METRICS_FILE).read_text())
    config = OmegaConf.to_container(cfg, resolve=True)
    saved_cfg = run / CONFIG_FILE
    if saved_cfg.exists() and OmegaConf.to_container(OmegaConf.load(saved_cfg)) != config:
        raise FileExistsError(f"{saved_cfg} differs from this config; use another folder")
    OmegaConf.save(cfg, saved_cfg)
    check_loss(cfg)
    configure_backends(cfg)
    started = datetime.now(UTC).isoformat(timespec="seconds")

    train_videos = load_videos(processed_root, "TRAIN")
    val_videos = load_videos(processed_root, "VAL")
    if cfg.model.head_bias != "train_mean_ef":
        raise ValueError(f"model.head_bias {cfg.model.head_bias!r}: only 'train_mean_ef'")
    head_bias = float(train_videos["EF"].mean())
    if make_model is None:
        if cfg.model.init != "kinetics400":
            raise ValueError(f"model.init {cfg.model.init!r}: sweep runs start from kinetics400")

        def make_model(bias: float) -> nn.Module:
            return build_model(True, bias, bool(cfg.model.activation_checkpointing))

    torch.manual_seed(int(cfg.seed))
    model = make_model(head_bias).to(device)
    optimizer = make_optimizer(model, cfg)
    micro_batch, grad_accum = int(cfg.train.micro_batch), int(cfg.train.grad_accum)
    precision, epochs = str(cfg.train.precision), int(cfg.train.epochs)
    generator = torch.Generator()
    loader = DataLoader(
        dataset_from_config(cfg, "TRAIN", TRAIN_MODE, processed_root, stats),
        batch_size=micro_batch,
        shuffle=True,
        drop_last=True,  # equal micro-batches: same BatchNorm batch size every step
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    warmup_steps = int(cfg.train.warmup_epochs) * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    base_lr = float(cfg.train.lr)

    start_epoch, best_mae, best_epoch = 0, math.inf, -1
    if (run / CHECKPOINT_LAST).exists():
        last = torch.load(run / CHECKPOINT_LAST, map_location="cpu", weights_only=True)
        model.load_state_dict(last["state_dict"])
        optimizer.load_state_dict(last["optimizer"])
        start_epoch = int(last["epoch"]) + 1
        best_mae, best_epoch = float(last["best_val_mae"]), int(last["best_epoch"])
        _rewrite_log(run, last["epoch"])
        log.info("resuming %s after epoch %d (best %.3f at epoch %d)",
                 run, last["epoch"], best_mae, best_epoch)  # fmt: skip
    meta = run_meta(config, int(cfg.seed), gpu_name(device) if device.type == "cuda" else "cpu",
                    started, "", machine=machine, vram_gb=_vram_gb(device),
                    task="A6", resumed_at_epoch=start_epoch)  # fmt: skip
    (run / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    writer = SummaryWriter(str(run / TB_DIR))
    log.info(
        "%s seed %d: %d train / %d val videos, %d micro-batches of %d per epoch, %d epochs, "
        "head bias %.2f, checkpointing %s",
        cfg.config_id, cfg.seed, len(train_videos), len(val_videos), len(loader), micro_batch,
        epochs, head_bias, cfg.model.activation_checkpointing,
    )  # fmt: skip

    for epoch in range(start_epoch, epochs):
        # Data order, augmentation and worker seeds depend on (seed, epoch) only (resume-safe).
        generator.manual_seed(int(cfg.seed) * 1000 + epoch)
        torch.manual_seed(int(cfg.seed) * 1000 + epoch)
        model.train()
        t0, losses, step = time.perf_counter(), [], epoch * steps_per_epoch
        set_lr(optimizer, base_lr * lr_factor(step, warmup_steps, total_steps))
        for i, batch in enumerate(loader):
            clip = batch["clip"].to(device, non_blocking=True)
            ef = batch["ef"].to(device, non_blocking=True)
            losses.append(train_micro_step(model, clip, ef, grad_accum, precision))
            if (i + 1) % grad_accum == 0 or i + 1 == len(loader):
                if not torch.isfinite(losses[-1]):
                    raise FloatingPointError(
                        f"loss {losses[-1].item()} at epoch {epoch}, batch {i}"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                set_lr(optimizer, base_lr * lr_factor(step, warmup_steps, total_steps))
            if (i + 1) % LOG_EVERY == 0:
                recent = torch.stack(losses[-LOG_EVERY:]).mean().item()
                rate = (i + 1) * micro_batch / (time.perf_counter() - t0)
                log.info("epoch %d  %d/%d  loss %.2f  %.1f clips/s", epoch, i + 1, len(loader),
                         recent, rate)  # fmt: skip
        train_loss = torch.stack(losses).mean().item()
        lr_now = optimizer.param_groups[0]["lr"]
        val_mae = mae(
            predict_split(
                model, val_videos, "VAL", "M1", config_id=cfg.config_id, seed=int(cfg.seed),
                T=int(cfg.T), period=int(cfg.period), processed_root=processed_root,
                stats=stats, device=device, precision=precision, batch_size=micro_batch,
                workers=workers, m2_overlap=float(cfg.eval.m2_overlap),
            )
        )  # fmt: skip
        seconds = time.perf_counter() - t0
        row = {"epoch": epoch, "train_loss": train_loss, "val_mae_m1": val_mae, "lr": lr_now,
               "seconds": round(seconds, 1), **_gpu_state(device)}  # fmt: skip
        path = run / LOG_FILE
        pd.DataFrame([row], columns=LOG_COLUMNS + LOG_EXTRA).to_csv(
            path, mode="a", header=not path.exists(), index=False
        )
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_mae_m1", val_mae, epoch)
        writer.add_scalar("lr", lr_now, epoch)
        writer.flush()
        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            save_checkpoint(run / CHECKPOINT_BEST, model, config, epoch, val_mae)
        save_checkpoint(run / CHECKPOINT_LAST, model, config, epoch, val_mae,
                        optimizer=optimizer.state_dict(), best_val_mae=best_mae,
                        best_epoch=best_epoch)  # fmt: skip
        log.info("epoch %d done: train loss %.2f, val MAE (M1) %.3f%s, %.0f s", epoch, train_loss,
                 val_mae, "  *best*" if best_epoch == epoch else "", seconds)  # fmt: skip
    writer.close()

    # Final: the best epoch on validation in both modes (contract prediction files).
    best = torch.load(run / CHECKPOINT_BEST, map_location="cpu", weights_only=True)
    model.load_state_dict(best["state_dict"])
    maes = {}
    for mode in MODES:
        preds = predict_split(
            model, val_videos, "VAL", mode, config_id=cfg.config_id, seed=int(cfg.seed),
            T=int(cfg.T), period=int(cfg.period), processed_root=processed_root, stats=stats,
            device=device, precision=precision, batch_size=micro_batch, workers=workers,
            m2_overlap=float(cfg.eval.m2_overlap),
        )  # fmt: skip
        preds.to_csv(predictions_path(run, "VAL", mode), index=False)
        maes[mode] = mae(preds)
    log_df = pd.read_csv(run / LOG_FILE)
    metrics = {
        "config_id": cfg.config_id,
        "seed": int(cfg.seed),
        "best_epoch": int(best["epoch"]),
        "best_val_mae_m1": float(best["val_mae"]),
        "val_mae_m1": maes["M1"],
        "val_mae_m2": maes["M2"],
        "epochs": epochs,
        "train_seconds": float(log_df["seconds"].sum()),
        "epoch_seconds_mean": float(log_df["seconds"].mean()),
    }
    (run / METRICS_FILE).write_text(json.dumps(metrics, indent=2) + "\n")
    meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    (run / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    (run / DONE_FILE).write_text(meta["finished"] + "\n")
    log.info("%s DONE: best epoch %d, val MAE M1 %.3f, M2 %.3f", run, metrics["best_epoch"],
             maes["M1"], maes["M2"])  # fmt: skip
    return metrics


def _vram_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2)
