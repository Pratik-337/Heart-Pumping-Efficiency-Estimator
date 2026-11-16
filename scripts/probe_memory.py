"""Task A4: does micro-batch 8 fit at each T, and how long does an epoch take on this GPU?

For each config (default fs8 fs16 fs32 fs64: one per T; the FW configs share their T and so
their compute):

1. One training step at ``train.micro_batch`` on random clips, without activation
   checkpointing; if it does not fit, again with checkpointing. If neither fits, the config is
   reported and skipped: the batch is never lowered silently (CLAUDE.md).
2. The largest micro-batch that fits with that setting (searched upwards to --max-batch).
3. ``--steps`` timed training micro-steps and evaluation batches on real data, then the
   epoch and 10-epoch run times they imply.
4. With --compile, the same timing with ``torch.compile`` (spec: compile only if >= 15% faster).

Writes <results_root>/probe/probe_memory_<machine>[_<tag>].csv and .json (provenance,
nvidia-smi).
Run only while no training queue uses this GPU.

    uv run python scripts/probe_memory.py --machine rtx4060 [--configs fs64] [--compile]
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from echoef.config import load_config, load_paths
from echoef.contracts import CONFIGS
from echoef.data.dataset import TRAIN_MODE, dataset_from_config, load_stats, load_videos
from echoef.models.factory import build_model
from echoef.train.probe import (
    epoch_seconds,
    largest_micro_batch,
    memory_trial,
    peak_memory_mb,
    time_eval,
    time_training,
)
from echoef.train.trainer import check_loss, configure_backends, make_optimizer
from echoef.utils import git_commit, git_dirty, library_versions, seed_everything, setup_logging
from echoef.utils.gpu import cap_to_free_vram, gpu_name, nvidia_smi, require_gpu_free
from echoef.utils.seeding import seed_worker

log = logging.getLogger("probe_memory")
BATCH_CANDIDATES = (8, 12, 16, 24, 32, 48, 64)


def _loader(cfg, split, mode, processed, stats, batch_size, workers, seed) -> DataLoader:
    return DataLoader(
        dataset_from_config(cfg, split, mode, processed, stats),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def probe_config(config_id: str, args, roots: dict, stats: dict, device: torch.device) -> dict:
    cfg = load_config(config_id)
    seed_everything(cfg.seed)
    configure_backends(cfg, cudnn_benchmark=not args.no_cudnn_benchmark)
    check_loss(cfg)
    processed = roots["processed_root"]
    mb = int(cfg.train.micro_batch)
    n_train = len(load_videos(processed, "TRAIN"))
    n_val = len(load_videos(processed, "VAL"))
    train_mean_ef = float(load_videos(processed, "TRAIN")["EF"].mean())
    row = {"config_id": config_id, "T": int(cfg.T), "period": int(cfg.period), "micro_batch": mb}

    def make_model(checkpointing: bool):
        return build_model(
            pretrained=False, head_bias=train_mean_ef, activation_checkpointing=checkpointing
        )

    trial = memory_trial(make_model, cfg, mb, device, activation_checkpointing=False)
    row["fits_without_checkpointing"] = trial.fits
    log.info("%s: %s", config_id, trial)
    if not trial.fits:
        trial = memory_trial(make_model, cfg, mb, device, activation_checkpointing=True)
        log.info("%s: %s", config_id, trial)
    row["activation_checkpointing"] = trial.activation_checkpointing
    row["fits"] = trial.fits
    row["peak_alloc_mb"], row["peak_reserved_mb"] = trial.peak_alloc_mb, trial.peak_reserved_mb
    if not trial.fits:
        log.error(
            "%s: micro-batch %d does not fit even with checkpointing; STOP and report",
            config_id,
            mb,
        )
        return row

    bigger = [b for b in BATCH_CANDIDATES if mb < b <= args.max_batch]
    trials = largest_micro_batch(make_model, cfg, device, trial.activation_checkpointing, bigger)
    for t in trials:
        log.info("%s: %s", config_id, t)
    row["largest_micro_batch"] = max([mb] + [t.micro_batch for t in trials if t.fits])
    row["largest_tried"] = max([mb] + [t.micro_batch for t in trials])

    model = build_model(
        pretrained=True,
        head_bias=train_mean_ef,
        activation_checkpointing=trial.activation_checkpointing,
    ).to(device)
    optimizer = make_optimizer(model, cfg)
    train_loader = _loader(cfg, "TRAIN", TRAIN_MODE, processed, stats, mb, args.workers, cfg.seed)
    val_loader = _loader(cfg, "VAL", "M1", processed, stats, mb, args.workers, cfg.seed)
    torch.cuda.reset_peak_memory_stats(device)
    row |= time_training(model, optimizer, train_loader, cfg, device, args.steps, args.warmup)
    row["peak_alloc_mb_real"] = peak_memory_mb(device)[0]
    row |= time_eval(model, val_loader, cfg, device, args.steps, args.warmup)
    smi = nvidia_smi() or {}
    row |= {f"smi_{k}": v for k, v in smi.items() if k not in ("name", "memory.total")}
    row["epoch_s_est"] = epoch_seconds(
        n_train, n_val, mb, row["train_step_ms"] / 1000, row["eval_step_ms"] / 1000
    )
    row["run_min_est"] = row["epoch_s_est"] * int(cfg.train.epochs) / 60
    log.info(
        "%s: %s",
        config_id,
        {k: row[k] for k in ("train_step_ms", "eval_step_ms", "epoch_s_est", "run_min_est")},
    )
    del model, optimizer

    if args.compile:
        try:
            compiled = torch.compile(
                build_model(
                    pretrained=True,
                    head_bias=train_mean_ef,
                    activation_checkpointing=trial.activation_checkpointing,
                ).to(device)
            )
            optimizer = make_optimizer(compiled, cfg)
            timing = time_training(
                compiled, optimizer, train_loader, cfg, device, args.steps, args.warmup
            )
            row["compile_train_step_ms"] = timing["train_step_ms"]
            row["compile_speedup"] = row["train_step_ms"] / timing["train_step_ms"] - 1
            log.info(
                "%s: compiled %.1f ms/step (%+.1f%%)",
                config_id,
                timing["train_step_ms"],
                100 * row["compile_speedup"],
            )
            del compiled, optimizer
        except Exception as exc:  # a compile failure is a result, not a crash of the probe
            row["compile_error"] = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("%s: torch.compile failed: %s", config_id, row["compile_error"])
    torch.cuda.empty_cache()
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Memory and speed probe per T (A4).")
    parser.add_argument("--machine", required=True, help="e.g. rtx4060 (assignment.yaml key)")
    parser.add_argument(
        "--configs", nargs="+", default=["fs8", "fs16", "fs32", "fs64"], choices=list(CONFIGS)
    )
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    parser.add_argument("--steps", type=int, default=50, help="timed micro-steps (spec: 50)")
    parser.add_argument("--warmup", type=int, default=10, help="untimed micro-steps first")
    parser.add_argument("--max-batch", type=int, default=64, help="largest micro-batch to try")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes")
    parser.add_argument("--compile", action="store_true", help="also time torch.compile")
    parser.add_argument("--no-cudnn-benchmark", action="store_true", help="autotuning off")
    parser.add_argument("--tag", default="", help="suffix for the output file names")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("probe_memory.py measures a GPU; CUDA is not available")
    roots = load_paths(args.paths)
    out_dir = roots["results_root"] / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"probe_memory_{args.machine}{'_' + args.tag if args.tag else ''}"
    setup_logging(stem.with_suffix(".log"))
    device = torch.device("cuda")
    require_gpu_free(roots["runs_root"])
    meta = {
        "task": "A4",
        "machine": args.machine,
        "gpu_name": gpu_name(device),
        "vram_total_mb": torch.cuda.get_device_properties(device).total_memory / 2**20,
        "vram_cap_mb": cap_to_free_vram(device),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "versions": library_versions(),
        "args": vars(args) | {"paths": str(args.paths)},
        "nvidia_smi_start": nvidia_smi(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    stats = load_stats(roots["processed_root"])
    rows = []
    for config_id in args.configs:
        row = probe_config(config_id, args, roots, stats, device)
        rows.append({"machine": args.machine, "gpu_name": meta["gpu_name"], **row})
        pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)  # after every config
    meta["nvidia_smi_end"] = nvidia_smi()
    meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("wrote %s.csv and .json", stem)


if __name__ == "__main__":
    main()
