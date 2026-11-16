"""Task A7: cost of every config on this GPU -> <results_root>/cost.csv (contracts.COST_COLUMNS).

    uv run python scripts/benchmark_cost.py --all            # after the training queue (GPU)
    uv run python scripts/benchmark_cost.py --configs fs8 --device cpu --warmup 0 --runs 1

Weights do not change speed or memory, so random weights are used; the model setting
(checkpointing for training at T = 64) comes from each config. Mean M2 clips are over the used
TEST videos' lengths (frame counts only; no labels or predictions are read). Epoch time comes
from the seed-0 run's metrics.json. Writes cost_meta.json with nvidia-smi before and after.
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

from echoef.config import load_config, load_paths
from echoef.contracts import CONFIGS, COST_COLUMNS, run_dir
from echoef.data.dataset import load_videos
from echoef.eval.cost import (
    count_params,
    epoch_time_s,
    gflops_per_clip,
    latency_ms,
    mean_m2_clips,
    peak_infer_mb,
)
from echoef.models.factory import build_model
from echoef.train.probe import memory_trial
from echoef.train.steps import configure_backends
from echoef.utils import git_commit, library_versions, seed_everything, setup_logging
from echoef.utils.gpu import cap_to_free_vram, gpu_name, nvidia_smi, require_gpu_free

log = logging.getLogger("benchmark_cost")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark cost per config (A7).")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=None)
    parser.add_argument("--all", action="store_true", help="all 7 configs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", type=int, default=50, help="timed runs (spec: 50)")
    parser.add_argument("--warmup", type=int, default=10, help="warm-up runs (spec: 10)")
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    args = parser.parse_args(argv)
    configs = list(CONFIGS) if args.all else (args.configs or parser.error("--configs or --all"))

    roots = load_paths(args.paths)
    setup_logging(roots["results_root"] / "cost.log")
    seed_everything(0)
    device = torch.device(args.device)
    if device.type == "cuda":
        require_gpu_free(roots["runs_root"])
        cap_to_free_vram(device)
    n_frames = load_videos(roots["processed_root"], "TEST")["n_frames"].tolist()
    meta = {
        "git_commit": git_commit(),
        "versions": library_versions(),
        "smi_start": nvidia_smi(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": args.runs,
        "warmup": args.warmup,
    }

    rows = []
    for config_id in configs:
        cfg = load_config(config_id)
        configure_backends(cfg)
        t, period, mb = int(cfg.T), int(cfg.period), int(cfg.train.micro_batch)
        precision = str(cfg.train.precision)
        model = build_model(pretrained=False, head_bias=55.0)
        clips = mean_m2_clips(n_frames, t, period)
        gflops = gflops_per_clip(model, t)
        row = {
            "config_id": config_id, "T": t, "period": period, "params": count_params(model),
            "gflops_clip": gflops, "mean_clips_m2": clips, "gflops_video_m2": gflops * clips,
            "latency_ms_m1": latency_ms(model, t, 1, device, precision, mb, args.warmup, args.runs),
            "latency_ms_m2": latency_ms(model, t, round(clips), device, precision, mb,
                                        args.warmup, args.runs),
            "peak_vram_mb_infer": peak_infer_mb(model, t, device, precision),
            "micro_batch": mb,
            "epoch_time_s": epoch_time_s(run_dir(roots["runs_root"], config_id, 0)),
            "gpu_name": gpu_name(device) if device.type == "cuda" else "cpu",
        }  # fmt: skip
        if device.type == "cuda":
            ckpt = bool(cfg.model.activation_checkpointing)
            trial = memory_trial(
                lambda c, ck=ckpt: build_model(False, 55.0, ck), cfg, mb, device, ckpt
            )
            row["peak_vram_mb_train"] = trial.peak_alloc_mb
        else:
            row["peak_vram_mb_train"] = None
        rows.append(row)
        log.info("%s: %s", config_id, {k: row[k] for k in ("gflops_clip", "latency_ms_m1",
                 "latency_ms_m2", "peak_vram_mb_infer", "peak_vram_mb_train")})  # fmt: skip
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = roots["results_root"]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[COST_COLUMNS].to_csv(out / "cost.csv", index=False)
    meta |= {"smi_end": nvidia_smi(), "finished": datetime.now(UTC).isoformat(timespec="seconds")}
    (out / "cost_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("wrote %s", out / "cost.csv")


if __name__ == "__main__":
    main()
