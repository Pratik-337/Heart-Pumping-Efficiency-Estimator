"""Tasks B6/B7: explanations and their numeric tests for trained configs (seed 0).

    uv run python scripts/run_xai.py --configs fs8 fw8 fs32 fs64 --n-videos 100 --device cuda \
        --split test --confirm-test                         # final run, after Gate C
    uv run python scripts/run_xai.py --configs fs8 --n-videos 6 --split val --device cpu  # smoke

Writes, under results_root:
- ``xai_sample.csv`` (``xai_sample_val.csv`` for --split val): the fixed-seed video sample;
- ``xai/<config_id>/xai_metrics.csv`` and ``examples.pdf`` (echo frames: git-ignored folder);
- ``xai/<sanity config>/sanity.csv``: randomisation check on --sanity-videos videos;
- ``tables/xai_summary.csv``, ``tables/xai_sanity.csv`` and ``figures/xai_summary.pdf``: numbers
  only, committable.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from echoef.config import load_paths
from echoef.contracts import CONFIGS, run_dir
from echoef.data.dataset import load_clip, load_stats, load_videos
from echoef.data.sampling import edes_start
from echoef.eval.evaluation import require_test_unlocked
from echoef.models.factory import load_checkpoint
from echoef.train.steps import configure_backends
from echoef.utils import seed_everything, setup_logging
from echoef.utils.gpu import require_gpu_free
from echoef.xai.pipeline import (
    METHODS,
    draw_sample,
    explain_video,
    metrics_frame,
    sanity_rows,
    sanity_summary,
    summarise,
)
from echoef.xai.render import example_panel, summary_chart

log = logging.getLogger("run_xai")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Explanations and their numeric tests (B6/B7).")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), required=True)
    parser.add_argument("--n-videos", type=int, default=100)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--confirm-test", action="store_true", help="required for --split test")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--sanity-config", default="fs32", help="'' to skip the sanity check")
    parser.add_argument("--sanity-videos", type=int, default=20)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    args = parser.parse_args(argv)

    split = args.split.upper()
    if split == "TEST":
        require_test_unlocked(args.confirm_test)
    roots = load_paths(args.paths)
    results, processed = roots["results_root"], roots["processed_root"]
    setup_logging(results / "xai" / f"run_xai_{args.split}.log")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        require_gpu_free(roots["runs_root"])
    stats = load_stats(processed)
    sample_file = results / (
        "xai_sample.csv" if split == "TEST" else f"xai_sample_{args.split}.csv"
    )
    if sample_file.exists():  # drawn once (spec); reused by every later run
        sample = pd.read_csv(sample_file)
        if len(sample) < args.n_videos:
            parser.error(f"{sample_file} holds {len(sample)} videos; delete it to draw a new one")
    else:
        sample = draw_sample(load_videos(processed, split), args.n_videos, args.seed)
        sample.to_csv(sample_file, index=False)
    sample = sample.head(args.n_videos)
    log.info(
        "%d videos in the %s sample (%s)",
        len(sample),
        split,
        sample["band"].value_counts().to_dict(),
    )

    all_metrics, sanity = [], None
    methods = tuple(args.methods)
    for config_id in args.configs:
        model, cfg = load_checkpoint(run_dir(roots["runs_root"], config_id, 0), device)
        configure_backends(OmegaConf.create(cfg))
        t, period = int(cfg["T"]), int(cfg["period"])
        # bf16 as trained on the GPU; fp32 on the CPU, where bf16 autocast is emulated and slow
        precision = str(cfg["train"]["precision"]) if device.type == "cuda" else "fp32"
        rows, examples, worst = [], {}, None
        for i, video in sample.iterrows():
            r, ex = explain_video(
                model,
                video,
                config_id,
                t,
                period,
                processed,
                stats,
                device,
                precision,
                methods,
                args.ig_steps,
            )
            rows += r
            if ex is not None:
                examples.setdefault(ex["band"], ex)
                if worst is None or abs(ex["ef_pred"] - ex["ef_true"]) > abs(
                    worst["ef_pred"] - worst["ef_true"]
                ):
                    worst = ex
            log.info(
                "%s: %d / %d videos (%s)",
                config_id,
                i + 1,
                len(sample),
                "skipped" if ex is None else "done",
            )
        metrics = metrics_frame(rows)
        out = results / "xai" / config_id
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "xai_metrics.csv", index=False)
        all_metrics.append(metrics)
        panel = [
            examples[b]
            | {
                "label": f"{b}: EF {examples[b]['ef_true']:.0f}, predicted "
                f"{examples[b]['ef_pred']:.0f}"
            }
            for b in ("EF <= 40", "EF 41-49", "EF >= 50")
            if b in examples
        ]
        if worst is not None and "ig" in methods and "occlusion" in methods:
            panel.append(
                worst
                | {
                    "label": f"largest error: EF {worst['ef_true']:.0f}, "
                    f"predicted {worst['ef_pred']:.0f}"
                }
            )
            example_panel(panel, stats, f"{config_id.upper()}, {args.split}").savefig(
                out / "examples.pdf"
            )
        if config_id == args.sanity_config:
            clips = {}
            for _, video in sample.iterrows():
                start = edes_start(
                    int(video["ed_frame"]),
                    int(video["es_frame"]),
                    t,
                    period,
                    int(video["n_frames"]),
                )
                if start is not None:
                    clips[video["FileName"]] = load_clip(
                        video["FileName"], t, period, start, processed, stats
                    ).to(device)
                if len(clips) == args.sanity_videos:
                    break
            sanity = sanity_rows(model, clips, config_id, precision, ig_steps=args.ig_steps)
            sanity.to_csv(out / "sanity.csv", index=False)
        log.info("%s done: %s", config_id, summarise(metrics).round(3).to_dict("records"))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarise(pd.concat(all_metrics, ignore_index=True), args.seed)
    (results / "tables").mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    suffix = "" if split == "TEST" else f"_{args.split}"
    summary.to_csv(results / "tables" / f"xai_summary{suffix}.csv", index=False)
    if sanity is not None and len(sanity):
        sanity_summary(sanity).to_csv(results / "tables" / f"xai_sanity{suffix}.csv", index=False)
    summary_chart(summary, sanity).savefig(results / "figures" / f"xai_summary{suffix}.pdf")
    log.info("wrote %s", results / "tables" / f"xai_summary{suffix}.csv")


if __name__ == "__main__":
    main()
