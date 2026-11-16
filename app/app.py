"""Task B8: local Gradio demo, video in, EF out (spec, "Prototype app"; D-14).

    uv run python app/app.py [--runs runs/fs32/seed0 runs/fs32/seed1] [--examples 5]

``--runs``: seed run folders of one config; default: every finished seed of the config chosen on
validation (D-13, ``results/tables/demo_choice.csv`` from ``aggregate_results.py``).
``--examples`` lists that many EchoNet TEST videos from the local raw folder (read at run time,
never copied into the app). Always ``share=False``: the app runs on this machine only.
Uploads (Gradio's cache), saliency animations and every other output go to one private temporary
folder that is deleted when the app stops (Ctrl+C or SIGTERM), so no copy of an echo video stays
behind in /tmp.
"""

import argparse
import logging
import os
import signal
import sys
import tempfile
import uuid
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # no telemetry or version pings: local only

import gradio as gr  # noqa: E402
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # app/ as a script folder

from inference import (  # noqa: E402
    BANNER,
    demo_runs,
    ig_animation,
    ig_maps,
    ig_overlay,
    importance_figure,
    load_ensemble,
    predict,
    read_video,
    report,
    video_info,
)

from echoef.config import load_paths  # noqa: E402
from echoef.data.dataset import load_stats  # noqa: E402
from echoef.data.layout import INDEX_FILE  # noqa: E402
from echoef.utils import setup_logging  # noqa: E402

log = logging.getLogger("app")


def make_runner(ens, stats: dict, processed_root: Path | None, work_dir: Path | None = None):
    """``run(video_path, want_ig) -> (result md, notes md, figure, IG strip, IG animation path)``.

    ``work_dir``: where IG animations are written (default: a temporary folder made on first use).
    """
    folder = {"path": Path(work_dir) if work_dir else None}

    def run(video_path: str | None, want_ig: bool):
        if not video_path:
            return "Upload a video first.", "", None, None, None
        try:
            frames, warnings = read_video(Path(video_path))
        except ValueError as exc:
            return f"**{exc}**", "", None, None, None
        result = predict(ens, frames, stats)
        info = video_info(Path(video_path).name, processed_root)
        text, notes = report(ens, result, warnings, info["reference"], len(frames))
        fig = importance_figure(ens, frames, stats, info["ed"], info["es"])
        strip = animation = None
        if want_ig:
            maps = ig_maps(ens, frames, stats)
            strip = ig_overlay(ens, frames, stats, maps=maps)
            if folder["path"] is None:
                folder["path"] = Path(tempfile.mkdtemp(prefix="ef_demo_"))
            gif = folder["path"] / f"ig_{uuid.uuid4().hex}.gif"
            animation = str(ig_animation(ens, maps, gif))
        log.info(
            "%s: EF %.1f in %.0f ms", Path(video_path).name, result["ef"], result["latency_ms"]
        )
        return text, notes, fig, strip, animation

    return run


def build_ui(run, examples: list[str]) -> gr.Blocks:
    """The Gradio page around ``run`` (from ``make_runner``)."""
    with gr.Blocks(title="EF estimator, research prototype", analytics_enabled=False) as demo:
        gr.Markdown(f"# Left-ventricular ejection fraction from an echo video\n{BANNER}")
        with gr.Row():
            video = gr.Video(
                label="Apical four-chamber echo video (AVI or MP4)", sources=["upload"]
            )
            with gr.Column():
                result_md = gr.Markdown()
                notes_md = gr.Markdown()
        want_ig = gr.Checkbox(label="Also show the Integrated Gradients overlay (a few seconds)")
        button = gr.Button("Estimate EF", variant="primary")
        plot = gr.Plot(label="Frame importance (temporal occlusion)")
        overlay = gr.Image(label="Integrated Gradients: 8 frames of the central clip", type="numpy")
        animation = gr.Image(
            label="Integrated Gradients: the whole central clip (left: input, right: overlay)",
            type="filepath",
        )
        if examples:
            gr.Examples(examples=[[e] for e in examples], inputs=[video])
        button.click(
            run,
            inputs=[video, want_ig],
            outputs=[result_md, notes_md, plot, overlay, animation],
        )
    return demo


def local_examples(raw_root: Path, processed_root: Path, n: int) -> list[str]:
    """Paths of ``n`` EchoNet TEST videos in the local raw folder (none are copied)."""
    if n <= 0 or not (processed_root / INDEX_FILE).exists():
        return []
    index = pd.read_parquet(processed_root / INDEX_FILE)
    test = index[(index["Split"] == "TEST") & index["has_trace"]].sample(n, random_state=0)
    paths = [raw_root / "Videos" / f"{name}.avi" for name in test["FileName"]]
    return [str(p) for p in paths if p.exists()]


def _exit_cleanly(*_) -> None:
    """SIGTERM -> normal exit, so the temporary folder is deleted. Later signals (``uv run``
    forwards one) are ignored, so they cannot cut the cleanup short."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sys.exit(0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local EF demo (B8).")
    parser.add_argument(
        "--runs", nargs="+", type=Path, default=None, help="seed folders (default: D-13 choice)"
    )
    parser.add_argument("--examples", type=int, default=0, help="TEST videos to list")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    roots = load_paths(args.paths)
    runs = args.runs or demo_runs(roots["results_root"], roots["runs_root"])
    ens = load_ensemble(runs, args.device)
    stats = load_stats(roots["processed_root"])
    examples = local_examples(roots["raw_root"], roots["processed_root"], args.examples)
    log.info("config %s, seeds %s on %s", ens.config_id, ens.seeds, ens.device)
    with tempfile.TemporaryDirectory(prefix="ef_demo_") as work:  # deleted on exit
        os.environ["GRADIO_TEMP_DIR"] = work  # Gradio's upload cache: copies of the echo videos
        signal.signal(signal.SIGTERM, _exit_cleanly)  # clean up on kill, not only on Ctrl+C
        demo = build_ui(make_runner(ens, stats, roots["processed_root"], Path(work)), examples)
        demo.launch(share=False, server_name="127.0.0.1", server_port=args.port,
                    allowed_paths=[work])  # fmt: skip  # never public


if __name__ == "__main__":
    main()
