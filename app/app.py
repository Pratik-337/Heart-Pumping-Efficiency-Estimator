"""Task B8: local Gradio demo, video in, EF out (spec, "Prototype app"; D-14).

    uv run python app/app.py --runs runs/fs32/seed0 runs/fs32/seed1 [--examples 5]

``--runs``: the seeds of the demo config (chosen on validation, D-13). ``--examples`` lists that
many EchoNet TEST videos from the local raw folder (read at run time, never copied into the app).
Always ``share=False``: the app runs on this machine only.
"""

import argparse
import logging
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # app/ as a script folder

from inference import (  # noqa: E402
    BANNER,
    ig_overlay,
    importance_figure,
    load_ensemble,
    predict,
    read_video,
    reference_ef,
    report,
)

from echoef.config import load_paths  # noqa: E402
from echoef.data.dataset import load_stats  # noqa: E402
from echoef.data.layout import INDEX_FILE  # noqa: E402
from echoef.utils import setup_logging  # noqa: E402

log = logging.getLogger("app")


def make_runner(ens, stats: dict, processed_root: Path | None):
    """``run(video_path, want_ig) -> (result markdown, notes markdown, figure, overlay)``."""

    def run(video_path: str | None, want_ig: bool):
        if not video_path:
            return "Upload a video first.", "", None, None
        try:
            frames, warnings = read_video(Path(video_path))
        except ValueError as exc:
            return f"**{exc}**", "", None, None
        result = predict(ens, frames, stats)
        ref = reference_ef(Path(video_path).name, processed_root)
        text, notes = report(ens, result, warnings, ref, len(frames))
        fig = importance_figure(ens, frames, stats)
        overlay = ig_overlay(ens, frames, stats) if want_ig else None
        log.info(
            "%s: EF %.1f in %.0f ms", Path(video_path).name, result["ef"], result["latency_ms"]
        )
        return text, notes, fig, overlay

    return run


def build_ui(run, examples: list[str]) -> gr.Blocks:
    """The Gradio page around ``run`` (from ``make_runner``)."""
    with gr.Blocks(title="EF estimator, research prototype") as demo:
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
        overlay = gr.Image(label="Integrated Gradients on the central clip", type="numpy")
        if examples:
            gr.Examples(examples=[[e] for e in examples], inputs=[video])
        button.click(run, inputs=[video, want_ig], outputs=[result_md, notes_md, plot, overlay])
    return demo


def local_examples(raw_root: Path, processed_root: Path, n: int) -> list[str]:
    """Paths of ``n`` EchoNet TEST videos in the local raw folder (none are copied)."""
    if n <= 0 or not (processed_root / INDEX_FILE).exists():
        return []
    index = pd.read_parquet(processed_root / INDEX_FILE)
    test = index[(index["Split"] == "TEST") & index["has_trace"]].sample(n, random_state=0)
    paths = [raw_root / "Videos" / f"{name}.avi" for name in test["FileName"]]
    return [str(p) for p in paths if p.exists()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local EF demo (B8).")
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="seed run folders")
    parser.add_argument("--examples", type=int, default=0, help="TEST videos to list")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    roots = load_paths(args.paths)
    ens = load_ensemble(args.runs, args.device)
    stats = load_stats(roots["processed_root"])
    examples = local_examples(roots["raw_root"], roots["processed_root"], args.examples)
    log.info("config %s, seeds %s on %s", ens.config_id, ens.seeds, ens.device)
    demo = build_ui(make_runner(ens, stats, roots["processed_root"]), examples)
    demo.launch(share=False, server_name="127.0.0.1", server_port=args.port)  # never public


if __name__ == "__main__":
    main()
