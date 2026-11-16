"""Gate D (task A11): the app handles 6 inputs, each in under 3 s on the RTX 4060.

    uv run python app/gate_d.py --runs runs/fs32/seed0 runs/fs32/seed1

Inputs: 5 EchoNet TEST videos from the local raw folder and 1 synthetic non-echo video (colour
noise, 160 x 120). Each must return an EF or a clear warning; the time is the full request
without the optional IG overlay. Shows test-set errors, so it needs the ``models-frozen`` tag.
Run only when no training queue uses the GPU.
"""

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import load_ensemble  # noqa: E402

from app import local_examples, make_runner  # noqa: E402
from echoef.config import load_paths  # noqa: E402
from echoef.data.dataset import load_stats  # noqa: E402
from echoef.eval.evaluation import require_test_unlocked  # noqa: E402
from echoef.utils import setup_logging  # noqa: E402
from echoef.utils.gpu import require_gpu_free  # noqa: E402

log = logging.getLogger("gate_d")
LIMIT_S = 3.0


def synthetic_video(folder: Path) -> Path:
    """A 2 s colour-noise clip, 160 x 120: nothing like an echo."""
    path = folder / "synthetic_noise.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (160, 120))
    rng = np.random.default_rng(0)
    for _ in range(100):
        writer.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    writer.release()
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gate D: 6 inputs, each under 3 s.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--paths", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging()
    require_test_unlocked(True)  # shows test errors: only after Gate C
    roots = load_paths(args.paths)
    if args.device == "cuda":
        require_gpu_free(roots["runs_root"])
    ens = load_ensemble(args.runs, args.device)
    run = make_runner(ens, load_stats(roots["processed_root"]), roots["processed_root"])
    with tempfile.TemporaryDirectory() as tmp:
        inputs = local_examples(roots["raw_root"], roots["processed_root"], 5)
        inputs.append(str(synthetic_video(Path(tmp))))
        run(inputs[0], False)  # warm-up: CUDA context and cuDNN autotuning
        ok = True
        for path in inputs:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            text, notes, _, _ = run(path, False)
            seconds = time.perf_counter() - t0
            answered = text.startswith("## EF") or "could not" in text
            synthetic = "synthetic" in Path(path).name
            warned = synthetic and "No warnings." not in notes
            passed = seconds < LIMIT_S and answered and (warned or not synthetic)
            ok &= passed
            log.info(
                "%-28s %5.2f s  %s  %s",
                Path(path).name,
                seconds,
                "PASS" if passed else "FAIL",
                text.splitlines()[0],
            )
    log.info("Gate D: %s", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
