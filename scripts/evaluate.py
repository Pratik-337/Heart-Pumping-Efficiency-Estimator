"""Tasks A5/A10: write contract prediction CSVs for trained runs and for Tier 0 (B0).

    uv run python scripts/evaluate.py --split val --tier0 fs32 --modes M2    # Gate A
    uv run python scripts/evaluate.py --split val --tier0                    # Tier 0, 7 settings
    uv run python scripts/evaluate.py --split val --runs runs/fs8/seed0
    uv run python scripts/evaluate.py --split test --confirm-test --all      # after Gate C only

``--all`` = every trained run with a DONE marker plus Tier 0 at all 7 settings. The TEST split
needs --confirm-test and the git tag ``models-frozen`` (CLAUDE.md, Hard rules).
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from echoef.config import load_config, load_paths
from echoef.contracts import CONFIGS, MODES
from echoef.data.dataset import load_stats
from echoef.eval.evaluation import (
    evaluate_model,
    finished_runs,
    require_test_unlocked,
    tier0_run,
    tier0_run_id,
    write_tier0_record,
)
from echoef.models.factory import load_b0, load_checkpoint
from echoef.train.trainer import configure_backends
from echoef.utils import seed_everything, setup_logging
from echoef.utils.gpu import require_gpu_free

log = logging.getLogger("evaluate")
GATE_A_CONFIG, GATE_A_MAX_MAE = "fs32", 5.0  # spec, task A5: B0 at T = 32, p = 2, val MAE (M2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict a split for runs and Tier 0 (A5/A10).")
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--confirm-test", action="store_true", help="required for --split test")
    parser.add_argument("--runs", nargs="+", type=Path, default=[], help="trained run folders")
    parser.add_argument(
        "--tier0", nargs="*", choices=list(CONFIGS), default=None,
        help="evaluate B0 at these settings (no names = all 7)",
    )  # fmt: skip
    parser.add_argument("--all", action="store_true", help="every DONE run plus Tier 0")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes")
    parser.add_argument("--paths", type=Path, default=None, help="default: configs/paths.yaml")
    args = parser.parse_args(argv)

    split = args.split.upper()
    if split == "TEST":
        require_test_unlocked(args.confirm_test)
    roots = load_paths(args.paths)
    setup_logging(roots["runs_root"] / f"evaluate_{args.split}.log")
    seed_everything(0)  # inference is deterministic up to cuDNN; seeded anyway (CLAUDE.md)
    device = torch.device(args.device)
    if device.type == "cuda":
        require_gpu_free(roots["runs_root"])
    processed, stats = roots["processed_root"], load_stats(roots["processed_root"])
    runs = list(args.runs) + (finished_runs(roots["runs_root"]) if args.all else [])
    tier0 = list(CONFIGS) if (args.all or args.tier0 == []) else (args.tier0 or [])
    if not runs and not tier0:
        parser.error("nothing to evaluate: give --runs, --tier0 or --all")
    log.info(
        "split %s, modes %s, %d trained runs, Tier 0 at %s", split, args.modes, len(runs), tier0
    )

    for run in runs:
        model, cfg = load_checkpoint(run, device)
        configure_backends(OmegaConf.create(cfg))
        evaluate_model(
            model, run, split, args.modes, config_id=cfg["config_id"], seed=cfg["seed"],
            T=cfg["T"], period=cfg["period"], processed_root=processed, stats=stats,
            device=device, precision=cfg["train"]["precision"],
            batch_size=cfg["train"]["micro_batch"], workers=args.workers,
            m2_overlap=cfg["eval"]["m2_overlap"],
        )  # fmt: skip

    if tier0:
        b0_cfg = load_config("b0_echonet")
        configure_backends(b0_cfg)
        model = load_b0(roots["weights_root"] / b0_cfg.model.weights_file, device)
        for config_id in tier0:
            started = datetime.now(UTC).isoformat(timespec="seconds")
            _, t, period = CONFIGS[config_id]
            cfg = load_config("b0_echonet", [f"T={t}", f"period={period}"])
            cfg.config_id = tier0_run_id(config_id)
            run = tier0_run(roots["runs_root"], config_id)
            maes = evaluate_model(
                model, run, split, args.modes, config_id=cfg.config_id, seed=0, T=t,
                period=period, processed_root=processed, stats=stats, device=device,
                precision=cfg.train.precision, batch_size=cfg.train.micro_batch,
                workers=args.workers, m2_overlap=cfg.eval.m2_overlap,
            )  # fmt: skip
            write_tier0_record(run, cfg, started, device)
            if split == "VAL" and config_id == GATE_A_CONFIG and "M2" in maes:
                verdict = "PASS" if maes["M2"] < GATE_A_MAX_MAE else "FAIL"
                log.info("Gate A: B0 val MAE (M2) %.3f, limit %.1f: %s",
                         maes["M2"], GATE_A_MAX_MAE, verdict)  # fmt: skip


if __name__ == "__main__":
    main()
