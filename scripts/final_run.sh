#!/usr/bin/env bash
# Tasks A7/A10: the unattended final run after Gate C (spec, "Final integration").
# Run in tmux on the RTX 4060 once the training queue has finished and `models-frozen` is tagged:
#     tmux new -s final 'bash scripts/final_run.sh 2>&1 | tee runs/final_run.log'
# Stops at the first failing step (set -e); rerun after fixing it: every step overwrites its outputs.
set -euo pipefail
cd "$(dirname "$0")/.."

step() { echo; echo "=== $(date '+%a %H:%M:%S')  $*"; }

step "checks: Gate C tag, no training queue, tests"
git rev-parse -q --verify refs/tags/models-frozen >/dev/null || { echo "tag models-frozen missing (Gate C)"; exit 1; }
uv run pytest -q

step "Tier 0 on validation again (predictions with the FP32 regression head)"
uv run python scripts/evaluate.py --split val --tier0

step "TEST split: every finished run and Tier 0, M1 and M2"
uv run python scripts/evaluate.py --split test --confirm-test --all

step "cost benchmark on this GPU"
uv run python scripts/benchmark_cost.py --all

step "explainability on the 4 priority configs"
uv run python scripts/run_xai.py --configs fs8 fw8 fs32 fs64 --n-videos 100 --device cuda

step "aggregation and figures"
uv run python scripts/aggregate_results.py
uv run python scripts/make_figures.py

step "final run finished"
