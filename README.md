# Heart Pumping Efficiency Estimator

Predicts left-ventricular ejection fraction (EF) from EchoNet-Dynamic echocardiogram videos with a fine-tuned R(2+1)D-18, and measures how many frames the model needs: test accuracy against GFLOPs, latency and VRAM for clip lengths 8–64 under two sampling regimes, plus explanations scored with numbers.

**Research prototype. Not clinically validated. Not for diagnosis.**

- Plan, interface contracts and task list: [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) (the source of truth)
- Decision log: [docs/decisions.md](docs/decisions.md)
- Report source (IEEEtran): [docs/report/](docs/report/)
- Rules for Claude Code: [CLAUDE.md](CLAUDE.md)

No result is quoted here: every number lives in `results/all_runs.csv` and the tables generated from it.

**Results:** [results/tables/metrics_summary.md](results/tables/metrics_summary.md) gives MAE, RMSE, MSE, R², bias, limits of agreement and AUROC for every model on the test and validation splits; `results/all_runs.csv` has every metric per seed with the cost columns, `results/cost.csv` the cost benchmark and `results/figures/` the figures.

## Status (17 Nov 2026)

All 14 training runs (7 configurations × 2 seeds) and the RTX 4050 hardware-control run are finished and frozen (tag `models-frozen`). The final run evaluated every model and Tier 0 on the test split once, ran the cost benchmark and the explainability tests, and generated `results/all_runs.csv`, the tables, the figures and the report's numbers; the demo passed Gate D. The report in `docs/report/` is complete in draft and compiles on Overleaf from `results/xai/report.zip`.

## Setup (WSL2 Ubuntu or Linux, NVIDIA GPU for training)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # once per machine
git clone https://github.com/Pratik-337/Heart-Pumping-Efficiency-Estimator.git
cd Heart-Pumping-Efficiency-Estimator
uv sync                                                 # Python 3.12 + the locked environment
cp configs/paths.example.yaml configs/paths.yaml        # then edit the roots for this machine
uv run pre-commit install                               # ruff + data-file guard on every commit
uv run pytest -q                                        # CPU only, synthetic data
```

`configs/paths.yaml` is git-ignored and holds the five roots every script reads: `raw_root`, `processed_root`, `runs_root`, `results_root`, `weights_root`. The locked PyTorch wheel has CUDA on Linux only; on native Windows it installs a CPU build, so use WSL2.

## Data

EchoNet-Dynamic is licensed for individual, non-commercial research and may not be redistributed. Register on the Stanford AIMI portal, download the dataset and unpack it so that `raw_root` holds `FileList.csv`, `VolumeTracings.csv` and `Videos/`. For the baseline B0, put EchoNet's released `r2plus1d_18_32_2_pretrained.pt` (echonet/dynamic, release v1.0.0) in `weights_root`.

Videos, frame caches, per-video predictions, checkpoints and figures that show echo frames are never committed; `.gitignore` and a pre-commit hook block them.

## Reproducing the study

Each step is one script; all read `configs/paths.yaml` and log to their output folder.

| Step | Command | Writes |
| --- | --- | --- |
| A2 Verify and cache the data | `uv run python scripts/verify_dataset.py`<br>`uv run python scripts/preprocess.py`<br>`uv run python scripts/eda.py` | `processed_root/` (frame cache, index, tracings, train-only stats, QC report); `results/eda/*.csv` |
| A3 Look at batches | `uv run python scripts/batch_grids.py` | `processed_root/inspect/batch_grids/` (echo frames: local only) |
| A4 Memory and speed per T | `uv run python scripts/probe_memory.py --machine rtx4060` | `results/probe/` |
| A5 Gate A and Tier 0 | `uv run python scripts/evaluate.py --split val --tier0` | `runs/b0-<config>/seed0/predictions_val_*.csv` |
| A6 Train | `git switch queue-d31` (or the tag `config-frozen`), then `tmux new -s sweep 'uv run python scripts/run_sweep.py --machine rtx4060'` | `runs/<config>/seed<k>/` |
| Freeze models (Gate C) | `uv run python scripts/check_runs.py` must report every run ok; then `git tag models-frozen` | — |
| A7–A10 Final run | `bash scripts/final_run.sh` | test predictions, `results/cost.csv`, `results/xai/`, `results/all_runs.csv`, tables, figures, report numbers, Gate D, `results/xai/report.zip` |
| A11 Demo | `uv run python app/app.py [--runs runs/<config>/seed0 runs/<config>/seed1] [--examples 5]` | local Gradio page on `127.0.0.1`; default model: the D-13 choice in `results/tables/demo_choice.csv` |

Training follows `configs/assignment.yaml` (which laptop trains which `[config, seed]`); the sweep refuses to start unless the training recipe matches the tag `config-frozen`, and it resumes interrupted runs from their last epoch. The test split can only be evaluated with `--confirm-test` after the tag `models-frozen` exists.

### After the training queue (Gate C and the final run)

On the RTX 4060, once `tmux attach -t sweep` shows the queue finished (and every RTX 4050 run is unpacked into `runs/`):

```bash
cd ~/projects/Heart-Pumping-Efficiency-Estimator          # the tree the queue ran from
git switch main && git merge --ff-only a-dev             # a-dev is shared with the -dev worktree
uv sync && uv run python scripts/check_runs.py           # every run ok, else stop
git tag models-frozen && git push origin main --tags     # Gate C
tmux new -s final 'bash scripts/final_run.sh 2>&1 | tee runs/final_run.log'
```

`final_run.sh` stops at the first failing step (rerun it after a fix; every step overwrites its outputs), except the last one: `check_report.py` lists any report sentence whose `% CLAIM:` the test numbers do not support, so that sentence gets rewritten. Then commit `results/` (never `results/xai/`) as M3.

### Tables and figures from saved predictions (no GPU)

```bash
uv run python scripts/aggregate_results.py   # results/all_runs.csv, results/tables/*.csv
uv run python scripts/make_figures.py        # results/figures/*.pdf (3.5 in, IEEE column)
uv run python scripts/make_tables.py         # results/numbers.tex, dataset.tex, tables/*.tex for the report
uv run python scripts/check_report.py        # every number, table and figure of docs/report/main.tex exists; claims hold
uv run python scripts/bundle_report.py       # results/xai/report.zip: main.tex + refs.bib + results/ for Overleaf
```

The report has no LaTeX build here: upload `results/xai/report.zip` to Overleaf (New project, Upload project). Add `--with-xai-examples` for the example panel with echo frames, only if your guide allows frames in the report.

## Checks before every push

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

## Layout

| Path | What lives there |
| --- | --- |
| `src/echoef/data/` | Preprocessing, tracings and LV masks, clip sampling, dataset, transforms |
| `src/echoef/models/` | R(2+1)D-18 EF regressor, B0 loader, checkpoints |
| `src/echoef/train/` | Training step and run, memory probe, resumable sweep |
| `src/echoef/eval/` | Prediction, metrics, statistics, cost, aggregation, figures, report tables |
| `src/echoef/xai/` | Occlusion, Integrated Gradients, 3D Grad-CAM, their numeric tests, figures |
| `scripts/` | Thin command-line wrappers around `src/echoef/` |
| `app/` | Local Gradio demo and the Gate D check |
| `configs/` | `base.yaml`, `experiments/<config_id>.yaml`, `assignment.yaml`, `paths.example.yaml` |
| `tests/` | pytest, CPU only, synthetic data |
| `results/` | Aggregated CSVs, tables and figures (committed); `xai/` (git-ignored) |
| `runs/` | One folder per run (git-ignored) |
| `docs/` | Spec, decision log, report source |
