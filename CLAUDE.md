# CLAUDE.md — Heart Pumping Efficiency Estimator

## Source of truth
- Read docs/PROJECT_SPEC.md before any task. It wins over code comments and over your own assumptions.
- Decisions live in docs/decisions.md. Do not change a decided item without a new entry approved by the user.
- Work on one task ID (e.g. A3 or B6) per session. Put the task ID in every commit message.
- Since D-30 (16 Nov) Pratik works alone and does every task, A1–A12 and B1–B10; the ownership table in the spec now only maps modules to task IDs.
- contracts.py, pyproject.toml, uv.lock and configs/base.yaml change only with the user's explicit agreement and a decision entry.
- Unit tests build their own synthetic data and run without a GPU; B's code may also run on the real data on this laptop (D-30).

## Hard rules
- Never modify, move or delete anything under data/raw/.
- Never commit data/, runs/, results/xai/, *.npy, *.pt, *.avi, configs/paths.yaml, any video file, or any figure that shows echo frames.
- Never evaluate on the TEST split except via `scripts/evaluate.py --split test --confirm-test`, and only after the tag `models-frozen` exists.
- Never type a result number into the report, README or code. Numbers come from results/all_runs.csv via scripts.
- Never change configs/base.yaml after the tag `config-frozen` without a decision entry.
- Never shrink batch size, epochs or data silently to make something fit. On OOM or slowness, stop and report.

## How code is written
- Library code in src/echoef/; scripts/ are thin CLI wrappers (argparse or OmegaConf CLI).
- No hard-coded hyperparameters or paths: everything comes from configs/*.yaml.
- Every run writes runs/<config>/seed<k>/{config.yaml, meta.json, log.csv, metrics.json, predictions_*.csv, best.pt, last.pt, tb/, DONE}.
- meta.json holds git commit, config hash, library versions, GPU name, seed, start/end time.
- Type hints on public functions; docstrings state tensor shapes, e.g. (B, C, T, H, W).
- pathlib for paths; logging, not print; no notebooks as a source of results.
- Determinism: call utils.seed_everything(seed) at the start of every script.

## Before you say a task is done
- `uv run ruff check . && uv run ruff format --check . && uv run pytest -q` all pass.
- New logic has a unit test that runs on synthetic data without a GPU.
- The task's "Done when" line in the spec is met; say how you checked it.

## Ask the user first
- Adding a dependency, changing the folder layout, changing a metric definition,
  or anything that affects which numbers end up in the report.

## Environment
- WSL2 Ubuntu, Python 3.12, uv. Data root and run root come from configs/paths.yaml.
- GPU (A's laptop only): RTX 4060 Laptop GPU, 8 GB. Use torch.autocast(dtype=torch.bfloat16) for training and inference.
- Never start a GPU job while the training queue runs (until about Mon 20:40); smoke-test with --device cpu.
