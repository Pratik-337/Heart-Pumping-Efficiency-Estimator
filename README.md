# Heart Pumping Efficiency Estimator

Predicts left-ventricular ejection fraction (EF) from EchoNet-Dynamic echocardiogram videos with a fine-tuned R(2+1)D-18, and measures how many frames the model needs: test accuracy against GFLOPs, latency and VRAM for clip lengths 8–64, plus explanations scored with numbers.

**Research prototype. Not clinically validated. Not for diagnosis.**

- Plan, interface contracts and task list: [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) (the source of truth)
- Decision log: [docs/decisions.md](docs/decisions.md)
- Rules for Claude Code: [CLAUDE.md](CLAUDE.md)

## Setup (WSL2 Ubuntu, Linux or Apple-silicon macOS)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # once per machine
git clone https://github.com/Pratik-337/Heart-Pumping-Efficiency-Estimator.git
cd Heart-Pumping-Efficiency-Estimator
uv sync                                                 # Python 3.12 + the locked environment
cp configs/paths.example.yaml configs/paths.yaml        # then edit the roots for this machine
uv run pre-commit install                               # ruff + data-file guard on every commit
uv run pytest -q
```

`configs/paths.yaml` is git-ignored and holds the five roots every script reads (`raw_root`, `processed_root`, `runs_root`, `results_root`, `weights_root`). On a machine without the dataset, point them at the synthetic data folder (see the comments in `paths.example.yaml`).

## Data

EchoNet-Dynamic is licensed for individual, non-commercial research and may not be redistributed. Videos, frame caches, per-video predictions, checkpoints and figures that show echo frames are never committed; `.gitignore` and a pre-commit hook block them. Anyone who needs the real data registers on the Stanford AIMI portal personally. Development without the data uses synthetic videos (`scripts/make_synthetic_data.py`, task B1).

## Checks before every push

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

## Layout

| Path | What lives there |
| --- | --- |
| `src/echoef/` | Library code: data pipeline, model, training, evaluation, explainability |
| `scripts/` | Thin command-line wrappers around `src/echoef/` |
| `configs/` | `base.yaml` (shared settings), `experiments/<config_id>.yaml`, `paths.example.yaml` |
| `tests/` | pytest, CPU only, synthetic data |
| `results/` | `all_runs.csv`, `cost.csv`, tables and figures (committed); `xai/` (git-ignored) |
| `runs/` | One folder per training run (git-ignored) |

The full tree and file ownership are in the spec, sections "Repository folder structure" and "Who owns what".
