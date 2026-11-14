"""The shared contracts match the spec's configuration table and the experiment YAML files."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from echoef import contracts

REPO = Path(__file__).resolve().parents[1]

# Window W = T x period in raw frames, from the spec's configuration table.
SPEC_WINDOWS = {"fs8": 16, "fs16": 32, "fs32": 64, "fs64": 128, "fw8": 64, "fw16": 64, "fw64": 64}


def test_windows_match_spec_table():
    windows = {cid: t * p for cid, (_regime, t, p) in contracts.CONFIGS.items()}
    assert windows == SPEC_WINDOWS


def test_fixed_stride_configs_use_period_2():
    for regime, _t, period in contracts.CONFIGS.values():
        if regime in ("FS", "BOTH"):
            assert period == 2


@pytest.mark.parametrize(
    "columns",
    [
        contracts.INDEX_COLUMNS,
        contracts.TRACINGS_COLUMNS,
        contracts.PREDICTION_COLUMNS,
        contracts.COST_COLUMNS,
        contracts.ALL_RUNS_COLUMNS,
        contracts.XAI_COLUMNS,
    ],
)
def test_column_lists_have_no_duplicates(columns):
    assert len(columns) == len(set(columns))


def test_run_dir_and_predictions_path(tmp_path):
    run = contracts.run_dir(tmp_path, "fs8", 0)
    assert run == tmp_path / "fs8" / "seed0"
    assert contracts.predictions_path(run, "VAL", "M1") == run / "predictions_val_M1.csv"


@pytest.mark.parametrize("config_id", sorted(contracts.CONFIGS))
def test_experiment_yaml_matches_contracts(config_id):
    cfg = OmegaConf.load(REPO / "configs" / "experiments" / f"{config_id}.yaml")
    assert (cfg.config_id, cfg.regime, cfg.T, cfg.period) == (
        config_id,
        *contracts.CONFIGS[config_id],
    )
