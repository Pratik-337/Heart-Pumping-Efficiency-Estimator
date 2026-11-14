"""Loading machine paths and merging experiment configs."""

from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError

from echoef import config
from echoef.contracts import CONFIGS

REPO = Path(__file__).resolve().parents[1]


def _write_paths(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "paths.yaml"
    path.write_text(text)
    return path


def test_example_paths_file_is_complete():
    roots = config.load_paths(REPO / "configs" / "paths.example.yaml")
    assert set(roots) == set(config.PATH_KEYS)
    assert roots["runs_root"] == REPO / "runs"
    assert roots["raw_root"] == Path.home() / "data/echonet/raw/EchoNet-Dynamic"


def test_paths_absolute_relative_and_home(tmp_path):
    path = _write_paths(
        tmp_path,
        "raw_root: /abs/raw\nprocessed_root: ~/proc\nruns_root: runs\n"
        "results_root: results\nweights_root: ~/w\n",
    )
    roots = config.load_paths(path)
    assert roots["raw_root"] == Path("/abs/raw")
    assert roots["processed_root"] == Path.home() / "proc"
    assert roots["runs_root"] == config.REPO_ROOT / "runs"


def test_missing_paths_file_names_the_template(tmp_path):
    with pytest.raises(FileNotFoundError, match="paths.example.yaml"):
        config.load_paths(tmp_path / "paths.yaml")


def test_paths_missing_or_unknown_key_raises(tmp_path):
    path = _write_paths(tmp_path, "raw_root: /r\nproc_root: /p\n")
    with pytest.raises(ValueError, match="missing keys"):
        config.load_paths(path)


@pytest.mark.parametrize("config_id", sorted(CONFIGS))
def test_every_experiment_merges_with_base(config_id):
    cfg = config.load_config(config_id)
    _regime, t, period = CONFIGS[config_id]
    assert (cfg.T, cfg.period) == (t, period)
    assert cfg.model.init == "kinetics400"
    assert cfg.train.micro_batch * cfg.train.grad_accum == 16
    assert cfg.train.epochs == 10


def test_b0_uses_echonet_weights():
    cfg = config.load_config("b0_echonet")
    assert cfg.model.init == "echonet"
    assert cfg.model.weights_file == "r2plus1d_18_32_2_pretrained.pt"


def test_override_existing_key():
    cfg = config.load_config("fs8", ["seed=2", "train.lr=3e-4"])
    assert cfg.seed == 2
    assert cfg.train.lr == pytest.approx(3e-4)


def test_typo_in_override_raises():
    with pytest.raises(ConfigKeyError):
        config.load_config("fs8", ["train.lrr=3e-4"])


def test_override_that_breaks_contract_raises():
    with pytest.raises(ValueError, match="contracts.CONFIGS"):
        config.load_config("fs64", ["T=16"])


def test_unknown_experiment_raises():
    with pytest.raises(FileNotFoundError, match="known"):
        config.load_config("fs12")


def test_unset_required_key_raises(tmp_path):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "base.yaml").write_text((REPO / "configs" / "base.yaml").read_text())
    (tmp_path / "experiments" / "half.yaml").write_text("config_id: fs8\nregime: FS\nT: 8\n")
    with pytest.raises(ValueError, match="period"):
        config.load_config("half", configs_dir=tmp_path)
