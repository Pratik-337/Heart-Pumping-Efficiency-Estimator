"""Load machine paths (configs/paths.yaml) and experiment configs.

An experiment config is ``configs/base.yaml`` merged with ``configs/experiments/<name>.yaml`` and
then with command-line overrides such as ``train.lr=3e-4``. ``base.yaml`` is loaded in struct
mode, so a key that does not exist there (for example a typo) raises instead of being ignored.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from echoef.contracts import CONFIGS

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
PATH_KEYS = ("raw_root", "processed_root", "runs_root", "results_root", "weights_root")
B0_CONFIG_ID = "b0"  # EchoNet's released weights; evaluation only (Gate A, Tier 0)


def load_paths(path: Path | None = None) -> dict[str, Path]:
    """Read the five machine roots from ``configs/paths.yaml``.

    ``~`` is expanded and relative paths are resolved from the repo root.

    Raises:
        FileNotFoundError: the file does not exist yet.
        ValueError: a root is missing or empty, or an unknown key is present.
    """
    path = CONFIGS_DIR / "paths.yaml" if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy configs/paths.example.yaml to configs/paths.yaml "
            "and set the roots for this machine."
        )
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    missing = [key for key in PATH_KEYS if not raw.get(key)]
    unknown = sorted(set(raw) - set(PATH_KEYS))
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {missing}, unknown keys {unknown}")

    roots = {}
    for key in PATH_KEYS:
        root = Path(str(raw[key])).expanduser()
        roots[key] = root if root.is_absolute() else REPO_ROOT / root
    return roots


def load_config(
    experiment: str,
    overrides: list[str] | None = None,
    configs_dir: Path = CONFIGS_DIR,
) -> DictConfig:
    """Merge ``base.yaml``, ``experiments/<experiment>.yaml`` and dot-list overrides.

    Args:
        experiment: experiment file name without ``.yaml``, e.g. ``"fs32"`` or ``"b0_echonet"``.
        overrides: dot-list strings such as ``["seed=1", "train.lr=3e-4"]``.
        configs_dir: folder holding ``base.yaml`` and ``experiments/``.

    Raises:
        FileNotFoundError: no such experiment file.
        omegaconf.errors.ConfigKeyError: an experiment or override sets a key not in base.yaml.
        ValueError: a required key is unset, or (regime, T, period) disagrees with CONFIGS.
    """
    exp_path = configs_dir / "experiments" / f"{experiment}.yaml"
    if not exp_path.exists():
        known = sorted(p.stem for p in (configs_dir / "experiments").glob("*.yaml"))
        raise FileNotFoundError(f"No experiment config {exp_path}; known: {known}")

    base = OmegaConf.load(configs_dir / "base.yaml")
    OmegaConf.set_struct(base, True)
    cfg = OmegaConf.merge(base, OmegaConf.load(exp_path), OmegaConf.from_dotlist(overrides or []))

    unset = sorted(OmegaConf.missing_keys(cfg))
    if unset:
        raise ValueError(f"Config {experiment!r} leaves required keys unset: {unset}")
    _check_against_contracts(cfg)
    return cfg


def load_assignment(path: Path | None = None) -> dict[str, list[tuple[str, int]]]:
    """Queue of ``(config_id, seed)`` runs per laptop from ``configs/assignment.yaml`` (D-29).

    Raises:
        ValueError: an unknown config_id, a seed that is not a non-negative int, a malformed
            entry, or a run listed twice (on one laptop or across laptops).
    """
    path = CONFIGS_DIR / "assignment.yaml" if path is None else Path(path)
    raw = OmegaConf.to_container(OmegaConf.load(path)) or {}
    queues: dict[str, list[tuple[str, int]]] = {}
    seen: set[tuple[str, int]] = set()
    for machine, runs in raw.items():
        queue = []
        for entry in runs or []:
            if not (isinstance(entry, list) and len(entry) == 2):
                raise ValueError(f"{path}: {machine}: entry {entry!r} is not [config_id, seed]")
            config_id, seed = entry
            if config_id not in CONFIGS:
                raise ValueError(f"{path}: {machine}: unknown config_id {config_id!r}")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ValueError(f"{path}: {machine}: seed {seed!r} is not an int >= 0")
            if (config_id, seed) in seen:
                raise ValueError(f"{path}: run {config_id} seed {seed} is listed twice")
            seen.add((config_id, seed))
            queue.append((config_id, seed))
        queues[str(machine)] = queue
    return queues


def _check_against_contracts(cfg: DictConfig) -> None:
    """A sweep config must match its (regime, T, period) row in contracts.CONFIGS."""
    config_id = cfg.config_id
    if config_id == B0_CONFIG_ID:
        return  # Tier 0 evaluates B0 at every setting, so T and period may be overridden
    if config_id not in CONFIGS:
        raise ValueError(f"Unknown config_id {config_id!r}; expected one of {sorted(CONFIGS)}")
    expected = CONFIGS[config_id]
    got = (cfg.regime, cfg.T, cfg.period)
    if got != expected:
        raise ValueError(
            f"{config_id}: (regime, T, period) = {got} but contracts.CONFIGS says {expected}"
        )
