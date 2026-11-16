"""Resumable training queue of one laptop (task A6; ``scripts/run_sweep.py``).

The queue is the laptop's list in ``configs/assignment.yaml`` (D-29), read again before every
run, so runs can be moved between laptops without restarting the sweep. Runs whose folder holds
``DONE`` are skipped, so the sweep can be restarted at any time; an unfinished run resumes from
its ``last.pt``. A run that fails is tried once more (resuming), then marked ``FAILED`` and the
queue moves on, so one bad run does not idle the GPU overnight.

Every laptop trains with the recipe frozen at the tag ``config-frozen``: the files that decide
how a model is trained must be unchanged since that tag (``FROZEN_PATHS``). The queue file may
change; each run's ``meta.json`` records the exact commit.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from echoef.config import REPO_ROOT
from echoef.contracts import MODES, predictions_path, run_dir
from echoef.train.trainer import DONE_FILE
from echoef.utils.meta import (
    git_changed_between,
    git_changed_since,
    git_dirty,
    git_ref_commit,
)

log = logging.getLogger(__name__)

CONFIG_FROZEN_TAG = "config-frozen"
FAILED_FILE = "FAILED"
# What decides how a model is trained; must match the config-frozen tag on every laptop.
FROZEN_PATHS = (
    "src",
    "configs/base.yaml",
    "configs/experiments",
    "scripts/train.py",
    "pyproject.toml",
    "uv.lock",
)


def require_frozen_commit(repo: Path = REPO_ROOT) -> str:
    """Refuse to train unless the training recipe is the one tagged ``config-frozen``.

    Returns:
        The HEAD commit hash.

    Raises:
        RuntimeError: no tag, a file in ``FROZEN_PATHS`` differs from the tag, or tracked files
            have uncommitted changes.
    """
    if git_ref_commit(CONFIG_FROZEN_TAG, repo) is None:
        raise RuntimeError(f"tag {CONFIG_FROZEN_TAG!r} does not exist; tag the frozen commit")
    if git_dirty(repo):
        raise RuntimeError("tracked files have uncommitted changes")
    changed = git_changed_since(CONFIG_FROZEN_TAG, FROZEN_PATHS, repo)
    if changed:
        raise RuntimeError(f"changed since {CONFIG_FROZEN_TAG}: {changed}; needs a decision entry")
    return git_ref_commit("HEAD", repo)


def run_queue(
    get_queue: Callable[[], list[tuple[str, int]]],
    runs_root: Path,
    train_one: Callable[[str, int], int],
    attempts: int = 2,
) -> dict[str, list[tuple[str, int]]]:
    """Train, in queue order, every run that is not DONE yet, re-reading the queue each time.

    Args:
        get_queue: returns this laptop's current ``[(config_id, seed), ...]``.
        train_one: ``(config_id, seed) -> exit code``; 0 means the run finished.
        attempts: tries per run (later tries resume from ``last.pt``).

    Returns:
        ``{"done": [...], "skipped": [...], "failed": [...]}`` lists of (config_id, seed);
        ``skipped`` = already DONE when the queue reached them.
    """
    summary: dict[str, list[tuple[str, int]]] = {"done": [], "skipped": [], "failed": []}
    handled: set[tuple[str, int]] = set()
    while True:
        pending = [item for item in get_queue() if item not in handled]
        if not pending:
            break
        config_id, seed = pending[0]
        handled.add((config_id, seed))
        run = run_dir(Path(runs_root), config_id, seed)
        if (run / DONE_FILE).exists():
            log.info("skip %s seed %d: DONE", config_id, seed)
            summary["skipped"].append((config_id, seed))
            continue
        for attempt in range(1, attempts + 1):
            log.info("start %s seed %d (attempt %d of %d)", config_id, seed, attempt, attempts)
            code = train_one(config_id, seed)
            if code == 0 and (run / DONE_FILE).exists():
                (run / FAILED_FILE).unlink(missing_ok=True)
                summary["done"].append((config_id, seed))
                break
            log.error("%s seed %d stopped with exit code %d", config_id, seed, code)
        else:
            run.mkdir(parents=True, exist_ok=True)
            (run / FAILED_FILE).write_text(f"exit code {code} after {attempts} attempts\n")
            summary["failed"].append((config_id, seed))
            log.error("%s seed %d FAILED; moving on (see %s/train.log)", config_id, seed, run)
    log.info("queue finished: %s", {k: len(v) for k, v in summary.items()})
    return summary


HWCONTROL_GLOB = "_hwcontrol_*"  # runs/_hwcontrol_<machine>/<config>/seed<k>/ (D-31)


def run_status(run: Path, epochs: int, repo: Path = REPO_ROOT) -> dict:
    """What Gate C needs to know about one run folder; ``problems`` is "ok" when nothing is
    missing: DONE, every epoch logged, ``best.pt``, validation predictions in M1 and M2, and a
    training recipe (``FROZEN_PATHS`` at the commit in ``meta.json``) equal to ``config-frozen``.
    """
    run = Path(run)
    try:
        meta = json.loads((run / "meta.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        meta = {}
    try:
        logged = len(pd.read_csv(run / "log.csv"))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logged = 0
    problems = []
    if not (run / DONE_FILE).exists():
        problems.append("not DONE")
    if logged < epochs:
        problems.append(f"{logged} of {epochs} epochs logged")
    if not (run / "best.pt").exists():
        problems.append("no best.pt")
    problems += [f"no VAL {m} predictions" for m in MODES
                 if not predictions_path(run, "VAL", m).exists()]  # fmt: skip
    commit = str(meta.get("git_commit", ""))
    changed = git_changed_between(CONFIG_FROZEN_TAG, commit, FROZEN_PATHS, repo) if commit else None
    if changed is None:
        problems.append(f"commit {commit[:7] or '?'} unknown here")
    elif changed:
        problems.append(f"recipe differs from {CONFIG_FROZEN_TAG}: {changed}")
    return {"epochs": logged, "gpu_name": meta.get("gpu_name", ""), "commit": commit[:7],
            "problems": "; ".join(problems) or "ok"}  # fmt: skip


def check_runs(
    runs_root: Path,
    assignment: dict[str, list[tuple[str, int]]],
    epochs: int,
    repo: Path = REPO_ROOT,
) -> pd.DataFrame:
    """Gate C: the status of every run in ``assignment.yaml`` and every hardware-control run.

    Returns:
        One row per run: machine, config_id, seed, epochs, gpu_name, commit, problems.
    """
    rows = []
    for machine, queue in assignment.items():
        for config_id, seed in queue:
            status = run_status(run_dir(runs_root, config_id, seed), epochs, repo)
            rows.append({"machine": machine, "config_id": config_id, "seed": seed} | status)
    for run in sorted(Path(runs_root).glob(f"{HWCONTROL_GLOB}/*/seed*")):
        machine = run.parents[1].name.removeprefix("_hwcontrol_") + " (control)"
        status = run_status(run, epochs, repo)
        rows.append({"machine": machine, "config_id": run.parent.name,
                     "seed": int(run.name.removeprefix("seed"))} | status)  # fmt: skip
    return pd.DataFrame(rows, columns=["machine", "config_id", "seed", "epochs", "gpu_name",
                                       "commit", "problems"])  # fmt: skip
