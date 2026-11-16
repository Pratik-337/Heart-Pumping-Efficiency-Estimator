"""Training queue: skipping, retries, the frozen-commit check and the GPU lock (no GPU)."""

import json
import os
import subprocess

import pytest

from echoef.train import sweep
from echoef.utils import gpu


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                   cwd=repo, check=True, capture_output=True)  # fmt: skip


def test_queue_skips_done_retries_and_marks_failures(tmp_path):
    (tmp_path / "fs8" / "seed0").mkdir(parents=True)
    (tmp_path / "fs8" / "seed0" / "DONE").touch()
    # fs32 fails once then succeeds on the resume; fw8 always fails; fs16 succeeds at once.
    succeeds_on_try = {"fs32": 2, "fw8": None, "fs16": 1}
    calls = []

    def train_one(config_id, seed):
        calls.append((config_id, seed))
        run = tmp_path / config_id / f"seed{seed}"
        run.mkdir(parents=True, exist_ok=True)
        if calls.count((config_id, seed)) == succeeds_on_try[config_id]:
            (run / "DONE").touch()
            return 0
        return 1

    queue = [("fs8", 0), ("fs32", 0), ("fw8", 0), ("fs16", 0)]
    summary = sweep.run_queue(lambda: queue, tmp_path, train_one)
    assert calls == [("fs32", 0), ("fs32", 0), ("fw8", 0), ("fw8", 0), ("fs16", 0)]
    assert summary == {"done": [("fs32", 0), ("fs16", 0)], "skipped": [("fs8", 0)],
                       "failed": [("fw8", 0)]}  # fmt: skip
    assert (tmp_path / "fw8" / "seed0" / sweep.FAILED_FILE).exists()


def test_queue_is_reread_before_every_run(tmp_path):
    queue = [("fs8", 0), ("fs16", 0), ("fs32", 0)]
    calls = []

    def train_one(config_id, seed):
        calls.append(config_id)
        if config_id == "fs8":
            queue.remove(("fs16", 0))  # fs16 moved to another laptop meanwhile
        (tmp_path / config_id / f"seed{seed}").mkdir(parents=True)
        (tmp_path / config_id / f"seed{seed}" / "DONE").touch()
        return 0

    sweep.run_queue(lambda: list(queue), tmp_path, train_one)
    assert calls == ["fs8", "fs32"]


def test_frozen_recipe_check(tmp_path):
    _git(tmp_path, "init", "-q")
    for name in ("src/a.py", "configs/assignment.yaml"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("1")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "one")
    with pytest.raises(RuntimeError, match="does not exist"):
        sweep.require_frozen_commit(tmp_path)
    _git(tmp_path, "tag", "config-frozen")
    assert len(sweep.require_frozen_commit(tmp_path)) == 40
    (tmp_path / "configs/assignment.yaml").write_text("2")  # the queue may change ...
    with pytest.raises(RuntimeError, match="uncommitted"):
        sweep.require_frozen_commit(tmp_path)
    _git(tmp_path, "commit", "-q", "-am", "move runs")
    assert len(sweep.require_frozen_commit(tmp_path)) == 40  # ... once committed
    (tmp_path / "src/a.py").write_text("2")  # the recipe may not
    _git(tmp_path, "commit", "-q", "-am", "change code")
    with pytest.raises(RuntimeError, match="src/a.py"):
        sweep.require_frozen_commit(tmp_path)


def test_gpu_lock_blocks_a_second_job_and_ignores_stale_locks(tmp_path):
    gpu.require_gpu_free(tmp_path)
    with gpu.queue_lock(tmp_path, "rtx4060"):
        assert gpu.queue_holder(tmp_path)["pid"] == os.getpid()
        with pytest.raises(RuntimeError, match="training queue is running"):
            gpu.require_gpu_free(tmp_path)
    assert not (tmp_path / gpu.QUEUE_LOCK).exists()
    dead = subprocess.Popen(["true"])
    dead.wait()
    (tmp_path / gpu.QUEUE_LOCK).write_text(json.dumps({"pid": dead.pid, "started": "x"}))
    gpu.require_gpu_free(tmp_path)  # stale lock: its process is gone


def test_check_runs_for_gate_c(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("1")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "one")
    _git(repo, "tag", "config-frozen")
    frozen = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                            text=True, check=True).stdout.strip()  # fmt: skip
    (repo / "src" / "a.py").write_text("2")
    _git(repo, "commit", "-q", "-am", "change the recipe")
    changed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                             text=True, check=True).stdout.strip()  # fmt: skip
    runs = tmp_path / "runs"

    def make(path, commit, epochs=3, done=True, preds=("M1", "M2")):
        path.mkdir(parents=True)
        (path / "meta.json").write_text(json.dumps({"git_commit": commit, "gpu_name": "G"}))
        (path / "log.csv").write_text("epoch\n" + "".join(f"{e}\n" for e in range(epochs)))
        (path / "best.pt").touch()
        for mode in preds:
            (path / f"predictions_val_{mode}.csv").touch()
        if done:
            (path / "DONE").touch()

    make(runs / "fs8" / "seed0", frozen)
    make(runs / "fs8" / "seed1", frozen, epochs=2, done=False)  # still training
    make(runs / "fs16" / "seed0", changed)  # trained with another recipe
    make(runs / "fw8" / "seed0", frozen, preds=("M1",))
    make(runs / "_hwcontrol_rtx4050" / "fs8" / "seed0", frozen)
    assignment = {"a": [("fs8", 0), ("fs8", 1)], "b": [("fs16", 0), ("fw8", 0), ("fs64", 0)]}
    table = sweep.check_runs(runs, assignment, 3, repo).set_index(["config_id", "seed"])
    problems = table["problems"]
    assert problems[("fs8", 0)].tolist() == ["ok", "ok"]  # the queued run and the control
    assert "not DONE" in problems[("fs8", 1)] and "2 of 3 epochs" in problems[("fs8", 1)]
    assert "recipe differs" in problems[("fs16", 0)] and "src/a.py" in problems[("fs16", 0)]
    assert problems[("fw8", 0)] == "no VAL M2 predictions"
    assert "not DONE" in problems[("fs64", 0)] and "no best.pt" in problems[("fs64", 0)]
    assert sorted(table["machine"].unique()) == ["a", "b", "rtx4050 (control)"]
