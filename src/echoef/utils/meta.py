"""Provenance for run folders and preprocessing outputs: git commit and library versions."""

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

from echoef.config import REPO_ROOT

PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "opencv-python-headless",
    "omegaconf",
    "torch",
    "torchvision",
    "captum",
    "scikit-learn",
    "scipy",
)


def _git(args: list[str], repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_commit(repo: Path = REPO_ROOT) -> str:
    """Full hash of HEAD, or ``"unknown"`` outside a git checkout."""
    return _git(["rev-parse", "HEAD"], repo) or "unknown"


def git_dirty(repo: Path = REPO_ROOT) -> bool | None:
    """True if tracked files have uncommitted changes; None outside a git checkout."""
    out = _git(["status", "--porcelain", "--untracked-files=no"], repo)
    return None if out is None else bool(out)


def library_versions(packages: tuple[str, ...] = PACKAGES) -> dict[str, str]:
    """Python version plus the installed version of each package (``"not installed"`` if absent)."""
    versions = {"python": platform.python_version()}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def git_tag_exists(tag: str, repo: Path = REPO_ROOT) -> bool:
    """True if ``tag`` names a tag in the repository (e.g. ``models-frozen``, Gate C)."""
    return _git(["rev-parse", "-q", "--verify", f"refs/tags/{tag}"], repo) is not None


def config_hash(config: dict) -> str:
    """Short SHA-256 of a resolved config, independent of key order."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_meta(
    config: dict, seed: int, gpu: str, started: str, finished: str, **extra: object
) -> dict:
    """``meta.json`` of a run: git commit, config hash, library versions, GPU, seed, times."""
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_hash": config_hash(config),
        "versions": library_versions(),
        "gpu_name": gpu,
        "seed": seed,
        "started": started,
        "finished": finished,
        **extra,
    }


def git_ref_commit(ref: str, repo: Path = REPO_ROOT) -> str | None:
    """Commit hash a ref (tag, branch, ``HEAD``) points to, or None if it does not exist."""
    return _git(["rev-parse", "-q", "--verify", f"{ref}^{{commit}}"], repo)


def git_changed_since(ref: str, paths: tuple[str, ...], repo: Path = REPO_ROOT) -> list[str]:
    """Files under ``paths`` whose committed content differs between ``ref`` and HEAD."""
    out = _git(["diff", "--name-only", ref, "HEAD", "--", *paths], repo)
    return [line for line in (out or "").splitlines() if line]


def git_changed_between(
    ref_a: str, ref_b: str, paths: tuple[str, ...], repo: Path = REPO_ROOT
) -> list[str] | None:
    """Files under ``paths`` that differ between two commits; None if either is unknown here."""
    if git_ref_commit(ref_a, repo) is None or git_ref_commit(ref_b, repo) is None:
        return None
    out = _git(["diff", "--name-only", ref_a, ref_b, "--", *paths], repo)
    return [line for line in (out or "").splitlines() if line]
