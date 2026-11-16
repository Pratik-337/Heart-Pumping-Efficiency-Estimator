"""Provenance for run folders and preprocessing outputs: git commit and library versions."""

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
