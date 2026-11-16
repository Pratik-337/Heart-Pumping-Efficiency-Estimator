"""Provenance helpers used in preprocess_meta.json and run meta.json."""

import re

from echoef.utils import git_commit, git_dirty, library_versions


def test_git_commit_is_a_hash_or_unknown():
    commit = git_commit()
    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_git_dirty_is_bool_or_none(tmp_path):
    assert git_dirty() in (True, False)
    assert git_dirty(tmp_path) is None  # not a git checkout


def test_library_versions_lists_python_and_numpy():
    versions = library_versions(("numpy", "no-such-package-xyz"))
    assert versions["python"].startswith("3.")
    assert versions["numpy"][0].isdigit()
    assert versions["no-such-package-xyz"] == "not installed"
