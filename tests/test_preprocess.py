"""Preprocessing on a tiny synthetic raw dataset with one of every QC problem (CPU only)."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from echoef.contracts import INDEX_COLUMNS, TRACINGS_COLUMNS
from echoef.data import preprocess as pp

REPO = Path(__file__).resolve().parents[1]

# name: (split, decoded frames, frame size, grey level, traced (frame, chord half-width) in order)
# EchoNet lists the ED frame (larger LV) first; half-width -1 = a frame traced twice (D-26).
VIDEOS = {
    "vid_a": ("TRAIN", 40, 112, 100, [(20, 10), (5, 4)]),  # normal: ED first
    "vid_b": ("TRAIN", 30, 112, 120, [(3, 4), (15, 12)]),  # ES listed first
    "vid_c": ("VAL", 25, 112, 250, [(10, 9), (2, 5)]),  # EF label wrong; bright (stats check)
    "vid_d": ("TEST", 20, 112, 90, []),  # FileList frame count wrong; no tracings
    "vid_e": ("TRAIN", 15, 64, 100, [(5, 9), (1, 4)]),  # 64 x 64: not cached
    "vid_f": ("VAL", 12, 112, 100, [(30, 9), (2, 4)]),  # traced frame past the end
    "vid_g": ("TEST", 10, 112, 100, [(4, 6)]),  # only one traced frame
    "vid_h": ("TRAIN", 22, 112, 110, [(16, -1), (8, 4)]),  # frame 16 traced twice
    "vid_i": ("TRAIN", 18, 112, 30, []),  # no tracings: not in the training stats (D-27)
}
EXPECTED_QC = {
    ("missing_video", "vid_missing"),
    ("orphan_video", "vid_orphan"),
    ("ef_mismatch", "vid_c"),
    ("frame_count_mismatch", "vid_d"),
    ("frame_size", "vid_e"),
    ("filelist_size_mismatch", "vid_e"),
    ("no_trace", "vid_d"),
    ("no_trace", "vid_missing"),
    ("orphan_trace", "vid_ghost"),
    ("trace_frame_count", "vid_g"),
    ("no_trace", "vid_i"),
    ("trace_frame_out_of_range", "vid_f"),
    ("edes_not_first", "vid_b"),
    ("irregular_tracing", "vid_h"),
}
# name: (n_frames, ed_frame, es_frame, has_trace)
EXPECTED_INDEX = {
    "vid_a": (40, 20, 5, True),
    "vid_b": (30, 15, 3, True),
    "vid_c": (25, 10, 2, True),
    "vid_d": (20, -1, -1, False),
    "vid_e": (0, -1, -1, False),
    "vid_f": (12, -1, -1, False),
    "vid_g": (10, -1, -1, False),
    "vid_h": (22, -1, -1, False),
    "vid_i": (18, -1, -1, False),
    "vid_missing": (0, -1, -1, False),
}


def _write_avi(path: Path, n: int, size: int, level: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (size, size))
    assert writer.isOpened()
    for i in range(n):
        frame = np.zeros((size, size, 3), np.uint8)
        r = int(size / 8 * (1.5 + np.sin(2 * np.pi * i / 20)))  # pulsing "ventricle"
        cv2.ellipse(frame, (size // 2, size // 2), (r, r + 4), 0, 0, 360, (level,) * 3, -1)
        writer.write(frame)
    writer.release()


def _trace_rows(name: str, frame: int, half_width: int) -> list[tuple]:
    """Long axis, then 11 chords: the outline is a (2 * half_width) x 10 rectangle.

    ``half_width = -1`` mimics a frame traced twice: both long axes, then the chords of the
    two tracings interleaved (the irregular layout found in EchoNet, D-26).
    """

    def chord(k: int, hw: int) -> tuple:
        return (f"{name}.avi", 56.0 - hw, 50.0 + k, 56.0 + hw, 50.0 + k, frame)

    axis = (f"{name}.avi", 56.0, 40.0, 56.0, 72.0, frame)
    if half_width == -1:
        return [axis, axis] + [c for k in range(11) for c in (chord(k, 8), chord(k, 9))]
    return [axis] + [chord(k, half_width) for k in range(11)]


def make_raw(root: Path) -> Path:
    """Raw-format dataset (FileList.csv, VolumeTracings.csv, Videos/) with planted problems."""
    videos = root / "Videos"
    videos.mkdir(parents=True)
    filelist, tracings = [], []
    for name, (split, n, size, level, traced) in VIDEOS.items():
        _write_avi(videos / f"{name}.avi", n, size, level)
        listed_frames = n + 1 if name == "vid_d" else n
        ef = 66.0 if name == "vid_c" else 60.0
        filelist.append((name, ef, 40.0, 100.0, 112, 112, 50, listed_frames, split))
        for frame, half_width in traced:
            tracings += _trace_rows(name, frame, half_width)
    filelist.append(("vid_missing", 60.0, 40.0, 100.0, 112, 112, 50, 30, "TRAIN"))
    _write_avi(videos / "vid_orphan.avi", 5, 112, 100)
    tracings += _trace_rows("vid_ghost", 1, 4) + _trace_rows("vid_ghost", 9, 8)
    pd.DataFrame(filelist, columns=pp.FILELIST_COLUMNS).to_csv(root / "FileList.csv", index=False)
    pd.DataFrame(tracings, columns=["FileName", "X1", "Y1", "X2", "Y2", "Frame"]).to_csv(
        root / "VolumeTracings.csv", index=False
    )
    return root


@pytest.fixture(scope="module")
def raw(tmp_path_factory) -> Path:
    return make_raw(tmp_path_factory.mktemp("raw") / "EchoNet-Dynamic")


@pytest.fixture(scope="module")
def processed(raw, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("processed")
    pp.run_preprocess(raw, out, ef_tolerance=1.0, workers=1)
    return out


def test_index_matches_contract_and_expectations(processed):
    index = pd.read_parquet(processed / pp.INDEX_FILE)
    assert list(index.columns) == INDEX_COLUMNS
    assert len(index) == len(EXPECTED_INDEX)
    got = {
        r.FileName: (r.n_frames, r.ed_frame, r.es_frame, r.has_trace)
        for r in index.itertuples(index=False)
    }
    assert got == EXPECTED_INDEX
    assert index["has_trace"].dtype == bool


def test_cache_holds_grayscale_uint8(processed):
    for name, (n_frames, *_rest) in EXPECTED_INDEX.items():
        path = processed / pp.CACHE_DIR / f"{name}.npy"
        assert path.exists() == (n_frames > 0), name
        if n_frames:
            arr = np.load(path, mmap_mode="r")
            assert arr.dtype == np.uint8 and arr.shape == (n_frames, 112, 112)
    assert not list((processed / pp.CACHE_DIR).glob("*.part"))


def test_stats_use_train_videos_only(processed):
    stats = json.loads((processed / pp.STATS_FILE).read_text())
    assert set(stats) == {"mean", "std"}
    train_set = ("vid_a", "vid_b", "vid_h")  # cached TRAIN with tracings; not vid_i (D-27)
    train = np.concatenate(
        [np.load(processed / pp.CACHE_DIR / f"{n}.npy").ravel() for n in train_set]
    ).astype(np.float64)
    assert stats["mean"] == pytest.approx(train.mean(), rel=1e-12)
    assert stats["std"] == pytest.approx(train.std(), rel=1e-12)
    for other in ("vid_c", "vid_i"):  # a VAL video; an untraced TRAIN video
        with_other = np.concatenate(
            [train, np.load(processed / pp.CACHE_DIR / f"{other}.npy").ravel()]
        )
        assert stats["mean"] != pytest.approx(with_other.mean(), rel=1e-3)


def test_qc_report_finds_every_planted_problem(processed):
    qc = pd.read_csv(processed / pp.QC_FILE, keep_default_na=False)
    assert list(qc.columns) == pp.QC_COLUMNS
    assert set(zip(qc["check"], qc["FileName"], strict=True)) == EXPECTED_QC


def test_tracings_copied_without_avi_suffix(raw, processed):
    tracings = pd.read_parquet(processed / pp.TRACINGS_FILE)
    assert list(tracings.columns) == TRACINGS_COLUMNS
    assert len(tracings) == len(pd.read_csv(raw / "VolumeTracings.csv"))
    assert not tracings["FileName"].str.endswith(".avi").any()


def test_meta_records_counts(processed):
    meta = json.loads((processed / pp.META_FILE).read_text())
    assert meta["split_counts"]["TRAIN"] == {"filelist": 6, "cached": 4, "has_trace": 2}
    # regular listed two-frame videos: a, c, e, f have ED first, b does not; h is irregular
    assert meta["edes_first_traced_is_ed"] == {"yes": 4, "no": 1, "irregular_videos": 1}
    assert meta["stats"]["n_videos"] == 3
    assert meta["channels"]["videos"] == len(VIDEOS)


def test_refuses_to_overwrite_without_flag(raw, processed):
    with pytest.raises(FileExistsError):
        pp.run_preprocess(raw, processed, ef_tolerance=1.0, workers=1)


def test_refuses_output_inside_raw(raw):
    with pytest.raises(ValueError, match="must not be inside raw_root"):
        pp.run_preprocess(raw, raw / "processed", ef_tolerance=1.0, workers=1)
    assert not (raw / "processed").exists()


def test_worker_pool_gives_identical_outputs(raw, processed, tmp_path):
    pp.run_preprocess(raw, tmp_path, ef_tolerance=1.0, workers=2)
    for name in (pp.INDEX_FILE, pp.TRACINGS_FILE):
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / name), pd.read_parquet(processed / name)
        )
    for name in (pp.STATS_FILE, pp.QC_FILE):
        assert (tmp_path / name).read_text() == (processed / name).read_text()
    for name in (pp.SCAN_FILE,):
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / name), pd.read_parquet(processed / name)
        )
    for name in ():
        assert (tmp_path / name).read_text() == (processed / name).read_text()
    for npy in (processed / pp.CACHE_DIR).glob("*.npy"):
        np.testing.assert_array_equal(np.load(npy), np.load(tmp_path / pp.CACHE_DIR / npy.name))


def test_verify_writes_qc_but_no_cache(raw, processed, tmp_path):
    qc = pp.verify_dataset(raw, tmp_path, ef_tolerance=1.0, workers=1)
    assert set(zip(qc["check"], qc["FileName"], strict=True)) == EXPECTED_QC
    assert (tmp_path / pp.QC_FILE).read_text() == (processed / pp.QC_FILE).read_text()
    assert not (tmp_path / pp.CACHE_DIR).exists()
    assert not (tmp_path / pp.INDEX_FILE).exists()


def test_scripts_run_end_to_end(raw, tmp_path):
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        f"raw_root: {raw}\nprocessed_root: {tmp_path / 'processed'}\n"
        f"runs_root: {tmp_path / 'runs'}\nresults_root: {tmp_path / 'results'}\n"
        f"weights_root: {tmp_path / 'weights'}\n"
    )
    for script in ("verify_dataset.py", "preprocess.py", "eda.py"):
        cmd = [sys.executable, str(REPO / "scripts" / script), "--paths", str(paths)]
        if script != "eda.py":
            cmd += ["--workers", "1"]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    assert (tmp_path / "processed" / pp.INDEX_FILE).exists()
    for table in ("split_summary", "padding_share", "edes_fit"):
        assert (tmp_path / "results" / "eda" / f"{table}.csv").exists()
