"""Raw EchoNet-Dynamic -> frame cache, index, tracings, train-only stats and QC report (task A2).

Implements the "Preprocessing pipeline" in docs/PROJECT_SPEC.md (section "Dataset:
EchoNet-Dynamic"). Reads ``raw_root`` only and writes only under ``processed_root``:

    videos/<FileName>.npy   uint8 (frames, 112, 112), grayscale by OpenCV's luma rule (D-08, D-25)
    index.parquet           contracts.INDEX_COLUMNS, one row per FileList.csv row
    tracings.parquet        contracts.TRACINGS_COLUMNS, copied from VolumeTracings.csv with the
                            ".avi" suffix removed from FileName so it matches the index
    stats.json              {"mean": float, "std": float}: every pixel of every frame of the cached
                            TRAIN videos that have tracings (the training set, D-27), in raw
                            0-255 grey levels, population SD
    qc_report.csv           QC_COLUMNS, one row per problem found
    video_scan.parquet      one row per decoded video: frame count, size, channel differences
    preprocess_meta.json    provenance: git commit, library versions, settings, counts, timings

Index conventions: ``n_frames`` is the decoded frame count, or 0 when the video is missing or
was not cached. ``ed_frame``/``es_frame`` are -1 and ``has_trace`` is False unless the video is
cached and has exactly two traced frames, each one regular tracing (D-26), inside its decoded
length. ED is the traced frame with the larger LV polygon area, ES the smaller (spec step 4). In
EchoNet-Dynamic the first traced frame is ED; a tie follows that order, and a video where area
disagrees with it is reported as ``edes_not_first``.

Videos without tracings stay in the index (``has_trace`` False) but are not part of the
training set (D-27); ``tracings.parquet`` tells them apart from irregularly traced ones.
"""

import json
import logging
import math
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from echoef.contracts import FRAME_SIZE, INDEX_COLUMNS, SPLITS, TRACINGS_COLUMNS
from echoef.data.layout import (  # processed layout; re-exported for scripts and tests
    CACHE_DIR,
    INDEX_FILE,
    META_FILE,
    QC_FILE,
    SCAN_FILE,
    STATS_FILE,
    TRACINGS_FILE,
)
from echoef.data.tracings import is_single_tracing, lv_polygon, polygon_area
from echoef.utils.meta import git_commit, git_dirty, library_versions

log = logging.getLogger(__name__)

# Raw dataset layout (fixed by EchoNet-Dynamic).
FILELIST_CSV = "FileList.csv"
TRACINGS_CSV = "VolumeTracings.csv"
VIDEOS_DIR = "Videos"
VIDEO_SUFFIX = ".avi"
FILELIST_COLUMNS = [
    "FileName",
    "EF",
    "ESV",
    "EDV",
    "FrameHeight",
    "FrameWidth",
    "FPS",
    "NumberOfFrames",
    "Split",
]

QC_COLUMNS = ["FileName", "Split", "check", "detail"]
EDES_COLUMNS = [
    "FileName",
    "n_traced_frames",
    "frame_rows",  # e.g. "46:21 61:21" = traced frame:rows
    "regular",  # every traced frame is one regular tracing (D-26)
    "ed_frame",
    "es_frame",
    "ed_area",
    "es_area",
    "first_is_ed",
]
GRAYSCALE_RULE = "cv2.COLOR_BGR2GRAY: 0.299 R + 0.587 G + 0.114 B (D-25)"
PROGRESS_EVERY = 500  # videos between progress log lines


@dataclass(frozen=True)
class VideoScan:
    """What decoding one video found. Pixel sums are over the grayscale frames."""

    file_name: str
    n_frames: int  # decoded frames; 0 if decoding failed
    height: int
    width: int
    cached: bool  # written to the frame cache
    channel_max_diff: int  # largest (max - min) over B, G, R at any pixel; 0 = identical channels
    channel_diff_share: float  # share of pixels whose B, G, R are not all equal
    pixel_sum: int
    pixel_sumsq: int
    error: str  # "" when decoding worked


# ----------------------------------------------------------------------------- raw tables


def _stem(name: str) -> str:
    return name.removesuffix(VIDEO_SUFFIX)


def read_filelist(raw_root: Path) -> pd.DataFrame:
    """``FileList.csv`` with ``FileName`` stripped of any ".avi" and ``Split`` upper-cased."""
    df = pd.read_csv(raw_root / FILELIST_CSV, dtype={"FileName": str, "Split": str})
    missing = [c for c in FILELIST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{raw_root / FILELIST_CSV} lacks columns {missing}")
    df["FileName"] = df["FileName"].map(_stem)
    df["Split"] = df["Split"].str.upper()
    unknown = sorted(set(df["Split"]) - set(SPLITS))
    if unknown:
        raise ValueError(f"{FILELIST_CSV}: unknown Split values {unknown}; expected {SPLITS}")
    duplicated = df["FileName"][df["FileName"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{FILELIST_CSV}: duplicated FileName {duplicated[:5]}")
    return df.reset_index(drop=True)


def read_tracings(raw_root: Path) -> pd.DataFrame:
    """``VolumeTracings.csv`` as ``TRACINGS_COLUMNS``, file order kept, ".avi" removed."""
    df = pd.read_csv(raw_root / TRACINGS_CSV, dtype={"FileName": str})
    missing = [c for c in TRACINGS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{raw_root / TRACINGS_CSV} lacks columns {missing}")
    df["FileName"] = df["FileName"].map(_stem)
    df["Frame"] = df["Frame"].astype("int64")
    return df[TRACINGS_COLUMNS].reset_index(drop=True)


def edes_from_tracings(tracings: pd.DataFrame) -> pd.DataFrame:
    """ED and ES frame of every traced video, chosen by LV polygon area (spec step 4, D-26).

    Args:
        tracings: ``TRACINGS_COLUMNS`` rows in file order.

    Returns:
        ``EDES_COLUMNS``, one row per traced video. ``ed_frame``/``es_frame`` are -1 (areas NaN)
        unless the video has exactly two traced frames and both are regular tracings.
        ``first_is_ed`` says whether ED is the first traced frame in file order.
    """
    rows = []
    for name, group in tracings.groupby("FileName", sort=True):
        frame_col = group["Frame"].to_numpy()
        frames = list(dict.fromkeys(frame_col.tolist()))  # order of first appearance
        coords = group[["X1", "Y1", "X2", "Y2"]].to_numpy()
        per_frame = [coords[frame_col == f] for f in frames]
        frame_rows = " ".join(f"{f}:{len(r)}" for f, r in zip(frames, per_frame, strict=True))
        regular = all(is_single_tracing(r) for r in per_frame)
        if len(frames) != 2 or not regular:
            rows.append((name, len(frames), frame_rows, regular, -1, -1, math.nan, math.nan, False))
            continue
        areas = [polygon_area(lv_polygon(r)) for r in per_frame]
        ed_i = 1 if areas[1] > areas[0] else 0  # tie -> the dataset's order (first = ED)
        es_i = 1 - ed_i
        rows.append(
            (name, 2, frame_rows, True, frames[ed_i], frames[es_i], areas[ed_i], areas[es_i],
             ed_i == 0)
        )  # fmt: skip
    return pd.DataFrame(rows, columns=EDES_COLUMNS)


def check_labels(filelist: pd.DataFrame, tolerance: float) -> list[dict]:
    """QC rows where EF differs from (EDV - ESV) / EDV x 100 by more than ``tolerance`` points."""
    recomputed = (filelist["EDV"] - filelist["ESV"]) / filelist["EDV"] * 100
    diff = (recomputed - filelist["EF"]).abs()
    qc = []
    for name, split, ef, re_ef, d in zip(
        filelist["FileName"], filelist["Split"], filelist["EF"], recomputed, diff, strict=True
    ):
        if not math.isfinite(d):
            qc.append(_qc(name, split, "label_missing", "EF, EDV or ESV is missing or EDV = 0"))
        elif d > tolerance:
            qc.append(_qc(name, split, "ef_mismatch", f"EF {ef:.2f}, recomputed {re_ef:.2f}"))
    return qc


def _qc(name: str, split: str, check: str, detail: str) -> dict:
    return {"FileName": name, "Split": split, "check": check, "detail": detail}


# ----------------------------------------------------------------------------- videos


def decode_video(path: Path) -> np.ndarray:
    """Decode every frame of a video.

    Returns:
        uint8 array (F, H, W, 3) in OpenCV's BGR order; (0, 0, 0, 3) if nothing decodes.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    finally:
        cap.release()
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames)


def to_gray(frames: np.ndarray) -> np.ndarray:
    """(F, H, W, 3) uint8 BGR -> (F, H, W) uint8 by OpenCV's luma rule (D-25)."""
    f, h, w, _ = frames.shape
    gray = cv2.cvtColor(np.ascontiguousarray(frames).reshape(f * h, w, 3), cv2.COLOR_BGR2GRAY)
    return gray.reshape(f, h, w)


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as fh:
        np.save(fh, array)
    part.replace(path)


def scan_video(file_name: str, video_path: Path, cache_dir: Path | None) -> VideoScan:
    """Decode one video, measure it and, if ``cache_dir`` is given and it is 112 x 112, cache it.

    The cached array is uint8 (F, 112, 112). Decoding errors are recorded, not raised.
    """
    try:
        frames = decode_video(video_path)
    except Exception as exc:  # a corrupt file must not stop the other 10,000 videos
        return VideoScan(file_name, 0, 0, 0, False, 0, 0.0, 0, 0, f"{type(exc).__name__}: {exc}")
    if len(frames) == 0:
        return VideoScan(file_name, 0, 0, 0, False, 0, 0.0, 0, 0, "no frames decoded")

    n, h, w = frames.shape[:3]
    spread = frames.max(axis=-1) - frames.min(axis=-1)  # uint8, no overflow: max >= min
    gray = to_gray(frames)
    counts = np.bincount(gray.ravel(), minlength=256).astype(np.int64)
    levels = np.arange(256, dtype=np.int64)
    cached = cache_dir is not None and (h, w) == (FRAME_SIZE, FRAME_SIZE)
    if cached:
        _save_npy_atomic(cache_dir / f"{file_name}.npy", gray)
    return VideoScan(
        file_name=file_name,
        n_frames=n,
        height=h,
        width=w,
        cached=cached,
        channel_max_diff=int(spread.max()),
        channel_diff_share=float(np.count_nonzero(spread) / spread.size),
        pixel_sum=int(counts @ levels),
        pixel_sumsq=int(counts @ (levels * levels)),
        error="",
    )


def _scan_job(job: tuple[str, Path, Path | None]) -> VideoScan:
    return scan_video(*job)


def _init_worker() -> None:
    cv2.setNumThreads(1)  # one decoding process per core; no nested OpenCV threads


def scan_videos(
    names: Iterable[str], videos_dir: Path, cache_dir: Path | None, workers: int
) -> dict[str, VideoScan]:
    """Run ``scan_video`` on every name, in ``workers`` processes (1 = in this process)."""
    jobs = [(n, videos_dir / f"{n}{VIDEO_SUFFIX}", cache_dir) for n in names]
    results: dict[str, VideoScan] = {}

    def collect(scans: Iterable[VideoScan]) -> None:
        for i, scan in enumerate(scans, 1):
            results[scan.file_name] = scan
            if i % PROGRESS_EVERY == 0 or i == len(jobs):
                log.info("decoded %d / %d videos", i, len(jobs))

    if workers <= 1:
        collect(_scan_job(job) for job in jobs)
    else:
        # spawn: no fork of a process that may hold OpenCV threads; same behaviour on macOS
        with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
            collect(pool.imap_unordered(_scan_job, jobs, chunksize=8))
    return results


def scan_table(scans: dict[str, VideoScan]) -> pd.DataFrame:
    """One row per scanned video, sorted by name (written to ``video_scan.parquet``)."""
    table = pd.DataFrame([vars(s) for s in scans.values()], columns=list(VideoScan.__annotations__))
    return table.sort_values("file_name").reset_index(drop=True)


def channel_summary(scans: dict[str, VideoScan]) -> dict:
    """How far R, G and B differ, over all decoded videos (spec step 2, D-25)."""
    decoded = [s for s in scans.values() if not s.error]
    shares = [s.channel_diff_share for s in decoded] or [0.0]
    return {
        "videos": len(decoded),
        "videos_identical": sum(s.channel_max_diff == 0 for s in decoded),
        "max_abs_diff": max((s.channel_max_diff for s in decoded), default=0),
        "median_diff_share": statistics.median(shares),
        "max_diff_share": max(shares),
    }


# ----------------------------------------------------------------------------- assembly


def _scan_qc(filelist: pd.DataFrame, scans: dict[str, VideoScan]) -> list[dict]:
    qc = []
    for row in filelist.itertuples(index=False):
        scan = scans.get(row.FileName)
        if scan is None:
            continue
        if scan.error:
            qc.append(_qc(row.FileName, row.Split, "decode_failed", scan.error))
            continue
        if scan.n_frames != row.NumberOfFrames:
            detail = f"decoded {scan.n_frames}, FileList.csv says {row.NumberOfFrames}"
            qc.append(_qc(row.FileName, row.Split, "frame_count_mismatch", detail))
        if (scan.height, scan.width) != (FRAME_SIZE, FRAME_SIZE):
            detail = f"decoded {scan.height}x{scan.width}, expected {FRAME_SIZE}x{FRAME_SIZE}"
            qc.append(_qc(row.FileName, row.Split, "frame_size", detail + "; not cached"))
        if (row.FrameHeight, row.FrameWidth) != (scan.height, scan.width):
            detail = (
                f"FileList.csv says {row.FrameHeight}x{row.FrameWidth}, "
                f"decoded {scan.height}x{scan.width}"
            )
            qc.append(_qc(row.FileName, row.Split, "filelist_size_mismatch", detail))
    return qc


def _tracing_qc(
    filelist: pd.DataFrame, edes: pd.DataFrame, scans: dict[str, VideoScan]
) -> list[dict]:
    split_of = dict(zip(filelist["FileName"], filelist["Split"], strict=True))
    listed, traced = set(split_of), set(edes["FileName"])
    qc = [
        _qc(n, split_of[n], "no_trace", "no rows in VolumeTracings.csv")
        for n in sorted(listed - traced)
    ]
    qc += [
        _qc(n, "", "orphan_trace", "traced video not listed in FileList.csv; ignored")
        for n in sorted(traced - listed)
    ]
    for r in edes.itertuples(index=False):
        if r.FileName not in listed:
            continue
        split = split_of[r.FileName]
        if r.n_traced_frames != 2:
            detail = f"{r.n_traced_frames} traced frames ({r.frame_rows} rows), expected 2"
            qc.append(_qc(r.FileName, split, "trace_frame_count", detail))
            continue
        if not r.regular:
            detail = f"frame:rows {r.frame_rows}; a frame is not one long axis + chords (D-26)"
            qc.append(_qc(r.FileName, split, "irregular_tracing", detail))
            continue
        scan = scans.get(r.FileName)
        if scan is not None and not scan.error and max(r.ed_frame, r.es_frame) >= scan.n_frames:
            detail = f"traced frames {r.ed_frame} and {r.es_frame}; video has {scan.n_frames}"
            qc.append(_qc(r.FileName, split, "trace_frame_out_of_range", detail))
        if not r.first_is_ed:
            detail = f"by area ED={r.ed_frame}, ES={r.es_frame}; the first traced frame is ES"
            qc.append(_qc(r.FileName, split, "edes_not_first", detail))
    return qc


def build_index(
    filelist: pd.DataFrame, scans: dict[str, VideoScan], edes: pd.DataFrame
) -> pd.DataFrame:
    """One row per FileList.csv row, columns ``contracts.INDEX_COLUMNS`` (conventions above)."""
    index = filelist[["FileName", "Split", "EF", "EDV", "ESV", "FPS"]].copy()
    n_frames = pd.Series(
        [scans[n].n_frames if n in scans and scans[n].cached else 0 for n in index["FileName"]],
        index=index.index,
        dtype="int64",
    )
    by_name = edes.set_index("FileName")
    ed = index["FileName"].map(by_name["ed_frame"]).fillna(-1).astype("int64")
    es = index["FileName"].map(by_name["es_frame"]).fillna(-1).astype("int64")
    ok = (ed >= 0) & (es >= 0) & (ed < n_frames) & (es < n_frames)
    index["n_frames"] = n_frames
    index["ed_frame"] = ed.where(ok, -1)
    index["es_frame"] = es.where(ok, -1)
    index["has_trace"] = ok.astype(bool)
    index = index.astype({"EF": "float64", "EDV": "float64", "ESV": "float64", "FPS": "int64"})
    return index[INDEX_COLUMNS]


def compute_stats(scans: dict[str, VideoScan], names: Iterable[str]) -> tuple[dict, dict]:
    """Pixel mean and population SD over every frame of the cached videos in ``names``.

    Returns:
        ``({"mean", "std"}, {"n_videos", "n_pixels"})``, in raw 0-255 grey levels. Sums are
        exact integers until the final division.
    """
    total = pixel_sum = pixel_sumsq = n_videos = 0
    for name in names:
        scan = scans.get(name)
        if scan is None or not scan.cached:
            continue
        total += scan.n_frames * scan.height * scan.width
        pixel_sum += scan.pixel_sum
        pixel_sumsq += scan.pixel_sumsq
        n_videos += 1
    if total == 0:
        raise ValueError("no cached video to compute normalisation statistics from")
    variance = (pixel_sumsq * total - pixel_sum * pixel_sum) / (total * total)
    return (
        {"mean": pixel_sum / total, "std": math.sqrt(variance)},
        {"n_videos": n_videos, "n_pixels": total},
    )


def check_roots(raw_root: Path, processed_root: Path) -> None:
    """Refuse a missing raw folder, or an output folder that is or lies inside the raw folder."""
    if not (raw_root / FILELIST_CSV).exists():
        raise FileNotFoundError(f"{raw_root / FILELIST_CSV} not found; check raw_root")
    raw, out = raw_root.resolve(), processed_root.resolve()
    if out == raw or out.is_relative_to(raw):
        raise ValueError(f"processed_root {out} must not be inside raw_root {raw} (read-only)")


def _inspect(
    raw_root: Path, cache_dir: Path | None, ef_tolerance: float, workers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, VideoScan], pd.DataFrame]:
    """Read the raw tables, decode every listed video (caching it if asked) and collect QC."""
    filelist = read_filelist(raw_root)
    tracings = read_tracings(raw_root)
    videos_dir = raw_root / VIDEOS_DIR
    on_disk = {_stem(p.name) for p in videos_dir.glob(f"*{VIDEO_SUFFIX}")}
    split_of = dict(zip(filelist["FileName"], filelist["Split"], strict=True))
    listed = set(split_of)

    qc = [
        _qc(n, split_of[n], "missing_video", f"no {VIDEOS_DIR}/{n}{VIDEO_SUFFIX}")
        for n in sorted(listed - on_disk)
    ]
    qc += [
        _qc(n, "", "orphan_video", "video file not listed in FileList.csv; ignored")
        for n in sorted(on_disk - listed)
    ]
    qc += check_labels(filelist, ef_tolerance)

    to_scan = sorted(listed & on_disk)
    log.info("decoding %d videos with %d worker(s)", len(to_scan), workers)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    scans = scan_videos(to_scan, videos_dir, cache_dir, workers)
    qc += _scan_qc(filelist, scans)

    edes = edes_from_tracings(tracings)
    qc += _tracing_qc(filelist, edes, scans)
    qc_df = pd.DataFrame(qc, columns=QC_COLUMNS).sort_values(["check", "FileName"])
    return filelist, tracings, edes, scans, qc_df.reset_index(drop=True)


def _log_qc(qc: pd.DataFrame, channels: dict) -> None:
    counts = qc["check"].value_counts().sort_index()
    log.info("QC problems: %s", counts.to_dict() if len(counts) else "none")
    log.info(
        "channels: %d of %d decoded videos have identical R, G, B; max |diff| %d; "
        "median share of differing pixels %.4f",
        channels["videos_identical"],
        channels["videos"],
        channels["max_abs_diff"],
        channels["median_diff_share"],
    )


def verify_dataset(
    raw_root: Path, processed_root: Path, *, ef_tolerance: float, workers: int
) -> pd.DataFrame:
    """Spec step 1, 2 and 5 without writing the cache: decode everything, write the QC report.

    Returns:
        The QC report (``QC_COLUMNS``), also written to ``processed_root/qc_report.csv``.
    """
    raw_root, processed_root = Path(raw_root), Path(processed_root)
    check_roots(raw_root, processed_root)
    filelist, _, _, scans, qc = _inspect(raw_root, None, ef_tolerance, workers)
    processed_root.mkdir(parents=True, exist_ok=True)
    qc.to_csv(processed_root / QC_FILE, index=False)
    log.info("split counts in FileList.csv: %s", filelist["Split"].value_counts().to_dict())
    _log_qc(qc, channel_summary(scans))
    log.info("wrote %s", processed_root / QC_FILE)
    return qc


def run_preprocess(
    raw_root: Path,
    processed_root: Path,
    *,
    ef_tolerance: float,
    workers: int,
    overwrite: bool = False,
) -> dict:
    """The full pipeline (spec steps 1-7). Writes the files listed in the module docstring.

    Raises:
        FileExistsError: ``processed_root`` already holds an index and ``overwrite`` is False.

    Returns:
        The provenance dict also written to ``preprocess_meta.json``.
    """
    raw_root, processed_root = Path(raw_root), Path(processed_root)
    check_roots(raw_root, processed_root)
    if (processed_root / INDEX_FILE).exists() and not overwrite:
        raise FileExistsError(f"{processed_root / INDEX_FILE} exists; pass overwrite=True")
    started = datetime.now(UTC)
    t0 = time.perf_counter()

    filelist, tracings, edes, scans, qc = _inspect(
        raw_root, processed_root / CACHE_DIR, ef_tolerance, workers
    )
    index = build_index(filelist, scans, edes)
    traced = index["FileName"].isin(tracings["FileName"])  # D-27: untraced videos not trained on
    train_names = index.loc[(index["Split"] == "TRAIN") & traced, "FileName"]
    stats, stats_n = compute_stats(scans, train_names)
    channels = channel_summary(scans)

    tracings.to_parquet(processed_root / TRACINGS_FILE, index=False)
    (processed_root / STATS_FILE).write_text(json.dumps(stats, indent=2) + "\n")
    qc.to_csv(processed_root / QC_FILE, index=False)
    scan_table(scans).to_parquet(processed_root / SCAN_FILE, index=False)
    index.to_parquet(processed_root / INDEX_FILE, index=False)  # last: marks a finished cache

    split_counts = {
        split: {
            "filelist": int((index["Split"] == split).sum()),
            "cached": int(((index["Split"] == split) & (index["n_frames"] > 0)).sum()),
            "has_trace": int(((index["Split"] == split) & index["has_trace"]).sum()),
        }
        for split in SPLITS
    }
    listed = edes[edes["FileName"].isin(index["FileName"])]
    first_is_ed = listed.loc[listed["regular"] & (listed["n_traced_frames"] == 2), "first_is_ed"]
    first_is_ed = first_is_ed.astype(bool)
    meta = {
        "task": "A2",
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "versions": library_versions(),
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "settings": {
            "ef_tolerance": ef_tolerance,
            "workers": workers,
            "frame_size": FRAME_SIZE,
            "grayscale": GRAYSCALE_RULE,
        },
        "split_counts": split_counts,
        "qc_counts": {k: int(v) for k, v in qc["check"].value_counts().sort_index().items()},
        "channels": channels,
        "stats": {
            **stats,
            **stats_n,
            "videos": "cached TRAIN with tracings (D-27)",
            "units": "grey levels 0-255",
        },
        "edes_first_traced_is_ed": {
            "yes": int(first_is_ed.sum()),
            "no": int((~first_is_ed).sum()),
            "irregular_videos": int((~listed["regular"]).sum()),
        },
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (processed_root / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")

    for split, c in split_counts.items():
        log.info(
            "%-5s FileList %5d | cached %5d | with tracings %5d",
            split,
            c["filelist"],
            c["cached"],
            c["has_trace"],
        )
    _log_qc(qc, channels)
    log.info("train stats: mean %.4f, std %.4f (%d videos)", *stats.values(), stats_n["n_videos"])
    log.info("wrote %s in %.0f s", processed_root, meta["seconds"])
    return meta
