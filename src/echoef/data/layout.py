"""File names inside ``processed_root`` (spec, "Interface contracts", "Files on disk").

Kept apart from ``preprocess`` so that the readers (``dataset``, ``tracings``) and the writer share
one definition without importing each other.
"""

from pathlib import Path

CACHE_DIR = "videos"
INDEX_FILE = "index.parquet"
TRACINGS_FILE = "tracings.parquet"
STATS_FILE = "stats.json"
QC_FILE = "qc_report.csv"
SCAN_FILE = "video_scan.parquet"
META_FILE = "preprocess_meta.json"


def video_path(processed_root: Path, file_name: str) -> Path:
    """Cached frames of one video: ``<processed_root>/videos/<file_name>.npy``."""
    return Path(processed_root) / CACHE_DIR / f"{file_name}.npy"
