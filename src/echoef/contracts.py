"""Shared contracts between A (pipeline) and B (analysis). Change only when both agree."""

from pathlib import Path

# Config ID -> (regime, T, period). fs32 belongs to both regimes.
CONFIGS = {
    "fs8": ("FS", 8, 2),
    "fs16": ("FS", 16, 2),
    "fs32": ("BOTH", 32, 2),
    "fs64": ("FS", 64, 2),
    "fw8": ("FW", 8, 8),
    "fw16": ("FW", 16, 4),
    "fw64": ("FW", 64, 1),
}
TIER0_PREFIX = "b0-"  # Tier 0 run IDs: b0-fs8 ... b0-fw64 (EchoNet weights, no training)
SPLITS = ("TRAIN", "VAL", "TEST")
MODES = ("M1", "M2")
FRAME_SIZE = 112

INDEX_COLUMNS = [
    "FileName",
    "Split",
    "EF",
    "EDV",
    "ESV",
    "FPS",
    "n_frames",
    "ed_frame",
    "es_frame",
    "has_trace",
]
TRACINGS_COLUMNS = ["FileName", "Frame", "X1", "Y1", "X2", "Y2"]
PREDICTION_COLUMNS = [
    "file_name",
    "split",
    "mode",
    "config_id",
    "seed",
    "T",
    "period",
    "ef_true",
    "ef_pred",
    "n_clips",
    "padded",
]
COST_COLUMNS = [
    "config_id",
    "T",
    "period",
    "params",
    "gflops_clip",
    "mean_clips_m2",
    "gflops_video_m2",
    "latency_ms_m1",
    "latency_ms_m2",
    "peak_vram_mb_infer",
    "peak_vram_mb_train",
    "micro_batch",
    "epoch_time_s",
    "gpu_name",
]
ALL_RUNS_COLUMNS = [
    "config_id",
    "regime",
    "T",
    "period",
    "seed",
    "split",
    "mode",
    "n",
    "mae",
    "mae_ci_lo",
    "mae_ci_hi",
    "rmse",
    "r2",
    "bias",
    "loa_lo",
    "loa_hi",
    "auroc_lt50",
    "auroc_le40",
    "mae_le40",
    "mae_41_49",
    "mae_ge50",
    "calib_slope",
]
XAI_COLUMNS = [
    "file_name",
    "config_id",
    "method",
    "hit",
    "lv_energy_ratio",
    "deletion_aopc",
    "random_aopc",
    "skipped",
]


def run_dir(runs_root: Path, config_id: str, seed: int) -> Path:
    return runs_root / config_id / f"seed{seed}"


def predictions_path(run: Path, split: str, mode: str) -> Path:
    return run / f"predictions_{split.lower()}_{mode}.csv"
