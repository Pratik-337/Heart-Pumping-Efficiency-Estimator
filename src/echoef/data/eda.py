"""Aggregated dataset summaries from ``index.parquet`` (task A2; exported for B's Dataset section).

Every output is a count, share or summary statistic per split; nothing per video, so the tables
may leave A's laptop. Frame statistics use cached videos only (``n_frames > 0``); label
statistics use every FileList.csv row.
"""

import pandas as pd

from echoef.contracts import CONFIGS, SPLITS
from echoef.data.sampling import is_padded, m1_start

ALL = "ALL"  # pseudo-split covering every row


def _by_split(index: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [(s, index[index["Split"] == s]) for s in SPLITS] + [(ALL, index)]


def ef_band_counts(ef: pd.Series, edges: tuple[float, float]) -> dict[str, int]:
    """Counts per EF band: ``ef <= lo``, ``lo < ef < hi``, ``ef >= hi`` (spec, "MAE by EF band").

    With ``edges = (40, 50)`` the keys are ``n_ef_le40``, ``n_ef_41_49``, ``n_ef_ge50``,
    matching the ``mae_le40``/``mae_41_49``/``mae_ge50`` columns of ``ALL_RUNS_COLUMNS``.
    """
    lo, hi = edges
    return {
        f"n_ef_le{lo:g}": int((ef <= lo).sum()),
        f"n_ef_{lo + 1:g}_{hi - 1:g}": int(((ef > lo) & (ef < hi)).sum()),
        f"n_ef_ge{hi:g}": int((ef >= hi).sum()),
    }


def split_summary(index: pd.DataFrame, ef_band_edges: tuple[float, float]) -> pd.DataFrame:
    """One row per split plus ``ALL``: counts, frame-count spread, FPS, EF spread and bands."""
    rows = []
    for split, part in _by_split(index):
        cached = part[part["n_frames"] > 0]
        traced = part[part["has_trace"]]
        frames = cached["n_frames"]
        rows.append(
            {
                "split": split,
                "n": len(part),
                "n_cached": len(cached),
                "n_has_trace": len(traced),
                "frames_min": frames.min(),
                "frames_median": frames.median(),
                "frames_mean": frames.mean(),
                "frames_max": frames.max(),
                "fps_median": cached["FPS"].median(),
                "ef_mean": part["EF"].mean(),
                "ef_sd": part["EF"].std(),
                **ef_band_counts(part["EF"], ef_band_edges),
                "edes_gap_median": (traced["ed_frame"] - traced["es_frame"]).abs().median(),
            }
        )
    return pd.DataFrame(rows)


def padding_share(index: pd.DataFrame) -> pd.DataFrame:
    """Per config and split: share of cached videos whose clips are zero-padded (D-28).

    A video is padded when its clip reads a frame at or past its end (``sampling.is_padded``).
    That happens only when it is shorter than the window W = T x period, and then every clip
    starts at 0, so training, M1 and M2 agree. A video up to ``period - 1`` frames shorter than
    W is not padded: its last read frame is still inside it. Replaces the normal approximation
    in the spec's "Short videos" table with exact counts.
    """
    rows = []
    for config_id, (regime, t, period) in CONFIGS.items():
        window = t * period
        for split, part in _by_split(index):
            frames = part.loc[part["n_frames"] > 0, "n_frames"]
            n_padded = sum(is_padded(m1_start(n, t, period), t, period, n) for n in frames)
            rows.append(
                {
                    "config_id": config_id,
                    "regime": regime,
                    "T": t,
                    "period": period,
                    "window": window,
                    "split": split,
                    "n": len(frames),
                    "n_padded": n_padded,
                    "share_padded": n_padded / len(frames) if len(frames) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def edes_fit(index: pd.DataFrame) -> pd.DataFrame:
    """Per config and split: share of traced videos whose ED and ES cannot share one clip.

    A video is skipped by the XAI tests when ``|ED - ES| >= W`` (spec, "ED/ES-centred clips").
    """
    rows = []
    for config_id, (regime, t, period) in CONFIGS.items():
        window = t * period
        for split, part in _by_split(index):
            traced = part[part["has_trace"]]
            gap = (traced["ed_frame"] - traced["es_frame"]).abs()
            n_skip = int((gap >= window).sum())
            rows.append(
                {
                    "config_id": config_id,
                    "regime": regime,
                    "T": t,
                    "period": period,
                    "window": window,
                    "split": split,
                    "n": len(traced),
                    "n_skip": n_skip,
                    "share_skip": n_skip / len(traced) if len(traced) else float("nan"),
                }
            )
    return pd.DataFrame(rows)
