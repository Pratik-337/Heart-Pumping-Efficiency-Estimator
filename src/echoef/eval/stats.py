"""Uncertainty and hypothesis tests: bootstrap CIs, paired Wilcoxon with Holm (task B3).

Spec, "Pre-specified analysis": 95% bootstrap CI over test videos with 10,000 resamples; paired
Wilcoxon signed-rank tests on per-video absolute errors, Holm-Bonferroni over the family of
comparisons at alpha = 0.05, with the median paired difference and its bootstrap CI as effect
size. CIs are percentile intervals from a fixed seed, so every rerun gives the same numbers.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

N_BOOT = 10_000
CONFIDENCE = 0.95
ALPHA = 0.05
PAIR_COLUMNS = ["comparison", "file_name", "err_a", "err_b"]
WILCOXON_COLUMNS = [
    "comparison", "n", "median_diff", "ci_lo", "ci_hi", "p", "p_holm", "significant",
]  # fmt: skip


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = 0,
    statistic=np.mean,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic`` (default: the mean, e.g. MAE from |errors|).

    Contract function: ``bootstrap_ci(abs_errors, n_boot=10000, seed=0) -> (lo, hi)``.

    Args:
        values: 1-D array, one value per video.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty 1-D array")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for start in range(0, n_boot, 1000):  # chunks keep memory small for 10,000 x n
        stop = min(n_boot, start + 1000)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        stats[start:stop] = statistic(values[idx], axis=1)
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return float(lo), float(hi)


def wilcoxon_holm(pairs: pd.DataFrame, alpha: float = ALPHA, seed: int = 0) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test per comparison, Holm-corrected across comparisons.

    Contract function. ``pairs`` has ``PAIR_COLUMNS``: per comparison and video, the absolute
    errors of config A (``err_a``) and config B (``err_b``) on the same video.

    Returns:
        ``WILCOXON_COLUMNS``, one row per comparison: ``median_diff`` = median of
        ``err_a - err_b`` (negative: A is more accurate) with its bootstrap CI, two-sided ``p``,
        Holm-adjusted ``p_holm`` and ``significant`` = ``p_holm < alpha``.
    """
    missing = [c for c in PAIR_COLUMNS if c not in pairs.columns]
    if missing:
        raise ValueError(f"pairs lacks columns {missing}")
    rows = []
    for comparison, group in pairs.groupby("comparison", sort=False):
        diff = (group["err_a"] - group["err_b"]).to_numpy(dtype=np.float64)
        p = float(wilcoxon(diff, zero_method="wilcox").pvalue) if np.any(diff) else 1.0
        lo, hi = bootstrap_ci(diff, seed=seed, statistic=np.median)
        rows.append((comparison, len(diff), float(np.median(diff)), lo, hi, p))
    out = pd.DataFrame(rows, columns=WILCOXON_COLUMNS[:6])
    if len(out):
        reject, p_holm, _, _ = multipletests(out["p"], alpha=alpha, method="holm")
        out["p_holm"], out["significant"] = p_holm, reject
    else:
        out["p_holm"], out["significant"] = [], []
    return out[WILCOXON_COLUMNS]
