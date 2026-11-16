"""Accuracy and agreement metrics of one predictions table (task B3).

Spec, "Evaluation protocol and metrics". EF errors are in EF points. With y = reference EF and
y_hat = prediction:

- MAE, RMSE, R^2 = 1 - SS_res / SS_tot;
- Bland-Altman: bias = mean(y_hat - y), limits of agreement = bias +- 1.96 SD(y_hat - y);
- AUROC for EF < 50% and EF <= 40%, the prediction used as a score (lower EF = positive);
- MAE by reference-EF band: <= 40, 41-49 (40 < EF < 50), >= 50;
- calibration slope: OLS slope of y regressed on y_hat (below 1: predictions too spread out);
- 95% bootstrap CI of the MAE over videos (``stats.bootstrap_ci``).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from echoef.eval.stats import bootstrap_ci

LOA_Z = 1.96
METRIC_KEYS = [
    "n", "mae", "mae_ci_lo", "mae_ci_hi", "rmse", "r2", "bias", "loa_lo", "loa_hi",
    "auroc_lt50", "auroc_le40", "mae_le40", "mae_41_49", "mae_ge50", "calib_slope",
]  # fmt: skip  # the metric columns of contracts.ALL_RUNS_COLUMNS


def _auroc(positive: np.ndarray, score: np.ndarray) -> float:
    if positive.all() or not positive.any():
        return float("nan")  # undefined with one class only
    return float(roc_auc_score(positive, score))


def _band_mae(abs_err: np.ndarray, mask: np.ndarray) -> float:
    return float(abs_err[mask].mean()) if mask.any() else float("nan")


def compute_metrics(df: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> dict[str, float]:
    """All metrics of one predictions table (``PREDICTION_COLUMNS``). Contract function.

    Returns:
        dict with ``METRIC_KEYS``.
    """
    y = df["ef_true"].to_numpy(dtype=np.float64)
    y_hat = df["ef_pred"].to_numpy(dtype=np.float64)
    if len(y) < 2:
        raise ValueError("need at least 2 predictions")
    err = y_hat - y
    abs_err = np.abs(err)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    bias, sd = float(err.mean()), float(err.std(ddof=1))
    ci_lo, ci_hi = bootstrap_ci(abs_err, n_boot=n_boot, seed=seed)
    slope = float(np.polyfit(y_hat, y, 1)[0]) if np.ptp(y_hat) > 0 else float("nan")
    return {
        "n": len(y),
        "mae": float(abs_err.mean()),
        "mae_ci_lo": ci_lo,
        "mae_ci_hi": ci_hi,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "bias": bias,
        "loa_lo": bias - LOA_Z * sd,
        "loa_hi": bias + LOA_Z * sd,
        "auroc_lt50": _auroc(y < 50, -y_hat),
        "auroc_le40": _auroc(y <= 40, -y_hat),
        "mae_le40": _band_mae(abs_err, y <= 40),
        "mae_41_49": _band_mae(abs_err, (y > 40) & (y < 50)),
        "mae_ge50": _band_mae(abs_err, y >= 50),
        "calib_slope": slope,
    }
