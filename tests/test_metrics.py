"""Metrics against hand-computed values (task B3)."""

import math

import numpy as np
import pandas as pd
import pytest

from echoef.contracts import ALL_RUNS_COLUMNS
from echoef.eval.metrics import METRIC_KEYS, compute_metrics


def _df(y, y_hat):
    return pd.DataFrame({"ef_true": y, "ef_pred": y_hat})


def test_metric_keys_are_the_all_runs_metric_columns():
    start = ALL_RUNS_COLUMNS.index("n")
    assert ALL_RUNS_COLUMNS[start:] == METRIC_KEYS


def test_hand_computed_values():
    # y = 30, 45, 60, 70; y_hat = 34, 43, 60, 65 -> errors +4, -2, 0, -5
    m = compute_metrics(_df([30, 45, 60, 70], [34, 43, 60, 65]), n_boot=200)
    assert m["n"] == 4
    assert m["mae"] == pytest.approx((4 + 2 + 0 + 5) / 4)
    assert m["rmse"] == pytest.approx(math.sqrt((16 + 4 + 0 + 25) / 4))
    ss_tot = sum((v - 51.25) ** 2 for v in (30, 45, 60, 70))
    assert m["r2"] == pytest.approx(1 - 45 / ss_tot)
    sd = np.std([4, -2, 0, -5], ddof=1)
    assert m["bias"] == pytest.approx(-0.75)
    assert (m["loa_lo"], m["loa_hi"]) == pytest.approx((-0.75 - 1.96 * sd, -0.75 + 1.96 * sd))
    assert m["mae_le40"] == 4 and m["mae_41_49"] == 2 and m["mae_ge50"] == pytest.approx(2.5)
    assert m["auroc_lt50"] == 1.0  # the two EF < 50 videos get the two lowest predictions
    assert m["auroc_le40"] == 1.0
    assert m["mae_ci_lo"] <= m["mae"] <= m["mae_ci_hi"]


def test_calibration_slope_and_perfect_prediction():
    y = np.array([20.0, 40.0, 55.0, 70.0])
    m = compute_metrics(_df(y, 50 + (y - 50) * 2), n_boot=100)  # predictions twice as spread
    assert m["calib_slope"] == pytest.approx(0.5)
    perfect = compute_metrics(_df(y, y), n_boot=100)
    assert perfect["mae"] == 0 and perfect["r2"] == 1
    assert perfect["calib_slope"] == pytest.approx(1.0)


def test_undefined_values_are_nan():
    m = compute_metrics(_df([55, 60, 70], [50, 62, 68]), n_boot=100)  # no EF < 50
    assert math.isnan(m["auroc_lt50"]) and math.isnan(m["mae_le40"])
    with pytest.raises(ValueError):
        compute_metrics(_df([50], [51]))
