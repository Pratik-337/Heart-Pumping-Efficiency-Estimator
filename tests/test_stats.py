"""Bootstrap CIs and the Wilcoxon-Holm family (task B3)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

from echoef.eval.stats import WILCOXON_COLUMNS, bootstrap_ci, wilcoxon_holm


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    x = np.random.default_rng(1).normal(5, 2, 400)
    lo, hi = bootstrap_ci(x, n_boot=2000, seed=0)
    assert lo < x.mean() < hi
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.15)  # close to the normal interval
    assert bootstrap_ci(x, n_boot=2000, seed=0) == (lo, hi)
    assert bootstrap_ci(np.full(10, 3.0), n_boot=100) == (3.0, 3.0)
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))


def test_wilcoxon_holm():
    rng = np.random.default_rng(0)
    n = 200
    base = rng.uniform(0, 10, n)
    pairs = pd.concat(
        [
            pd.DataFrame({"comparison": "clear", "file_name": range(n), "err_a": base,
                          "err_b": base + 1 + rng.normal(0, 0.5, n)}),
            pd.DataFrame({"comparison": "none", "file_name": range(n), "err_a": base,
                          "err_b": base + rng.normal(0, 0.5, n)}),
            pd.DataFrame({"comparison": "same", "file_name": range(n), "err_a": base,
                          "err_b": base}),
        ]
    )  # fmt: skip
    out = wilcoxon_holm(pairs).set_index("comparison")
    assert list(out.reset_index().columns) == WILCOXON_COLUMNS
    clear = out.loc["clear"]
    assert clear["significant"] and clear["median_diff"] < 0 and clear["ci_hi"] < 0
    diff = pairs[pairs.comparison == "clear"].eval("err_a - err_b")
    assert clear["p"] == pytest.approx(wilcoxon(diff).pvalue)
    assert out.loc["same", "p"] == 1.0 and not out.loc["same", "significant"]
    # Holm: the smallest p is multiplied by the family size (3), the next by 2, ...
    ordered = out.sort_values("p")
    assert ordered["p_holm"].iloc[0] == pytest.approx(min(1, 3 * ordered["p"].iloc[0]))
    assert (out["p_holm"] >= out["p"]).all()
