"""Report figures from synthetic all_runs tables (task B4; no GPU, no real data)."""

import numpy as np
import pandas as pd
import pytest

from echoef.contracts import CONFIGS
from echoef.eval import figures as fg


@pytest.fixture
def table():
    rows = []
    for cid, (regime, t, period) in CONFIGS.items():
        for prefix in ("", "b0-"):
            for mode in ("M1", "M2"):
                m = 4.0 + 8 / t + (1.5 if prefix else 0)
                rows.append({"config_id": prefix + cid, "regime": regime, "T": t,
                             "period": period, "seed": 0, "split": "VAL", "mode": mode, "mae": m,
                             "mae_ci_lo": m - 0.2, "mae_ci_hi": m + 0.2, "mae_le40": m * 1.5,
                             "mae_41_49": m * 1.2, "mae_ge50": m * 0.9, "gflops_clip": 10.0 * t,
                             "gflops_video_m2": 40.0 * t})  # fmt: skip
    return pd.DataFrame(rows)


def test_figures_are_one_column_wide(table):
    for fig in (fg.mae_vs_t(table, "VAL", "M1"), fg.pareto(table, "VAL"),
                fg.mae_by_band(table, "VAL", "M1")):  # fmt: skip
        assert fig.get_size_inches()[0] == pytest.approx(fg.WIDTH_IN)


def test_mae_vs_t_has_four_series_with_legend(table):
    ax = fg.mae_vs_t(table, "VAL", "M1").axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["FS, trained", "FS, B0", "FW, trained", "FW, B0"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["8", "16", "32", "64"]


def test_pareto_front_is_non_dominated():
    pts = [(1.0, 5.0, "a"), (2.0, 4.0, "b"), (3.0, 4.5, "c"), (4.0, 3.0, "d"), (0.5, 6.0, "e")]
    assert [p[2] for p in fg._pareto_front(pts)] == ["e", "a", "b", "d"]


def test_bars_and_bland_altman(table):
    ax = fg.mae_by_band(table, "VAL", "M1").axes[0]
    assert len(ax.patches) == 3 * len(CONFIGS)
    y = np.linspace(20, 80, 50)
    preds = pd.DataFrame({"ef_true": y, "ef_pred": y + 2.0})  # constant bias of +2
    ax = fg.bland_altman(preds, "t").axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any("bias" in t and "+2.0" in t for t in texts)


def test_text_is_at_least_8_pt(table):
    fig = fg.mae_vs_t(table, "VAL", "M1")
    sizes = [t.get_fontsize() for t in fig.findobj(match=lambda a: hasattr(a, "get_fontsize"))]
    assert min(sizes) >= 8
