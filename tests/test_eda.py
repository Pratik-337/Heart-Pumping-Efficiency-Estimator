"""Aggregated EDA tables on a hand-made index."""

import pandas as pd
import pytest

from echoef.contracts import INDEX_COLUMNS
from echoef.data.eda import edes_fit, ef_band_counts, padding_share, split_summary

INDEX = pd.DataFrame(
    {
        "FileName": ["a", "b", "c", "d", "e"],
        "Split": ["TRAIN", "TRAIN", "VAL", "TEST", "TEST"],
        "EF": [40.0, 45.0, 50.0, 30.0, 70.0],
        "EDV": [100.0] * 5,
        "ESV": [50.0] * 5,
        "FPS": [50] * 5,
        "n_frames": [15, 100, 200, 64, 0],  # e is not cached
        "ed_frame": [0, 10, 100, 5, -1],
        "es_frame": [10, 40, 20, 60, -1],  # gaps 10, 30, 80, 55
        "has_trace": [True, True, True, True, False],
    }
)[INDEX_COLUMNS]


def _row(df: pd.DataFrame, **match) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in match.items():
        mask &= df[key] == value
    assert mask.sum() == 1
    return df[mask].iloc[0]


def test_band_edges_are_inclusive_as_specified():
    ef = pd.Series([40.0, 40.1, 49.9, 50.0])
    assert ef_band_counts(ef, (40, 50)) == {"n_ef_le40": 1, "n_ef_41_49": 2, "n_ef_ge50": 1}


def test_split_summary():
    s = split_summary(INDEX, (40, 50))
    assert list(s["split"]) == ["TRAIN", "VAL", "TEST", "ALL"]
    test = _row(s, split="TEST")
    assert (test["n"], test["n_cached"], test["n_has_trace"]) == (2, 1, 1)
    assert test["frames_max"] == 64  # the uncached video does not count
    every = _row(s, split="ALL")
    assert (every["n_ef_le40"], every["n_ef_41_49"], every["n_ef_ge50"]) == (2, 1, 2)
    assert _row(s, split="TRAIN")["edes_gap_median"] == pytest.approx(20.0)


def test_padding_share_counts_videos_shorter_than_window():
    p = padding_share(INDEX)
    assert _row(p, config_id="fs8", split="TRAIN")["n_padded"] == 1  # 15 < 16
    fs64_test = _row(p, config_id="fs64", split="TEST")  # 64 < 128; uncached video excluded
    assert (fs64_test["n"], fs64_test["n_padded"]) == (1, 1)
    assert _row(p, config_id="fs64", split="ALL")["share_padded"] == pytest.approx(3 / 4)


def test_edes_fit_skips_when_gap_reaches_window():
    e = edes_fit(INDEX)
    assert _row(e, config_id="fs8", split="TRAIN")["n_skip"] == 1  # gap 30 >= 16
    assert _row(e, config_id="fw8", split="VAL")["n_skip"] == 1  # gap 80 >= 64
    assert _row(e, config_id="fw8", split="TEST")["n_skip"] == 0  # gap 55 < 64
