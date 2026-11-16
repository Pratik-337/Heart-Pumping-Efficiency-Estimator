"""Report tables and macros from synthetic results (task B4/A10)."""

import re

import pandas as pd

from echoef.contracts import CONFIGS
from echoef.eval import tables as tb


def test_macro_names_are_letters_only():
    assert tb.macro_name("mae", "fs32", "TEST", "M1") == "maeFsThirtyTwoTestMOne"
    assert (
        tb.macro_name("mae_ci_lo", "b0-fs8", "VAL", "M2", "ens") == "maeciloBzeroFsEightValMTwoEns"
    )
    assert tb.macro_name("gflops_video_m2", "fw64") == "gflopsvideomTwoFwSixtyFour"


def test_tables_and_numbers(tmp_path):
    rows = [{"config_id": c, "regime": r, "T": t, "period": p, "seed": s, "split": "TEST",
             "mode": "M1", "n": 10, "mae": 4.0, "mae_ci_lo": 3.8, "mae_ci_hi": 4.2, "rmse": 5.0,
             "r2": 0.8, "bias": 0.1, "loa_lo": -9.0, "loa_hi": 9.0, "auroc_lt50": 0.95,
             "auroc_le40": 0.96}
            for c, (r, t, p) in CONFIGS.items() for s in (0, 1, "ens")]  # fmt: skip
    pd.DataFrame(rows).to_csv(tmp_path / "all_runs.csv", index=False)
    written = tb.write_tables(tmp_path)
    assert {p.name for p in written} == {"table_configs.tex", "numbers.tex", "table_main.tex"}
    numbers = (tmp_path / "numbers.tex").read_text()
    names = re.findall(r"\\newcommand\{\\(\w+)\}", numbers)
    assert "maeFsThirtyTwoTestMOne" in names and "maeFsThirtyTwoTestMOneEns" in names
    assert len(names) == len(set(names))  # no duplicate macro
    assert not any(ch.isdigit() for n in names for ch in n)
    main = (tmp_path / "tables" / "table_main.tex").read_text()
    assert main.count("\\\\") == 1 + len(CONFIGS)  # header + one row per config
    assert "FS-8" in (tmp_path / "tables" / "table_configs.tex").read_text()


def test_dataset_macros_from_eda(tmp_path):
    eda = tmp_path / "eda"
    eda.mkdir()
    cols = [
        "split",
        "n",
        "n_has_trace",
        "frames_median",
        "frames_min",
        "frames_max",
        "ef_mean",
        "ef_sd",
        "n_ef_le40",
        "n_ef_41_49",
        "n_ef_ge50",
        "edes_gap_median",
        "fps_median",
    ]
    pd.DataFrame(
        [["ALL", 10030, 9931, 171.0, 28, 1002, 55.75, 12.37, 1264, 982, 7784, 16.0, 50.0]],
        columns=cols,
    ).to_csv(eda / "split_summary.csv", index=False)
    for name, col in (("padding_share.csv", "share_padded"), ("edes_fit.csv", "share_skip")):
        pd.DataFrame({"config_id": ["fs64"], "split": ["ALL"], col: [0.131]}).to_csv(
            eda / name, index=False
        )
    tex = tb.eda_tex(eda)
    assert "\\newcommand{\\edaNAll}{10,030}" in tex
    assert "\\newcommand{\\edaEfmeanAll}{55.8}" in tex
    assert "\\newcommand{\\edaSharepaddedFsSixtyFour}{13.1}" in tex
    assert "\\newcommand{\\edaFpsmedianAll}{50}" in tex
