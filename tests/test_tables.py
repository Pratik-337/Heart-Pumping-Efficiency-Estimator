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


def _all_runs(split="TEST"):
    """Two seeds and an ensemble of every config, plus Tier 0, in M1 and M2; fw16 is best."""
    rows = []
    for c, (r, t, p) in CONFIGS.items():
        for s in (0, 1, "ens", "b0"):
            for mode in ("M1", "M2"):
                base = 4.0 + 0.1 * t / 8 - (0.5 if c == "fw16" else 0.0)
                mae = base + (0.3 if mode == "M1" else 0.0) + (0.1 if s == 1 else 0.0)
                rows.append(
                    {
                        "config_id": ("b0-" + c) if s == "b0" else c,
                        "regime": r,
                        "T": t,
                        "period": p,
                        "seed": 0 if s == "b0" else s,
                        "split": split,
                        "mode": mode,
                        "n": 1276,
                        "mae": mae + (2.0 if s == "b0" else 0.0),
                        "mae_ci_lo": mae - 0.2,
                        "mae_ci_hi": mae + 0.2,
                        "rmse": mae + 1.0,
                        "r2": 0.7,
                        "bias": -0.2,
                        "loa_lo": -9.0,
                        "loa_hi": 8.6,
                        "auroc_lt50": 0.9,
                        "auroc_le40": 0.95,
                        "mae_le40": 6.0,
                        "mae_41_49": 5.0,
                        "mae_ge50": 3.5,
                        "calib_slope": 1.1,
                        "gflops_clip": 40.5 * t / 8,
                        "gflops_video_m2": 300.0 + t,
                    }
                )
    return pd.DataFrame(rows)


def _macros(text):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text))


def test_derived_macros_follow_the_data():
    m = _macros("\n".join(tb.derived_tex(_all_runs())))
    assert m["bestConfigTestMOne"] == "FW-16" and m["bestMeanConfigTestMOne"] == "FW-16"
    assert m["worstConfigTestMOne"] == "FS-64"
    assert m["tierZeroBestConfigTestMOne"] == "FW-16"  # Tier 0 rows ranked separately
    assert m["maeMeanFwSixteenTestMOne"] == "4.05" and m["maeSdFwSixteenTestMOne"] == "0.07"
    assert m["mTwoBetterCountTest"] == "7" and m["trainedCountTest"] == "7"
    assert m["mTwoGainMaxTest"] == "0.30"
    assert "FW-16 (M1)" in m["paretoFrontTest"]  # cheapest clip and lowest M1 MAE
    assert not any(ch.isdigit() for name in m for ch in name)


def test_analysis_table_macros_and_tables(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame(
        {
            "comparison": ["fs8 vs fs16", "fs8 vs fw8"],
            "n": [1276, 1276],
            "median_diff": [0.6, -0.1],
            "ci_lo": [0.4, -0.3],
            "ci_hi": [0.8, 0.1],
            "p": [1e-9, 0.4],
            "p_holm": [2e-9, 0.4],
            "significant": [True, False],
            "split": "TEST",
            "mode": "M1",
        }
    ).to_csv(tables / "wilcoxon_holm.csv", index=False)
    pd.DataFrame(
        {
            "config_id": ["fs8"],
            "seed": [0],
            "split": ["VAL"],
            "mode": ["M1"],
            "machine": ["rtx4050"],
            "gpu_name": ["a"],
            "mae_control": [5.81],
            "main_gpu_name": ["b"],
            "mae_main": [5.85],
            "hardware_gap": [-0.04],
            "seed_gap": [-0.09],
        }
    ).to_csv(tables / "hardware_control.csv", index=False)
    pd.DataFrame(
        {
            "config_id": ["fs64", "fs8"],
            "seed": [0, 0],
            "split": "TEST",
            "mode": "M1",
            "n": 1276,
            "n_padded": [150, 0],
            "mae_all": [4.2, 5.8],
            "mae_unpadded": [4.1, 5.8],
        }
    ).to_csv(tables / "unpadded.csv", index=False)
    pd.DataFrame(
        {
            "config_id": ["fs8", "fs8", "fs32", "fs32"],
            "method": ["occlusion", "ig", "occlusion", "ig"],
            "n": [60, 60, 98, 98],
            "n_skipped": [40, 40, 2, 2],
            "hit_rate": [0.3, None, 0.62, None],
            "chance_rate": [0.45, None, 0.25, None],
            "hit_p": [0.9, None, 0.0001, None],
            "aopc_ratio": [0.9, None, 1.8, None],
            "lv_energy_ratio": [None, 1.4, None, 2.1],
        }
    ).to_csv(tables / "xai_summary.csv", index=False)
    pd.DataFrame(
        {
            "config_id": "fs32",
            "method": ["ig", "ig"],
            "stage": ["fc", "stem"],
            "n": [20, 20],
            "spearman_mean": [0.9, 0.05],
            "spearman_sd": [0.1, 0.1],
        }
    ).to_csv(tables / "xai_sanity.csv", index=False)
    pd.DataFrame(
        {
            "config_id": ["fs8", "fw16"],
            "n_seeds": [2, 2],
            "split": "VAL",
            "mode": "M2",
            "mae": [4.9, 3.94],
            "gflops_video_m2": [None, None],
            "pareto": [None, None],
            "chosen": [False, True],
        }
    ).to_csv(tables / "demo_choice.csv", index=False)
    m = _macros("\n".join(tb.tables_tex(tables)))
    assert m["wilSigFsEightFsSixteenTest"] == "significant"
    assert (
        m["wilDirFsEightFsSixteenTest"] == "larger" and m["wilDirFsEightFwEightTest"] == "smaller"
    )
    assert m["wilPFsEightFsSixteenTest"] == "$<$0.001" and m["wilPFsEightFwEightTest"] == "0.400"
    assert m["wilSigCountTest"] == "1" and m["wilCountTest"] == "2"
    assert m["hwGapAbsFsEightValMOne"] == "0.04" and m["hwSeedGapAbsFsEightValMOne"] == "0.09"
    assert m["unpadNFsSixtyFourTestMOne"] == "150" and "unpadNFsEightTestMOne" not in m
    assert m["xaiHitFsThirtyTwoOcclusion"] == "62" and m["xaiPFsThirtyTwoOcclusion"] == "$<$0.001"
    assert m["xaiSkippedFsEightOcclusion"] == "40" and m["xaiLvFsThirtyTwoIg"] == "2.10"
    assert m["xaiHitSigCount"] == "1" and m["xaiAopcAboveOneCount"] == "1"
    assert m["xaiLvAboveOneCount"] == "2" and m["xaiConfigCount"] == "2"
    assert m["sanityIgStem"] == "0.05" and m["sanityConfig"] == "FS-32"
    assert m["demoConfig"] == "FW-16" and m["demoSeeds"] == "2"
    wil = tb.table_wilcoxon(pd.read_csv(tables / "wilcoxon_holm.csv"))
    assert "FS-8 vs FS-16" in wil and "$<$0.001" in wil and wil.count("\\\\") == 3


def test_full_write_and_report_check(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    results = tmp_path / "results"
    (results / "tables").mkdir(parents=True)
    _all_runs().to_csv(results / "all_runs.csv", index=False)
    cost = pd.DataFrame(
        [
            {
                "config_id": c,
                "T": t,
                "period": p,
                "params": 31_300_638,
                "gflops_clip": 40.5 * t / 8,
                "mean_clips_m2": 4.0,
                "gflops_video_m2": 162.0 * t / 8,
                "latency_ms_m1": 5.0,
                "latency_ms_m2": 20.0,
                "peak_vram_mb_infer": 300.0,
                "peak_vram_mb_train": 1500.0 * t / 8,
                "micro_batch": 8,
                "epoch_time_s": 97.0 * t / 8,
                "gpu_name": "RTX",
            }
            for c, (_, t, p) in CONFIGS.items()
        ]
    )
    cost.to_csv(results / "cost.csv", index=False)
    written = {p.name for p in tb.write_tables(results)}
    assert {"numbers.tex", "table_main.tex", "table_cost.tex"} <= written
    m = _macros((results / "numbers.tex").read_text())
    assert m["paramsMillions"] == "31.3" and m["epochtimesFsEight"] == "97"
    assert m["nFsEightTestMOne"] == "1,276"
    main = (results / "tables" / "table_main.tex").read_text()
    assert main.count("\\\\") == 1 + len(CONFIGS) and "$\\pm$" in main
    assert (results / "tables" / "table_cost.tex").read_text().count("\\\\") == 2 + len(CONFIGS)

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_report.py"
    spec = importlib.util.spec_from_file_location("check_report", script)
    check_report = importlib.util.module_from_spec(spec)
    sys.modules["check_report"] = check_report
    spec.loader.exec_module(check_report)
    tex = (
        "\\res{bestConfigTestMOne} \\res{noSuchMacro} \\inputtable{table_main} "
        "\\inputtable{table_xai} \\resfig{mae_vs_t}{0.5}"
    )
    missing = check_report.check(tex, results)
    assert missing["macros"] == ["noSuchMacro"]
    assert missing["tables"] == ["table_xai"] and missing["figures"] == ["mae_vs_t"]
    assert check_report.check(tex, results, figures=False)["figures"] == []
