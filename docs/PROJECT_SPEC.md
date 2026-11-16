# Heart Pumping Efficiency Estimator — Project Spec & Build Plan

Nov 14, 2026 · @Master Bruce

## How to use this doc

This doc is the single source of truth (SSOT) for the Machine Learning semester project. If the code and this doc disagree, the doc wins until a Decision log entry changes it.

**Version 2, 14 Nov evening: re-scoped for the 18 Nov deadline and a two-person team.** Training drops to 10 epochs with 1 guaranteed seed. A zero-training experiment (Tier 0) is added as a safety net. A 4-day plan now splits the work between Pratik (A) and the partner (B), with fixed interface contracts and a step-by-step merge guide. Changed sections: Assumptions, Experiment design, Model and training, Explainability, Compute budget, 4-day plan, Interface contracts, Git workflow, IEEE report plan, Claude Code rules, Risks, Decision log.

- **Readers:** you (builder and presenter), the faculty reviewer, and Claude Code (implementer).
- **For Claude Code:** export this doc as Markdown and commit it as `docs/PROJECT_SPEC.md`. The repo's `CLAUDE.md` (section "Claude Code operating rules") points to it.
- **Requirement words:** MUST = required for submission; SHOULD = do unless time runs out; MAY = stretch goal.
- **IDs:** days are Day 0–Day 4, tasks are A1–A12 (Pratik) and B1–B10 (partner), decisions are D-nn, configs are FS-/FW-\<frames>. Commits and issues reference these IDs.
- **Changing the plan:** add a row to the Decision log first, then change the code. Never edit a result number by hand; every number in the report comes from `results/all_runs.csv`.

## Project summary and research questions

The project predicts left-ventricular ejection fraction (EF) from EchoNet-Dynamic echocardiogram videos with a fine-tuned R(2+1)D-18 video network. Its contribution is a controlled, cost-measured study of how many frames the model needs, plus explanations scored with numbers rather than judged by eye.

**What EF is.** EF is the share of blood the left ventricle pushes out per beat. It is computed from the end-diastolic volume (EDV, ventricle fullest) and end-systolic volume (ESV, ventricle emptiest):

```latex
\mathrm{EF} = \frac{\mathrm{EDV} - \mathrm{ESV}}{\mathrm{EDV}} \times 100\%
```

Worked example with the first row of the dataset's label file: EDV = 49.22 mL, ESV = 15.07 mL. Step 1: 49.22 − 15.07 = 34.15 mL ejected. Step 2: 34.15 ÷ 49.22 = 0.6938. Step 3: × 100 = 69.4%, which matches the stored EF of 69.37% ([FileList sample](https://huggingface.co/spaces/alaa-lab/Med-Real2Sim/raw/971afbd4c052213d64326d531fb3e4669baf404f/EchoNet-Dynamic/FileList.csv)).

**Honest positioning.** EchoNet-Dynamic's own paper already swept clip length (1–96 frames at period 1) and sampling period (1–8 at a fixed 64-frame window), but plotted only validation loss: no test metrics, no runtime or memory, one run each ([plot script](https://huggingface.co/spaces/alaa-lab/Med-Real2Sim/blob/d3d13a82cb49ea718ab68610de5838ed0eb635c4/dynamic/scripts/plot_hyperparameter_sweep.py)). This project must not claim the frame sweep itself is new. It claims a test-set, multi-seed, cost-measured and explanation-linked version of it, run on an 8 GB consumer GPU.

**Research questions**

1. **RQ1 — Frame efficiency.** How does test MAE change with clip length T ∈ {8, 16, 32, 64}, and how much of the change comes from sampling density versus how much of the heartbeat the clip covers?
2. **RQ2 — Cost.** What are the GFLOPs, latency, peak VRAM and training time of each configuration on an RTX 4060, and which configurations sit on the accuracy–cost Pareto frontier?
3. **RQ3 — Inference strategy.** Does averaging several clips per video change which configuration is best, and what does it cost?
4. **RQ4 — Explainability.** Do frame and pixel attributions concentrate on end-diastole/end-systole frames and inside the left ventricle more than chance, do they pass faithfulness and sanity tests, and how does this change with T?

**Deliverables (MUST):** trained models and results tables; a local prototype that takes a video and returns EF; an IEEE-style report; a clean GitHub repo (code only, no data).

**Non-goals:** beating state of the art, clinical validation, LV segmentation, EchoNet's beat-by-beat pipeline, and external-dataset testing.

## Assumptions and open inputs

The plan now assumes 4 days to an 18 Nov deadline, two people, the dataset already on Pratik's laptop, an IEEE report, and an RTX 4060 Laptop GPU. Rows marked "given" come from you; the rest are assumptions with their effect stated.

| Item | Known or assumed | Effect on the plan |
| --- | --- | --- |
| Deadline | 18 Nov 2026 (given). Assumed end of day; the plan finishes by Wed 12:00 to leave a buffer | About 48 GPU-hours available, Sun 10:00 to Tue 10:00 |
| Team | 2 people (given). A = Pratik, who has the GPU laptop and the data; B = partner | Work split by file ownership so merges don't conflict |
| Dataset | Downloaded from the Stanford AIMI portal by Pratik (given) | Data never leaves A's laptop; B builds against synthetic data |
| Report | IEEE format (given). Assumed IEEEtran conference template, two columns, about 6 pages | See IEEE report plan |
| GPU | RTX 4060 **Laptop** GPU, 8 GB (assumed, because you said the merge happens on your laptop) | Runs about 1.30× slower than desktop figures. Two more laptops (RTX 4050, RTX 3050 4 GB, given) share the training runs; see Compute budget, Step 7 |
| Partner's machine | Any laptop; no NVIDIA GPU needed; Windows (with WSL), macOS or Linux | B's code is tested on CPU with synthetic videos |
| Frame list | {8, 16, 32, 64} fixed by the proposal | — |
| Disk and RAM on A's laptop | ≥ 16 GB RAM; ≥ 40 GB free inside WSL | Cache 21.8 GB + raw videos + checkpoints |

**Inputs still needed**

- [ ] Exact submission time on 18 Nov and the page limit.
- [ ] Partner's name (to replace "B"), their operating system, and their GitHub username.
- [ ] Confirm the GPU is the laptop RTX 4060.
- [ ] Is there a demo or viva on 18 Nov, or only the report?
- [ ] Run `nvidia-smi` on the RTX 4050 laptop and note the exact GPU name and memory. The RTX 4050 Laptop GPU normally has 6 GB, not 8; an 8 GB card may be a different model.

## Dataset: EchoNet-Dynamic

EchoNet-Dynamic provides 10,030 apical four-chamber echo videos at 112×112 pixels, one per patient, each with EF, EDV and ESV labels and expert LV tracings at end-diastole (ED) and end-systole (ES). The project uses the official split stored in `FileList.csv`.

| Property | Value | Source |
| --- | --- | --- |
| Videos | 10,030 apical-4-chamber videos, Stanford, 2016–2018, one per patient | [EchoNet site](https://echonet.github.io/dynamic/), [Nature paper](https://utstat.toronto.edu/reid/sta2101f/s41586-020-2145-8.pdf) |
| Frame size | 112×112, cropped and masked to remove text and identifiers | [EchoNet site](https://echonet.github.io/dynamic/) |
| Frames per video | 24–1,002; mean about 173 (SD 47) | [ML4H 2019 paper](https://douyang.github.io/NeuroIPS_2019_ML4H%20Workshop_Paper.pdf), [PMC analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11450854) |
| Frame rate | Mean about 51 fps (sample rows show 50 and 53) | [ML4H 2019 paper](https://douyang.github.io/NeuroIPS_2019_ML4H%20Workshop_Paper.pdf) |
| EF distribution | Mean about 55%, SD about 12, left-skewed | [PMC analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11450854) |
| Split | 7,465 train / 1,288 val / 1,277 test | [arXiv 2609.19730](https://arxiv.org/pdf/2609.19730) |
| Label files | `FileList.csv`: FileName, EF, ESV, EDV, FrameHeight, FrameWidth, FPS, NumberOfFrames, Split. `VolumeTracings.csv`: FileName, X1, Y1, X2, Y2, Frame | [FileList sample](https://huggingface.co/spaces/alaa-lab/Med-Real2Sim/raw/971afbd4c052213d64326d531fb3e4669baf404f/EchoNet-Dynamic/FileList.csv), [EchoNet loader](https://huggingface.co/spaces/alaa-lab/Med-Real2Sim/blob/main/echonet/datasets/echo.py) |
| Missing tracings | A small number of videos lack tracings; EchoNet's loader drops them | [EchoNet loader](https://huggingface.co/spaces/alaa-lab/Med-Real2Sim/blob/main/echonet/datasets/echo.py) |
| Access and licence | Individual registration on the Stanford AIMI portal; non-commercial research only; no redistribution; no clinical use | [EchoNet site](https://echonet.github.io/dynamic/) |

The Nature paper's text lists 1,277 validation and 1,288 test patients, while its figures use n = 1,277 for test. The `Split` column is authoritative; Task A2 logs the actual counts.

**Preprocessing pipeline (task A2, Saturday night)**

1. **Verify.** Every `FileName` has a video; every video decodes; the decoded frame count matches `NumberOfFrames`. Mismatches go to `data/processed/qc_report.csv`. *Reason:* silent decode failures corrupt training without raising errors.
2. **Check channels.** Compare the R, G and B channels of every video and log the result. They are not strictly identical, so the cache stores OpenCV's luma conversion (D-25). *Reason:* echo B-mode is grayscale; one channel cuts storage by 3×.
3. **Cache decoded frames.** Save each video as `uint8` array (frames, 112, 112) in `data/processed/videos/<FileName>.npy`, read later with NumPy memory-mapping. *Reason:* a memory-mapped read loads only the 8–64 frames a clip needs; AVI decoding must read from the start of the file.
4. **Label ED and ES by area, not file order.** Build each traced LV polygon (EchoNet's method: skip each frame's first row, which is the long axis, then take the X1,Y1 points followed by the X2,Y2 points reversed) and compute its area. The larger-area frame is ED, the smaller is ES. Videos whose traced frames are not one regular tracing each get no ED/ES (D-26). *Reason:* EchoNet's loader infers ED/ES from row order, which is fragile.
5. **Check labels.** Recompute EF from EDV and ESV; flag rows where it differs from stored EF by more than 1 point.
6. **Normalisation statistics.** Compute pixel mean and SD on the training split only (the videos that are trained on, D-27) and save `stats.json`. *Reason:* using validation or test pixels would leak information.
7. **Index.** Write `data/processed/index.parquet` with FileName, EF, EDV, ESV, FPS, decoded frame count, Split, ED frame, ES frame, has\_trace.

**Storage estimate.** One frame is 112 × 112 = 12,544 bytes. A 173-frame video is 173 × 12,544 = 2,170,112 bytes (2.17 MB). All 10,030 videos are 10,030 × 2.17 MB = 21.8 GB for the grayscale cache. RGB would be 3× that, 65.3 GB.

**Short videos.** A clip needs T × p raw frames. EchoNet pads short videos with zero frames (zero = mean grey after normalisation). This plan keeps that rule for comparability, then reports test MAE on the unpadded subset as a sensitivity check. A rough normal approximation (mean 173, SD 47) suggests the share of videos shorter than the window is:

| Window (raw frames) | z = (window − 173) ÷ 47 | Approx. share padded |
| --- | --- | --- |
| 16 | −3.34 | 0.04% |
| 32 | −3.00 | 0.13% |
| 64 | −2.32 | 1.0% |
| 128 | −0.96 | 16.9% |

The 128-frame window (FS-64) may pad about 1 in 6 videos, so its sensitivity check matters most. Frame counts are skewed, not normal; Task A2 replaces these with exact counts.

## Improvements over the original proposal

Ten changes turn the proposal from a frame sweep plus pictures into a controlled, statistically tested study with measured explanations. None adds a new model family, so scope and compute stay realistic.

| # | Original proposal | Improvement | Reason |
| --- | --- | --- | --- |
| 1 | "Compare 8, 16, 32 and 64 frames" as the investigation | Position the study against EchoNet's existing validation-loss sweep; claim test-set, multi-seed, cost-measured results instead | A reviewer who knows Extended Data Fig. 1 of the Nature paper would otherwise call the study a repeat |
| 2 | Frame count is the only variable | Two sampling regimes: fixed stride (period 2, window grows with T) and fixed window (64 raw frames, period shrinks) | Fewer frames at a fixed stride also means a shorter clip. Two regimes separate "sees less of the beat" from "samples the beat more sparsely" |
| 3 | "Start with an existing pretrained model where feasible" | Baseline B0 = EchoNet's released 32-frame weights, evaluated as-is. All sweep models start from the same Kinetics-400 weights | B0 checks the pipeline. Starting every T from 32-frame EchoNet weights would favour T = 32 |
| 4 | One training run per setting | Bootstrap 95% CIs and paired Wilcoxon tests with Holm correction; 1 guaranteed seed per config, up to 3 if GPU time allows | Gaps between neighbouring T can be smaller than seed-to-seed noise |
| 5 | "Processing time and computational cost" left undefined | Measure GFLOPs, latency, peak VRAM and training time with a fixed protocol; plot a Pareto frontier | "Efficiency" needs both an accuracy axis and a cost axis |
| 6 | Prediction mode unspecified | Report single-clip (M1) and multi-clip average (M2) separately | Clip averaging changes both accuracy and cost per video |
| 7 | MAE and RMSE only | Add R², Bland–Altman bias and limits, AUROC for EF < 50% and EF ≤ 40%, error by EF band | MAE hides systematic bias at low EF, where errors matter clinically |
| 8 | "Explore visualisation methods" | Score explanations: alignment with ED/ES frames, attribution mass inside the LV tracing, deletion test, weight-randomisation sanity check | Turns "looks plausible" into numbers and backs the brief's influence-not-causation caveat |
| 9 | Single EF number in the prototype | Seed ensemble gives EF ± spread when 2 or more seeds finish | The seeds are trained anyway, so the uncertainty band is free |
| 10 | Any video accepted | Input checks plus an out-of-distribution warning | A new video rarely matches the dataset's 112×112 masked apical-4-chamber format |

## Frame-efficiency experiment design

The study trains 7 configurations × 3 seeds = 21 runs, split across three laptops (seed 0 is the only MUST), and evaluates each in 2 inference modes. A zero-training experiment, Tier 0, runs first and guarantees a complete frame-count result even if training fails. Only the clip length T and sampling period p change; everything else is frozen.

**Configurations.** T = frames fed to the model; p = take every p-th raw frame; window W = T × p raw frames; seconds assume 50 fps.

| ID | Regime | T | p | W (raw frames) | Seconds | Tier |
| --- | --- | --- | --- | --- | --- | --- |
| FS-8 | Fixed stride | 8 | 2 | 16 | 0.32 | 1 (MUST) |
| FS-16 | Fixed stride | 16 | 2 | 32 | 0.64 | 1 (MUST) |
| FS-32 = FW-32 | Both (shared run) | 32 | 2 | 64 | 1.28 | 1 (MUST) |
| FS-64 | Fixed stride | 64 | 2 | 128 | 2.56 | 1 (MUST) |
| FW-8 | Fixed window | 8 | 8 | 64 | 1.28 | 2 (SHOULD) |
| FW-16 | Fixed window | 16 | 4 | 64 | 1.28 | 2 (SHOULD) |
| FW-64 | Fixed window | 64 | 1 | 64 | 1.28 | 2 (SHOULD) |

The 64-frame window at period 2 is EchoNet's tuned setting, about 1.28 s ([Nature paper](https://utstat.toronto.edu/reid/sta2101f/s41586-020-2145-8.pdf)). It anchors both regimes, which is why FS-32 and FW-32 are one run.

**Tier 0 — frame reduction without retraining (MUST, runs first on Sunday).** Evaluate EchoNet's released 32-frame weights (B0) at all 7 (T, p) settings on validation and test, in M1 and M2. It answers a deployment question: can a model trained on 32 frames simply be fed fewer frames? It needs no training, about 40 minutes of GPU time, and gives the report a full frame-count curve even if the sweep breaks. Expect errors to rise away from T = 32, p = 2, since that is the only setting B0 has seen.

**Why coverage matters.** One heartbeat lasts 60 ÷ HR seconds, so a window of w seconds holds a full beat only when HR ≥ 60 ÷ w.

- FS-8, 0.32 s: needs HR ≥ 60 ÷ 0.32 = 187.5 bpm. At normal heart rates it never sees a full beat, so it may miss ED or ES.
- FS-16, 0.64 s: needs HR ≥ 93.75 bpm. Covers a full beat only in fast hearts.
- 64-frame window, 1.28 s: needs HR ≥ 46.9 bpm. Covers a full beat in almost all patients.
- FS-64, 2.56 s: needs HR ≥ 23.4 bpm for one beat; covers two beats when HR ≥ 46.9 bpm.

So a drop in accuracy for FS-8 could come from missing half the beat, not from having only 8 frames. FW-8 sees a whole 1.28 s window with 8 frames and answers which one it is.

**Held constant across all runs:** architecture, Kinetics-400 initialisation, optimiser and learning-rate schedule, epoch budget and early-stopping rule, micro-batch 8 and effective batch 16, augmentation, loss, data split, normalisation statistics, and checkpoint selection rule.

**Inference modes**

- **M1, single clip:** one clip starting at the video's centre minus W ÷ 2 (deterministic). This isolates each model's per-clip ability.
- **M2, multi-clip average:** clips start every W ÷ 2 frames (50% overlap) across the whole video; the prediction is their mean. Clip count n = ⌊(F − W) ÷ (W ÷ 2)⌋ + 1, or 1 if F < W.

Worked example for an average 173-frame video under FS-32 (W = 64): (173 − 64) = 109; 109 ÷ 32 = 3.4, floor 3; 3 + 1 = 4 clips. Under FS-8 (W = 16): (173 − 16) ÷ 8 = 19.6, floor 19; 19 + 1 = 20 clips. Per-video cost therefore differs from per-clip cost; the Compute budget section works it out.

**Pre-specified analysis**

- **Primary endpoint:** test MAE in M1. *Reason:* compares the models' per-clip ability without the confound of different clip counts. Fixing it before training prevents choosing the metric after seeing results.
- **Secondary endpoints:** M2 test MAE, RMSE, R², Bland–Altman, AUROC, and all cost metrics.
- **Uncertainty:** 95% bootstrap CI over test videos (10,000 resamples) for every config; mean ± SD across seeds only for configs where 2 or 3 seeds finished.
- **Hypothesis tests:** paired Wilcoxon signed-rank tests on per-video absolute errors (seed 0, or seed-averaged where more seeds finished). Family of 9 comparisons: neighbouring T within FS (3), neighbouring T within FW (3), and FS vs FW at T = 8, 16, 64 (3). Holm–Bonferroni correction at α = 0.05. Report the median paired difference with its bootstrap CI as the effect size.
- **Test-set lock:** model selection uses validation only. The test split is evaluated once, on Tue 17 Nov at about 07:00, after training stops and the models-frozen tag exists.

## Model and training design

Every sweep run fine-tunes torchvision's R(2+1)D-18 from Kinetics-400 weights with a one-output regression head, AdamW, cosine learning-rate decay, bf16 mixed precision and an effective batch of 16. The table gives the reason for each choice.

| Decision | Choice | Reason |
| --- | --- | --- |
| Backbone | `torchvision.models.video.r2plus1d_18`, Kinetics-400 weights: 31,505,325 parameters, 40.52 GFLOPS per 16×112×112 clip ([torchvision docs](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.r2plus1d_18.html)) | EchoNet chose it over R3D and MC3 in its sweep, and Kinetics pretraining helped slightly ([Nature paper](https://utstat.toronto.edu/reid/sta2101f/s41586-020-2145-8.pdf)). It fits 8 GB with bf16 and accepts any T through adaptive pooling |
| Head | Replace `fc` with `Linear(512, 1)`; set its bias to the training-set mean EF | EchoNet does the same with 55.6 ([training code](https://github.com/echonet/dynamic/blob/master/echonet/utils/video.py)). Starting at the mean avoids a huge first-epoch loss |
| Input | Grayscale frame copied to 3 channels; normalised with training-set mean and SD | The pretrained first layer expects 3 channels |
| Loss | Mean squared error | Same as EchoNet, so results stay comparable. MAE is reported, not optimised |
| Optimiser | AdamW, weight decay 0.01, 1 warm-up epoch, cosine decay over 10 epochs; keep the epoch with the best validation MAE | Adaptive optimisers are less sensitive to one shared learning rate across 7 configurations. 10 epochs (EchoNet used 45) is what a 4-day deadline allows; absolute MAE will be higher than EchoNet's, but every config gets the same budget |
| Learning rate | Fixed at 1e-4 for all runs, the value EchoNet also used (with SGD); no tuning sweep | No time for a sweep; one shared rate keeps the comparison fair. Limitation: it may be sub-optimal for every config, so absolute errors are conservative |
| Batch | Micro-batch 8, gradient accumulation 2 = effective batch 16 (the Nature paper's batch size) | BatchNorm statistics depend on micro-batch size, so it stays 8 for every T |
| Memory at T = 64 | Activation checkpointing on each residual block of the 4 stages, because micro-batch 8 does not fit without it (measured in A4; per-stage checkpointing was not enough) | Trades extra compute for memory without changing the maths. Fallback: micro-batch 4 with frozen BatchNorm, logged as a deviation |
| Precision | `torch.autocast` with bf16; TF32 on for remaining FP32 maths | The RTX 4060 (Ada) has bf16 tensor cores; bf16 needs no loss scaling, unlike fp16 |
| Augmentation (train only) | Random clip start; 12-pixel zero padding then random 112×112 crop | EchoNet's augmentation ([Nature paper](https://utstat.toronto.edu/reid/sta2101f/s41586-020-2145-8.pdf)) simulates small probe shifts. No horizontal flip: it would put the LV on the wrong side of an apical four-chamber view |
| Epoch | One random clip per training video | Same number of gradient updates for every T |
| Checkpoint | Keep the epoch with best validation MAE (M1) | Matches the primary endpoint |
| Seeds | Seed 0 for every config; seeds 1 and 2 only if GPU time remains. Seeds set for Python, NumPy, PyTorch and DataLoader workers; cuDNN not forced deterministic | Forced determinism slows 3D convolutions. Extra seeds, where they finish, measure run-to-run noise |
| `torch.compile` | Off by default; turn on only if the A4 timing run shows ≥ 15% speed-up | Compile time and Triton issues under WSL are a risk for little gain |

**Memory budget, first part.** Training state per parameter is 4 bytes (weight) + 4 (gradient) + 8 (two AdamW moments) = 16 bytes. 31,505,325 × 16 = 504 MB. That leaves about 7.5 GB for activations, which grow with T × micro-batch. Task A4 measures the real peak for each T.

**Alternatives considered and rejected**

- **Video transformers** such as EchoCoTr: test MAE 3.95 vs EchoNet's 4.05 ([EchoCoTr](https://ar5iv.labs.arxiv.org/html/2209.04242)). The 0.1-point gain does not justify more VRAM and training time on 8 GB.
- **Echo foundation models** such as EchoPrime (trained on 12 million videos, [paper](https://ai.stanford.edu/~bryanhe/publications/echoprime.pdf)) and PanEcho (EF MAE 4.4% internal, [preprint](https://medrxiv.org/content/10.1101/2024.11.16.24317431v1.full)). They target multi-view studies and would turn the project into model adaptation.
- **EchoGNN**, explainable by design ([MICCAI 2022](https://github.com/MasoudMo/echognn)). Changing architecture would confound the frame study; it is cited as related work for the XAI discussion.

## Evaluation protocol and metrics

Every model is scored on the same accuracy, agreement and cost metrics by one script, and the results land in `results/all_runs.csv`. EF errors are in EF percentage points (a prediction of 52% against a label of 55% is an error of 3 points).

**Accuracy metrics** (y = reference EF, ŷ = prediction, n = number of videos):

```latex
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n} |\hat{y}_i - y_i| \qquad \mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2} \qquad R^2 = 1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_i - \bar{y})^2}
```

**Agreement (Bland–Altman):** bias = mean(ŷ − y); limits of agreement = bias ± 1.96 × SD(ŷ − y).

| Metric | Definition | Why it is reported |
| --- | --- | --- |
| MAE (primary) | Formula above | Directly readable in EF points; EchoNet's headline metric |
| RMSE | Formula above | Penalises large misses more than MAE |
| R² | Formula above | Share of EF variance explained; comparable to published 0.81 |
| Bland–Altman bias and limits | As above, plus plot | Standard for comparing two measurement methods; shows systematic over- or under-estimation |
| AUROC, EF < 50% | Prediction used as a score | The Nature paper's cardiomyopathy threshold ([Nature paper](https://utstat.toronto.edu/reid/sta2101f/s41586-020-2145-8.pdf)) |
| AUROC, EF ≤ 40% | Prediction used as a score | Reduced-EF cut-off in the 2021 universal definition of heart failure ([PubMed](https://www.pubmed.gov/33605000/)) |
| MAE by EF band | ≤ 40, 41–49, ≥ 50 | Low-EF patients are fewer, and errors there matter most |
| Calibration slope | Regress y on ŷ | A slope below 1 means predictions are pulled toward the mean |

**Cost metrics (fixed protocol)**

| Metric | How it is measured |
| --- | --- |
| Parameters | Count of trainable parameters |
| GFLOPs per clip and per video | `torch.utils.flop_counter.FlopCounterMode` on one clip; per video = clips (M2) × per-clip. Report the convention: 1 multiply-add = 2 FLOPs |
| Latency per video | Batch 1, bf16, after 10 warm-up runs, median of 50 runs, timed with CUDA events after `torch.cuda.synchronize()` |
| Throughput | Clips per second at the largest batch that fits |
| Peak VRAM | `torch.cuda.max_memory_allocated()` for inference and for one training step |
| Training time | Wall-clock per epoch and per run, from logs |
| GPU state | `nvidia-smi` clocks, temperature and power logged during benchmarks |

**How to read the numbers**

- **Published reference:** EchoNet-Dynamic reports MAE 4.1, RMSE 5.3 and R² 0.81 on its test set, using segmentation-guided beat-by-beat averaging ([Nature paper](https://utstat.toronto.edu/reid/sta2101f/s41586-020-2145-8.pdf)). Clip-level models here do not use that pipeline, so B0 is the fair reference.
- **Clinical scale:** human inter-observer variation in EF is cited as 7.6–13.9% in the same paper. A statistically significant gap of 0.3 points between two configs is clinically negligible, and the report says so.
- **External validity:** EchoNet's MAE rose from 4.1 to 6.0 on another hospital's data. Results here are single-centre.

## Explainability design

Three attribution methods run on the seed-0 models of 4 priority configs (FS-8, FW-8, FS-32, FS-64) over 100 test videos, and four numeric tests score them; the other 3 configs run only if time allows. Attributions show what moves the model's output, not what causes the heart's function; the report states this wherever a map appears.

**Sample.** 100 test videos drawn once with a fixed seed and saved to `results/xai_sample.csv`: 35 with EF ≤ 40, 15 with EF 41–49, 50 with EF ≥ 50. *Reason:* a random draw would contain few low-EF cases, where explanations matter most.

**ED/ES-centred clips.** For each video and config, the clip is placed so both traced frames fall inside it. With d = |ED − ES| raw frames, start = max(0, min(ED, ES) − (W − d) ÷ 2). If d ≥ W, both frames cannot fit; the video is skipped for that config and the skip count is reported. Systole lasts roughly 0.3 s, about 15 frames at 50 fps, so FS-8's 16-frame window will often fail to hold both. That count is itself a result for RQ1.

**Methods**

| Method | Question it answers | Resolution | Cost per clip | Settings and reason |
| --- | --- | --- | --- | --- |
| Temporal occlusion | Which frames change the EF prediction most? | One score per input frame | T + 1 forward passes | Replace one frame with the linear interpolation of its neighbours, not a black frame. A black frame is out of distribution and exaggerates importance |
| Integrated Gradients ([Captum](https://captum.ai/api/guided_grad_cam.html)) | Which pixels in which frames push EF up or down? | Every voxel | 32 steps × (forward + backward) | Baseline = all-zero input after normalisation (the mean grey video); repeat with a blurred-video baseline as a sensitivity check. Raise to 64 steps if Captum's convergence delta is large |
| 3D Grad-CAM (Captum `LayerGradCam` on `layer4`, trilinear upsampling) | Which regions does the last convolutional stage use? | layer4 is T/8 × 7 × 7 | 1 forward + backward | `relu_attributions=False`, then absolute value, because a regression output has no class and both directions matter. Used for spatial maps only |

**Grad-CAM's temporal blind spot.** R(2+1)D-18 halves the time axis in layer2, layer3 and layer4, so layer4 has T/8 time steps: 1 for T = 8, 2 for T = 16, 4 for T = 32, 8 for T = 64. At T = 8 Grad-CAM carries no frame information at all. Frame-level questions therefore use occlusion and Integrated Gradients.

**Numeric tests**

1. **Temporal plausibility (ED/ES hit rate).** Take the 2 most important frames by occlusion. A hit is a frame within ±1 sampled frame of ED or ES. Compare the hit rate to the chance rate from 1,000 random frame rankings (permutation test).
2. **Spatial plausibility (LV energy ratio).** On the ED and ES frames, ratio = (share of |attribution| inside the traced LV polygon) ÷ (LV area ÷ frame area). Above 1 means attribution concentrates on the LV more than a uniform map would.
3. **Faithfulness (deletion test).** Remove frames from most to least important (interpolation replacement), up to T ÷ 2 frames, and record |change in EF| after each step. Area over the curve is compared with 10 random deletion orders; higher than random means the ranking reflects what the model uses.
4. **Sanity check (model randomisation).** On FS-32 only, randomise weights from `fc` back to the stem, one stage at a time, on 20 videos. The Spearman rank correlation between original and randomised maps should fall toward 0. A method whose maps barely change is not explaining the model ([Adebayo et al. 2018](https://arxiv.org/abs/1810.03292)).

**Outputs:** `results/xai/<config>/xai_metrics.csv`; a figure per config with 4 example videos (one per EF band plus the worst error) showing frame overlays and the occlusion curve with ED and ES marked; one summary chart of the four test scores against T for both regimes.

**Compute estimate on the laptop GPU (100 videos; achieved throughput 8 ÷ 1.30 = 6.14 TFLOPS).** FS-64: Integrated Gradients 32 steps × 3 × 324.2 GFLOPs = 31.1 TFLOPs per clip ÷ 6.14 = 5.1 s; × 100 = 8.4 min. Occlusion 65 × 324.2 GFLOPs = 21.1 TFLOPs, 3.4 s; × 100 = 5.7 min. Deletion test ≈ occlusion, 5.7 min. FS-64 total 19.9 min. FS-32 7.1 min; FS-8 and FW-8 1.3 min each; sanity check (20 videos × 5 randomisation stages × IG at T = 32) 4.2 min. Total 19.9 + 7.1 + 1.3 + 1.3 + 4.2 = 33.8 min, about 45 min with loading.

## Prototype app

A local Gradio app takes an AVI or MP4 echo video and returns predicted EF with an uncertainty band, an EF band label, a frame-importance chart and an optional saliency video. It runs on your machine only and is never shared through a public link.

**Request flow**

1. **Upload** an AVI or MP4 through `gr.Video` ([Gradio docs](https://www.gradio.app/docs/gradio/video)).
2. **Validate:** the file decodes; frame count ≥ the model's window W (else warn and pad, as in training); frames are converted to grayscale. If not 112×112, centre-crop to square and resize, with a visible warning that the input is out of distribution.
3. **Predict** with the default model = the configuration chosen from the Pareto frontier on Tuesday (task A11), as an ensemble of whichever seeds finished (1 to 3), in M2 multi-clip mode.
4. **Show:**
   - EF % = mean of the seed predictions; band = lowest to highest seed prediction, shown only when 2 or more seeds exist. *Note:* this captures model variance only, so it understates true uncertainty.
   - EF band label: ≤ 40% reduced, 41–49% mildly reduced, ≥ 50% preserved, following the 2021 universal definition's EF cut-offs ([PubMed](https://www.pubmed.gov/33605000/)). Labelled "EF band", never a diagnosis, because heart failure also requires symptoms.
   - Frame-importance chart from temporal occlusion; Integrated Gradients overlay video on request (slower, a few seconds).
   - Latency of the prediction in ms.
5. **Reference check:** if the file name matches an EchoNet test video, show the reference EF and the absolute error. Otherwise show "No reference EF available, so no error score", as the proposal specifies.
6. **Banner:** "Research prototype. Not clinically validated. Not for diagnosis."

**Rules:** `launch(share=False)`; no dataset videos stored in the repo or the app folder; demo examples are loaded from your local data folder at run time. *Reason:* the Research Use Agreement forbids redistribution ([EchoNet site](https://echonet.github.io/dynamic/)).

**Done when:** 5 test videos and 1 synthetic non-echo video each return a result or a clear warning in under 3 s for the prediction path on the RTX 4060.

## Compute budget on the RTX 4060

**Update 16 Nov (D-29):** the RTX 4050 and RTX 3050 are not available. Every run trains on the RTX 4060 in the order of `configs/assignment.yaml`, with the run times measured in Step 4b; the three-laptop plans in Step 7 and Step 8 are kept for the record only.

Training is split across three laptops: the RTX 4060 runs every 64-frame model, and the RTX 4050 and RTX 3050 run the shorter clips. All 21 runs (7 configs × 3 seeds) finish by about Mon 20:40, or about Mon 10:50 if the 4050 can fit 64-frame training. A single 4060 would finish only 20 of 21 by Tue 05:30. Task A4 replaces these estimates with measured epoch times on each laptop on Sunday morning.

**Local GPU vs Colab/Kaggle: use the local GPU.**

- **No session limits.** The longest single run (FS-64 or FW-64, 10 epochs) is estimated at about 5 hours on the laptop GPU, longer than free notebook sessions reliably last.
- **Comparable cost numbers.** Latency and VRAM results only mean something if every config is measured on the same GPU.
- **Data stays with you.** The agreement allows a copy for your own research only and forbids sharing ([EchoNet site](https://echonet.github.io/dynamic/)). Keeping it on your machine is the simplest way to comply.
- **The 21.8 GB frame cache persists** instead of being rebuilt each session.
- **Cost of the choice:** 8 GB VRAM forces bf16, micro-batch 8 and possibly checkpointing at T = 64. A laptop GPU may throttle when hot, so clocks are logged.

**Step 1 — compute per clip.** torchvision lists 40.52 GFLOPS for a 16-frame 112×112 clip ([torchvision docs](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.r2plus1d_18.html)). This plan treats that as multiply-adds and doubles it: 81.04 GFLOPs. Every convolution runs along the time axis, so cost scales linearly with T: forward GFLOPs = 81.04 × T ÷ 16. Task A4 checks this with PyTorch's FLOP counter.

**Step 2 — compute per epoch.** Training costs about 3 forward passes per clip (1 forward + backward ≈ 2 forwards), over 7,465 training videos. Validation is 1 forward over 1,288 videos.

**Step 3 — assumed GPU throughput.** Desktop RTX 4060 peak FP32 is 15.11 TFLOPS; the laptop listing is 11.6 ([Notebookcheck](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4060-vs-NVIDIA-GeForce-RTX-4060-Laptop-GPU-vs-NVIDIA-GeForce-RTX-4060-Ti-8G_11593_11455_11592.247598.0.html)). A laptop 4060 listing shows bf16 tensor throughput at 2× FP32 (23.84 vs 11.92, [hmc-tech](https://hmc-tech.com/gpus/nvidia-geforce-rtx-4060-laptop-85w-vs-nvidia-geforce-rtx-4060-max-q)); the same ratio gives the desktop about 30.2. Real 3D-CNN training reaches only part of peak, so this plan assumes 8 TFLOPS achieved (26% of 30.2). Laptop factor = 15.11 ÷ 11.6 = 1.30× slower.

**Step 4 — epoch and run time on the laptop GPU.** Achieved throughput = 8 ÷ 1.30 = 6.14 TFLOPS. Per epoch: training = 7,465 × 3 × forward GFLOPs (4 × with checkpointing, which recomputes the forward pass); validation = 1,288 × forward GFLOPs. Each run is 10 epochs.

| T | Forward GFLOPs per clip | Train + val GFLOPs per epoch | Epoch time (÷ 6,140 GFLOPs/s) | Epoch time with checkpointing | Run time, 10 epochs |
| --- | --- | --- | --- | --- | --- |
| 8 | 40.52 | 907,445 + 52,190 = 959,635 | 156 s = 2.6 min | 3.4 min | 26 min |
| 16 | 81.04 | 1,814,891 + 104,380 = 1,919,271 | 313 s = 5.2 min | 6.9 min | 52 min |
| 32 | 162.08 | 3,629,782 + 208,759 = 3,838,541 | 625 s = 10.4 min | 13.7 min | 104 min |
| 64 | 324.16 | 7,259,563 + 417,518 = 7,677,081 | 1,250 s = 20.8 min | 27.4 min | 274 min (checkpointing assumed) |

**Step 4b — measured on the RTX 4060 (task A4, 16 Nov).** `scripts/probe_memory.py` ran one config per T (the FW configs share their T, so their memory and compute) with micro-batch 8, bf16 and AdamW, then timed 50 training micro-steps and 50 validation batches on real data after 10 warm-up steps. PyTorch was capped at the VRAM free at start (NVIDIA GeForce RTX 4060 Laptop GPU, about 6.8 GB of 8 GB; the Windows display holds the rest). Raw numbers: `results/probe/probe_memory_rtx4060*.csv`.

| T | Checkpointing | Peak VRAM, micro-batch 8 (first step) | Largest micro-batch | Training step, autotuning on / off | Epoch | Run, 10 epochs (Step 4 estimate) |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | no | 1.5 GB (1.5 GB) | 48 | 99 / 327 ms | 1.6 min | 16 min (26 min) |
| 16 | no | 2.5 GB (2.5 GB) | 24 | 199 / 381 ms | 3.3 min | 33 min (52 min) |
| 32 | no | 4.3 GB (4.4 GB) | 12 | 392 / 429 ms | 6.4 min | 64 min (104 min) |
| 64 | per block | 5.0 GB (5.9 GB) | 8 | 1,082 / 1,189 ms | 17.6 min | 176 min (274 min) |

- **Checkpointing per residual block, not per stage.** At T = 64, micro-batch 8 did not fit without checkpointing, and checkpointing the 4 stages saved only about 10%: `layer1` runs at full T and 56 × 56 and holds most activation memory, so recomputing the whole stage at once needs nearly all of it again. Checkpointing each of the 8 residual blocks fits. A unit test shows outputs, gradients and BatchNorm statistics are identical with and without it.
- **cuDNN autotuning on.** It is 1.1–3.3× faster; its extra workspace appears only while it tunes, in the first step, and stays under the cap at T = 64.
- **`torch.compile` off.** On FS-32 it was 9.5% faster, below the 15% threshold.
- **Times are faster than Step 4's estimate.** One seed of all 7 configs takes 514 min; with 10% for end-of-run M2 validation and checkpoint saves, 9.4 h (3.0 h for the 5 configs with T ≤ 32, 6.4 h for FS-64 and FW-64). GPU clock about 2.6 GHz, 79–82 °C during the probe.

**Step 5 — one full seed.** Queue order, cheapest useful runs first: FS-8 26 + FS-32 104 + FS-16 52 + FW-8 26 + FW-16 52 + FS-64 274 + FW-64 274 = 808 min = 13.5 h. Add 10% for end-of-run validation in M2 and checkpoint saves: 13.5 × 1.1 = 14.8 h per seed.

**Step 6 — inference cost per average video in M2** (173 frames; clip counts from the Experiment design formula).

| Config | Window W | Clips per video | GFLOPs per video (clips × fwd) | Test set in M2 (1,277 videos ÷ 8 TFLOPS) |
| --- | --- | --- | --- | --- |
| FS-8 | 16 | 20 | 810.4 | 129 s |
| FS-16 | 32 | 9 | 729.4 | 116 s |
| FS-32 | 64 | 4 | 648.3 | 104 s |
| FS-64 | 128 | 1 | 324.2 | 52 s |
| FW-8 | 64 | 4 | 162.1 | 26 s |
| FW-16 | 64 | 4 | 324.2 | 52 s |
| FW-64 | 64 | 4 | 1,296.6 | 207 s |

This already shows why the two regimes matter for efficiency: under M2, FS-8 spends 5× the compute of FW-8 per video (810.4 ÷ 162.1). Whether it buys accuracy is what the study measures.

Step 6 assumes the desktop's 8 TFLOPS; on the laptop multiply those times by 1.30. Evaluating one full set of 7 models on validation and test, M1 and M2, takes about 37 minutes on the laptop.

**Step 7 — split the runs across the three laptops; do not combine them into one training job.**

*Why not one combined job (distributed data-parallel)?* Synchronous training sends the full gradient after every optimiser step.

1. Gradient size = 31,505,325 parameters × 4 bytes = 126.0 MB.
2. A ring all-reduce over 3 machines sends 2 × (3 − 1) ÷ 3 × 126.0 = 168.0 MB per machine per step.
3. One epoch = 7,465 ÷ 16 = 467 optimiser steps, so 467 × 168.0 MB = 78.4 GB moved per epoch.
4. At about 30 MB/s on home Wi-Fi that is 78,400 ÷ 30 = 2,613 s = 43.6 min of transfer per epoch; on gigabit Ethernet (about 110 MB/s) it is 11.9 min.
5. The RTX 4060 alone needs 10.4 min per FS-32 epoch.

Every step would also wait for the slowest card, and the 4 GB card would cap the batch for all three. A combined job would be slower than the 4060 on its own.

*Dividing works because the 21 runs are independent.* Each laptop runs its own queue of (config, seed) runs and shares nothing while training.

| Laptop GPU | VRAM | Peak FP32 | Speed vs RTX 4060 | Runs it can take |
| --- | --- | --- | --- | --- |
| RTX 4060 Laptop | 8 GB | 11.6 TFLOPS | 1.00 | All, including T = 64 |
| RTX 4050 Laptop | 6 GB as specified ([Notebookcheck](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4050-Laptop-GPU-vs-Radeon-Graphics-Ryzen-7000_11457_11422.247598.0.html)); you said 8 GB, so check `nvidia-smi` | 9.0 TFLOPS ([Notebookcheck](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4050-Laptop-GPU-Benchmarks-and-Specs.675695.0.html)) | 9.0 ÷ 11.6 = 0.78 | T ≤ 32; T = 64 only if the probe fits micro-batch 8 with checkpointing |
| RTX 3050 4GB Laptop | 4 GB | 2,048 cores × 2 × 1.5 GHz = 6.1 TFLOPS ([Notebookcheck](https://www.notebookcheck.net/GeForce-RTX-3050-vs-GeForce-RTX-3050-4GB-Laptop-GPU_11144_10592.247598.0.html)) | 6.1 ÷ 11.6 = 0.53 | T = 8 and T = 16, if the probe fits micro-batch 8 |

**Run time per run** (10 epochs plus 10% overhead) = RTX 4060 time ÷ speed:

| T | RTX 4060 | RTX 4050 (÷ 0.78) | RTX 3050 (÷ 0.53) |
| --- | --- | --- | --- |
| 8 | 28.6 min | 36.9 min | 54.0 min |
| 16 | 57.3 min | 73.9 min | 108.2 min |
| 32 | 114.6 min | 147.7 min | 216.4 min, if it fits |
| 64 | 301.4 min | 388.5 min, if it fits | does not fit |

**Plan B, the baseline (the 4050 cannot train T = 64).** Each config keeps all three seeds on one GPU model.

| Laptop | Queue | Working | Hours | Done |
| --- | --- | --- | --- | --- |
| RTX 4060 | FS-8 seed 0 as a hardware control; FS-64 and FW-64, seeds 0–2 | 28.6 + 6 × 301.4 = 1,837 min | 30.6 | Mon 20:37 (start Sun 14:00) |
| RTX 4050 | FS-32 and FS-16, seeds 0–2 | 3 × 147.7 + 3 × 73.9 = 665 min | 11.1 | Mon 01:34 (start Sun 14:30) |
| RTX 3050 | FS-8, FW-8 and FW-16, seeds 0–2 | 6 × 54.0 + 3 × 108.2 = 649 min | 10.8 | Mon 01:18 (start Sun 14:30) |

If the 3050 cannot fit T = 16, FW-16 moves to the 4050: 665 + 3 × 73.9 = 887 min = 14.8 h, done Mon 05:15. The 4060 still finishes last, so the end time does not change.

**Plan A (the 4050 passes the T = 64 probe).** The 64-frame work is shared, and everything ends about 10 hours earlier.

| Laptop | Queue | Working | Hours | Done |
| --- | --- | --- | --- | --- |
| RTX 4060 | FS-8 seed 0 control; FS-64 and FW-64, seeds 0–1 | 28.6 + 4 × 301.4 = 1,234 min | 20.6 | Mon 10:34 |
| RTX 4050 | FS-64 and FW-64 seed 2; FS-32 seeds 0–2 | 2 × 388.5 + 3 × 147.7 = 1,220 min | 20.3 | Mon 10:50 |
| RTX 3050 | FS-8, FW-8, FS-16, FW-16, seeds 0–2 | 6 × 54.0 + 6 × 108.2 = 973 min | 16.2 | Mon 06:43 |

**Rules that keep the three laptops comparable**

- **Same code and config everywhere:** all three run the commit tagged `config-frozen`; check `git rev-parse HEAD` matches before starting.
- **Same recipe:** micro-batch 8 with accumulation 2 on every laptop. A config that does not fit at micro-batch 8 is not assigned to that laptop; the batch is never lowered.
- **Hardware control:** FS-8 seed 0 trains on both the RTX 4060 and the RTX 3050. The report compares that validation-MAE gap with the seed-to-seed spread.
- **Identical inputs:** copy `processed/` (cache, index, tracings, stats; about 21.8 GB) from the 4060 by USB drive or LAN instead of re-running preprocessing. Each extra laptop needs at least 35 GB free in WSL.
- **Cost numbers from one GPU only:** latency, VRAM and epoch time in the report come from the RTX 4060 benchmark, never from sweep logs, which mix GPUs.
- **Central test evaluation:** copy every `runs/<config>/seed<k>/` folder back to the 4060 (names are unique, so it is a plain copy) and run test evaluation there.
- **Licence:** these are your laptops; keep the copies on machines only you use and delete them after submission.

**Step 8 — revised GPU calendar (Plan B).**

| Start | Laptop | Job | Ends (estimate) |
| --- | --- | --- | --- |
| Sun 10:00 | RTX 4060 | Gate A, Tier 0 on validation, memory probes and timing on all three laptops (A4, A5) | Sun 12:00 |
| Sun 14:00 | RTX 4060 | FS-8 control, then FS-64 and FW-64 × 3 seeds | Mon 20:37 |
| Sun 14:30 | RTX 4050 | FS-32 and FS-16 × 3 seeds | Mon 01:34 |
| Sun 14:30 | RTX 3050 | FS-8, FW-8, FW-16 × 3 seeds | Mon 01:18 |
| Mon 21:00 | RTX 4060 | After M2: collect runs, freeze, then unattended `final_run.sh`: test evaluation (1.5 h), cost benchmark (0.5 h), XAI (0.75 h), aggregation and figures | Tue 00:00 |

Results are ready on Tuesday morning instead of Tuesday noon, and every config has all 3 seeds. Under Plan A the final run can start at Mon 11:00 for evaluation and cost, with XAI after M2.

**Disk:** frame cache 21.8 GB + checkpoints (up to 20 runs × 126 MB = 2.5 GB, after deleting `last.pt` from finished runs) + raw videos + XAI outputs. Keep at least 40 GB free inside WSL.

**Laptop care:** keep it plugged in, set Windows to Best performance, disable sleep and automatic updates until Tue 10:00, and raise it for airflow. Log GPU clocks so throttling shows up in the timing data.

## Tech stack

The stack is Python 3.12 + PyTorch/torchvision on WSL2, managed with uv, with Captum for explanations and Gradio for the demo. Every component is free, runs offline, and was picked for a stated reason; exact versions are pinned in `uv.lock` on Day 0 (task A1).

| Layer | Choice | Reason | Rejected alternative |
| --- | --- | --- | --- |
| OS | Windows + WSL2 Ubuntu 24.04 LTS; GPU through the Windows NVIDIA driver (no Linux driver installed inside WSL) | Linux tooling and fork-based DataLoader workers; Windows-native workers use spawn, which starts slower and breaks easily with pickling | Native Windows Python |
| Data location | Inside the WSL filesystem, e.g. `~/data/echonet` | Files under `/mnt/c` cross the Windows–Linux boundary and read much slower; memory-mapped clip reads are I/O-bound | Data on the C: drive |
| Python | 3.12 | Supported by current PyTorch (2.15 needs ≥ 3.11, [PyTorch notice](https://dev-discuss.pytorch.org/latest.md)) and mature across all libraries below | 3.13+ (fewer tested wheels) |
| Environment | uv with `pyproject.toml` and `uv.lock` | One lockfile makes the environment reproducible for the examiner; installs are fast | conda (slow solver, mixing two package managers) |
| Deep learning | PyTorch stable (2.14 or later) with its default CUDA 13 wheels, plus torchvision | CUDA 13 is the PyPI default since 2.11 ([PyTorch](https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325.md)); torchvision ships R(2+1)D-18 with Kinetics weights; bf16 autocast and a FLOP counter are built in | TensorFlow/Keras (no pretrained R(2+1)D in the standard library) |
| Video decoding | OpenCV (`cv2`) once, in preprocessing | The tool EchoNet used for its AVI files | PyAV, decord (extra binary dependencies) |
| Frame storage | NumPy `.npy` per video, memory-mapped | Reads only the frames a clip needs; no decoding in the training loop | HDF5 (file locking with multiple workers), LMDB (more setup) |
| Tables | pandas + pyarrow (Parquet index) | Typed columns, fast load | Plain CSV for the index |
| Configuration | OmegaConf YAML: `base.yaml` + one file per experiment, command-line overrides | Each run saves its full merged config next to its results; easy to read in a viva | Hydra (output-folder conventions and plugins are overkill for 7 configs) |
| Tracking | TensorBoard for curves; per-run `metrics.json` and `predictions.csv`; one aggregated `results/all_runs.csv` | Offline, no account; the CSV is the only source for report numbers | Weights & Biases (online account; data leaves the machine) |
| Metrics and statistics | scikit-learn (MAE, RMSE, R², AUROC); SciPy (`wilcoxon`, `bootstrap`); statsmodels (`multipletests`, Holm) | Standard, citable implementations | Hand-written statistics |
| Explainability | Captum (`IntegratedGradients`, `Occlusion`, `LayerGradCam`) | Supports 5D video tensors, including trilinear interpolation for Grad-CAM ([Captum API](https://captum.ai/api/guided_grad_cam.html)) | pytorch-grad-cam (2D-image focus) |
| Cost measurement | `torch.utils.flop_counter.FlopCounterMode`, CUDA events, `torch.cuda.max_memory_allocated`, `nvidia-smi` logs | Built into PyTorch; no extra dependency | fvcore, thop (extra dependencies for the same job) |
| Plots | matplotlib, saved as vector PDF | Sharp figures in LaTeX | Plotly (interactive, not needed in a paper) |
| Demo | Gradio, `share=False` | Video upload and playback component in a few lines | Streamlit (reruns the whole script on every interaction; Gradio's function-in, result-out model fits a predict call) |
| Code quality | ruff (lint + format), pytest, pre-commit | Catches errors before long GPU runs; one tool for style | flake8 + black + isort (three tools) |
| Report | LaTeX, IEEE conference template (Overleaf) | Standard ML paper format; PDF figures drop straight in | Word |
| Version control | Git + private GitHub repo until submission; data and runs git-ignored | Reproducible history; nothing licensed is uploaded | — |

## Repository folder structure

The repo separates importable library code (`src/echoef/`), runnable entry points (`scripts/`), configuration (`configs/`) and generated outputs (`runs/`, `results/`). Data and runs never enter Git.

```text
Heart-Pumping-Efficiency-Estimator/
├── CLAUDE.md                     # Claude Code rules; points to docs/PROJECT_SPEC.md
├── README.md                     # setup, how to reproduce each table and figure
├── pyproject.toml                # dependencies, ruff and pytest settings
├── uv.lock                       # pinned environment
├── .gitignore                    # data/, runs/, results/xai/, *.npy, *.pt, *.avi, configs/paths.yaml
├── .pre-commit-config.yaml
├── configs/
│   ├── base.yaml                 # frozen shared settings (tag config-frozen, Sunday)
│   ├── paths.example.yaml        # template; each person copies it to paths.yaml (git-ignored)
│   └── experiments/
│       ├── b0_echonet.yaml       # released EchoNet weights, eval only
│       ├── fs8.yaml  fs16.yaml  fs32.yaml  fs64.yaml
│       └── fw8.yaml  fw16.yaml  fw64.yaml
├── data/                         # git-ignored; lives on the WSL ext4 disk
│   ├── raw/EchoNet-Dynamic/      # Videos/, FileList.csv, VolumeTracings.csv (read-only)
│   ├── processed/
│   │   ├── videos/<FileName>.npy # uint8 (frames, 112, 112)
│   │   ├── index.parquet
│   │   ├── stats.json            # train-only mean and SD
│   │   └── qc_report.csv
│   └── weights/                  # r2plus1d_18_32_2_pretrained.pt (B0)
├── src/echoef/
│   ├── config.py                 # OmegaConf load + merge + validation
│   ├── contracts.py              # shared constants, column names, config IDs (agreed by A and B)
│   ├── data/
│   │   ├── preprocess.py         # decode, cache, index, ED/ES by area, stats
│   │   ├── tracings.py           # polygon from tracings, LV mask, area
│   │   ├── sampling.py           # FS/FW clip indices, M1/M2 clip starts, padding
│   │   ├── dataset.py            # EchoClipDataset (memmap reads)
│   │   └── transforms.py         # normalise, pad-and-crop, gray→3 channels
│   ├── models/
│   │   └── factory.py            # r2plus1d_18 + regression head, B0 loader, checkpointing
│   ├── train/
│   │   └── trainer.py            # AMP loop, accumulation, early stop, resume
│   ├── eval/
│   │   ├── predict.py            # M1/M2 inference, seed ensembles
│   │   ├── metrics.py            # MAE, RMSE, R², Bland–Altman, AUROC, bands
│   │   ├── stats.py              # bootstrap CIs, Wilcoxon, Holm
│   │   └── cost.py               # FLOPs, latency, VRAM, throughput
│   ├── xai/
│   │   ├── attribution.py        # occlusion, Integrated Gradients, 3D Grad-CAM
│   │   ├── xai_metrics.py        # hit rate, LV energy ratio, deletion, randomisation
│   │   └── render.py             # overlay frames and videos
│   └── utils/                    # seeding, logging, run-folder metadata (git hash, env)
├── scripts/                      # thin command-line wrappers around src/
│   ├── make_synthetic_data.py    # fake dataset with the real layout (B1)
│   ├── make_fake_runs.py         # fake runs and predictions in contract format (B1)
│   ├── verify_dataset.py         # A2
│   ├── preprocess.py             # A2
│   ├── eda.py                    # A2
│   ├── probe_memory.py           # A4
│   ├── train.py                  # one config, one seed
│   ├── run_sweep.py              # resumable queue of (config, seed)
│   ├── evaluate.py               # --split val|test (test needs --confirm-test)
│   ├── benchmark_cost.py
│   ├── run_xai.py
│   ├── aggregate_results.py      # → results/all_runs.csv
│   └── make_figures.py           # → results/figures/*.pdf
├── app/
│   ├── app.py                    # Gradio UI
│   └── inference.py              # validation, ensemble predict, overlays
├── tests/                        # pytest; run on synthetic data, no GPU needed
│   ├── test_sampling.py  test_tracings.py  test_dataset.py
│   └── test_metrics.py   test_stats.py     test_xai_shapes.py
├── runs/                         # git-ignored: runs/<config>/seed<k>/
│                                 #   config.yaml, meta.json, best.pt, last.pt, log.csv,
│                                 #   metrics.json, predictions_*.csv, tb/, DONE
├── results/                      # tables/, figures/, all_runs.csv committed; xai/ git-ignored
├── notebooks/                    # exploration only; never the source of a result
└── docs/
    ├── PROJECT_SPEC.md           # export of this doc
    ├── decisions.md              # decision log, newest first
    └── report/                   # LaTeX source
```

**Why this layout**

- **`src/` package + thin `scripts/`:** tests import the same code the scripts run, and Claude Code edits logic in one place.
- **One YAML per experiment:** a config ID in the report maps to exactly one file in the repo.
- **`runs/<config>/seed<k>/` with a `DONE` marker:** `run_sweep.py` skips finished runs, so a crash or reboot costs at most one epoch (`last.pt` saved every epoch).
- **`meta.json` per run:** git commit hash, config hash, library versions, GPU name and seed, so every number traces back to the code that made it.

## 4-day build plan and team split

Pratik (A) owns the GPU laptop, the dataset and every script that touches them. The partner (B) owns metrics, statistics, figures, explainability, the demo and most of the report, built on synthetic data until merged. The halves meet through fixed interface contracts and three scheduled merges, and all final runs happen on A's laptop.

&#91;embedded content: 4-day plan · three GPU lanes, Pratik and partner · 4 gates, 4 merges\]

Three GPU lanes start on Sunday afternoon and the RTX 4060 finishes last, about Mon 20:40 under Plan B. Merges M1 and M2 bring B's code onto `main`; the overnight final run feeds M3 on Tuesday at 08:00.

### Who owns what

Each file has exactly one owner, so the two branches never edit the same file and merges cannot conflict.

| Area | Owner | Files |
| --- | --- | --- |
| Repo, environment, contracts | A | `pyproject.toml`, `uv.lock`, `configs/`, `src/echoef/contracts.py`, `CLAUDE.md`, `README.md` |
| Data pipeline | A | `src/echoef/data/`; `scripts/verify_dataset.py`, `preprocess.py`, `eda.py` |
| Model, training, inference, cost | A | `src/echoef/models/`, `train/`, `eval/predict.py`, `eval/cost.py`, `utils/`; `scripts/probe_memory.py`, `train.py`, `run_sweep.py`, `evaluate.py`, `benchmark_cost.py` |
| Synthetic data and fake runs | B | `scripts/make_synthetic_data.py`, `scripts/make_fake_runs.py` |
| Metrics, statistics, aggregation, figures | B | `src/echoef/eval/metrics.py`, `eval/stats.py`; `scripts/aggregate_results.py`, `make_figures.py` |
| Explainability | B | `src/echoef/xai/`, `scripts/run_xai.py` |
| Demo app | B | `app/` |
| Tests | Owner of the module tested | `tests/test_<module>.py` |
| IEEE report | B leads; A writes Method and Experimental setup | Overleaf project |
| Anything that reads real data | A only | — |

**Data rule for B.** B never receives the dataset, the frame cache or per-video predictions. The licence forbids sharing and requires each user to register individually ([EchoNet site](https://echonet.github.io/dynamic/)). B works with synthetic videos and with aggregated outputs (`all_runs.csv`, tables, figures). If B wants real data, B registers on the portal personally.

### Day-by-day tasks

Each task has an owner prefix (A or B), a time box and a "done when" check. Times are targets; the gates and merges are fixed.

#### Day 0 — Sat 14 Nov (tonight)

**Pratik**

1. **A1 Repo and environment (19:00–20:30).** In WSL, install uv and create a private GitHub repo `Pratik-337/Heart-Pumping-Efficiency-Estimator`; add B as a collaborator. With Claude Code, create the skeleton from the folder structure: `pyproject.toml` with every library in the Tech stack, `uv.lock`, `CLAUDE.md`, `configs/paths.example.yaml`, and `src/echoef/contracts.py` copied from Interface contracts. Export this doc to `docs/PROJECT_SPEC.md`. Push to `main` (merge **M0**). Check `nvidia-smi`, `torch.cuda.is_available()` and `torch.cuda.is_bf16_supported()`. Download the B0 weights into `data/weights/`. *Done when:* B can clone and `uv sync` succeeds.
2. **A2 Data pipeline (20:30–01:00).** Move the dataset into `~/data/echonet/raw/EchoNet-Dynamic/` inside WSL. Build `verify_dataset.py`, `preprocess.py` (frame cache, `index.parquet`, `tracings.parquet`, ED/ES by polygon area, train-only `stats.json`) and `eda.py`. Start preprocessing before you sleep; it runs unattended. While it runs, prepare the RTX 4050 and RTX 3050 laptops: WSL2, NVIDIA driver, uv, clone the repo, uv sync, and record each GPU's exact name and memory from nvidia-smi. *Done when:* the cache and index exist and split counts are logged.

**Partner**

1. **B1 Setup and synthetic data (20:30–23:30).** Install Git and uv; accept the GitHub invite; clone; `uv sync`; create branch `b-dev`. Write `make_synthetic_data.py` (30 raw-format videos of a pulsing ellipse, `FileList.csv`, `VolumeTracings.csv`, EF from the largest and smallest area) and `make_fake_runs.py` (run folders and prediction CSVs for all 7 configs and Tier 0, in contract format, with config-dependent noise so the plots look realistic). *Done when:* both scripts run and pass a schema test.
2. **B2 Report skeleton (23:30–01:00).** Create an Overleaf project from the IEEE conference template (IEEEtran) and share it with A. Add the section headings from the IEEE report plan and a `refs.bib` built from the Sources section.

#### Day 1 — Sun 15 Nov

**Pratik**

3. **A3 Dataset and sampling (08:00–10:00).** `sampling.py`, `transforms.py`, `tracings.py` and `dataset.py` exactly as the contracts specify, with unit tests. Save and inspect one batch image grid per config. Copy the processed folder (about 21.8 GB) to the other two laptops by USB drive or LAN.
4. **A4 Model and probes (10:00–11:00).** `factory.py` (Kinetics model, B0 loader, checkpointing switch). Run `probe_memory.py` (largest micro-batch per T, plus 50 timed training steps per T) on all three laptops. The result picks Plan A or Plan B (Compute budget, Step 7); write it into `configs/assignment.yaml` and the measured epoch times into the Compute budget section.
5. **A5 Gate A and Tier 0 (11:00–12:00).** `predict.py` and `evaluate.py` writing contract CSVs. **Gate A:** B0 at T = 32, p = 2 has validation MAE (M2) below 5.0. Then run Tier 0: B0 at all 7 settings on validation.
6. **A6 Trainer and sweep start (12:00–14:00).** `trainer.py`, `train.py` and a resumable `run_sweep.py --machine <name>` that runs that laptop's list from `configs/assignment.yaml`. One-epoch FS-8 smoke run on real data. Tag `config-frozen`, pull that commit on all three laptops, and start each queue in `tmux`: `rtx4060` here at 14:00, `rtx4050` and `rtx3050` by 14:30.
7. **Gate B (about 17:00, when FS-32 seed 0 finishes on the RTX 4050).** Its validation MAE (M1) must be at most B0's validation MAE (M1) + 1.0. If not, stop all queues and debug before anything else.
8. **A7 Cost benchmark, final-run script and EDA summary (16:30–20:00).** Write `benchmark_cost.py` and `scripts/final_run.sh` (test evaluation → cost benchmark → XAI → aggregation → figures, stopping on the first error); both run on Monday night. Export the EDA summary as an aggregated CSV for B's Dataset section.
9. **Merge M1 (21:00).** Review and merge B's pull request. Pull, then run `aggregate_results.py` and `make_figures.py` on the real validation predictions (Tier 0 plus finished runs). Check the plots make sense.

**Partner**

3. **B3 Metrics and statistics (09:00–13:00).** `metrics.py` (MAE, RMSE, R², Bland–Altman, AUROC for EF < 50% and ≤ 40%, MAE by EF band, calibration slope) and `stats.py` (bootstrap CI, paired Wilcoxon, Holm), with tests against hand-computed values.
4. **B4 Aggregation and figures (13:00–19:00).** `aggregate_results.py` writes `all_runs.csv` and `wilcoxon_holm.csv`. `make_figures.py` draws MAE vs T for both regimes with the Tier 0 curve, Pareto plots, Bland–Altman and MAE by EF band, as PDFs 3.5 inches wide (one IEEE column). Develop on the fake runs.
5. **B5 Report: Introduction and Related work (19:00–20:30).**
6. **Open the pull request `b-dev` → `main` by 20:30** for M1.

#### Day 2 — Mon 16 Nov

**Pratik**

10. **A8 Monitor (07:00–09:00).** Under Plan B the RTX 4050 and RTX 3050 queues finished around 01:30. Check curves and validation predictions on each laptop, copy their run folders to the 4060, rerun aggregation, and confirm the 4060's 64-frame queue is still running.
11. **A9 Report: Method and Experimental setup (09:00–17:00)** in Overleaf: model, sampling regimes table, training, inference modes, hardware, metrics, statistics. Answer B's contract questions.
12. **Merge M2 (21:00), then start the final run.** Review and merge B's pull request (XAI and app). When the RTX 4060 queue has finished (about 20:40 under Plan B), copy any remaining run folders onto the 4060, tag `models-frozen` (**Gate C**) and start `scripts/final_run.sh` in `tmux`: test evaluation, cost benchmark, XAI on the 4 priority configs, aggregation and figures. It runs unattended, about 3 hours.

**Partner**

7. **B6 Explainability (09:00–15:00).** `attribution.py` (occlusion, Integrated Gradients, Grad-CAM), `xai_metrics.py` (hit rate, LV energy ratio, deletion AOPC, randomisation sanity check) and `render.py`. Test on synthetic videos with a randomly initialised model at T = 8, on CPU.
8. **B7 `run_xai.py` (15:00–17:00)**, loading checkpoints and writing outputs exactly as the contracts specify.
9. **B8 Demo app (17:00–20:00):** `app/inference.py` and `app/app.py`, tested with a fake checkpoint.
10. **B5 continued (evening):** Dataset section from A's EDA summary; XAI subsection of Method. Open the M2 pull request by 20:30.

#### Day 3 — Tue 17 Nov

**Pratik**

13. **A10 Check the overnight run (07:00–08:00).** Read the `final_run.sh` logs and rerun any failed step. Commit and push `all_runs.csv`, `cost.csv`, tables and figures (merge **M3**, 08:00).
14. **A11 Demo and spare GPU time (08:00–12:00).** Choose the demo model on validation. **Gate D:** the app handles 6 inputs, each under 3 s. If time allows, run XAI on the other 3 configs (about 15 minutes).
15. **Afternoon:** fill Results tables with B and check every number against `all_runs.csv`.

**Partner**

11. **B9 Results, Discussion, Limitations, Ethics, Abstract and Conclusion (08:00–22:00)**, using the M3 figures uploaded to Overleaf.
12. **On call in the morning** to fix your own code if A hits a bug during A10 or A11.

Both read the full draft by 22:00.

#### Day 4 — Wed 18 Nov

16. **A12 Reproducibility check (08:00–10:00).** Fresh clone into `/tmp/check`, `uv sync`, `uv run pytest`, regenerate one table and one figure from saved predictions. Final README; tag `v1.0`.
17. **B10 Proofread and format (08:00–11:00).** IEEE layout, page limit, figure captions, every figure cited in the text, reference style.
18. **Submit by 12:00.** The afternoon is buffer.

#### Fallbacks

- **Training starts after 18:00 Sunday:** drop seed 2; run seed 1 for the FS configs only.
- **Gate B fails:** after the fix, train only FS-8, FS-32 and FS-64. Tier 0 still gives a full curve.
- **XAI not merged by Tue 09:00:** run occlusion only and skip Integrated Gradients.
- **App not working by Gate D:** show a screen recording, if a demo is assessed at all.

## Interface contracts

Both people code against the names, shapes and file formats below, so B's code runs on A's laptop without edits. They are frozen at merge M0; a change needs both people to agree and a Decision log row.

**`src/echoef/contracts.py`** (A commits it at M0; both import from it):

```python
"""Shared contracts between A (pipeline) and B (analysis). Change only when both agree."""
from pathlib import Path

# Config ID -> (regime, T, period). fs32 belongs to both regimes.
CONFIGS = {
    "fs8": ("FS", 8, 2), "fs16": ("FS", 16, 2), "fs32": ("BOTH", 32, 2), "fs64": ("FS", 64, 2),
    "fw8": ("FW", 8, 8), "fw16": ("FW", 16, 4), "fw64": ("FW", 64, 1),
}
TIER0_PREFIX = "b0-"   # Tier 0 run IDs: b0-fs8 ... b0-fw64 (EchoNet weights, no training)
SPLITS = ("TRAIN", "VAL", "TEST")
MODES = ("M1", "M2")
FRAME_SIZE = 112

INDEX_COLUMNS = ["FileName", "Split", "EF", "EDV", "ESV", "FPS",
                 "n_frames", "ed_frame", "es_frame", "has_trace"]
TRACINGS_COLUMNS = ["FileName", "Frame", "X1", "Y1", "X2", "Y2"]
PREDICTION_COLUMNS = ["file_name", "split", "mode", "config_id", "seed", "T", "period",
                      "ef_true", "ef_pred", "n_clips", "padded"]
COST_COLUMNS = ["config_id", "T", "period", "params", "gflops_clip", "mean_clips_m2",
                "gflops_video_m2", "latency_ms_m1", "latency_ms_m2", "peak_vram_mb_infer",
                "peak_vram_mb_train", "micro_batch", "epoch_time_s", "gpu_name"]
ALL_RUNS_COLUMNS = ["config_id", "regime", "T", "period", "seed", "split", "mode", "n",
                    "mae", "mae_ci_lo", "mae_ci_hi", "rmse", "r2", "bias", "loa_lo", "loa_hi",
                    "auroc_lt50", "auroc_le40", "mae_le40", "mae_41_49", "mae_ge50", "calib_slope"]
XAI_COLUMNS = ["file_name", "config_id", "method", "hit", "lv_energy_ratio",
               "deletion_aopc", "random_aopc", "skipped"]

def run_dir(runs_root: Path, config_id: str, seed: int) -> Path:
    return runs_root / config_id / f"seed{seed}"

def predictions_path(run: Path, split: str, mode: str) -> Path:
    return run / f"predictions_{split.lower()}_{mode}.csv"
```

**Files on disk**

| Artifact | Path | Format |
| --- | --- | --- |
| Machine paths | `configs/paths.yaml` (git-ignored, copied from `paths.example.yaml`) | Keys: `raw_root`, `processed_root`, `runs_root`, `results_root`, `weights_root` |
| Frame cache | `<processed_root>/videos/<FileName>.npy` | uint8, shape (frames, 112, 112) |
| Index | `<processed_root>/index.parquet` | `INDEX_COLUMNS`; `ed_frame`/`es_frame` = −1 when missing |
| Tracings | `<processed_root>/tracings.parquet` | `TRACINGS_COLUMNS`, copied from `VolumeTracings.csv` |
| Normalisation | `<processed_root>/stats.json` | `{"mean": float, "std": float}`, training split only |
| Run folder | `<runs_root>/<config_id>/seed<k>/` | `config.yaml`, `meta.json`, `best.pt`, `last.pt`, `log.csv`, `metrics.json`, prediction CSVs, `tb/`, `DONE` |
| Checkpoint | `best.pt` | dict with `state_dict`, `config` (dict incl. T, period, config\_id, seed), `epoch`, `val_mae` |
| Training log | `log.csv` | epoch, train\_loss, val\_mae\_m1, lr, seconds |
| Predictions | `predictions_<split>_<mode>.csv` | `PREDICTION_COLUMNS`; `padded` = 1 if any clip was zero-padded |
| Cost | `<results_root>/cost.csv` | `COST_COLUMNS`, one row per config |
| Aggregated results | `<results_root>/all_runs.csv` | `ALL_RUNS_COLUMNS` plus the cost columns joined on `config_id`; `seed` = "ens" for the seed ensemble |
| XAI | `<results_root>/xai/<config_id>/xai_metrics.csv` | `XAI_COLUMNS`, one row per video and method |

**Additions for three laptops.** `configs/assignment.yaml` maps each laptop (`rtx4060`, `rtx4050`, `rtx3050`) to its list of `[config_id, seed]` runs, and `run_sweep.py --machine <name>` runs only that list. `meta.json` records `gpu_name`, `vram_gb` and the git commit; `aggregate_results.py` copies `gpu_name` into `all_runs.csv` so the report can state which GPU trained each run.

**Functions A provides (B calls them)**

| Function | Signature | Returns |
| --- | --- | --- |
| `echoef.data.dataset.load_clip` | `(file_name, T, period, start, processed_root, stats)` | float32 tensor (3, T, 112, 112), normalised; frames past the video end are zeros |
| `echoef.data.sampling.clip_starts` | `(n_frames, T, period, mode)` | list of start frames: 1 for M1, n for M2 |
| `echoef.data.sampling.edes_start` | `(ed, es, T, period, n_frames)` | start frame that holds both ED and ES, or None |
| `echoef.data.tracings.lv_mask` | `(file_name, frame, processed_root)` | bool array (112, 112) |
| `echoef.models.factory.build_model` | `(pretrained=True)` | `nn.Module`: input (B, 3, T, 112, 112) → output (B,) EF % |
| `echoef.models.factory.load_checkpoint` | `(run, device)` | `(model in eval mode, config dict)` |
| `echoef.eval.predict.predict_video` | `(model, file_name, T, period, mode, processed_root, stats, device)` | `(ef_pred, n_clips)` |

**Functions and scripts B provides (A runs them on Tuesday)**

| Function or script | Signature or command | Output |
| --- | --- | --- |
| `echoef.eval.metrics.compute_metrics` | `(df)` with `PREDICTION_COLUMNS` | dict with the metric keys of `ALL_RUNS_COLUMNS` |
| `echoef.eval.stats.bootstrap_ci` | `(abs_errors, n_boot=10000, seed=0)` | `(lo, hi)` |
| `echoef.eval.stats.wilcoxon_holm` | `(pairs_df)` | DataFrame: comparison, median diff, CI, p, p\_holm, significant |
| `echoef.xai.attribution.occlusion_curve` | `(model, clip)` | array (T,) |
| `echoef.xai.attribution.integrated_gradients` | `(model, clip, baseline="zero" or "blur", steps=32)` | array (T, 112, 112) |
| `echoef.xai.attribution.gradcam3d` | `(model, clip)` | array (T, 112, 112) |
| `scripts/aggregate_results.py` | `[--runs-root R] [--results-root S], defaults from paths.yaml` | `S/all_runs.csv`, `S/tables/*.csv` |
| `scripts/make_figures.py` | `--results-root S` | `S/figures/*.pdf` |
| `scripts/run_xai.py` | `--configs fs8 fw8 fs32 fs64 --n-videos 100 --device cuda` | `S/xai/<config_id>/` metrics and figures |
| `app/app.py` | `--runs <run folders>` | local Gradio app |

Every script reads `configs/paths.yaml`, so the same commands work on both laptops; only the paths differ.

## Git workflow and merge guide

Pratik works on branch `a-dev`, the partner on `b-dev`, and both merge into `main` through pull requests at fixed times. Because every file has one owner, each merge should be conflict-free; a conflict is a signal that someone edited a file they do not own.

| Merge | When | What lands on `main` | Who merges |
| --- | --- | --- | --- |
| M0 | Sat 20:30 | Skeleton, `contracts.py`, `pyproject.toml` + `uv.lock`, `CLAUDE.md`, this spec | A pushes directly |
| A's early merges | Sun 10:00 and 14:00 | A3–A6: loaders, model, predict, trainer | A, so B can test against the real loaders on synthetic data |
| M1 | Sun 21:00 | B3–B4: metrics, statistics, aggregation, figures | A reviews and merges B's pull request |
| M2 | Mon 21:00 | B6–B8: XAI, `run_xai.py`, demo app | A reviews and merges B's pull request |
| M3 | Tue 08:00 | Final `all_runs.csv`, `cost.csv`, tables and figures | A pushes; B pulls |

### One-time setup: Pratik (WSL Ubuntu, Saturday 19:00)

1. Install the tools and set your Git identity:

```bash
sudo apt update && sudo apt install -y git tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
git config --global user.name "Pratik-337"
git config --global user.email "<email on your GitHub account>"
```

2. On github.com: New repository → name `Heart-Pumping-Efficiency-Estimator` → Private → Create. Then Settings → Collaborators → Add people → your partner's username. Sign in from the terminal with the GitHub CLI (`gh auth login`) or a personal access token, so `git push` works.
3. Clone, build the skeleton with Claude Code (task A1), and push:

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/Pratik-337/Heart-Pumping-Efficiency-Estimator.git
cd Heart-Pumping-Efficiency-Estimator
# ... Claude Code creates the skeleton (task A1) ...
cp configs/paths.example.yaml configs/paths.yaml   # set raw_root etc. to ~/data/echonet/...
uv sync
uv run pytest -q
git add -A && git commit -m "A1: skeleton, contracts, environment (M0)"
git push origin main
git switch -c a-dev && git push -u origin a-dev
```

### One-time setup: partner (Saturday, after M0)

1. Install Git and uv. On Windows, use WSL Ubuntu and the same commands as Pratik's step 1; on macOS or Linux the uv install line is the same.
2. Accept the GitHub invitation (email, or github.com/notifications).
3. Clone, configure and create your branch:

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/Pratik-337/Heart-Pumping-Efficiency-Estimator.git
cd Heart-Pumping-Efficiency-Estimator
cp configs/paths.example.yaml configs/paths.yaml   # point every root at ~/synthetic
uv sync
uv run pytest -q
git switch -c b-dev && git push -u origin b-dev
```

### Daily loop (both people, every work session)

1. Pull in whatever the other person has merged:

```bash
git switch main && git pull
git switch b-dev            # Pratik: a-dev
git merge main
uv sync && uv run pytest -q
```

2. Work in small steps; stage only your own files; put the task ID in the message:

```bash
git add src/echoef/eval/metrics.py tests/test_metrics.py
git commit -m "B3: bootstrap CI for MAE"
git push
```

3. Before a merge point, all three checks must pass: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q`. Then on GitHub: Pull requests → New → base `main`, compare `b-dev` → title "M1: metrics, statistics, figures" → Create.
4. Pratik reviews the Files changed tab: it should list only partner-owned files. Merge with "Create a merge commit".

### Final integration on Pratik's RTX 4060 laptop (Monday 21:00, unattended overnight)

First copy each extra laptop's `runs/` folders onto the RTX 4060 (USB drive, or `rsync -a` over the LAN); folder names are unique per config and seed, so nothing is overwritten. Training runs inside `tmux` (start: `tmux new -s sweep`; detach: Ctrl+B then D; return: `tmux attach -t sweep`). When the 4060's queue has finished, run these steps; `scripts/final_run.sh` runs the same commands in order:

```bash
cd ~/projects/Heart-Pumping-Efficiency-Estimator
git switch main && git pull && uv sync && uv run pytest -q
git tag models-frozen && git push --tags                       # Gate C
uv run python scripts/evaluate.py --split test --confirm-test --all
uv run python scripts/benchmark_cost.py --all
uv run python scripts/run_xai.py --configs fs8 fw8 fs32 fs64 --n-videos 100 --device cuda
uv run python scripts/aggregate_results.py
uv run python scripts/make_figures.py
git add results/all_runs.csv results/cost.csv results/tables results/figures
git commit -m "A10-A11: final results (M3)" && git push
```

Scripts take their folders from `configs/paths.yaml`, so no path flags are needed. The partner then runs `git switch main && git pull` and uploads `results/figures/*.pdf` to Overleaf.

### Rules that keep merges painless

- **Edit only files you own.** Need a change in the other person's file? Message them and they make it.
- **Only Pratik changes `pyproject.toml` and `uv.lock`.** Every library is added at M0; if the partner needs another, Pratik adds it.
- **Never commit** data, `runs/`, `configs/paths.yaml`, videos, `.npy`, `.pt`, or any figure that shows echo frames. XAI panels with frames go from Pratik's laptop straight to Overleaf.
- **On a conflict, stop.** Run `git merge --abort` and message the file's owner instead of resolving it yourself.
- **Never run a second GPU job while training runs** on Pratik's laptop; use `--device cpu` for smoke tests until its queue finishes (about Mon 20:40).

## IEEE report plan

The report is an IEEE conference paper of about 6 pages (IEEEtran template, two columns) written in a shared Overleaf project. The partner leads and owns the narrative; Pratik writes Method and Experimental setup and supplies every number.

| Section | Owner | Content | Source in this spec | Pages | Draft by |
| --- | --- | --- | --- | --- | --- |
| Title, abstract, index terms | B | Problem, approach, 3 headline numbers, takeaway; at most 200 words | `all_runs.csv` | — | Tue 20:00 |
| I. Introduction | B | Why EF matters; automated EF already exists; the gap (cost-measured frame study, scored explanations); 3 contribution bullets | Project summary, Improvements | 0.6 | Sun 21:00 |
| II. Related work | B | EchoNet-Dynamic and its validation-loss sweep; EchoCoTr; EchoGNN; EchoPrime and PanEcho; video XAI and sanity checks | Sources | 0.5 | Sun 21:00 |
| III. Dataset | B | Facts, split, preprocessing, padded share per window | Dataset section + A's EDA summary | 0.4 | Mon 21:00 |
| IV. Method | A (XAI subsection B) | Model; sampling regimes (Table I); training; inference modes; Tier 0; XAI methods and tests | Experiment design, Model and training, Explainability | 1.3 | Mon 17:00 |
| V. Experimental setup | A | Hardware, metrics, statistics, cost protocol | Evaluation protocol, Compute budget | 0.4 | Mon 17:00 |
| VI. Results | A numbers, B text | A. Tier 0; B. Trained frame sweep; C. Cost and Pareto; D. Multi-clip; E. Explainability | `results/` | 1.6 | Tue 18:00 |
| VII. Discussion and limitations | B | Density vs coverage; clinical scale of the gaps; limitations | Ethics, data use and limitations | 0.5 | Tue 21:00 |
| VIII. Conclusion | B | 3–4 sentences | — | 0.2 | Tue 21:00 |
| References | B | BibTeX, IEEE numeric style | Sources | 0.5 | Tue 21:00 |

Total: 0.6 + 0.5 + 0.4 + 1.3 + 0.4 + 1.6 + 0.5 + 0.2 + 0.5 = 6.0 pages. Adjust once the page limit is known.

**Figures and tables**

| Item | Content | Made by |
| --- | --- | --- |
| Fig. 1 | Pipeline: video → clip sampler (FS or FW) → R(2+1)D-18 → EF, with the XAI branch | B (draw.io or TikZ) |
| Table I | The 7 configurations: T, p, window, seconds | A, from Experiment design |
| Fig. 2 | Test MAE vs T: FS line, FW line, Tier 0 line, 95% CI bars | `make_figures.py` |
| Table II | Per config: MAE with CI, RMSE, R², AUROC (EF < 50%), GFLOPs per video, latency | `aggregate_results.py` |
| Fig. 3 | Pareto plot: MAE vs GFLOPs per video, M1 and M2 markers | `make_figures.py` |
| Table III | XAI scores per config: hit rate vs chance, LV energy ratio, deletion AOPC vs random | `aggregate_results.py` |
| Fig. 4 | XAI panel: frames with IG overlay and the occlusion curve with ED/ES marked | `render.py`; A uploads it, since it shows echo frames |
| Fig. 5 (if space) | Bland–Altman plot for the chosen model | `make_figures.py` |

**Writing rules**

- Every number in the text and tables is copied from `all_runs.csv`; the partner pastes it and Pratik checks it.
- State the constraints plainly: 10 epochs, 1–3 seeds, single-centre data, not clinically validated.
- Cite EchoNet's own frame sweep in Related work and position the contribution against it.
- Figures are vector PDFs at one-column width (3.5 in) with fonts of at least 8 pt.

## Claude Code operating rules

Claude Code works one task ID at a time, reads this spec first, and must never touch raw data, the test split or result numbers outside the defined scripts. Save the block below as `CLAUDE.md` in the repo root.

```markdown
# CLAUDE.md — Heart Pumping Efficiency Estimator

## Source of truth
- Read docs/PROJECT_SPEC.md before any task. It wins over code comments and over your own assumptions.
- Decisions live in docs/decisions.md. Do not change a decided item without a new entry approved by the user.
- Work on one task ID (e.g. A3 or B6) per session. Put the task ID in every commit message.
- Ask which person you are working for (A = Pratik, B = partner). Edit only files that person owns (ownership table in the spec).
- contracts.py, pyproject.toml, uv.lock and configs/base.yaml change only with both people's agreement.
- Person B's machine has no real data: develop and test only on synthetic data from scripts/make_synthetic_data.py.

## Hard rules
- Never modify, move or delete anything under data/raw/.
- Never commit data/, runs/, results/xai/, *.npy, *.pt, *.avi, configs/paths.yaml, any video file, or any figure that shows echo frames.
- Never evaluate on the TEST split except via `scripts/evaluate.py --split test --confirm-test`, and only after the tag `models-frozen` exists.
- Never type a result number into the report, README or code. Numbers come from results/all_runs.csv via scripts.
- Never change configs/base.yaml after the tag `config-frozen` without a decision entry.
- Never shrink batch size, epochs or data silently to make something fit. On OOM or slowness, stop and report.

## How code is written
- Library code in src/echoef/; scripts/ are thin CLI wrappers (argparse or OmegaConf CLI).
- No hard-coded hyperparameters or paths: everything comes from configs/*.yaml.
- Every run writes runs/<config>/seed<k>/{config.yaml, meta.json, log.csv, metrics.json, predictions_*.csv, best.pt, last.pt, tb/, DONE}.
- meta.json holds git commit, config hash, library versions, GPU name, seed, start/end time.
- Type hints on public functions; docstrings state tensor shapes, e.g. (B, C, T, H, W).
- pathlib for paths; logging, not print; no notebooks as a source of results.
- Determinism: call utils.seed_everything(seed) at the start of every script.

## Before you say a task is done
- `uv run ruff check . && uv run ruff format --check . && uv run pytest -q` all pass.
- New logic has a unit test that runs on synthetic data without a GPU.
- The task's "Done when" line in the spec is met; say how you checked it.

## Ask the user first
- Adding a dependency, changing the folder layout, changing a metric definition,
  or anything that affects which numbers end up in the report.

## Environment
- WSL2 Ubuntu, Python 3.12, uv. Data root and run root come from configs/paths.yaml.
- GPU (A's laptop only): RTX 4060 Laptop GPU, 8 GB. Use torch.autocast(dtype=torch.bfloat16) for training and inference.
- Never start a GPU job while the training queue runs (until about Mon 20:40); smoke-test with --device cpu.
```

**How to prompt Claude Code, one task at a time**

- Start a session with: "Read `docs/PROJECT_SPEC.md` and `CLAUDE.md`. I am A. Implement A3. Show me the tests first, then the code."
- After each task: run the "Done when" check yourself, then commit. *Reason:* you must be able to explain every module in the viva, so review each one as it lands.
- Long GPU runs are started by you in a separate terminal (`tmux`), not inside a Claude Code session, so a closed session never kills a 5-hour run.

## Risks and mitigations

The biggest risks are a late training start and lost overnight GPU time. Tier 0 and the seed-ordered queues mean a complete result exists even if both happen. Three laptops add two risks: an extra laptop failing overnight (its unfinished runs move to whichever laptop is free first) and mixed hardware (measured by the FS-8 hardware control run).

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Training not running by Sun 18:00 | Medium | High: fewer seeds | One Claude Code session per task; Tier 0 already gives a full curve; fallback drops seed 2 |
| Pipeline bug (normalisation, channel order, frame index) | Medium | High: silently wrong results | Gate A, Gate B, unit tests, batch image grids |
| Out of memory at T = 64 | Medium | Medium | Activation checkpointing; fallback micro-batch 4 with frozen BatchNorm, logged |
| Laptop throttles, sleeps or updates overnight | Medium | Medium | Plugged in, best-performance mode, sleep and updates off, `tmux`, resumable queue with `last.pt` |
| A second GPU job kills a training run | Medium | High | CPU-only smoke tests while any queue runs |
| B's code breaks on real data | Medium | Medium | Contracts; B tests through A's loaders on synthetic data after A's early merges; CPU smoke test at M2 |
| Merge conflicts | Low | Medium | One owner per file; contracts frozen at M0; on conflict, abort and message the owner |
| Report unfinished | Medium | High | Intro, related work, dataset and method drafted Sun–Mon; Tuesday only fills results |
| Differences between configs are not significant | Medium | Low | A valid finding (fewer frames suffice); report effect sizes and CIs |
| Data leakage | Low | High | Official split; train-only statistics; test lock (Gate C); demo model chosen on validation |
| Dataset reaches the partner by mistake | Low | High: licence breach | Partner uses synthetic data only; data and `paths.yaml` are git-ignored |

## Ethics, data use and limitations

The dataset may be used only for your own non-commercial research, cannot be shared, and must never inform clinical decisions; the project's code, report and demo are designed around those terms ([EchoNet site](https://echonet.github.io/dynamic/)).

**Data-use rules the project follows**

- No videos, frame caches, per-video predictions or download links in GitHub, the report appendix or the demo.
- The agreement also forbids derivative works. The local frame cache is treated as a private working copy and is deleted after submission. Trained weights are not published: whether they count as derivative works is unclear, so the safe default is to keep them private. Confirm with your faculty guide if they want weights submitted.
- Figures in the report may show a few example frames, as the original paper does. Ask your guide whether this is acceptable for your submission.
- No attempt to re-identify anyone; the agreement requires reporting any accidental re-identification.
- The demo and report state that the system is not clinically validated and not for diagnosis.

**Limitations to state in the report**

- **Single centre.** All data comes from one hospital; EchoNet's own error rose from 4.1 to 6.0 MAE on another hospital's videos ([Nature paper](https://utstat.toronto.edu/reid/sta2101f/s41586-020-2145-8.pdf)).
- **Label noise.** Reference EF comes from human tracings, and human inter-observer variation is 7.6–13.9% in the same paper. Very small MAE gaps are below the labels' own precision.
- **Apical four-chamber view only.** The model cannot be trusted on other views.
- **Tuning bias.** One untuned learning rate (1e-4, D-18) is shared by every config and may suit some clip lengths better than others; Kinetics weights were trained on 16-frame clips, which may slightly favour that length.
- **Frames, not seconds.** Windows are set in frames, and frame rate varies around 50 fps, so a 64-frame window is not exactly 1.28 s for every video.
- **Explanations show influence, not mechanism.** High attribution means the model's output depends on that input, not that the region causes the heart's function.
- **Uncertainty band is narrow by construction.** The 3-seed spread captures model variance only.

## Decision log

Twenty-nine decisions are fixed as of 16 Nov 2026; D-15 to D-24 re-scope the plan for the 18 Nov deadline, two people and three laptops, D-25 to D-27 record what A2 found in the raw data, D-28 defines a padded clip (A3), and D-29 moves all training to the RTX 4060 (A4). New rows go at the top, and `docs/decisions.md` mirrors this table.

| ID | Date | Decision | Alternatives considered | Reason |
| --- | --- | --- | --- | --- |
| D-29 | 16 Nov 2026 | All training runs on the RTX 4060; the RTX 4050 and 3050 laptops are not available (supersedes D-23's three-laptop split). The queue in `configs/assignment.yaml`: seed 0 of FS-8, FS-32, FS-16, FW-8, FW-16, FS-64, FW-64; then seed 1 of FS-8, FS-32, FS-16, FW-8, FW-16; then seed 1 of FS-64 and FW-64. Runs not finished at the `models-frozen` tag are not used. No hardware-control run | The spec's fallback (seed 1 for the FS configs only); seed 0 only; wait for the other laptops | Run times measured in A4 (Compute budget, Step 4b) fit seed 0 of all 7 configs and seed 1 of all 7 by Tuesday morning; cutting the queue early still leaves complete seeds for the cheaper configs. With one GPU there is no hardware difference to control |
| D-28 | 16 Nov 2026 | A clip is padded when it reads a frame at or past the video's end (`sampling.is_padded`); a video is padded for a config when its clips are. This rule sets the `padded` column of the prediction CSVs, the unpadded-subset check (D-09) and `results/eda/padding_share.csv` | Count every video shorter than the window W = T × p (EchoNet's padding condition) | A video up to p − 1 frames shorter than W has a strided clip that ends inside it, so it holds no zero frames; counting it would put unaffected videos in the padded group. The two rules differ for FS-32, FS-64, FW-8 and FW-16 by 3 to 50 videos |
| D-27 | 15 Nov 2026 | Videos with no tracings are left out of training, evaluation and the training-set normalisation statistics, as EchoNet's loader does; they stay in `index.parquet` and `qc_report.csv` | Keep all 10,030 videos | 6 videos (5 train, 1 test) have no tracings, and FileList.csv also gives them wrong frame sizes; dropping them matches EchoNet's loader, so B0 and the sweep are scored on the same videos |
| D-26 | 15 Nov 2026 | ED/ES frames and LV outlines only from regular tracings: each of the video's two traced frames must be one long axis followed by chords perpendicular to it. Otherwise `has_trace` = False, `ed_frame` = `es_frame` = −1 and `qc_report.csv` lists the video as `irregular_tracing`; the video stays in training and evaluation | Take the first traced frame as ED; pool all chords into one outline | 93 videos (66 train, 15 val, 12 test) have a frame traced several times with the rows interleaved; no outline built from them matches the EF labels. On all 9,931 regular videos the area rule agrees with the file order (first traced frame = ED) |
| D-25 | 15 Nov 2026 | The frame cache stays grayscale (D-08), converted with OpenCV's luma rule (`cv2.COLOR_BGR2GRAY`: 0.299 R + 0.587 G + 0.114 B), although the raw R, G and B channels are not identical. Per-video channel differences are in `processed/video_scan.parquet`, a summary in `processed/preprocess_meta.json` | Store RGB; keep one channel (G) | Full scan by A2: only 155 of 10,030 videos have identical channels. Most differ slightly in about 3% of pixels (luma then moves pixels by about 0.03 grey levels on average), while 263 videos (2.6%) are colour-tinted, with over half their pixels differing; luma keeps the image structure in both cases. RGB would triple the cache (about 65 GB) and the read I/O, and per-channel statistics would break the `stats.json` contract |
| D-24 | 14 Nov 2026 | No combined (distributed) training across laptops | One distributed job over Wi-Fi or Ethernet | Gradient sync costs 11.9–43.6 min per epoch vs 10.4 min of compute on the 4060 alone |
| D-23 | 14 Nov 2026 | Split the 21 runs across RTX 4060, 4050 and 3050 laptops; same commit, same micro-batch; FS-8 seed 0 as hardware control; cost metrics and test evaluation on the 4060 only | Single 4060 | All 3 seeds by Mon \~20:40 instead of 20 of 21 runs by Tue 05:30 |
| D-22 | 14 Nov 2026 | Work split by file ownership; contracts frozen at M0; merges at fixed times (M0–M3) | Shared files; ad-hoc merges | Two people, four days: no time to resolve conflicts |
| D-21 | 14 Nov 2026 | Partner works on synthetic data and aggregated results only; all real-data runs on Pratik's laptops | Share the dataset | Licence forbids sharing and requires individual registration |
| D-20 | 14 Nov 2026 | XAI on 4 priority configs, 100 videos; sanity check on 20 videos | All 7 configs, 200 videos | Fits in about 45 minutes of GPU time |
| D-19 | 14 Nov 2026 | Tier 0: EchoNet weights evaluated at all 7 settings without training | No safety net | A full frame curve for about 40 minutes of GPU time |
| D-18 | 14 Nov 2026 | Fixed learning rate 1e-4, no sweep (supersedes the tuning part of D-05) | LR check on FS-32 | No time; one shared rate stays fair |
| D-17 | 14 Nov 2026 | 10 epochs per run, best-validation checkpoint | 25 epochs with early stopping | One seed of all 7 configs fits in 14.8 h on the 4060 |
| D-16 | 14 Nov 2026 | Seed 0 MUST; seeds 1 and 2 as GPU time allows (supersedes D-11's 3 seeds) | 3 seeds for every config | 4-day deadline |
| D-15 | 14 Nov 2026 | Re-scope to the 18 Nov deadline; finish by Wed 12:00 | The original 6-week plan | Deadline given by the user |
| D-14 | 14 Nov 2026 | Local Gradio demo, `share=False` | Public Hugging Face Space | Licence forbids redistribution |
| D-13 | 14 Nov 2026 | Demo model chosen on validation M2 MAE among Pareto-optimal configs | Choose on test | Avoids test leakage |
| D-12 | 14 Nov 2026 | XAI: occlusion and IG for frames, Grad-CAM for space only, scored by 4 numeric tests | Grad-CAM pictures only | Grad-CAM has T/8 time steps; numbers beat visual judgement |
| D-11 | 14 Nov 2026 | Wilcoxon + Holm over 9 comparisons; bootstrap CIs (seed count now set by D-16) | Single run, no tests | Seed noise can exceed gaps between configs |
| D-10 | 14 Nov 2026 | Primary endpoint = test MAE in M1 | M2 as primary | Isolates per-clip ability from clip count |
| D-09 | 14 Nov 2026 | Zero-padding of short videos (EchoNet rule) plus unpadded-subset sensitivity check | Loop or repeat frames | Comparability with B0; the check shows the effect |
| D-08 | 14 Nov 2026 | Memory-mapped uint8 frame cache, grayscale | Decode AVI each step | Reads only needed frames; keeps the GPU fed at T = 8 |
| D-07 | 14 Nov 2026 | OmegaConf YAML + TensorBoard + `all_runs.csv` | Hydra; W&B | Simple, offline, one source for numbers |
| D-06 | 14 Nov 2026 | bf16 autocast, micro-batch 8, accumulation 2, checkpointing at T = 64 if needed | fp16; smaller batches | Fits 8 GB; equal BatchNorm batch for all T |
| D-05 | 14 Nov 2026 | AdamW + cosine decay (learning rate now set by D-18) | SGD as EchoNet | Converges in fewer epochs |
| D-04 | 14 Nov 2026 | Two sampling regimes: fixed stride (p = 2) and fixed 64-frame window | Frame count only | Separates density from beat coverage |
| D-03 | 14 Nov 2026 | All sweep runs start from Kinetics-400; EchoNet weights used only as baseline B0 and Tier 0 | Fine-tune from EchoNet weights | EchoNet weights were trained at T = 32 and would favour it |
| D-02 | 14 Nov 2026 | R(2+1)D-18 backbone | Transformers; echo foundation models; EchoGNN | Proven on this dataset; fits 8 GB; keeps the study about frames |
| D-01 | 14 Nov 2026 | Train on the local RTX 4060 under WSL2 (extended by D-23) | Colab, Kaggle | No session limits; consistent cost measurements; data stays local |

## Sources

Pages consulted for this spec on 14 Nov 2026, plus the method papers the report must cite.

**Dataset and baseline**

- Ouyang et al., [Video-based AI for beat-to-beat assessment of cardiac function](https://utstat.toronto.edu/reid/sta2101f/s41586-020-2145-8.pdf), Nature 580, 252–256 (2020); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/32269341/)
- [EchoNet-Dynamic dataset page and Research Use Agreement](https://echonet.github.io/dynamic/)
- Ouyang et al., [EchoNet-Dynamic dataset paper, ML4H workshop 2019](https://douyang.github.io/NeuroIPS_2019_ML4H%20Workshop_Paper.pdf)
- [echonet/dynamic GitHub repository](https://github.com/echonet/dynamic) and its [training code](https://github.com/echonet/dynamic/blob/master/echonet/utils/video.py)
- [EchoNet dataset loader (mirror)](https://huggingface.co/spaces/alaa-lab/Med-Real2Sim/blob/main/echonet/datasets/echo.py) and [hyperparameter-sweep plot script (mirror)](https://huggingface.co/spaces/alaa-lab/Med-Real2Sim/blob/d3d13a82cb49ea718ab68610de5838ed0eb635c4/dynamic/scripts/plot_hyperparameter_sweep.py)
- [zea implementation of EchoNet, with release-weight URLs](https://zea.readthedocs.io/en/latest/_modules/zea/models/echonet.html)
- [The segmentation ceiling (arXiv 2609.19730)](https://arxiv.org/pdf/2609.19730), for split sizes
- [EchoNet embeddings study (PMC11450854)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11450854), for frame-count and EF statistics

**Related models**

- [EchoCoTr (arXiv 2209.04242)](https://ar5iv.labs.arxiv.org/html/2209.04242)
- Mokhtari et al., [EchoGNN, MICCAI 2022](https://arxiv.org/abs/2208.14003v1) and [code](https://github.com/MasoudMo/echognn)
- [EchoPrime](https://ai.stanford.edu/~bryanhe/publications/echoprime.pdf)
- [PanEcho (medRxiv)](https://medrxiv.org/content/10.1101/2024.11.16.24317431v1.full)

**Tools and hardware**

- [torchvision r2plus1d\_18 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.r2plus1d_18.html)
- [PyTorch: CUDA 13.0 as stable PyPI wheels from 2.11](https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325.md); [PyTorch dev-discuss latest (2.14/2.15 notices)](https://dev-discuss.pytorch.org/latest.md)
- [Captum Grad-CAM API (interpolation modes for 5D)](https://captum.ai/api/guided_grad_cam.html)
- [Gradio Video component](https://www.gradio.app/docs/gradio/video)
- [Notebookcheck: RTX 4060 desktop vs laptop](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4060-vs-NVIDIA-GeForce-RTX-4060-Laptop-GPU-vs-NVIDIA-GeForce-RTX-4060-Ti-8G_11593_11455_11592.247598.0.html); [hmc-tech: RTX 4060 laptop tensor throughput](https://hmc-tech.com/gpus/nvidia-geforce-rtx-4060-laptop-85w-vs-nvidia-geforce-rtx-4060-max-q)

**Clinical definitions**

- Bozkurt et al., [Universal definition and classification of heart failure (2021)](https://www.pubmed.gov/33605000/)

**Method papers to cite in the report**

- Tran et al., A Closer Look at Spatiotemporal Convolutions for Action Recognition, CVPR 2018 (R(2+1)D)
- Sundararajan et al., Axiomatic Attribution for Deep Networks, ICML 2017 (Integrated Gradients)
- Selvaraju et al., Grad-CAM, ICCV 2017
- Adebayo et al., [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292), NeurIPS 2018
- Bland and Altman, Statistical methods for assessing agreement between two methods of clinical measurement, Lancet 1986
- Holm, A simple sequentially rejective multiple test procedure, Scandinavian Journal of Statistics 1979
