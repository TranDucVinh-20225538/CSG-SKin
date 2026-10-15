# Paper B R5 / Q4 — a label-free collapse statistic from ID data only

Committed before any Q4 run (no Q4 output exists at this commit). Fixed now; deviations appended.

## Question

The mechanism says OOD features collapse onto the NV (majority-class) centroid. Can a statistic computed from ID data
only (no OOD data, no OOD labels) rank runs by their OOD AUROC, across every run (R3, R4, R5; all λ, seeds, backbones,
tasks), with a leave-one-task-out check once Q1 exists?

## Runs (manifest; code: `scripts/r5_q4_manifest.py`)

`results/paperB/r5/q4/manifest.csv`, one row per run with feature directory, representation, ID train / test split names,
OOD split names, primary OOD set, number of classes, majority class, status. Ready now: 35 lesion-level runs
(`z_lesion_norm`) and 31 image-level Phase 13 runs (same keys). Added when features exist: single-encoder Phase 15.1 runs
(P5 extraction; `z`), P4 adversary-LR runs (GPU part), and every R5/R6 GPU run (P7b, P8, Q1, Q2, Q3, R6 R1/R2, …) through
`--extra <csv>` lists in the same columns. Runs without cached features (Phase 15.2 MMD/CORAL, ResNet-50 Phase 2.5,
Camelyon, HAM/BCN Phase 16) stay `pending` until features are extracted; they are listed, not dropped.

## Statistics (code: `scripts/r5_q4_collapse.py`; fixed now)

Fit on ID train (labels give class means and the prior); evaluated on ID test features without their labels:
- S1 `maj_share_excess` (PRIMARY): share of ID-test points whose nearest class mean (pooled-covariance Mahalanobis + 1e-3·I,
  the detector's geometry) is the majority class, minus the majority prior in ID train.
- S1r `maj_share` (raw share); S2 `fisher_ratio` = tr(S_between) / tr(S_within) on ID train; S3 `part_ratio` =
  participation ratio of the ID-test covariance / dimension; S4 `pred_maj_excess` = share of ID-test argmax = majority
  minus the prior.
Outcome: Mahalanobis AUROC on the run's primary OOD set (ISIC runs: pad_heldout; Fitzpatrick also stored), from the same
features.

## Analysis

Spearman ρ across ready runs (all; within each family), 95% CI from 2,000 run resamples (rng 20261005); partial ρ given λ
(rank residuals) to separate the statistic from the λ setting itself. Threshold transfer: outcome "AUROC < 0.5";
threshold = Youden-optimal on all other groups, tested on the held-out group; groups = task when ≥ 2 tasks are ready
(leave-one-task-out after Q1), otherwise family (leave-one-protocol-out now: lesion vs image split).

## PREDICTED OUTCOME (written now)

- S1 on ID test barely moves with λ (R2 item 2: ID NV-nearest fraction 0.526–0.532 at every λ), so ρ(S1, AUROC) across
  the 66 ISIC runs is weak: |ρ| ≤ 0.3, CI including 0; partial ρ given λ ≈ 0.
- S3 (participation ratio) is the strongest ID-only statistic: ρ = 0.4–0.7 (rank collapses with λ; R3 C spectra:
  effective rank ≈ 5–6 of 16 at λ = 2).
- Leave-one-family-out transfer: sensitivity ≥ 0.8 for S3, ≤ 0.5 for S1.
- Reading I expect: an ID-only statistic tracks λ (the alignment strength) but not the OOD-specific pull toward NV; the
  deployment-data signals of R6 R4 do better.
Alternative (≈ 30 %): S1 rises by 0.02–0.05 with λ and reaches |ρ| ≈ 0.5.

## Gates (flag, never drop)

Families with < 5 ready runs are reported in the per-run table but get no within-family ρ.

## Outputs

`results/paperB/r5/q4/`: `manifest.csv`, `per_run/*.json`, `summary.csv`, `correlations.csv`, `transfer.csv`, `REPORT.md`.

## Cost

GPU 0 (extraction for pending families is booked under their own items). CPU ≈ 0.3 h on 8 CPUs for 66–100 runs.

## Stop rules

Statistic set and primary fixed now; no new statistic is added after seeing correlations (any later addition is a
labelled exploratory deviation).

## Deviations
