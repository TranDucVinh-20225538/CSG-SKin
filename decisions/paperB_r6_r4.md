# Paper B R6 / R4 — label-free warning signals on unlabeled deployment data

Committed before any R4 run (no R4 output exists at this commit). Fixed now; deviations appended.

## Question

Using unlabeled deployment data only, do (a) the fraction predicted as the majority class, (b) the BBSE estimate of the
target class proportions vs the source prior, and (c) the ID-only collapse statistic of R5 Q4 rank runs by OOD AUROC and
flag every inverted run — across all runs (R3, R4, R5, R6; every λ, seed, backbone, task), with the threshold chosen on
one task and tested on the others?

## Runs

Same manifest as Q4 (`results/paperB/r5/q4/manifest.csv`; ready now: 35 lesion-level + 31 image-level ISIC runs; later
R5/R6 runs and Q1 via `--extra`). Deployment set = each run's primary OOD split (ISIC runs: pad_heldout, 716 images);
its labels are used by no signal.

## Signals (code: `scripts/r6_r4_warning_signals.py`; fixed now)

- W1 `pred_maj_frac_excess`: share of deployment images predicted as the majority class minus the ID-train prior.
- W2 `bbse_tv`: BBSE (Lipton et al. 2018): confusion matrix C[i, j] = P(ŷ = i, y = j) on labelled ID test, deployment
  prediction histogram μ̂, w = C⁻¹μ̂ (least squares, 1e-6·I), q̂ = clip(w·p_src, 0) renormalised; TV(q̂, p_src).
  W2b `bbse_maj_ratio` = q̂_maj / p_src,maj. Condition number of C reported.
- W3 `q4_maj_share_excess`: the Q4 primary statistic (ID test only).
- W3d `deploy_maj_share_excess`: the same geometry on deployment features (majority-centroid share minus prior).
Validation only (deployment labels, where they exist): true TV(q, p_src) and BBSE error TV(q̂, q).

## Analysis

Spearman ρ with the deployment Mahalanobis AUROC across ready runs and within family, 95% CI from 2,000 run resamples;
partial ρ given λ. Threshold transfer for "AUROC < 0.5": Youden threshold on all other groups, sensitivity / specificity
on the held-out group (task when Q1 exists, otherwise family). "Flags every inverted run" = held-out sensitivity 1.0.

## PREDICTED OUTCOME (written now)

- W1 and W3d track the inversion closely: ρ ≤ −0.7 (more majority predictions / majority-centroid share → lower AUROC);
  pad_heldout predicted-NV share 0.10–0.20 at λ = 0 vs ≥ 0.5 at λ ≥ 0.5 (R2 item 2: 105 → 424 of 716 images).
  Held-out sensitivity 1.0 for W1 across the lesion/image families.
- W2 (BBSE TV) is distorted by the same collapse: the confusion matrix is estimated on ID, so a collapsed classifier
  makes q̂ overstate NV; ρ(W2, AUROC) weaker (|ρ| 0.3–0.6) and BBSE error large (TV error ≥ 0.3 at λ ≥ 0.5; true
  ISIC→pad_heldout TV ≈ 0.55).
- W3 (ID-only) weak, as predicted in Q4 (|ρ| ≤ 0.3).

## Gates (flag, never drop)

BBSE condition number > 1e3 → that run's W2 flagged "ill-conditioned". Families with < 5 runs: no within-family ρ.

## Outputs

`results/paperB/r6/r4/`: `per_run/*.json`, `summary.csv`, `correlations.csv`, `transfer.csv`, `REPORT.md`.

## Cost

GPU 0. CPU ≈ 0.3 h on 8 CPUs.

## Stop rules

Signal set and thresholding rule fixed now.

## Deviations
