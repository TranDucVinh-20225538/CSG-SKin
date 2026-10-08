# R2 — Relative Mahalanobis Distance (RMD)

Commit: `892241d`. Pre-commit: `results/paperB/r2/PRECOMMIT_RMD.json` (`958684a`).

**Verdict:** As predicted: RMD removes part of the inversion and does not repair the collapse

**RMD applies only to Mahalanobis. kNN, MSP, Energy and cosine also collapsed with λ and have no RMD analogue; whatever RMD recovers is a partial fix of the monitoring failure, not a fix.**

Score: min_k (MD_k − MD_0). MD_k: class means, pooled within-class covariance + 1e-3 I. MD_0: one Gaussian on ISIC train ignoring labels (full covariance + 1e-3 I). All statistics fit on ISIC train only.

## Pre-committed test (z_lesion^norm, ID = full ISIC test, λ = 2, mean of 5 seeds)

f = (AUROC_RMD − AUROC_MD) / (0.5 − AUROC_MD): fraction of the below-chance gap removed. Predicted f = 0.5 on both sets; correct if 0.25 ≤ f ≤ 0.75 and AUROC_RMD ≤ 0.60.

| OOD set | AUROC MD | AUROC RMD | f observed | f predicted |
|---|---|---|---|---|
| pad_heldout | 0.427 | 0.477 | 0.69 | 0.50 |
| fitzpatrick17k | 0.391 | 0.467 | 0.69 | 0.50 |

## z_lesion^norm (16-d), primary — pad_heldout (mean ± s.d. across seeds)

| λ | n | MD, ID full | RMD, ID full | f (per-seed mean) | MD, ID lesion-disjoint | RMD, ID lesion-disjoint |
|---:|---:|---|---|---|---|---|
| 0 | 5 | 0.843 ± 0.023 | 0.898 ± 0.018 | — | 0.890 ± 0.015 | 0.919 ± 0.015 |
| 0.25 | 3 | 0.513 ± 0.018 | 0.556 ± 0.015 | 13.06 ± 0.00 | 0.597 ± 0.032 | 0.581 ± 0.025 |
| 0.5 | 3 | 0.465 ± 0.015 | 0.528 ± 0.013 | 2.08 ± 1.04 | 0.553 ± 0.012 | 0.558 ± 0.014 |
| 1 | 3 | 0.455 ± 0.036 | 0.500 ± 0.023 | 2.25 ± 2.53 | 0.535 ± 0.036 | 0.523 ± 0.023 |
| 2 | 5 | 0.427 ± 0.025 | 0.477 ± 0.019 | 0.70 ± 0.28 | 0.515 ± 0.026 | 0.503 ± 0.017 |
| 4 | 3 | 0.455 ± 0.039 | 0.508 ± 0.022 | 3.08 ± 3.56 | 0.545 ± 0.040 | 0.542 ± 0.024 |
| 8 | 5 | 0.494 ± 0.027 | 0.531 ± 0.024 | 7.85 ± 9.85 | 0.584 ± 0.016 | 0.561 ± 0.028 |

## z_lesion^norm (16-d), primary — fitzpatrick17k (mean ± s.d. across seeds)

| λ | n | MD, ID full | RMD, ID full | f (per-seed mean) | MD, ID lesion-disjoint | RMD, ID lesion-disjoint |
|---:|---:|---|---|---|---|---|
| 0 | 5 | 0.764 ± 0.038 | 0.817 ± 0.034 | — | 0.830 ± 0.027 | 0.850 ± 0.030 |
| 0.25 | 3 | 0.439 ± 0.061 | 0.513 ± 0.043 | 0.88 ± 0.09 | 0.536 ± 0.052 | 0.546 ± 0.044 |
| 0.5 | 3 | 0.449 ± 0.025 | 0.515 ± 0.028 | 1.60 ± 0.98 | 0.547 ± 0.029 | 0.553 ± 0.037 |
| 1 | 3 | 0.457 ± 0.088 | 0.526 ± 0.050 | 1.00 ± 0.12 | 0.548 ± 0.086 | 0.563 ± 0.051 |
| 2 | 5 | 0.391 ± 0.068 | 0.467 ± 0.047 | 0.79 ± 0.27 | 0.490 ± 0.072 | 0.501 ± 0.049 |
| 4 | 3 | 0.431 ± 0.036 | 0.506 ± 0.024 | 1.28 ± 0.62 | 0.533 ± 0.036 | 0.551 ± 0.020 |
| 8 | 5 | 0.474 ± 0.060 | 0.533 ± 0.045 | 7.35 ± 5.41 | 0.578 ± 0.046 | 0.574 ± 0.046 |

## Lesion backbone, pre-projection (1536-d), secondary — pad_heldout (mean ± s.d. across seeds)

| λ | n | MD, ID full | RMD, ID full | f (per-seed mean) | MD, ID lesion-disjoint | RMD, ID lesion-disjoint |
|---:|---:|---|---|---|---|---|
| 0 | 5 | 0.749 ± 0.052 | 0.905 ± 0.015 | — | 0.841 ± 0.030 | 0.926 ± 0.013 |
| 0.25 | 3 | 0.629 ± 0.011 | 0.600 ± 0.017 | — | 0.721 ± 0.010 | 0.635 ± 0.025 |
| 0.5 | 3 | 0.557 ± 0.015 | 0.559 ± 0.007 | — | 0.659 ± 0.014 | 0.597 ± 0.004 |
| 1 | 3 | 0.548 ± 0.049 | 0.537 ± 0.012 | — | 0.645 ± 0.047 | 0.568 ± 0.010 |
| 2 | 5 | 0.489 ± 0.040 | 0.518 ± 0.014 | 1.31 ± 0.18 | 0.595 ± 0.038 | 0.553 ± 0.013 |
| 4 | 3 | 0.497 ± 0.041 | 0.540 ± 0.011 | 4.85 ± 4.31 | 0.594 ± 0.036 | 0.578 ± 0.007 |
| 8 | 5 | 0.525 ± 0.035 | 0.555 ± 0.019 | 1.99 ± 0.00 | 0.623 ± 0.031 | 0.588 ± 0.021 |

## Lesion backbone, pre-projection (1536-d), secondary — fitzpatrick17k (mean ± s.d. across seeds)

| λ | n | MD, ID full | RMD, ID full | f (per-seed mean) | MD, ID lesion-disjoint | RMD, ID lesion-disjoint |
|---:|---:|---|---|---|---|---|
| 0 | 5 | 0.813 ± 0.051 | 0.822 ± 0.031 | — | 0.883 ± 0.028 | 0.857 ± 0.026 |
| 0.25 | 3 | 0.603 ± 0.076 | 0.556 ± 0.030 | — | 0.712 ± 0.055 | 0.600 ± 0.033 |
| 0.5 | 3 | 0.587 ± 0.031 | 0.554 ± 0.026 | — | 0.697 ± 0.027 | 0.600 ± 0.034 |
| 1 | 3 | 0.612 ± 0.102 | 0.554 ± 0.048 | — | 0.707 ± 0.089 | 0.596 ± 0.049 |
| 2 | 5 | 0.516 ± 0.064 | 0.519 ± 0.017 | 0.94 ± 0.00 | 0.631 ± 0.056 | 0.566 ± 0.014 |
| 4 | 3 | 0.532 ± 0.048 | 0.525 ± 0.038 | -1.44 ± 0.00 | 0.634 ± 0.047 | 0.571 ± 0.032 |
| 8 | 5 | 0.595 ± 0.099 | 0.565 ± 0.044 | 0.86 ± 0.00 | 0.695 ± 0.082 | 0.609 ± 0.040 |

## Caveats

1. f is defined only where MD is below 0.5; it is computed from seed means for the verdict and per seed in the tables (seeds with MD ≥ 0.5 excluded from the per-seed mean).
2. The ISIC test split is image-level; 3041 of 5067 test images share a lesion with train or val. The lesion-disjoint columns (n = 2026) remove that overlap from the ID side.
3. backbone_raw_lesion received adversarial gradient; it is reported for completeness, not as a remedy.
