# R2 Item 2 — does the inversion run through the NV centroid?

Commit: `4c62482`. Pre-commit: `results/paperB/r2/PRECOMMIT_ITEM1_ITEM2.json` (`e131a42`).

**Verdict:** All three hold -> the mechanism is identified.

Standard class-conditional Mahalanobis (pooled covariance + 1e-3 I, fit on ISIC train). Nearest centroid = argmin of the same 8 distances. ID = full ISIC test. Distances are squared Mahalanobis. Mean ± s.d. across seeds.

## (a) z_lesion^norm, OOD = pad_heldout (primary)

| λ | n | AUROC all | NV-nearest frac (OOD) | NV-nearest frac (ID) | AUROC NV-nearest | AUROC non-NV-nearest | predicted NV (OOD) |
|---:|---:|---|---|---|---|---|---|
| 0 | 5 | 0.843 ± 0.023 | 0.332 ± 0.054 | 0.528 ± 0.004 | 0.807 ± 0.026 | 0.861 ± 0.025 | 105 |
| 0.25 | 3 | 0.513 ± 0.018 | 0.531 ± 0.031 | 0.532 ± 0.003 | 0.356 ± 0.021 | 0.691 ± 0.009 | 373 |
| 0.5 | 3 | 0.465 ± 0.015 | 0.599 ± 0.012 | 0.530 ± 0.004 | 0.318 ± 0.026 | 0.686 ± 0.017 | 425 |
| 1 | 3 | 0.455 ± 0.036 | 0.561 ± 0.036 | 0.526 ± 0.008 | 0.281 ± 0.033 | 0.678 ± 0.010 | 396 |
| 2 | 5 | 0.427 ± 0.025 | 0.601 ± 0.029 | 0.531 ± 0.007 | 0.282 ± 0.023 | 0.644 ± 0.005 | 424 |
| 4 | 3 | 0.455 ± 0.039 | 0.588 ± 0.033 | 0.526 ± 0.004 | 0.307 ± 0.022 | 0.666 ± 0.037 | 417 |
| 8 | 5 | 0.494 ± 0.027 | 0.586 ± 0.049 | 0.530 ± 0.010 | 0.346 ± 0.025 | 0.707 ± 0.020 | 412 |

## (b) z_lesion^norm, OOD = pad_heldout (primary)

Ratio = median distance(OOD → class-c centroid) / median distance(ID test images of class c → class-c centroid).

| λ | ratio MEL | ratio NV | ratio BCC | ratio AK | ratio BKL | ratio DF | ratio VASC | ratio SCC | med OOD→NV | med ID-NV→NV |
|---:|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.60 ± 0.18 | 6.63 ± 0.45 | 2.83 ± 0.51 | 1.87 ± 0.16 | 1.84 ± 0.18 | 2.92 ± 0.73 | 4.04 ± 1.62 | 2.05 ± 0.23 | 49.8 | 7.5 |
| 0.25 | 3.44 ± 0.85 | 4.05 ± 0.66 | 4.21 ± 0.61 | 2.65 ± 0.37 | 2.65 ± 0.25 | 2.98 ± 1.03 | 4.83 ± 1.62 | 2.54 ± 0.03 | 29.6 | 7.4 |
| 0.5 | 3.41 ± 0.40 | 2.70 ± 0.37 | 3.77 ± 0.12 | 2.74 ± 0.12 | 2.56 ± 0.36 | 2.78 ± 0.30 | 5.46 ± 1.86 | 2.54 ± 0.70 | 19.5 | 7.2 |
| 1 | 3.06 ± 0.30 | 3.05 ± 0.95 | 4.25 ± 0.32 | 3.06 ± 0.20 | 2.63 ± 0.23 | 3.85 ± 0.93 | 3.77 ± 1.01 | 2.60 ± 0.35 | 23.8 | 7.8 |
| 2 | 2.90 ± 0.22 | 2.37 ± 0.53 | 4.10 ± 0.45 | 2.74 ± 0.25 | 2.62 ± 0.24 | 3.49 ± 0.39 | 5.63 ± 0.55 | 2.65 ± 0.36 | 17.5 | 7.4 |
| 4 | 2.99 ± 0.16 | 2.77 ± 0.79 | 4.03 ± 0.27 | 2.67 ± 0.43 | 2.84 ± 0.36 | 3.45 ± 0.33 | 4.31 ± 0.23 | 2.81 ± 0.32 | 20.2 | 7.4 |
| 8 | 2.95 ± 0.56 | 2.81 ± 0.55 | 4.11 ± 0.20 | 2.67 ± 0.35 | 2.79 ± 0.37 | 3.10 ± 0.22 | 5.52 ± 1.17 | 2.46 ± 0.41 | 22.0 | 7.7 |

## Pre-committed criteria (primary block)

| Criterion | Holds | Detail |
|---|---|---|
| C1 NV-nearest fraction rises | yes | λ=0.25: Δ=0.199 vs 2SE=0.060; λ=2: Δ=0.268 vs 2SE=0.055 |
| C2 inversion concentrated in NV-nearest subset | yes | λ=0.25: NV 0.356 / non-NV 0.691; λ=0.5: NV 0.318 / non-NV 0.686; λ=1: NV 0.281 / non-NV 0.678; λ=2: NV 0.282 / non-NV 0.644; λ=4: NV 0.307 / non-NV 0.666; λ=8: NV 0.346 / non-NV 0.707 |
| C3 OOD→NV distance shrinks vs class-matched reference | yes | λ=0.25: Δratio=2.579 vs 2SE=0.864; λ=2: Δratio=4.262 vs 2SE=0.616 |

Spearman(NV-nearest fraction, predicted-NV count on pad_heldout) across all runs: ρ = 0.982 (p = 1.2e-19).

## (a) repeat on pre-projection backbone, OOD = pad_heldout

| λ | n | AUROC all | NV-nearest frac (OOD) | NV-nearest frac (ID) | AUROC NV-nearest | AUROC non-NV-nearest | predicted NV (OOD) |
|---:|---:|---|---|---|---|---|---|
| 0 | 5 | 0.749 ± 0.052 | 0.484 ± 0.148 | 0.550 ± 0.003 | 0.726 ± 0.057 | 0.768 ± 0.045 | 105 |
| 0.25 | 3 | 0.629 ± 0.011 | 0.561 ± 0.022 | 0.550 ± 0.003 | 0.490 ± 0.029 | 0.805 ± 0.002 | 373 |
| 0.5 | 3 | 0.557 ± 0.015 | 0.628 ± 0.024 | 0.551 ± 0.002 | 0.417 ± 0.016 | 0.794 ± 0.004 | 425 |
| 1 | 3 | 0.548 ± 0.049 | 0.583 ± 0.034 | 0.550 ± 0.004 | 0.380 ± 0.058 | 0.784 ± 0.011 | 396 |
| 2 | 5 | 0.489 ± 0.040 | 0.603 ± 0.040 | 0.549 ± 0.004 | 0.330 ± 0.025 | 0.730 ± 0.026 | 424 |
| 4 | 3 | 0.497 ± 0.041 | 0.602 ± 0.026 | 0.546 ± 0.003 | 0.341 ± 0.022 | 0.732 ± 0.053 | 417 |
| 8 | 5 | 0.525 ± 0.035 | 0.606 ± 0.038 | 0.550 ± 0.008 | 0.381 ± 0.051 | 0.751 ± 0.019 | 412 |

## (b) repeat on pre-projection backbone, OOD = pad_heldout

Ratio = median distance(OOD → class-c centroid) / median distance(ID test images of class c → class-c centroid).

| λ | ratio MEL | ratio NV | ratio BCC | ratio AK | ratio BKL | ratio DF | ratio VASC | ratio SCC | med OOD→NV | med ID-NV→NV |
|---:|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.61 ± 0.26 | 2.76 ± 0.33 | 1.07 ± 0.07 | 0.85 ± 0.10 | 1.04 ± 0.11 | 0.83 ± 0.08 | 1.14 ± 0.10 | 0.84 ± 0.07 | 2177.4 | 791.5 |
| 0.25 | 1.54 ± 0.13 | 2.41 ± 0.21 | 0.93 ± 0.07 | 0.75 ± 0.06 | 0.98 ± 0.09 | 0.77 ± 0.16 | 1.15 ± 0.08 | 0.76 ± 0.09 | 1975.4 | 824.9 |
| 0.5 | 1.19 ± 0.11 | 1.91 ± 0.07 | 0.76 ± 0.07 | 0.65 ± 0.07 | 0.80 ± 0.09 | 0.69 ± 0.17 | 1.06 ± 0.07 | 0.64 ± 0.07 | 1554.7 | 812.8 |
| 1 | 1.09 ± 0.16 | 1.69 ± 0.17 | 0.82 ± 0.10 | 0.70 ± 0.07 | 0.80 ± 0.12 | 0.71 ± 0.06 | 1.01 ± 0.10 | 0.69 ± 0.10 | 1537.9 | 911.1 |
| 2 | 0.88 ± 0.08 | 1.48 ± 0.21 | 0.69 ± 0.07 | 0.59 ± 0.06 | 0.68 ± 0.07 | 0.61 ± 0.07 | 0.91 ± 0.21 | 0.63 ± 0.05 | 1308.7 | 883.9 |
| 4 | 1.00 ± 0.16 | 1.54 ± 0.30 | 0.70 ± 0.11 | 0.67 ± 0.09 | 0.72 ± 0.08 | 0.65 ± 0.11 | 0.94 ± 0.11 | 0.65 ± 0.09 | 1362.8 | 886.6 |
| 8 | 1.12 ± 0.19 | 1.60 ± 0.06 | 0.84 ± 0.13 | 0.75 ± 0.11 | 0.82 ± 0.11 | 0.68 ± 0.08 | 0.93 ± 0.15 | 0.72 ± 0.10 | 1523.3 | 952.3 |

## (a) z_lesion^norm, OOD = Fitzpatrick17k (secondary)

| λ | n | AUROC all | NV-nearest frac (OOD) | NV-nearest frac (ID) | AUROC NV-nearest | AUROC non-NV-nearest | predicted NV (OOD) |
|---:|---:|---|---|---|---|---|---|
| 0 | 5 | 0.764 ± 0.038 | 0.528 ± 0.025 | 0.528 ± 0.004 | 0.722 ± 0.034 | 0.810 ± 0.042 | 1582 |
| 0.25 | 3 | 0.439 ± 0.061 | 0.795 ± 0.063 | 0.532 ± 0.003 | 0.385 ± 0.067 | 0.659 ± 0.010 | 3039 |
| 0.5 | 3 | 0.449 ± 0.025 | 0.761 ± 0.021 | 0.530 ± 0.004 | 0.374 ± 0.042 | 0.684 ± 0.053 | 2938 |
| 1 | 3 | 0.457 ± 0.088 | 0.766 ± 0.088 | 0.526 ± 0.008 | 0.387 ± 0.079 | 0.703 ± 0.019 | 2938 |
| 2 | 5 | 0.391 ± 0.068 | 0.809 ± 0.055 | 0.531 ± 0.007 | 0.332 ± 0.056 | 0.647 ± 0.037 | 3110 |
| 4 | 3 | 0.431 ± 0.036 | 0.798 ± 0.021 | 0.526 ± 0.004 | 0.373 ± 0.042 | 0.661 ± 0.030 | 3090 |
| 8 | 5 | 0.474 ± 0.060 | 0.749 ± 0.099 | 0.530 ± 0.010 | 0.411 ± 0.053 | 0.679 ± 0.024 | 2839 |

## (b) z_lesion^norm, OOD = Fitzpatrick17k (secondary)

Ratio = median distance(OOD → class-c centroid) / median distance(ID test images of class c → class-c centroid).

| λ | ratio MEL | ratio NV | ratio BCC | ratio AK | ratio BKL | ratio DF | ratio VASC | ratio SCC | med OOD→NV | med ID-NV→NV |
|---:|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.66 ± 0.38 | 5.31 ± 0.65 | 3.45 ± 0.46 | 2.06 ± 0.15 | 1.95 ± 0.16 | 3.05 ± 0.79 | 4.29 ± 1.62 | 2.25 ± 0.22 | 39.9 | 7.5 |
| 0.25 | 3.42 ± 0.76 | 1.62 ± 0.51 | 4.31 ± 0.53 | 2.64 ± 0.39 | 2.56 ± 0.18 | 2.88 ± 1.04 | 4.68 ± 1.77 | 2.55 ± 0.06 | 12.2 | 7.4 |
| 0.5 | 3.40 ± 0.30 | 1.75 ± 0.35 | 3.89 ± 0.13 | 2.77 ± 0.17 | 2.51 ± 0.40 | 2.74 ± 0.32 | 5.33 ± 1.77 | 2.58 ± 0.73 | 12.6 | 7.2 |
| 1 | 3.23 ± 0.11 | 1.90 ± 0.97 | 4.37 ± 0.58 | 3.07 ± 0.13 | 2.49 ± 0.24 | 3.73 ± 0.90 | 3.73 ± 1.01 | 2.59 ± 0.36 | 14.9 | 7.8 |
| 2 | 3.00 ± 0.22 | 1.35 ± 0.37 | 4.23 ± 0.50 | 2.73 ± 0.23 | 2.50 ± 0.24 | 3.46 ± 0.39 | 5.52 ± 0.58 | 2.67 ± 0.37 | 10.0 | 7.4 |
| 4 | 3.07 ± 0.14 | 1.55 ± 0.36 | 4.10 ± 0.29 | 2.67 ± 0.46 | 2.77 ± 0.46 | 3.37 ± 0.28 | 4.24 ± 0.27 | 2.83 ± 0.32 | 11.4 | 7.4 |
| 8 | 2.94 ± 0.49 | 1.93 ± 0.60 | 4.18 ± 0.15 | 2.63 ± 0.36 | 2.60 ± 0.36 | 3.03 ± 0.24 | 5.42 ± 1.11 | 2.47 ± 0.43 | 15.2 | 7.7 |

## (a) pre-projection backbone, OOD = Fitzpatrick17k (secondary)

| λ | n | AUROC all | NV-nearest frac (OOD) | NV-nearest frac (ID) | AUROC NV-nearest | AUROC non-NV-nearest | predicted NV (OOD) |
|---:|---:|---|---|---|---|---|---|
| 0 | 5 | 0.813 ± 0.051 | 0.637 ± 0.079 | 0.550 ± 0.003 | 0.799 ± 0.056 | 0.838 ± 0.040 | 1582 |
| 0.25 | 3 | 0.603 ± 0.076 | 0.809 ± 0.062 | 0.550 ± 0.003 | 0.565 ± 0.092 | 0.781 ± 0.036 | 3039 |
| 0.5 | 3 | 0.587 ± 0.031 | 0.790 ± 0.013 | 0.551 ± 0.002 | 0.531 ± 0.039 | 0.799 ± 0.053 | 2938 |
| 1 | 3 | 0.612 ± 0.102 | 0.805 ± 0.067 | 0.550 ± 0.004 | 0.564 ± 0.108 | 0.827 ± 0.024 | 2938 |
| 2 | 5 | 0.516 ± 0.064 | 0.813 ± 0.049 | 0.549 ± 0.004 | 0.465 ± 0.053 | 0.747 ± 0.048 | 3110 |
| 4 | 3 | 0.532 ± 0.048 | 0.822 ± 0.010 | 0.546 ± 0.003 | 0.486 ± 0.054 | 0.745 ± 0.021 | 3090 |
| 8 | 5 | 0.595 ± 0.099 | 0.775 ± 0.086 | 0.550 ± 0.008 | 0.552 ± 0.104 | 0.769 ± 0.029 | 2839 |

## (c) Re-reading Table 4 (ID reweighted to PAD class mix)

Phase 2.5 runB_orth1 features, OOD = pad_full, 5 seeds. Weighted AUROC = Σ_c share_c · AUROC(OOD vs ID class c).

| Quantity | mean ± s.d. |
|---|---|
| Unweighted (Table 4 col 1) | 0.409 ± 0.022 |
| Reweighted, observed (Table 4 col 3) | 0.240 ± 0.014 |
| Reweighted, from per-class decomposition | 0.240 ± 0.014 |
| Only NV share moved to PAD level (10.6%) | 0.301 ± 0.016 |
| Fraction of the drop reproduced by the NV-only change | 0.636 ± 0.019 |

| Class | ISIC share | PAD-weighted share | AUROC(OOD vs ID class), mean ± s.d. |
|---|---|---|---|
| MEL | 0.178 | 0.023 | 0.364 ± 0.022 |
| NV | 0.508 | 0.106 | 0.540 ± 0.032 |
| BCC | 0.131 | 0.368 | 0.237 ± 0.020 |
| AK | 0.034 | 0.318 | 0.155 ± 0.015 |
| BKL | 0.104 | 0.102 | 0.257 ± 0.015 |
| DF | 0.009 | 0.000 | 0.130 ± 0.021 |
| VASC | 0.010 | 0.000 | 0.162 ± 0.029 |
| SCC | 0.025 | 0.084 | 0.141 ± 0.015 |

Magnitude consistent with the NV hypothesis (pre-committed: NV-only change reproduces ≥ half the drop): **yes**.

## Caveats

- The class-composition test in Table 4 reweighted the ID set only; (a)–(b) here locate the OOD samples in feature space.
- Fitzpatrick17k is secondary; it does not enter the criteria.
- **The NV ratio stays above 1 at every λ; this sets the wording of the mechanism claim.** On z_lesion^norm / pad_heldout the NV ratio falls from 6.63 to 2.37 at λ=2 but never reaches 1: OOD images never get closer to the NV centroid than real ID nevi are. Per class in (c) (Phase 2.5 features, OOD = pad_full), AUROC(OOD vs ID-NV) is 0.540, near chance and not inverted, while AUROC(OOD vs every non-NV ID class) is 0.130–0.364. The inversion therefore arises because OOD moves into the NV region, where it looks less atypical than ID images of the non-NV classes look relative to their own centroids. It does not arise because OOD lands closer to the NV centroid than nevi do. The claim should be worded as "OOD collapses toward the NV centroid", not "OOD becomes more NV-like than nevi".
- **C1 is close to tautological with the prediction shift.** With a shared covariance, the nearest Mahalanobis centroid is a linear-discriminant classification, so the NV-nearest fraction largely restates how often OOD images are predicted NV (105 → 424 of 716 on pad_heldout; Spearman ρ = 0.982 across runs). C1 confirms the shift but adds little independent evidence; C2, C3 and (c) carry the mechanism.
- **At the pre-projection backbone the shift is broad, not NV-specific.** From λ=0 to λ=2 the OOD-to-centroid ratio shrinks for every class, not only NV: NV 2.76 → 1.48, MEL 1.61 → 0.88, BCC 1.07 → 0.69, AK 0.85 → 0.59, SCC 0.84 → 0.63. The NV-nearest fraction rises only from 0.48 ± 0.15 to 0.60. The NV-specific concentration is a property of the 16-d projection z_lesion^norm, not of the backbone.
