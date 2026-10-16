# Paper B: BCN→HAM Mahalanobis at λ = 0 — label shift vs dispersion

Design: `decisions/paperB_r5_bcnham_maha0.md`. Commit `25016a6`. Seeds (42, 52, 62). Detectors fit on BCN train; OOD = HAM-heldout; ID test = BCN-test. Lesion-clustered 95% CI (PREREGISTER 2000 draws, rng 20261005). Classes with n < 20 on either side are excluded from the reading (CI still shown).

## (1) Class mix (this split)

BCN-test n = 1265, HAM-heldout n = 3040. TV = 0.364. Majority: BCN-test NV (0.306) → HAM-heldout NV (0.667).

| class | n BCN-test | π BCN-test | n HAM-heldout | π HAM | eligible |
|---|---:|---:|---:|---:|:---:|
| MEL | 281 | 0.222 | 360 | 0.118 | yes |
| NV | 387 | 0.306 | 2028 | 0.667 | yes |
| BCC | 272 | 0.215 | 158 | 0.052 | yes |
| AK | 97 | 0.077 | 32 | 0.011 | yes |
| BKL | 138 | 0.109 | 329 | 0.108 | yes |
| DF | 27 | 0.021 | 37 | 0.012 | yes |
| VASC | 10 | 0.008 | 34 | 0.011 | no |
| SCC | 53 | 0.042 | 62 | 0.020 | yes |

## (3) Pooled AUROC (lesion CI) and OOD-side reweight to BCN-test mix

| detector / weighting | mean ± s.d. | 95% CI (lesion) |
|---|---|---|
| Mahalanobis unweighted | 0.369 ± 0.039 | [0.337, 0.402] |
| kNN k=50 unweighted | 0.658 ± 0.041 | [0.626, 0.688] |
| Mahalanobis, HAM reweighted → BCN-test mix | 0.489 ± 0.038 | [0.453, 0.524] |

Share of gap to 0.5 recovered by reweighting: **91.9%** (unweighted 0.369 → reweighted 0.489).

## (2) Per-class class-matched AUROC (HAM class c vs BCN-test class c)

| class | n_BCN | n_HAM | Maha mean ± sd [CI] | kNN mean ± sd [CI] | in reading |
|---|---:|---:|---|---|:---:|
| MEL | 281 | 360 | 0.443 ± 0.047 [0.362, 0.527] | 0.620 ± 0.052 [0.540, 0.693] | yes |
| NV | 387 | 2028 | 0.483 ± 0.065 [0.418, 0.546] | 0.639 ± 0.067 [0.584, 0.693] | yes |
| BCC | 272 | 158 | 0.498 ± 0.080 [0.437, 0.561] | 0.656 ± 0.021 [0.585, 0.729] | yes |
| AK | 97 | 32 | 0.685 ± 0.150 [0.570, 0.792] | 0.595 ± 0.053 [0.447, 0.733] | yes |
| BKL | 138 | 329 | 0.526 ± 0.008 [0.445, 0.615] | 0.623 ± 0.027 [0.542, 0.705] | yes |
| DF | 27 | 37 | 0.577 ± 0.022 [0.360, 0.768] | 0.717 ± 0.085 [0.593, 0.852] | yes |
| VASC | 10 | 34 | 0.616 ± 0.110 [0.380, 0.773] | 0.325 ± 0.118 [0.023, 0.606] | no |
| SCC | 53 | 62 | 0.407 ± 0.072 [0.275, 0.528] | 0.714 ± 0.056 [0.594, 0.826] | yes |

## (4) Per-class dispersion (median Mahalanobis² to own class centroid)

| class | n_BCN | n_HAM | median d² BCN-test | median d² HAM | HAM/BCN | in reading |
|---|---:|---:|---:|---:|---:|:---:|
| MEL | 281 | 360 | 1709.83 | 1565.64 | 0.92 | yes |
| NV | 387 | 2028 | 1279.62 | 1236.65 | 0.97 | yes |
| BCC | 272 | 158 | 2025.13 | 2006.89 | 0.99 | yes |
| AK | 97 | 32 | 2427.74 | 2920.41 | 1.20 | yes |
| BKL | 138 | 329 | 2194.83 | 2223.94 | 1.01 | yes |
| DF | 27 | 37 | 2493.89 | 2627.01 | 1.05 | yes |
| VASC | 10 | 34 | 2915.94 | 3190.72 | 1.09 | no |
| SCC | 53 | 62 | 2477.56 | 2237.30 | 0.90 | yes |

## Reading

**both.** Mixed: reweighting moves Mahalanobis toward 0.5 (share of gap recovered = 91.9%) but pooled reweighted AUROC is 0.489; 4/7 eligible class-matched Mahalanobis AUROCs are < 0.5. Both label shift and within-class dispersion contribute; do not pick one.

Pre-existing headline (A1 / R6-R5): Mahalanobis λ = 0 ≈ 0.369, kNN ≈ 0.658 — reproduced here as unweighted pooled estimates above.

