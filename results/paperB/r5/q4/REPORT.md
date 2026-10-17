# Paper B R5 Q4 — label-free collapse statistic vs OOD AUROC

Design: `decisions/paperB_r5_q4.md`. Commit `a7195ca`. Ready runs: 69 of 69 in the manifest (csg_dual_lesion 35, csg_dual_image 31, single_dann_image 3). Outcome: Mahalanobis AUROC on each run's primary OOD set. ρ = Spearman across runs, 95% CI from 2,000 run resamples.

| subset | statistic | n runs | ρ [95% CI] | partial ρ given λ |
|---|---|---:|---|---|
| all | maj_share_excess | 69 | -0.372 [-0.573, -0.132] | -0.336 |
| all | maj_share | 69 | -0.372 [-0.573, -0.132] | -0.336 |
| all | fisher_ratio | 69 | 0.103 [-0.164, 0.366] | -0.017 |
| all | part_ratio | 69 | -0.136 [-0.359, 0.095] | -0.070 |
| all | pred_maj_excess | 69 | -0.132 [-0.358, 0.122] | -0.090 |
| csg_dual_image | maj_share_excess | 31 | -0.001 [-0.357, 0.351] | -0.042 |
| csg_dual_image | maj_share | 31 | -0.001 [-0.357, 0.351] | -0.042 |
| csg_dual_image | fisher_ratio | 31 | 0.303 [-0.100, 0.606] | 0.130 |
| csg_dual_image | part_ratio | 31 | -0.036 [-0.381, 0.324] | 0.102 |
| csg_dual_image | pred_maj_excess | 31 | 0.098 [-0.296, 0.480] | 0.013 |
| csg_dual_lesion | maj_share_excess | 35 | -0.344 [-0.618, 0.003] | -0.299 |
| csg_dual_lesion | maj_share | 35 | -0.344 [-0.618, 0.003] | -0.299 |
| csg_dual_lesion | fisher_ratio | 35 | -0.123 [-0.453, 0.243] | -0.176 |
| csg_dual_lesion | part_ratio | 35 | -0.179 [-0.469, 0.147] | -0.069 |
| csg_dual_lesion | pred_maj_excess | 35 | -0.087 [-0.407, 0.273] | 0.007 |

## Threshold transfer (primary statistic; outcome = AUROC < 0.5; held out by family)

| held out | n | n inverted | threshold | sensitivity | specificity |
|---|---:|---:|---|---|---|
| csg_dual_image | 31 | 20 | 0.045 | 0.000 | 1.000 |
| csg_dual_lesion | 35 | 27 | 0.024 | 0.926 | 0.500 |
| single_dann_image | 3 | 1 | 0.024 | 1.000 | 0.000 |

## Plain reading

Across 69 runs the excess majority-centroid share on ID test has Spearman ρ = -0.37 [-0.57, -0.13] with Mahalanobis OOD AUROC on the primary OOD set; holding λ fixed (partial ρ) it is -0.34.

## LaTeX rows (subset & n & ρ [CI] & partial ρ), primary statistic

```latex
all & 69 & -0.37 [-0.57, -0.13] & -0.34 \\
csg\_dual\_image & 31 & -0.00 [-0.36, 0.35] & -0.04 \\
csg\_dual\_lesion & 35 & -0.34 [-0.62, 0.00] & -0.30 \\
```

