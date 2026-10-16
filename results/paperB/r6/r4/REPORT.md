# Paper B R6 R4 — label-free warning signals on unlabeled deployment data

Design: `decisions/paperB_r6_r4.md`. Commit `a7195ca`. Ready runs: 69 (csg_dual_lesion 35, csg_dual_image 31, single_dann_image 3). Deployment set = each run's primary OOD split; outcome = Mahalanobis AUROC there.

| subset | signal | n runs | ρ [95% CI] | partial ρ given λ |
|---|---|---:|---|---|
| all | pred_maj_frac_excess | 69 | -0.715 [-0.845, -0.516] | -0.649 |
| all | bbse_tv | 69 | 0.268 [-0.010, 0.508] | 0.094 |
| all | bbse_maj_ratio | 69 | -0.599 [-0.752, -0.391] | -0.496 |
| all | q4_maj_share_excess | 69 | -0.372 [-0.573, -0.132] | -0.336 |
| all | deploy_maj_share_excess | 69 | -0.731 [-0.844, -0.554] | -0.669 |
| csg_dual_image | pred_maj_frac_excess | 31 | -0.834 [-0.930, -0.642] | -0.766 |
| csg_dual_image | bbse_tv | 31 | 0.208 [-0.271, 0.565] | -0.038 |
| csg_dual_image | bbse_maj_ratio | 31 | -0.837 [-0.926, -0.645] | -0.771 |
| csg_dual_image | q4_maj_share_excess | 31 | -0.001 [-0.357, 0.351] | -0.042 |
| csg_dual_image | deploy_maj_share_excess | 31 | -0.849 [-0.927, -0.671] | -0.788 |
| csg_dual_lesion | pred_maj_frac_excess | 35 | -0.624 [-0.832, -0.298] | -0.530 |
| csg_dual_lesion | bbse_tv | 35 | 0.598 [0.272, 0.822] | 0.506 |
| csg_dual_lesion | bbse_maj_ratio | 35 | -0.645 [-0.838, -0.356] | -0.563 |
| csg_dual_lesion | q4_maj_share_excess | 35 | -0.344 [-0.618, 0.003] | -0.299 |
| csg_dual_lesion | deploy_maj_share_excess | 35 | -0.654 [-0.854, -0.355] | -0.581 |

## Threshold transfer (outcome AUROC < 0.5; held out by family)

| signal | held out | n | n inverted | threshold | sensitivity | specificity |
|---|---|---:|---:|---|---|---|
| pred_maj_frac_excess | csg_dual_image | 31 | 20 | 0.024 | 0.900 | 0.727 |
| pred_maj_frac_excess | csg_dual_lesion | 35 | 27 | 0.048 | 0.852 | 0.875 |
| pred_maj_frac_excess | single_dann_image | 3 | 1 | 0.048 | 1.000 | 0.500 |
| bbse_tv | csg_dual_image | 31 | 20 | 0.087 | 0.950 | 0.091 |
| bbse_tv | csg_dual_lesion | 35 | 27 | 0.092 | 0.926 | 0.000 |
| bbse_tv | single_dann_image | 3 | 1 | 0.104 | 0.000 | 0.500 |
| bbse_maj_ratio | csg_dual_image | 31 | 20 | 0.708 | 1.000 | 0.455 |
| bbse_maj_ratio | csg_dual_lesion | 35 | 27 | 0.941 | 0.741 | 0.875 |
| bbse_maj_ratio | single_dann_image | 3 | 1 | 0.874 | 1.000 | 0.000 |
| q4_maj_share_excess | csg_dual_image | 31 | 20 | 0.045 | 0.000 | 1.000 |
| q4_maj_share_excess | csg_dual_lesion | 35 | 27 | 0.024 | 0.926 | 0.500 |
| q4_maj_share_excess | single_dann_image | 3 | 1 | 0.024 | 1.000 | 0.000 |
| deploy_maj_share_excess | csg_dual_image | 31 | 20 | 0.043 | 0.850 | 0.727 |
| deploy_maj_share_excess | csg_dual_lesion | 35 | 27 | 0.066 | 0.778 | 0.875 |
| deploy_maj_share_excess | single_dann_image | 3 | 1 | 0.055 | 1.000 | 0.500 |

BBSE validation (deployment labels used only here): true TV 0.571 ± 0.000 ; BBSE TV error 0.666 ± 0.060 (mean ± s.d. over runs).

## Plain reading

Across runs, Spearman ρ with deployment Mahalanobis AUROC: pred_maj_frac_excess -0.72 [-0.84, -0.52]; bbse_tv 0.27 [-0.01, 0.51]; bbse_maj_ratio -0.60 [-0.75, -0.39]; q4_maj_share_excess -0.37 [-0.57, -0.13]; deploy_maj_share_excess -0.73 [-0.84, -0.55].

## LaTeX rows (signal & n & ρ [CI]), all runs

```latex
pred\_maj\_frac\_excess & 69 & -0.72 [-0.84, -0.52] \\
bbse\_tv & 69 & 0.27 [-0.01, 0.51] \\
bbse\_maj\_ratio & 69 & -0.60 [-0.75, -0.39] \\
q4\_maj\_share\_excess & 69 & -0.37 [-0.57, -0.13] \\
deploy\_maj\_share\_excess & 69 & -0.73 [-0.84, -0.55] \\
```

