# (5) S1/S2 — FPR@95 for flipped Mahalanobis at λ ≥ 0.5

Source: `r5/s_flip/summary.csv` column `ood_acc_flip_mean` (OOD acceptance at val 95th-percentile threshold on −score = FPR@95TPR for the flipped detector). Lesion-level Table 1 block.

**Range across pad_heldout / Fitzpatrick17k / MED-NODE / SD-198 / SCIN × λ ∈ {0.5,1,2,4,8}:** **[0.842, 0.983]** (min/max of cell means).

| set | λ | FPR@95 flipped (mean ± sd) | CI |
|---|---:|---|---|
| fitzpatrick17k | 0.5 | 0.898 ± 0.022 | [0.892, 0.904] |
| fitzpatrick17k | 1 | 0.875 ± 0.046 | [0.868, 0.882] |
| fitzpatrick17k | 2 | 0.920 ± 0.024 | [0.915, 0.925] |
| fitzpatrick17k | 4 | 0.914 ± 0.058 | [0.909, 0.918] |
| fitzpatrick17k | 8 | 0.939 ± 0.042 | [0.935, 0.943] |
| mednode | 0.5 | 0.888 ± 0.052 | [0.862, 0.913] |
| mednode | 1 | 0.886 ± 0.056 | [0.860, 0.911] |
| mednode | 2 | 0.871 ± 0.079 | [0.845, 0.896] |
| mednode | 4 | 0.902 ± 0.070 | [0.881, 0.924] |
| mednode | 8 | 0.919 ± 0.055 | [0.898, 0.939] |
| pad_heldout | 0.5 | 0.857 ± 0.032 | [0.837, 0.875] |
| pad_heldout | 1 | 0.842 ± 0.040 | [0.822, 0.861] |
| pad_heldout | 2 | 0.889 ± 0.043 | [0.872, 0.903] |
| pad_heldout | 4 | 0.917 ± 0.051 | [0.904, 0.929] |
| pad_heldout | 8 | 0.950 ± 0.021 | [0.940, 0.960] |
| scin | 0.5 | 0.978 ± 0.003 | [0.976, 0.979] |
| scin | 1 | 0.972 ± 0.011 | [0.970, 0.975] |
| scin | 2 | 0.982 ± 0.003 | [0.981, 0.984] |
| scin | 4 | 0.981 ± 0.009 | [0.979, 0.982] |
| scin | 8 | 0.983 ± 0.010 | [0.982, 0.984] |
| sd198 | 0.5 | 0.881 ± 0.025 | [0.876, 0.886] |
| sd198 | 1 | 0.868 ± 0.060 | [0.862, 0.873] |
| sd198 | 2 | 0.904 ± 0.038 | [0.900, 0.908] |
| sd198 | 4 | 0.901 ± 0.058 | [0.897, 0.905] |
| sd198 | 8 | 0.930 ± 0.046 | [0.927, 0.933] |
