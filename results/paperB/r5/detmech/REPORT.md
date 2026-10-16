# Paper B R5 — per-detector mechanism

Design: `decisions/paperB_r5_detmech.md`. last.ckpt features for A4; item4 for old recipe.

## old_recipe

### pad_heldout

| detector | λ = 0 AUROC | λ = 2 AUROC | Δ (2−0) |
|---|---|---|---|
| mahalanobis_classcond | 0.808 ± 0.027 | 0.403 ± 0.028 | -0.405 |
| knn_k50 | 0.912 ± 0.022 | 0.427 ± 0.022 | -0.484 |
| cosine_max | 0.866 ± 0.026 | 0.420 ± 0.021 | -0.446 |
| MSP | 0.947 ± 0.010 | 0.455 ± 0.027 | -0.492 |
| Energy_T1 | 0.965 ± 0.008 | 0.468 ± 0.041 | -0.497 |

### fitzpatrick17k

| detector | λ = 0 AUROC | λ = 2 AUROC | Δ (2−0) |
|---|---|---|---|
| mahalanobis_classcond | 0.717 ± 0.034 | 0.402 ± 0.018 | -0.315 |
| knn_k50 | 0.801 ± 0.033 | 0.432 ± 0.013 | -0.370 |
| cosine_max | 0.758 ± 0.027 | 0.425 ± 0.019 | -0.333 |
| MSP | 0.823 ± 0.012 | 0.454 ± 0.019 | -0.369 |
| Energy_T1 | 0.895 ± 0.024 | 0.530 ± 0.048 | -0.365 |

## a4

### pad_heldout

| detector | λ = 0 AUROC | λ = 2 AUROC | Δ (2−0) |
|---|---|---|---|
| mahalanobis_classcond | 0.786 ± 0.019 | 0.452 ± 0.014 | -0.334 |
| knn_k50 | 0.891 ± 0.014 | 0.475 ± 0.028 | -0.416 |
| cosine_max | 0.816 ± 0.020 | 0.464 ± 0.021 | -0.351 |
| MSP | 0.767 ± 0.019 | 0.473 ± 0.031 | -0.295 |
| Energy_T1 | 0.861 ± 0.017 | 0.465 ± 0.048 | -0.396 |

### fitzpatrick17k

| detector | λ = 0 AUROC | λ = 2 AUROC | Δ (2−0) |
|---|---|---|---|
| mahalanobis_classcond | 0.821 ± 0.023 | 0.458 ± 0.061 | -0.363 |
| knn_k50 | 0.863 ± 0.036 | 0.521 ± 0.047 | -0.342 |
| cosine_max | 0.785 ± 0.030 | 0.464 ± 0.053 | -0.321 |
| MSP | 0.741 ± 0.028 | 0.478 ± 0.062 | -0.263 |
| Energy_T1 | 0.822 ± 0.054 | 0.516 ± 0.098 | -0.307 |

## Decomposition (Mahalanobis, pad_heldout; mean over seeds)

|Δ| = |reweighted − identity|; **≈ 0 is the identity check** (reweighting algebra), not a failed detector. Compare those two columns to **unweighted pooled AUROC** (ordinary ID-vs-OOD AUROC).

| recipe | λ | unweighted pooled AUROC | measured AUROC, ID reweighted to PAD mix | Σ w_c·AUROC_c (identity) | |Δ| | NV OOD AUROC | non-NV OOD AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_recipe | 0 | 0.808 ± 0.027 | 0.631 ± 0.055 | 0.631 ± 0.055 | 0.000 ± 0.000 | 0.688 ± 0.020 | 0.824 ± 0.028 |
| old_recipe | 2 | 0.403 ± 0.028 | 0.235 ± 0.028 | 0.235 ± 0.028 | 0.000 ± 0.000 | 0.286 ± 0.033 | 0.419 ± 0.029 |
| a4 | 0 | 0.786 ± 0.019 | 0.639 ± 0.028 | 0.639 ± 0.028 | 0.000 ± 0.000 | 0.620 ± 0.027 | 0.809 ± 0.020 |
| a4 | 2 | 0.452 ± 0.014 | 0.303 ± 0.016 | 0.303 ± 0.016 | 0.000 ± 0.000 | 0.325 ± 0.039 | 0.470 ± 0.014 |

## Plain words

Compare λ = 0 vs λ = 2 for Mahalanobis/cosine (inversion on pad_heldout / Fitz) against Energy/kNN/MSP (collapse on Fitz, milder on pad_heldout). Per-run JSON holds per-class AUROC and decomposition for every detector. In the decomposition table, |Δ| ≈ 0 confirms the reweight↔identity algebra; the unweighted pooled AUROC is the ordinary detection number.

