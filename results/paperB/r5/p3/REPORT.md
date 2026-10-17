# Paper B R5 P3 — frozen ImageNet monitor vs the lesion-level model

Design: `decisions/paperB_r5_p3.md`. Commit `a7195ca`. ID = lesion-level ISIC test; fit = lesion-level ISIC train; target (R2 pre-registration carried over): Mahalanobis ≥ 0.90 on pad_heldout and on Fitzpatrick17k.

## Frozen encoders

| Encoder | n fit / n ID (coverage) | Detector | pad_heldout [95% CI] | Fitzpatrick17k, image [95% CI] | Fitzpatrick17k, cluster [95% CI] |
|---|---|---|---|---|---|
| imagenet_r50_mapped_R2_features | 13613 / 4273 (0.843) | maha | 0.827 [0.806, 0.847] | 0.984 [0.980, 0.988] | 0.984 [0.980, 0.988] |
| imagenet_r50_mapped_R2_features | 13613 / 4273 (0.843) | maha_lw | 0.828 [0.807, 0.848] | 0.985 [0.981, 0.988] | 0.985 [0.981, 0.988] |
| imagenet_r50_mapped_R2_features | 13613 / 4273 (0.843) | knn | 0.904 [0.889, 0.918] | 0.978 [0.974, 0.982] | 0.978 [0.974, 0.982] |

## Model (z_lesion^norm), mean ± s.d. over seeds

| λ | n | Maha pad (full ID) | Maha pad (covered ID) | Maha Fitz (full) | kNN pad (full) | kNN Fitz (full) |
|---:|---:|---|---|---|---|---|
| 0 | 5 | 0.808 ± 0.027 | 0.806 ± 0.027 | 0.717 ± 0.034 | 0.912 ± 0.022 | 0.801 ± 0.033 |
| 0.25 | 5 | 0.525 ± 0.121 | 0.524 ± 0.121 | 0.496 ± 0.167 | 0.574 ± 0.161 | 0.515 ± 0.132 |
| 0.5 | 5 | 0.426 ± 0.008 | 0.425 ± 0.007 | 0.411 ± 0.047 | 0.440 ± 0.012 | 0.439 ± 0.046 |
| 1 | 5 | 0.387 ± 0.016 | 0.386 ± 0.015 | 0.375 ± 0.042 | 0.407 ± 0.012 | 0.409 ± 0.028 |
| 2 | 5 | 0.403 ± 0.028 | 0.402 ± 0.028 | 0.402 ± 0.018 | 0.427 ± 0.022 | 0.432 ± 0.013 |
| 4 | 5 | 0.433 ± 0.020 | 0.432 ± 0.020 | 0.376 ± 0.037 | 0.424 ± 0.028 | 0.386 ± 0.030 |
| 8 | 5 | 0.454 ± 0.044 | 0.453 ± 0.044 | 0.412 ± 0.051 | 0.440 ± 0.040 | 0.415 ± 0.042 |

Model pooled bootstrap (PREREGISTER draws), Mahalanobis: λ = 0 pad_heldout: 0.808 [0.792, 0.822]; λ = 0 fitzpatrick17k: 0.717 [0.703, 0.730]; λ = 2 pad_heldout: 0.403 [0.379, 0.426]; λ = 2 fitzpatrick17k: 0.402 [0.389, 0.414]

imagenet_r50_mapped_R2_features: Mahalanobis 0.827 on pad_heldout and 0.984 on Fitzpatrick17k (target 0.90); the model gives 0.808 / 0.717 at λ = 0 and 0.403 / 0.402 at λ = 2 on the same ID test.

## LaTeX rows (encoder & Maha pad_heldout [CI] & Maha Fitzpatrick [CI])

```latex
imagenet\_r50\_mapped\_R2\_features & 0.827 [0.806, 0.847] & 0.984 [0.980, 0.988] \\
model $\lambda=0$ & 0.808\,$\pm$\,0.027 & 0.717\,$\pm$\,0.034 \\
model $\lambda=2$ & 0.403\,$\pm$\,0.028 & 0.402\,$\pm$\,0.018 \\
```

