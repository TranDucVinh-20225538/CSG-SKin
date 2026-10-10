# R3 B1 — lesion-level λ ∈ {0, 2} at n = 5

Commit: `d0d105b`. Seeds 42, 52, 62 (R2 Item 4) + 72, 82 (this item) = the image-level endpoint seed set. Same code and recipe as Item 4 (`scripts/train_r2_item4_lesion_split.py`, 40 epochs).

**Verdict:** Paired 95% interval includes 0 -> report the drop as a trend with the interval; do not call it a degradation.

## ID balanced accuracy, paired by seed (λ = 2 − λ = 0)

| seed | λ = 0 | λ = 2 | difference |
|---:|---|---|---|
| 42 | 0.495 | 0.436 | -0.059 |
| 52 | 0.521 | 0.465 | -0.056 |
| 62 | 0.497 | 0.491 | -0.006 |
| 72 | 0.484 | 0.484 | +0.001 |
| 82 | 0.482 | 0.489 | +0.007 |

Mean difference -0.0227, s.d. 0.0322, n = 5; 95% t-interval [-0.0628, +0.0173] (t, df = 4).

## Table 1 columns, lesion-level split, mean ± s.d.

| λ | n | Leakage bal acc | ID bal acc | ID ECE | OOD ECE (pad_heldout, 6-cls) | Maha pad_heldout | Cosine pad_heldout | kNN pad_heldout | MSP pad_heldout | Energy pad_heldout | Fitz AUROC (Maha) | Cross-domain bal acc (pad_heldout, 6-cls) |
|---:|---:|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.927 ± 0.018 | 0.496 ± 0.016 | 0.181 ± 0.026 | 0.219 ± 0.046 | 0.808 ± 0.027 | 0.866 ± 0.026 | 0.912 ± 0.022 | 0.947 ± 0.010 | 0.965 ± 0.008 | 0.717 ± 0.034 | 0.275 ± 0.023 |
| 2 | 5 | 0.555 ± 0.046 | 0.473 ± 0.023 | 0.181 ± 0.027 | 0.717 ± 0.022 | 0.403 ± 0.028 | 0.420 ± 0.021 | 0.427 ± 0.022 | 0.455 ± 0.027 | 0.468 ± 0.041 | 0.402 ± 0.018 | 0.254 ± 0.015 |

## Per seed

| λ | seed | Leakage bal acc | ID bal acc | ID ECE | OOD ECE (pad_heldout, 6-cls) | Maha pad_heldout | Cosine pad_heldout | kNN pad_heldout | MSP pad_heldout | Energy pad_heldout | Fitz AUROC (Maha) | Cross-domain bal acc (pad_heldout, 6-cls) |
|---:|---:|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 42 | 0.913 | 0.495 | 0.202 | 0.250 | 0.795 | 0.842 | 0.889 | 0.935 | 0.968 | 0.685 | 0.277 |
| 0 | 52 | 0.952 | 0.521 | 0.184 | 0.181 | 0.849 | 0.891 | 0.935 | 0.955 | 0.973 | 0.772 | 0.236 |
| 0 | 62 | 0.910 | 0.497 | 0.161 | 0.164 | 0.802 | 0.874 | 0.921 | 0.952 | 0.956 | 0.724 | 0.282 |
| 0 | 72 | 0.919 | 0.484 | 0.150 | 0.226 | 0.777 | 0.887 | 0.927 | 0.955 | 0.969 | 0.706 | 0.285 |
| 0 | 82 | 0.941 | 0.482 | 0.211 | 0.273 | 0.815 | 0.835 | 0.887 | 0.936 | 0.958 | 0.698 | 0.294 |
| 2 | 42 | 0.515 | 0.436 | 0.212 | 0.744 | 0.371 | 0.392 | 0.393 | 0.422 | 0.447 | 0.388 | 0.233 |
| 2 | 52 | 0.554 | 0.465 | 0.157 | 0.705 | 0.391 | 0.409 | 0.418 | 0.465 | 0.468 | 0.417 | 0.257 |
| 2 | 62 | 0.625 | 0.491 | 0.157 | 0.693 | 0.436 | 0.448 | 0.449 | 0.462 | 0.451 | 0.416 | 0.252 |
| 2 | 72 | 0.515 | 0.484 | 0.171 | 0.707 | 0.429 | 0.425 | 0.435 | 0.433 | 0.436 | 0.409 | 0.255 |
| 2 | 82 | 0.568 | 0.489 | 0.207 | 0.738 | 0.386 | 0.424 | 0.442 | 0.491 | 0.538 | 0.377 | 0.275 |
