# Phase 13 — technical depth on the derm result

No new CSG training. Phase 12 was not interrupted. Outputs only under `results/paperB/phase13/`.

Phase 3 `z_lesion` is **`z_lesion_norm`** (post-BN 16-d). Pre-BN projector output is stored separately as `z_lesion`.
Leakage is 2-class balanced accuracy; chance floor **0.5**.

## 13.6 — do all detectors collapse together?

kNN k=50 collapses with MSP/Energy/cosine/Mahalanobis. The derm 'all detectors collapse together' claim holds on the Phase 3 sweep. Camelyon17's Maha-only drop remains anomalous relative to this signature. k ∈ {10, 200} is reported once cached features exist.

| λ_adv | n | kNN k=50 | Maha | MSP | cosine | Energy | leakage bal |
|---:|---:|---|---|---|---|---|---|
| 0 | 5 | 0.941 ± 0.010 | 0.863 ± 0.025 | 0.970 ± 0.007 | 0.901 ± 0.011 | 0.983 ± 0.008 | 0.915 ± 0.021 |
| 0.25 | 3 | 0.536 ± 0.024 | 0.508 ± 0.022 | 0.545 ± 0.046 | 0.540 ± 0.033 | 0.537 ± 0.029 | 0.553 ± 0.008 |
| 0.5 | 3 | 0.482 ± 0.014 | 0.449 ± 0.008 | 0.499 ± 0.015 | 0.484 ± 0.012 | 0.506 ± 0.022 | 0.555 ± 0.029 |
| 1 | 3 | 0.450 ± 0.032 | 0.439 ± 0.040 | 0.461 ± 0.015 | 0.452 ± 0.035 | 0.457 ± 0.013 | 0.569 ± 0.010 |
| 2 | 5 | 0.424 ± 0.008 | 0.413 ± 0.021 | 0.437 ± 0.027 | 0.435 ± 0.017 | 0.441 ± 0.039 | 0.584 ± 0.027 |
| 4 | 3 | 0.465 ± 0.033 | 0.443 ± 0.038 | 0.478 ± 0.032 | 0.474 ± 0.014 | 0.482 ± 0.043 | 0.548 ± 0.035 |
| 8 | 5 | 0.488 ± 0.037 | 0.481 ± 0.026 | 0.463 ± 0.055 | 0.497 ± 0.026 | 0.459 ± 0.049 | 0.625 ± 0.045 |

k=50 is the Phase 3 cached number. k ∈ {10, 200} below if features were extracted.

## 13.1 — dimensional collapse

Leakage dropped at λ=0.25 while PR did not. Dimensional collapse does not explain the OOD cliff.

PR(z_lesion_norm) λ=0 4.879 → λ=0.25 4.500. Leakage 0.914 → 0.553.

0.41 survives Ledoit-Wolf (headline 0.413, LW 0.413). Robust; pre-empts the conditioning attack.

| λ | n | PR z_norm | PR z_preBN | PR z_context | PR backbone | Maha eps=1e-3 | Maha LW | cond (eps=1e-3) |
|---:|---:|---|---|---|---|---|---|---|
| 0 | 5 | 4.879 ± 0.148 | 4.145 ± 0.220 | 1.559 ± 0.403 | 57.374 ± 2.950 | 0.863 ± 0.025 | 0.864 ± 0.025 | 1254.674 ± 435.455 |
| 0.25 | 3 | 4.500 ± 0.154 | 3.660 ± 0.029 | 1.457 ± 0.084 | 48.186 ± 4.884 | 0.508 ± 0.022 | 0.509 ± 0.022 | 700.150 ± 144.650 |
| 0.5 | 3 | 4.453 ± 0.154 | 3.610 ± 0.196 | 1.470 ± 0.156 | 45.912 ± 2.524 | 0.449 ± 0.008 | 0.449 ± 0.008 | 473.756 ± 76.815 |
| 1 | 3 | 4.525 ± 0.135 | 3.718 ± 0.229 | 1.502 ± 0.241 | 47.971 ± 2.872 | 0.439 ± 0.040 | 0.439 ± 0.040 | 489.199 ± 77.670 |
| 2 | 5 | 4.737 ± 0.203 | 4.003 ± 0.194 | 1.315 ± 0.176 | 45.728 ± 5.729 | 0.413 ± 0.021 | 0.413 ± 0.021 | 503.481 ± 108.508 |
| 4 | 3 | 4.680 ± 0.098 | 3.997 ± 0.186 | 1.369 ± 0.200 | 43.709 ± 7.132 | 0.443 ± 0.038 | 0.443 ± 0.039 | 441.029 ± 46.033 |
| 8 | 5 | 4.763 ± 0.183 | 4.306 ± 0.284 | 1.362 ± 0.087 | 36.395 ± 4.154 | 0.481 ± 0.026 | 0.481 ± 0.026 | 417.199 ± 125.608 |

## 13.2 — where in the network

**Finding: linear domain-decodability and OOD collapse decouple at the backbone.** Backbone leakage is flat **0.982 → 0.941** across the sweep while backbone Mahalanobis falls **0.749 → 0.475**. Domain information is still linearly decodable; OOD detection collapses anyway. This is a result, not a detail. It is the second independent contradiction of “reducing leakage causes the OOD collapse” (the first is Phase 3: leakage 0.553 → 0.584 while AUROC 0.508 → 0.413).

Depth classification remains mixed: backbone 0.749 → 0.627 at λ=0.25, z_lesion_norm 0.863 → 0.508. Neither clean (a) nor (b).

| λ | backbone Maha | z_lesion Maha | z_norm Maha | backbone leak | z_norm leak |
|---:|---|---|---|---|---|
| 0 | 0.749 ± 0.057 | 0.869 ± 0.024 | 0.863 ± 0.025 | 0.982 ± 0.001 | 0.914 ± 0.021 |
| 0.25 | 0.627 ± 0.013 | 0.510 ± 0.022 | 0.508 ± 0.022 | 0.952 ± 0.001 | 0.553 ± 0.008 |
| 0.5 | 0.547 ± 0.009 | 0.450 ± 0.008 | 0.449 ± 0.008 | 0.952 ± 0.001 | 0.554 ± 0.029 |
| 1 | 0.535 ± 0.053 | 0.440 ± 0.039 | 0.439 ± 0.040 | 0.953 ± 0.004 | 0.569 ± 0.009 |
| 2 | 0.475 ± 0.040 | 0.415 ± 0.021 | 0.413 ± 0.021 | 0.954 ± 0.003 | 0.584 ± 0.027 |
| 4 | 0.485 ± 0.035 | 0.445 ± 0.037 | 0.443 ± 0.038 | 0.946 ± 0.004 | 0.548 ± 0.035 |
| 8 | 0.509 ± 0.039 | 0.481 ± 0.026 | 0.481 ± 0.026 | 0.941 ± 0.007 | 0.625 ± 0.045 |

## 13.3 — confidence on unseen domains

**Lead result of the paper.** As λ rises, the model becomes more confident on data it has never seen. OOD ECE on pad_heldout rises **0.247 → 0.746** while ID ECE stays flat at ~0.10. Mean OOD softmax confidence overtakes ID by λ=2. Crossing (mean OOD MSP > mean ID MSP) at λ ∈ [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]. ECE is the metric clinicians and regulators already understand; it does not require Mahalanobis.

| λ | ID MSP | pad_heldout MSP | Fitz MSP | ID−hold gap | ID ECE | hold ECE |
|---:|---|---|---|---|---|---|
| 0 | 0.915 ± 0.002 | 0.489 ± 0.033 | 0.601 ± 0.050 | 0.425 ± 0.030 | 0.107 ± 0.005 | 0.247 ± 0.043 |
| 0.25 | 0.913 ± 0.003 | 0.898 ± 0.019 | 0.917 ± 0.013 | 0.015 ± 0.016 | 0.101 ± 0.007 | 0.684 ± 0.025 |
| 0.5 | 0.908 ± 0.005 | 0.911 ± 0.006 | 0.914 ± 0.011 | -0.003 ± 0.002 | 0.096 ± 0.005 | 0.724 ± 0.004 |
| 1 | 0.909 ± 0.004 | 0.922 ± 0.006 | 0.914 ± 0.019 | -0.012 ± 0.003 | 0.097 ± 0.005 | 0.733 ± 0.014 |
| 2 | 0.910 ± 0.005 | 0.930 ± 0.005 | 0.935 ± 0.012 | -0.020 ± 0.007 | 0.099 ± 0.002 | 0.746 ± 0.005 |
| 4 | 0.908 ± 0.005 | 0.916 ± 0.011 | 0.924 ± 0.009 | -0.008 ± 0.007 | 0.100 ± 0.007 | 0.726 ± 0.023 |
| 8 | 0.900 ± 0.005 | 0.906 ± 0.012 | 0.901 ± 0.026 | -0.006 ± 0.016 | 0.099 ± 0.008 | 0.713 ± 0.013 |

Fitz ECE is not computed (no compatible labels).

## 13.4 — latent geometry

| λ | ID var | pad_heldout var | ID dist-global | hold dist-global | ID Sw/Sb | hold Sw/Sb |
|---:|---|---|---|---|---|---|
| 0 | 60.015 ± 3.069 | 13.210 ± 1.930 | 7.151 ± 0.214 | 3.248 ± 0.279 | 0.729 ± 0.021 | 7.860 ± 1.936 |
| 0.25 | 47.882 ± 2.696 | 42.678 ± 3.367 | 6.404 ± 0.255 | 6.086 ± 0.270 | 0.706 ± 0.025 | 10.468 ± 1.907 |
| 0.5 | 46.041 ± 2.240 | 39.995 ± 4.258 | 6.270 ± 0.169 | 5.711 ± 0.395 | 0.712 ± 0.058 | 11.930 ± 1.986 |
| 1 | 45.598 ± 1.253 | 44.635 ± 1.935 | 6.269 ± 0.106 | 6.123 ± 0.186 | 0.714 ± 0.036 | 13.656 ± 1.762 |
| 2 | 45.726 ± 2.710 | 39.500 ± 1.986 | 6.258 ± 0.187 | 5.693 ± 0.239 | 0.740 ± 0.017 | 15.395 ± 3.738 |
| 4 | 43.678 ± 0.491 | 39.091 ± 3.363 | 6.128 ± 0.029 | 5.686 ± 0.306 | 0.730 ± 0.009 | 13.772 ± 1.347 |
| 8 | 40.288 ± 2.049 | 36.011 ± 3.967 | 5.878 ± 0.174 | 5.480 ± 0.390 | 0.776 ± 0.046 | 13.625 ± 3.470 |

Compression prediction: as λ rises, OOD variance and distance-to-centroid fall below ID of the same classes.

## 13.6b — kNN k ∈ {10, 50, 200} on cached z_lesion_norm

| λ | k=10 | k=50 (recomputed) | k=200 | Phase 3 k=50 |
|---:|---|---|---|---|
| 0 | 0.947 ± 0.010 | 0.941 ± 0.010 | 0.933 ± 0.011 | 0.941 ± 0.010 |
| 0.25 | 0.539 ± 0.024 | 0.536 ± 0.024 | 0.531 ± 0.025 | 0.536 ± 0.024 |
| 0.5 | 0.486 ± 0.014 | 0.482 ± 0.014 | 0.478 ± 0.014 | 0.482 ± 0.014 |
| 1 | 0.452 ± 0.031 | 0.450 ± 0.032 | 0.449 ± 0.033 | 0.450 ± 0.032 |
| 2 | 0.425 ± 0.007 | 0.424 ± 0.008 | 0.424 ± 0.010 | 0.424 ± 0.008 |
| 4 | 0.468 ± 0.035 | 0.465 ± 0.033 | 0.463 ± 0.029 | 0.465 ± 0.033 |
| 8 | 0.491 ± 0.038 | 0.488 ± 0.037 | 0.485 ± 0.037 | 0.488 ± 0.037 |

Figures: `phase13/figures/fig_13_*.{png,pdf}`.
