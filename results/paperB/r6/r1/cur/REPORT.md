# Paper B R6 R1 — causal dose test (recipe: cur)

Design: `decisions/paperB_r6_r1.md`. Commit `1ea63f0`. Three streams per step: ISIC classifier stream (full ISIC train, natural order), ISIC adversary stream resampled to the shift level, PAD adversary stream unchanged. PAD labels set only the resampling proportions and enter no loss. Primary checkpoint = `last.ckpt`. Every shift cell is scored on pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN (same detectors + cluster/patient bootstrap).

## Run quality flags (R1 deviation 2026-10-16 10:05)

Degraded = ID bal acc < 0.25 (2× chance 0.125): reported, flagged, **excluded from the main rule A verdict**; verdict also shown with them included. ID drop > 0.10 below matched λ = 0: flagged, **kept** in the main verdict. No seed reruns unless > 1 of 5 degrade in a cell → replacement seeds [92, 102].

| cell | λ | seed | ID bal | λ=0 ID bal | flags |
|---|---:|---:|---:|---:|---|
| λ=0 baseline (shift 100%) | 0 | 42 | 0.509 | 0.509 | — |
| λ=0 baseline (shift 100%) | 0 | 52 | 0.469 | 0.469 | — |
| λ=0 baseline (shift 100%) | 0 | 62 | 0.471 | 0.471 | — |
| λ=0 baseline (shift 100%) | 0 | 72 | 0.476 | 0.476 | — |
| λ=0 baseline (shift 100%) | 0 | 82 | 0.482 | 0.482 | — |
| shift 100% | 0.5 | 42 | 0.480 | 0.509 | — |
| shift 100% | 0.5 | 52 | 0.505 | 0.469 | — |
| shift 100% | 0.5 | 62 | 0.470 | 0.471 | — |
| shift 100% | 0.5 | 72 | 0.486 | 0.476 | — |
| shift 100% | 0.5 | 82 | 0.475 | 0.482 | — |
| shift 50% | 0.5 | 42 | 0.489 | 0.509 | — |
| shift 50% | 0.5 | 52 | 0.488 | 0.469 | — |
| shift 50% | 0.5 | 62 | 0.491 | 0.471 | — |
| shift 0% | 0.5 | 42 | 0.469 | 0.509 | — |
| shift 0% | 0.5 | 52 | 0.475 | 0.469 | — |
| shift 0% | 0.5 | 62 | 0.450 | 0.471 | — |
| shift 0% | 0.5 | 72 | 0.479 | 0.476 | — |
| shift 100% | 2 | 42 | 0.499 | 0.509 | — |
| shift 100% | 2 | 52 | 0.479 | 0.469 | — |
| shift 100% | 2 | 62 | 0.480 | 0.471 | — |
| shift 100% | 2 | 72 | 0.473 | 0.476 | — |
| shift 100% | 2 | 82 | 0.500 | 0.482 | — |
| shift 50% | 2 | 42 | 0.493 | 0.509 | — |
| shift 50% | 2 | 52 | 0.488 | 0.469 | — |
| shift 50% | 2 | 62 | 0.491 | 0.471 | — |
| shift 0% | 2 | 42 | 0.479 | 0.509 | — |
| shift 0% | 2 | 52 | 0.139 | 0.469 | degraded (ID bal 0.139 < 0.25); ID drop vs λ=0 (0.139 − 0.469 = -0.330) |
| shift 0% | 2 | 62 | 0.476 | 0.471 | — |
| shift 0% | 2 | 72 | 0.480 | 0.476 | — |
| shift 0% | 2 | 82 | 0.498 | 0.482 | — |

## Key comparison: λ = 2 shift 0% vs R1 shift 100% (last.ckpt), all unseen sets

Same three-stream recipe; primary = `last.ckpt`. Cells are R1's own shift runs (not two-stream item4). Shift 50% (seeds 42/52/62) is included when present. For each set: five detectors (point mean ± s.d.) and Mahalanobis / kNN / cosine rule-A bootstrap (cluster or patient units). Degraded seeds excluded from the primary rule A column; all-seeds column secondary.

### pad_heldout (λ = 2)

| shift | n seeds | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha [rule A excl. degraded] | Maha [all seeds] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| shift 100% | 5 | 0.368 ± 0.017 | 0.371 ± 0.016 | 0.368 ± 0.016 | 0.392 ± 0.028 | 0.404 ± 0.045 | 0.368 ± 0.017 [0.344, 0.391], 5/5 seed CIs below 0.5, rule A met | 0.368 ± 0.017 [0.344, 0.391], 5/5 seed CIs below 0.5, rule A met |
| shift 50% | 3 | 0.454 ± 0.009 | 0.455 ± 0.021 | 0.422 ± 0.028 | 0.446 ± 0.018 | 0.436 ± 0.020 | 0.454 ± 0.009 [0.432, 0.476], 3/3 seed CIs below 0.5, rule A met | 0.454 ± 0.009 [0.432, 0.476], 3/3 seed CIs below 0.5, rule A met |
| shift 0% | 5 | 0.458 ± 0.003 | 0.581 ± 0.010 | 0.479 ± 0.002 | 0.535 ± 0.018 | 0.510 ± 0.041 | 0.458 ± 0.003 [0.438, 0.477], 4/4 seed CIs below 0.5, rule A met; excluded degraded [52] | 0.514 ± 0.123 [0.497, 0.529], 4/5 seed CIs below 0.5, rule A not met |

### Fitzpatrick17k (λ = 2)

| shift | n seeds | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha [rule A excl. degraded] | Maha [all seeds] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| shift 100% | 5 | 0.340 ± 0.065 | 0.360 ± 0.054 | 0.364 ± 0.048 | 0.396 ± 0.056 | 0.462 ± 0.079 | 0.340 ± 0.065 [0.328, 0.352], 5/5 seed CIs below 0.5, rule A met | 0.340 ± 0.065 [0.328, 0.352], 5/5 seed CIs below 0.5, rule A met |
| shift 50% | 3 | 0.441 ± 0.076 | 0.431 ± 0.067 | 0.433 ± 0.064 | 0.464 ± 0.015 | 0.505 ± 0.014 | 0.441 ± 0.076 [0.428, 0.454], 2/3 seed CIs below 0.5, rule A not met | 0.441 ± 0.076 [0.428, 0.454], 2/3 seed CIs below 0.5, rule A not met |
| shift 0% | 5 | 0.691 ± 0.031 | 0.677 ± 0.045 | 0.631 ± 0.034 | 0.610 ± 0.053 | 0.609 ± 0.086 | 0.691 ± 0.031 [0.680, 0.702], 0/4 seed CIs below 0.5, rule A not met; excluded degraded [52] | 0.719 ± 0.069 [0.710, 0.729], 0/5 seed CIs below 0.5, rule A not met |

### MED-NODE (λ = 2)

| shift | n seeds | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha [rule A excl. degraded] | Maha [all seeds] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| shift 100% | 5 | 0.298 ± 0.032 | 0.327 ± 0.035 | 0.362 ± 0.035 | 0.388 ± 0.058 | 0.433 ± 0.060 | 0.298 ± 0.032 [0.271, 0.326], 5/5 seed CIs below 0.5, rule A met | 0.298 ± 0.032 [0.271, 0.326], 5/5 seed CIs below 0.5, rule A met |
| shift 50% | 3 | 0.299 ± 0.052 | 0.319 ± 0.059 | 0.354 ± 0.051 | 0.428 ± 0.043 | 0.466 ± 0.040 | 0.299 ± 0.052 [0.269, 0.331], 3/3 seed CIs below 0.5, rule A met | 0.299 ± 0.052 [0.269, 0.331], 3/3 seed CIs below 0.5, rule A met |
| shift 0% | 5 | 0.541 ± 0.027 | 0.531 ± 0.028 | 0.542 ± 0.025 | 0.533 ± 0.035 | 0.497 ± 0.076 | 0.541 ± 0.027 [0.512, 0.572], 0/4 seed CIs below 0.5, rule A not met; excluded degraded [52] | 0.591 ± 0.114 [0.568, 0.617], 0/5 seed CIs below 0.5, rule A not met |

### SD-198 (λ = 2)

| shift | n seeds | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha [rule A excl. degraded] | Maha [all seeds] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| shift 100% | 5 | 0.325 ± 0.069 | 0.348 ± 0.052 | 0.355 ± 0.052 | 0.381 ± 0.052 | 0.448 ± 0.072 | 0.325 ± 0.069 [0.315, 0.336], 5/5 seed CIs below 0.5, rule A met | 0.325 ± 0.069 [0.315, 0.336], 5/5 seed CIs below 0.5, rule A met |
| shift 50% | 3 | 0.414 ± 0.047 | 0.413 ± 0.046 | 0.420 ± 0.047 | 0.448 ± 0.015 | 0.493 ± 0.018 | 0.414 ± 0.047 [0.402, 0.425], 3/3 seed CIs below 0.5, rule A met | 0.414 ± 0.047 [0.402, 0.425], 3/3 seed CIs below 0.5, rule A met |
| shift 0% | 5 | 0.694 ± 0.031 | 0.675 ± 0.046 | 0.633 ± 0.032 | 0.604 ± 0.058 | 0.599 ± 0.094 | 0.694 ± 0.031 [0.683, 0.704], 0/4 seed CIs below 0.5, rule A not met; excluded degraded [52] | 0.720 ± 0.065 [0.712, 0.729], 0/5 seed CIs below 0.5, rule A not met |

### SCIN (λ = 2)

| shift | n seeds | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha [rule A excl. degraded] | Maha [all seeds] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| shift 100% | 5 | 0.519 ± 0.073 | 0.528 ± 0.058 | 0.522 ± 0.054 | 0.499 ± 0.051 | 0.550 ± 0.072 | 0.519 ± 0.073 [0.507, 0.532], 1/5 seed CIs below 0.5, rule A not met | 0.519 ± 0.073 [0.507, 0.532], 1/5 seed CIs below 0.5, rule A not met |
| shift 50% | 3 | 0.566 ± 0.051 | 0.554 ± 0.032 | 0.557 ± 0.029 | 0.538 ± 0.047 | 0.571 ± 0.062 | 0.566 ± 0.051 [0.553, 0.578], 0/3 seed CIs below 0.5, rule A not met | 0.566 ± 0.051 [0.553, 0.578], 0/3 seed CIs below 0.5, rule A not met |
| shift 0% | 5 | 0.732 ± 0.035 | 0.677 ± 0.050 | 0.652 ± 0.037 | 0.609 ± 0.069 | 0.619 ± 0.097 | 0.732 ± 0.035 [0.722, 0.742], 0/4 seed CIs below 0.5, rule A not met; excluded degraded [52] | 0.719 ± 0.041 [0.711, 0.728], 0/5 seed CIs below 0.5, rule A not met |

### λ = 2 Mahalanobis point Δ (shift 0% − shift 100%), excl. degraded

| set | mean Δ | shift 0% | shift 100% | shift 50% |
|---|---:|---:|---:|---:|
| pad_heldout | +0.090 | 0.458 ± 0.003 | 0.368 ± 0.017 | 0.454 ± 0.009 |
| Fitzpatrick17k | +0.351 | 0.691 ± 0.031 | 0.340 ± 0.065 | 0.441 ± 0.076 |
| MED-NODE | +0.243 | 0.541 ± 0.027 | 0.298 ± 0.032 | 0.299 ± 0.052 |
| SD-198 | +0.368 | 0.694 ± 0.031 | 0.325 ± 0.069 | 0.414 ± 0.047 |
| SCIN | +0.212 | 0.732 ± 0.035 | 0.519 ± 0.073 | 0.566 ± 0.051 |

### Flags (Table 1 leakage vs the λ = 0 cell; Mahalanobis pad_heldout headroom)

| cell | λ | leakage | Δ vs λ = 0 | flag |
|---|---:|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.928 ± 0.020 | — | — |
| shift 100% | 0.5 | 0.628 ± 0.062 | -0.299 | — |
| shift 50% | 0.5 | 0.629 ± 0.045 | -0.299 | — |
| shift 0% | 0.5 | 0.790 ± 0.007 | -0.137 | — |
| shift 100% | 2 | 0.634 ± 0.054 | -0.293 | — |
| shift 50% | 2 | 0.658 ± 0.020 | -0.269 | — |
| shift 0% | 2 | 0.782 ± 0.019 | -0.145 | — |

## Table 1 columns

Leakage = logistic domain probe on z_lesion_norm, ISIC test vs all of PAD (Table 1 definition). Mean ± s.d. over seeds; n = seeds present.

| cell | λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc (pad_heldout, 6-cls) |
|---|---:|---:|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 5 | 0.928 ± 0.020 | 0.481 ± 0.016 | 0.207 ± 0.003 | 0.305 ± 0.042 | 0.825 ± 0.022 | 0.712 ± 0.013 | 0.277 ± 0.024 |
| shift 100% | 0.5 | 5 | 0.628 ± 0.062 | 0.483 ± 0.013 | 0.214 ± 0.007 | 0.775 ± 0.010 | 0.361 ± 0.029 | 0.385 ± 0.023 | 0.222 ± 0.016 |
| shift 50% | 0.5 | 3 | 0.629 ± 0.045 | 0.489 ± 0.002 | 0.204 ± 0.005 | 0.641 ± 0.033 | 0.490 ± 0.058 | 0.482 ± 0.143 | 0.293 ± 0.012 |
| shift 0% | 0.5 | 4 | 0.790 ± 0.007 | 0.468 ± 0.013 | 0.212 ± 0.003 | 0.442 ± 0.020 | 0.499 ± 0.029 | 0.674 ± 0.053 | 0.380 ± 0.025 |
| shift 100% | 2 | 5 | 0.634 ± 0.054 | 0.486 ± 0.012 | 0.205 ± 0.007 | 0.757 ± 0.018 | 0.368 ± 0.017 | 0.340 ± 0.065 | 0.231 ± 0.029 |
| shift 50% | 2 | 3 | 0.658 ± 0.020 | 0.491 ± 0.003 | 0.208 ± 0.004 | 0.646 ± 0.019 | 0.454 ± 0.009 | 0.441 ± 0.076 | 0.311 ± 0.014 |
| shift 0% | 2 | 5 | 0.782 ± 0.019 | 0.414 ± 0.154 | 0.188 ± 0.057 | 0.361 ± 0.144 | 0.514 ± 0.123 | 0.719 ± 0.069 | 0.366 ± 0.103 |

### Detectors on pad_heldout (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.825 ± 0.022 [0.810, 0.839] | 0.892 ± 0.013 [0.880, 0.904] | 0.844 ± 0.015 [0.832, 0.856] | 0.928 ± 0.008 | 0.959 ± 0.008 | 0/5 | not met | 0.365 ± 0.070 |
| shift 100% | 0.5 | 0.361 ± 0.029 [0.338, 0.383] | 0.376 ± 0.026 [0.355, 0.397] | 0.381 ± 0.014 [0.362, 0.400] | 0.407 ± 0.035 | 0.444 ± 0.048 | 5/5 | met | 0.653 ± 0.041 |
| shift 50% | 0.5 | 0.490 ± 0.058 [0.466, 0.512] | 0.487 ± 0.032 [0.463, 0.510] | 0.460 ± 0.024 [0.438, 0.481] | 0.455 ± 0.017 | 0.402 ± 0.034 | 2/3 | not met | 0.433 ± 0.056 |
| shift 0% | 0.5 | 0.499 ± 0.029 [0.479, 0.519] | 0.605 ± 0.017 [0.587, 0.623] | 0.510 ± 0.031 [0.491, 0.528] | 0.518 ± 0.015 | 0.492 ± 0.051 | 0/4 | not met | 0.126 ± 0.021 |
| shift 100% | 2 | 0.368 ± 0.017 [0.344, 0.391] | 0.371 ± 0.016 [0.349, 0.391] | 0.368 ± 0.016 [0.348, 0.387] | 0.392 ± 0.028 | 0.404 ± 0.045 | 5/5 | met | 0.628 ± 0.039 |
| shift 50% | 2 | 0.454 ± 0.009 [0.432, 0.476] | 0.455 ± 0.021 [0.433, 0.476] | 0.422 ± 0.028 [0.402, 0.441] | 0.446 ± 0.018 | 0.436 ± 0.020 | 3/3 | met | 0.464 ± 0.034 |
| shift 0% | 2 | 0.514 ± 0.123 [0.438, 0.477] | 0.583 ± 0.010 [0.562, 0.599] | 0.495 ± 0.035 [0.459, 0.497] | 0.512 ± 0.056 | 0.504 ± 0.038 | 4/4 | met | 0.110 ± 0.031 |

### Detectors on fitzpatrick17k (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.712 ± 0.013 [0.698, 0.725] | 0.770 ± 0.015 [0.758, 0.780] | 0.731 ± 0.015 [0.721, 0.740] | 0.795 ± 0.034 | 0.875 ± 0.041 | 0/5 | not met | 0.601 ± 0.085 |
| shift 100% | 0.5 | 0.385 ± 0.023 [0.373, 0.398] | 0.418 ± 0.030 [0.404, 0.431] | 0.423 ± 0.021 [0.412, 0.436] | 0.450 ± 0.046 | 0.524 ± 0.062 | 5/5 | met | 0.777 ± 0.054 |
| shift 50% | 0.5 | 0.482 ± 0.143 [0.469, 0.494] | 0.463 ± 0.123 [0.450, 0.477] | 0.460 ± 0.108 [0.448, 0.472] | 0.452 ± 0.065 | 0.482 ± 0.046 | 2/3 | not met | 0.710 ± 0.165 |
| shift 0% | 0.5 | 0.674 ± 0.053 [0.662, 0.685] | 0.655 ± 0.053 [0.642, 0.668] | 0.621 ± 0.043 [0.609, 0.632] | 0.554 ± 0.063 | 0.542 ± 0.067 | 0/4 | not met | 0.462 ± 0.091 |
| shift 100% | 2 | 0.340 ± 0.065 [0.328, 0.352] | 0.360 ± 0.054 [0.348, 0.373] | 0.364 ± 0.048 [0.353, 0.376] | 0.396 ± 0.056 | 0.462 ± 0.079 | 5/5 | met | 0.821 ± 0.055 |
| shift 50% | 2 | 0.441 ± 0.076 [0.428, 0.454] | 0.431 ± 0.067 [0.418, 0.445] | 0.433 ± 0.064 [0.420, 0.446] | 0.464 ± 0.015 | 0.505 ± 0.014 | 2/3 | not met | 0.731 ± 0.090 |
| shift 0% | 2 | 0.719 ± 0.069 [0.680, 0.702] | 0.663 ± 0.051 [0.665, 0.690] | 0.634 ± 0.030 [0.619, 0.642] | 0.566 ± 0.109 | 0.569 ± 0.115 | 0/4 | not met | 0.335 ± 0.138 |

### Detectors on mednode (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.559 ± 0.019 [0.531, 0.587] | 0.603 ± 0.010 [0.573, 0.632] | 0.602 ± 0.005 [0.574, 0.630] | 0.674 ± 0.016 | 0.732 ± 0.015 | 0/5 | not met | 0.548 ± 0.069 |
| shift 100% | 0.5 | 0.310 ± 0.055 [0.284, 0.337] | 0.355 ± 0.056 [0.326, 0.383] | 0.389 ± 0.048 [0.358, 0.420] | 0.430 ± 0.050 | 0.490 ± 0.043 | 5/5 | met | 0.762 ± 0.067 |
| shift 50% | 0.5 | 0.303 ± 0.054 [0.272, 0.336] | 0.323 ± 0.031 [0.292, 0.355] | 0.360 ± 0.027 [0.328, 0.394] | 0.404 ± 0.029 | 0.401 ± 0.052 | 3/3 | met | 0.716 ± 0.093 |
| shift 0% | 0.5 | 0.529 ± 0.056 [0.501, 0.560] | 0.520 ± 0.046 [0.490, 0.552] | 0.532 ± 0.044 [0.503, 0.564] | 0.521 ± 0.042 | 0.489 ± 0.050 | 1/4 | not met | 0.532 ± 0.090 |
| shift 100% | 2 | 0.298 ± 0.032 [0.271, 0.326] | 0.327 ± 0.035 [0.298, 0.356] | 0.362 ± 0.035 [0.330, 0.394] | 0.388 ± 0.058 | 0.433 ± 0.060 | 5/5 | met | 0.754 ± 0.063 |
| shift 50% | 2 | 0.299 ± 0.052 [0.269, 0.331] | 0.319 ± 0.059 [0.289, 0.351] | 0.354 ± 0.051 [0.322, 0.388] | 0.428 ± 0.043 | 0.466 ± 0.040 | 3/3 | met | 0.741 ± 0.082 |
| shift 0% | 2 | 0.591 ± 0.114 [0.512, 0.572] | 0.561 ± 0.071 [0.501, 0.563] | 0.564 ± 0.053 [0.514, 0.571] | 0.531 ± 0.031 | 0.479 ± 0.077 | 0/4 | not met | 0.429 ± 0.136 |

### Detectors on sd198 (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.708 ± 0.019 [0.695, 0.722] | 0.771 ± 0.007 [0.761, 0.782] | 0.733 ± 0.007 [0.724, 0.743] | 0.806 ± 0.026 | 0.885 ± 0.035 | 0/5 | not met | 0.605 ± 0.083 |
| shift 100% | 0.5 | 0.365 ± 0.029 [0.353, 0.377] | 0.399 ± 0.037 [0.386, 0.411] | 0.412 ± 0.032 [0.401, 0.422] | 0.441 ± 0.056 | 0.519 ± 0.077 | 5/5 | met | 0.808 ± 0.052 |
| shift 50% | 0.5 | 0.451 ± 0.133 [0.439, 0.461] | 0.431 ± 0.113 [0.419, 0.443] | 0.437 ± 0.103 [0.427, 0.448] | 0.425 ± 0.058 | 0.453 ± 0.057 | 2/3 | not met | 0.754 ± 0.134 |
| shift 0% | 0.5 | 0.669 ± 0.066 [0.658, 0.680] | 0.647 ± 0.058 [0.635, 0.660] | 0.617 ± 0.048 [0.606, 0.628] | 0.542 ± 0.064 | 0.528 ± 0.062 | 0/4 | not met | 0.492 ± 0.098 |
| shift 100% | 2 | 0.325 ± 0.069 [0.315, 0.336] | 0.348 ± 0.052 [0.336, 0.359] | 0.355 ± 0.052 [0.346, 0.365] | 0.381 ± 0.052 | 0.448 ± 0.072 | 5/5 | met | 0.851 ± 0.062 |
| shift 50% | 2 | 0.414 ± 0.047 [0.402, 0.425] | 0.413 ± 0.046 [0.401, 0.425] | 0.420 ± 0.047 [0.409, 0.431] | 0.448 ± 0.015 | 0.493 ± 0.018 | 3/3 | met | 0.772 ± 0.056 |
| shift 0% | 2 | 0.720 ± 0.065 [0.683, 0.704] | 0.657 ± 0.057 [0.663, 0.687] | 0.637 ± 0.029 [0.623, 0.644] | 0.556 ± 0.118 | 0.560 ± 0.119 | 0/4 | not met | 0.366 ± 0.155 |

### Detectors on scin (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.723 ± 0.018 [0.710, 0.736] | 0.770 ± 0.023 [0.760, 0.781] | 0.735 ± 0.023 [0.725, 0.744] | 0.767 ± 0.050 | 0.850 ± 0.048 | 0/5 | not met | 0.585 ± 0.096 |
| shift 100% | 0.5 | 0.534 ± 0.015 [0.521, 0.547] | 0.560 ± 0.026 [0.547, 0.573] | 0.560 ± 0.022 [0.549, 0.571] | 0.530 ± 0.031 | 0.579 ± 0.055 | 0/5 | not met | 0.769 ± 0.028 |
| shift 50% | 0.5 | 0.586 ± 0.101 [0.575, 0.599] | 0.560 ± 0.084 [0.547, 0.572] | 0.557 ± 0.083 [0.545, 0.568] | 0.507 ± 0.102 | 0.545 ± 0.109 | 1/3 | not met | 0.750 ± 0.119 |
| shift 0% | 0.5 | 0.713 ± 0.039 [0.702, 0.724] | 0.656 ± 0.036 [0.643, 0.669] | 0.638 ± 0.039 [0.627, 0.649] | 0.566 ± 0.068 | 0.567 ± 0.060 | 0/4 | not met | 0.659 ± 0.071 |
| shift 100% | 2 | 0.519 ± 0.073 [0.507, 0.532] | 0.528 ± 0.058 [0.515, 0.541] | 0.522 ± 0.054 [0.511, 0.533] | 0.499 ± 0.051 | 0.550 ± 0.072 | 1/5 | not met | 0.786 ± 0.071 |
| shift 50% | 2 | 0.566 ± 0.051 [0.553, 0.578] | 0.554 ± 0.032 [0.541, 0.568] | 0.557 ± 0.029 [0.546, 0.569] | 0.538 ± 0.047 | 0.571 ± 0.062 | 0/3 | not met | 0.745 ± 0.033 |
| shift 0% | 2 | 0.719 ± 0.041 [0.722, 0.742] | 0.670 ± 0.046 [0.665, 0.690] | 0.643 ± 0.039 [0.642, 0.663] | 0.597 ± 0.065 | 0.582 ± 0.117 | 0/4 | not met | 0.505 ± 0.224 |

### Per-class Mahalanobis AUROC and decomposition check on pad_heldout

ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs the OOD images of that class group.

| cell | λ | ID MEL | ID NV | ID BCC | ID AK | ID BKL | ID DF | ID VASC | ID SCC | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.844 ± 0.028 | 0.926 ± 0.012 | 0.695 ± 0.042 | 0.525 ± 0.049 | 0.727 ± 0.026 | 0.358 ± 0.041 | 0.455 ± 0.072 | 0.456 ± 0.066 | 0.659 ± 0.039 | 0.659 ± 0.039 | 0.688 ± 0.040 | 0.844 ± 0.020 |
| shift 100% | 0.5 | 0.312 ± 0.025 | 0.487 ± 0.038 | 0.197 ± 0.020 | 0.123 ± 0.025 | 0.214 ± 0.026 | 0.124 ± 0.012 | 0.137 ± 0.039 | 0.106 ± 0.017 | 0.207 ± 0.022 | 0.207 ± 0.022 | 0.260 ± 0.026 | 0.375 ± 0.032 |
| shift 50% | 0.5 | 0.454 ± 0.056 | 0.603 ± 0.062 | 0.356 ± 0.054 | 0.276 ± 0.073 | 0.349 ± 0.060 | 0.193 ± 0.017 | 0.220 ± 0.083 | 0.231 ± 0.043 | 0.353 ± 0.058 | 0.353 ± 0.058 | 0.320 ± 0.022 | 0.513 ± 0.063 |
| shift 0% | 0.5 | 0.452 ± 0.045 | 0.613 ± 0.022 | 0.378 ± 0.061 | 0.306 ± 0.054 | 0.361 ± 0.036 | 0.185 ± 0.043 | 0.147 ± 0.012 | 0.263 ± 0.028 | 0.375 ± 0.046 | 0.375 ± 0.046 | 0.523 ± 0.065 | 0.496 ± 0.028 |
| shift 100% | 2 | 0.324 ± 0.021 | 0.487 ± 0.026 | 0.219 ± 0.017 | 0.139 ± 0.010 | 0.231 ± 0.012 | 0.140 ± 0.012 | 0.118 ± 0.026 | 0.118 ± 0.015 | 0.222 ± 0.007 | 0.222 ± 0.007 | 0.288 ± 0.043 | 0.379 ± 0.017 |
| shift 50% | 2 | 0.417 ± 0.012 | 0.580 ± 0.016 | 0.295 ± 0.010 | 0.232 ± 0.018 | 0.296 ± 0.010 | 0.150 ± 0.011 | 0.173 ± 0.026 | 0.185 ± 0.015 | 0.304 ± 0.007 | 0.304 ± 0.007 | 0.324 ± 0.023 | 0.473 ± 0.008 |
| shift 0% | 2 | 0.451 ± 0.123 | 0.607 ± 0.089 | 0.445 ± 0.175 | 0.366 ± 0.193 | 0.394 ± 0.171 | 0.256 ± 0.250 | 0.262 ± 0.233 | 0.326 ± 0.217 | 0.426 ± 0.172 | 0.426 ± 0.172 | 0.546 ± 0.029 | 0.509 ± 0.138 |

### Per-class Mahalanobis AUROC and decomposition check on pad_full

ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs the OOD images of that class group.

| cell | λ | ID MEL | ID NV | ID BCC | ID AK | ID BKL | ID DF | ID VASC | ID SCC | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.867 ± 0.025 | 0.939 ± 0.010 | 0.724 ± 0.045 | 0.555 ± 0.054 | 0.754 ± 0.026 | 0.375 ± 0.042 | 0.476 ± 0.074 | 0.480 ± 0.071 | 0.679 ± 0.043 | 0.679 ± 0.043 | 0.759 ± 0.040 | 0.854 ± 0.020 |
| shift 100% | 0.5 | 0.291 ± 0.021 | 0.467 ± 0.038 | 0.182 ± 0.017 | 0.114 ± 0.021 | 0.197 ± 0.021 | 0.117 ± 0.015 | 0.127 ± 0.033 | 0.099 ± 0.014 | 0.188 ± 0.018 | 0.188 ± 0.018 | 0.244 ± 0.029 | 0.354 ± 0.029 |
| shift 50% | 0.5 | 0.434 ± 0.045 | 0.589 ± 0.055 | 0.336 ± 0.044 | 0.257 ± 0.060 | 0.329 ± 0.046 | 0.183 ± 0.011 | 0.208 ± 0.073 | 0.216 ± 0.033 | 0.329 ± 0.047 | 0.329 ± 0.047 | 0.286 ± 0.029 | 0.495 ± 0.050 |
| shift 0% | 0.5 | 0.409 ± 0.041 | 0.572 ± 0.022 | 0.339 ± 0.056 | 0.269 ± 0.049 | 0.321 ± 0.032 | 0.164 ± 0.035 | 0.129 ± 0.013 | 0.231 ± 0.023 | 0.332 ± 0.042 | 0.332 ± 0.042 | 0.472 ± 0.061 | 0.458 ± 0.024 |
| shift 100% | 2 | 0.303 ± 0.015 | 0.464 ± 0.021 | 0.204 ± 0.021 | 0.131 ± 0.012 | 0.214 ± 0.014 | 0.133 ± 0.011 | 0.114 ± 0.025 | 0.113 ± 0.017 | 0.204 ± 0.012 | 0.204 ± 0.012 | 0.265 ± 0.050 | 0.358 ± 0.016 |
| shift 50% | 2 | 0.393 ± 0.017 | 0.559 ± 0.025 | 0.275 ± 0.010 | 0.215 ± 0.021 | 0.275 ± 0.007 | 0.142 ± 0.008 | 0.162 ± 0.028 | 0.173 ± 0.015 | 0.280 ± 0.010 | 0.280 ± 0.010 | 0.291 ± 0.046 | 0.450 ± 0.014 |
| shift 0% | 2 | 0.405 ± 0.150 | 0.564 ± 0.112 | 0.402 ± 0.198 | 0.326 ± 0.214 | 0.352 ± 0.194 | 0.233 ± 0.261 | 0.239 ± 0.245 | 0.292 ± 0.236 | 0.381 ± 0.195 | 0.381 ± 0.195 | 0.480 ± 0.078 | 0.470 ± 0.155 |

### Rule A with degraded seeds included (secondary; all unseen sets)

| cell | λ | pad_heldout | Fitzpatrick17k | MED-NODE | SD-198 | SCIN |
|---|---:|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.825 ± 0.022 [0.810, 0.839], 0/5 seed CIs below 0.5, rule A not met | 0.712 ± 0.013 [0.698, 0.725], 0/5 seed CIs below 0.5, rule A not met | 0.559 ± 0.019 [0.531, 0.587], 0/5 seed CIs below 0.5, rule A not met | 0.708 ± 0.019 [0.695, 0.722], 0/5 seed CIs below 0.5, rule A not met | 0.723 ± 0.018 [0.710, 0.736], 0/5 seed CIs below 0.5, rule A not met |
| shift 100% | 0.5 | 0.361 ± 0.029 [0.338, 0.383], 5/5 seed CIs below 0.5, rule A met | 0.385 ± 0.023 [0.373, 0.398], 5/5 seed CIs below 0.5, rule A met | 0.310 ± 0.055 [0.284, 0.337], 5/5 seed CIs below 0.5, rule A met | 0.365 ± 0.029 [0.353, 0.377], 5/5 seed CIs below 0.5, rule A met | 0.534 ± 0.015 [0.521, 0.547], 0/5 seed CIs below 0.5, rule A not met |
| shift 50% | 0.5 | 0.490 ± 0.058 [0.466, 0.512], 2/3 seed CIs below 0.5, rule A not met | 0.482 ± 0.143 [0.469, 0.494], 2/3 seed CIs below 0.5, rule A not met | 0.303 ± 0.054 [0.272, 0.336], 3/3 seed CIs below 0.5, rule A met | 0.451 ± 0.133 [0.439, 0.461], 2/3 seed CIs below 0.5, rule A not met | 0.586 ± 0.101 [0.575, 0.599], 1/3 seed CIs below 0.5, rule A not met |
| shift 0% | 0.5 | 0.499 ± 0.029 [0.479, 0.519], 0/4 seed CIs below 0.5, rule A not met | 0.674 ± 0.053 [0.662, 0.685], 0/4 seed CIs below 0.5, rule A not met | 0.529 ± 0.056 [0.501, 0.560], 1/4 seed CIs below 0.5, rule A not met | 0.669 ± 0.066 [0.658, 0.680], 0/4 seed CIs below 0.5, rule A not met | 0.713 ± 0.039 [0.702, 0.724], 0/4 seed CIs below 0.5, rule A not met |
| shift 100% | 2 | 0.368 ± 0.017 [0.344, 0.391], 5/5 seed CIs below 0.5, rule A met | 0.340 ± 0.065 [0.328, 0.352], 5/5 seed CIs below 0.5, rule A met | 0.298 ± 0.032 [0.271, 0.326], 5/5 seed CIs below 0.5, rule A met | 0.325 ± 0.069 [0.315, 0.336], 5/5 seed CIs below 0.5, rule A met | 0.519 ± 0.073 [0.507, 0.532], 1/5 seed CIs below 0.5, rule A not met |
| shift 50% | 2 | 0.454 ± 0.009 [0.432, 0.476], 3/3 seed CIs below 0.5, rule A met | 0.441 ± 0.076 [0.428, 0.454], 2/3 seed CIs below 0.5, rule A not met | 0.299 ± 0.052 [0.269, 0.331], 3/3 seed CIs below 0.5, rule A met | 0.414 ± 0.047 [0.402, 0.425], 3/3 seed CIs below 0.5, rule A met | 0.566 ± 0.051 [0.553, 0.578], 0/3 seed CIs below 0.5, rule A not met |
| shift 0% | 2 | 0.514 ± 0.123 [0.497, 0.529], 4/5 seed CIs below 0.5, rule A not met | 0.719 ± 0.069 [0.710, 0.729], 0/5 seed CIs below 0.5, rule A not met | 0.591 ± 0.114 [0.568, 0.617], 0/5 seed CIs below 0.5, rule A not met | 0.720 ± 0.065 [0.712, 0.729], 0/5 seed CIs below 0.5, rule A not met | 0.719 ± 0.041 [0.711, 0.728], 0/5 seed CIs below 0.5, rule A not met |

### Cross-domain behaviour on pad_heldout

| cell | λ | AK recall (6-cls) | NV recall | NV predicted share | ID NV-nearest | pad_heldout NV-nearest | leakage (class-matched ISIC subsample) |
|---|---:|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.145 ± 0.056 | 0.523 ± 0.084 | 0.196 ± 0.038 | 0.543 ± 0.008 | 0.365 ± 0.070 | 0.908 ± 0.021 |
| shift 100% | 0.5 | 0.008 ± 0.003 | 0.895 ± 0.036 | 0.644 ± 0.041 | 0.557 ± 0.012 | 0.653 ± 0.041 | 0.722 ± 0.035 |
| shift 50% | 0.5 | 0.129 ± 0.035 | 0.864 ± 0.052 | 0.429 ± 0.051 | 0.543 ± 0.009 | 0.433 ± 0.056 | 0.615 ± 0.031 |
| shift 0% | 0.5 | 0.452 ± 0.031 | 0.622 ± 0.057 | 0.135 ± 0.024 | 0.540 ± 0.017 | 0.126 ± 0.021 | 0.554 ± 0.029 |
| shift 100% | 2 | 0.010 ± 0.011 | 0.882 ± 0.027 | 0.617 ± 0.041 | 0.551 ± 0.009 | 0.628 ± 0.039 | 0.728 ± 0.021 |
| shift 50% | 2 | 0.137 ± 0.040 | 0.848 ± 0.037 | 0.463 ± 0.037 | 0.547 ± 0.006 | 0.464 ± 0.034 | 0.652 ± 0.044 |
| shift 0% | 2 | 0.430 ± 0.091 | 0.530 ± 0.181 | 0.121 ± 0.018 | 0.480 ± 0.123 | 0.110 ± 0.031 | 0.585 ± 0.040 |

## Where OOD images go (nearest ISIC class centroid, Mahalanobis)

Share of pad_heldout whose nearest ISIC-train class mean (Mahalanobis) is each class. Reported as **Δ vs λ = 0** pad_heldout distribution (paper baseline NV share ≈ 0.332 at λ = 0) and **Δ vs ID test** at the same cell (classifier / ID test unchanged by adversary-stream resampling). Raw shares in `cells.json`. Means over non-degraded seeds when any remain; otherwise all seeds.

| cell | λ | vs | MEL | NV | BCC | AK | BKL | DF | VASC | SCC |
|---|---:|---|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | raw (baseline) | 0.377 | 0.365 | 0.066 | 0.001 | 0.189 | 0.000 | 0.001 | 0.002 |
| shift 100% | 0.5 | Δ vs λ=0 pad_heldout | -0.185 | +0.289 | +0.021 | +0.006 | -0.139 | +0.000 | +0.002 | +0.005 |
| shift 100% | 0.5 | Δ vs ID test (same cell) | +0.028 | +0.096 | -0.051 | -0.012 | -0.045 | -0.004 | -0.006 | -0.006 |
| shift 100% | 0.5 | raw pad_heldout | 0.192 | 0.653 | 0.087 | 0.008 | 0.050 | 0.000 | 0.003 | 0.008 |
| shift 100% | 0.5 | raw ID test | 0.164 | 0.557 | 0.137 | 0.019 | 0.095 | 0.004 | 0.009 | 0.013 |
| shift 50% | 0.5 | Δ vs λ=0 pad_heldout | -0.240 | +0.069 | +0.146 | +0.076 | -0.094 | +0.000 | +0.001 | +0.043 |
| shift 50% | 0.5 | Δ vs ID test (same cell) | -0.031 | -0.110 | +0.070 | +0.047 | +0.005 | -0.003 | -0.007 | +0.028 |
| shift 50% | 0.5 | raw pad_heldout | 0.137 | 0.433 | 0.211 | 0.077 | 0.095 | 0.000 | 0.001 | 0.045 |
| shift 50% | 0.5 | raw ID test | 0.168 | 0.543 | 0.141 | 0.029 | 0.090 | 0.003 | 0.008 | 0.017 |
| shift 0% | 0.5 | Δ vs λ=0 pad_heldout | -0.327 | -0.239 | +0.343 | +0.253 | -0.098 | +0.000 | -0.000 | +0.067 |
| shift 0% | 0.5 | Δ vs ID test (same cell) | -0.111 | -0.414 | +0.266 | +0.218 | -0.001 | -0.003 | -0.006 | +0.051 |
| shift 0% | 0.5 | raw pad_heldout | 0.050 | 0.126 | 0.409 | 0.255 | 0.091 | 0.000 | 0.000 | 0.069 |
| shift 0% | 0.5 | raw ID test | 0.161 | 0.540 | 0.143 | 0.036 | 0.093 | 0.003 | 0.007 | 0.019 |
| shift 100% | 2 | Δ vs λ=0 pad_heldout | -0.205 | +0.264 | +0.031 | +0.009 | -0.108 | +0.001 | +0.001 | +0.006 |
| shift 100% | 2 | Δ vs ID test (same cell) | +0.007 | +0.078 | -0.045 | -0.010 | -0.016 | -0.003 | -0.006 | -0.004 |
| shift 100% | 2 | raw pad_heldout | 0.172 | 0.628 | 0.097 | 0.010 | 0.082 | 0.001 | 0.002 | 0.009 |
| shift 100% | 2 | raw ID test | 0.165 | 0.551 | 0.142 | 0.020 | 0.098 | 0.004 | 0.008 | 0.013 |
| shift 50% | 2 | Δ vs λ=0 pad_heldout | -0.259 | +0.100 | +0.133 | +0.091 | -0.095 | +0.002 | +0.001 | +0.028 |
| shift 50% | 2 | Δ vs ID test (same cell) | -0.048 | -0.083 | +0.065 | +0.059 | +0.001 | -0.002 | -0.007 | +0.015 |
| shift 50% | 2 | raw pad_heldout | 0.118 | 0.464 | 0.198 | 0.092 | 0.094 | 0.002 | 0.001 | 0.030 |
| shift 50% | 2 | raw ID test | 0.166 | 0.547 | 0.133 | 0.033 | 0.093 | 0.004 | 0.009 | 0.015 |
| shift 0% | 2 | Δ vs λ=0 pad_heldout | -0.329 | -0.242 | +0.330 | +0.258 | -0.090 | +0.000 | -0.001 | +0.075 |
| shift 0% | 2 | Δ vs ID test (same cell) | -0.113 | -0.413 | +0.252 | +0.220 | +0.004 | -0.003 | -0.007 | +0.058 |
| shift 0% | 2 | raw pad_heldout | 0.047 | 0.123 | 0.396 | 0.259 | 0.099 | 0.000 | 0.000 | 0.077 |
| shift 0% | 2 | raw ID test | 0.160 | 0.535 | 0.143 | 0.038 | 0.095 | 0.003 | 0.007 | 0.019 |

## Dose tables (Mahalanobis by shift; every unseen set)

Primary rule A excludes degraded seeds; all-seeds column secondary. Point detectors for each set are in the detector sections above.

### Dose — pad_heldout

| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |
|---:|---|---|---|---|---|
| 0.5 | shift 100% | 0.361 ± 0.029 [0.338, 0.383], 5/5 seed CIs below 0.5, rule A met | 0.361 ± 0.029 [0.338, 0.383], 5/5 seed CIs below 0.5, rule A met | 0.628 ± 0.062 | 0.653 ± 0.041 |
| 0.5 | shift 50% | 0.490 ± 0.058 [0.466, 0.512], 2/3 seed CIs below 0.5, rule A not met | 0.490 ± 0.058 [0.466, 0.512], 2/3 seed CIs below 0.5, rule A not met | 0.629 ± 0.045 | 0.433 ± 0.056 |
| 0.5 | shift 0% | 0.499 ± 0.029 [0.479, 0.519], 0/4 seed CIs below 0.5, rule A not met | 0.499 ± 0.029 [0.479, 0.519], 0/4 seed CIs below 0.5, rule A not met | 0.790 ± 0.007 | 0.126 ± 0.021 |
| 2 | shift 100% | 0.368 ± 0.017 [0.344, 0.391], 5/5 seed CIs below 0.5, rule A met | 0.368 ± 0.017 [0.344, 0.391], 5/5 seed CIs below 0.5, rule A met | 0.634 ± 0.054 | 0.628 ± 0.039 |
| 2 | shift 50% | 0.454 ± 0.009 [0.432, 0.476], 3/3 seed CIs below 0.5, rule A met | 0.454 ± 0.009 [0.432, 0.476], 3/3 seed CIs below 0.5, rule A met | 0.658 ± 0.020 | 0.464 ± 0.034 |
| 2 | shift 0% | 0.458 ± 0.003 [0.438, 0.477], 4/4 seed CIs below 0.5, rule A met; excluded degraded [52] | 0.514 ± 0.123 [0.497, 0.529], 4/5 seed CIs below 0.5, rule A not met | 0.791 ± 0.008 | 0.123 ± 0.016 |

### Dose — Fitzpatrick17k

| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |
|---:|---|---|---|---|---|
| 0.5 | shift 100% | 0.385 ± 0.023 [0.373, 0.398], 5/5 seed CIs below 0.5, rule A met | 0.385 ± 0.023 [0.373, 0.398], 5/5 seed CIs below 0.5, rule A met | 0.628 ± 0.062 | 0.777 ± 0.054 |
| 0.5 | shift 50% | 0.482 ± 0.143 [0.469, 0.494], 2/3 seed CIs below 0.5, rule A not met | 0.482 ± 0.143 [0.469, 0.494], 2/3 seed CIs below 0.5, rule A not met | 0.629 ± 0.045 | 0.710 ± 0.165 |
| 0.5 | shift 0% | 0.674 ± 0.053 [0.662, 0.685], 0/4 seed CIs below 0.5, rule A not met | 0.674 ± 0.053 [0.662, 0.685], 0/4 seed CIs below 0.5, rule A not met | 0.790 ± 0.007 | 0.462 ± 0.091 |
| 2 | shift 100% | 0.340 ± 0.065 [0.328, 0.352], 5/5 seed CIs below 0.5, rule A met | 0.340 ± 0.065 [0.328, 0.352], 5/5 seed CIs below 0.5, rule A met | 0.634 ± 0.054 | 0.821 ± 0.055 |
| 2 | shift 50% | 0.441 ± 0.076 [0.428, 0.454], 2/3 seed CIs below 0.5, rule A not met | 0.441 ± 0.076 [0.428, 0.454], 2/3 seed CIs below 0.5, rule A not met | 0.658 ± 0.020 | 0.731 ± 0.090 |
| 2 | shift 0% | 0.691 ± 0.031 [0.680, 0.702], 0/4 seed CIs below 0.5, rule A not met; excluded degraded [52] | 0.719 ± 0.069 [0.710, 0.729], 0/5 seed CIs below 0.5, rule A not met | 0.791 ± 0.008 | 0.389 ± 0.079 |

### Dose — MED-NODE

| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |
|---:|---|---|---|---|---|
| 0.5 | shift 100% | 0.310 ± 0.055 [0.284, 0.337], 5/5 seed CIs below 0.5, rule A met | 0.310 ± 0.055 [0.284, 0.337], 5/5 seed CIs below 0.5, rule A met | 0.628 ± 0.062 | 0.762 ± 0.067 |
| 0.5 | shift 50% | 0.303 ± 0.054 [0.272, 0.336], 3/3 seed CIs below 0.5, rule A met | 0.303 ± 0.054 [0.272, 0.336], 3/3 seed CIs below 0.5, rule A met | 0.629 ± 0.045 | 0.716 ± 0.093 |
| 0.5 | shift 0% | 0.529 ± 0.056 [0.501, 0.560], 1/4 seed CIs below 0.5, rule A not met | 0.529 ± 0.056 [0.501, 0.560], 1/4 seed CIs below 0.5, rule A not met | 0.790 ± 0.007 | 0.532 ± 0.090 |
| 2 | shift 100% | 0.298 ± 0.032 [0.271, 0.326], 5/5 seed CIs below 0.5, rule A met | 0.298 ± 0.032 [0.271, 0.326], 5/5 seed CIs below 0.5, rule A met | 0.634 ± 0.054 | 0.754 ± 0.063 |
| 2 | shift 50% | 0.299 ± 0.052 [0.269, 0.331], 3/3 seed CIs below 0.5, rule A met | 0.299 ± 0.052 [0.269, 0.331], 3/3 seed CIs below 0.5, rule A met | 0.658 ± 0.020 | 0.741 ± 0.082 |
| 2 | shift 0% | 0.541 ± 0.027 [0.512, 0.572], 0/4 seed CIs below 0.5, rule A not met; excluded degraded [52] | 0.591 ± 0.114 [0.568, 0.617], 0/5 seed CIs below 0.5, rule A not met | 0.791 ± 0.008 | 0.490 ± 0.019 |

### Dose — SD-198

| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |
|---:|---|---|---|---|---|
| 0.5 | shift 100% | 0.365 ± 0.029 [0.353, 0.377], 5/5 seed CIs below 0.5, rule A met | 0.365 ± 0.029 [0.353, 0.377], 5/5 seed CIs below 0.5, rule A met | 0.628 ± 0.062 | 0.808 ± 0.052 |
| 0.5 | shift 50% | 0.451 ± 0.133 [0.439, 0.461], 2/3 seed CIs below 0.5, rule A not met | 0.451 ± 0.133 [0.439, 0.461], 2/3 seed CIs below 0.5, rule A not met | 0.629 ± 0.045 | 0.754 ± 0.134 |
| 0.5 | shift 0% | 0.669 ± 0.066 [0.658, 0.680], 0/4 seed CIs below 0.5, rule A not met | 0.669 ± 0.066 [0.658, 0.680], 0/4 seed CIs below 0.5, rule A not met | 0.790 ± 0.007 | 0.492 ± 0.098 |
| 2 | shift 100% | 0.325 ± 0.069 [0.315, 0.336], 5/5 seed CIs below 0.5, rule A met | 0.325 ± 0.069 [0.315, 0.336], 5/5 seed CIs below 0.5, rule A met | 0.634 ± 0.054 | 0.851 ± 0.062 |
| 2 | shift 50% | 0.414 ± 0.047 [0.402, 0.425], 3/3 seed CIs below 0.5, rule A met | 0.414 ± 0.047 [0.402, 0.425], 3/3 seed CIs below 0.5, rule A met | 0.658 ± 0.020 | 0.772 ± 0.056 |
| 2 | shift 0% | 0.694 ± 0.031 [0.683, 0.704], 0/4 seed CIs below 0.5, rule A not met; excluded degraded [52] | 0.720 ± 0.065 [0.712, 0.729], 0/5 seed CIs below 0.5, rule A not met | 0.791 ± 0.008 | 0.423 ± 0.100 |

### Dose — SCIN

| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |
|---:|---|---|---|---|---|
| 0.5 | shift 100% | 0.534 ± 0.015 [0.521, 0.547], 0/5 seed CIs below 0.5, rule A not met | 0.534 ± 0.015 [0.521, 0.547], 0/5 seed CIs below 0.5, rule A not met | 0.628 ± 0.062 | 0.769 ± 0.028 |
| 0.5 | shift 50% | 0.586 ± 0.101 [0.575, 0.599], 1/3 seed CIs below 0.5, rule A not met | 0.586 ± 0.101 [0.575, 0.599], 1/3 seed CIs below 0.5, rule A not met | 0.629 ± 0.045 | 0.750 ± 0.119 |
| 0.5 | shift 0% | 0.713 ± 0.039 [0.702, 0.724], 0/4 seed CIs below 0.5, rule A not met | 0.713 ± 0.039 [0.702, 0.724], 0/4 seed CIs below 0.5, rule A not met | 0.790 ± 0.007 | 0.659 ± 0.071 |
| 2 | shift 100% | 0.519 ± 0.073 [0.507, 0.532], 1/5 seed CIs below 0.5, rule A not met | 0.519 ± 0.073 [0.507, 0.532], 1/5 seed CIs below 0.5, rule A not met | 0.634 ± 0.054 | 0.786 ± 0.071 |
| 2 | shift 50% | 0.566 ± 0.051 [0.553, 0.578], 0/3 seed CIs below 0.5, rule A not met | 0.566 ± 0.051 [0.553, 0.578], 0/3 seed CIs below 0.5, rule A not met | 0.658 ± 0.020 | 0.745 ± 0.033 |
| 2 | shift 0% | 0.732 ± 0.035 [0.722, 0.742], 0/4 seed CIs below 0.5, rule A not met; excluded degraded [52] | 0.719 ± 0.041 [0.711, 0.728], 0/5 seed CIs below 0.5, rule A not met | 0.791 ± 0.008 | 0.595 ± 0.110 |

## Continuity reference: two-stream runs of the same recipe (separate table)

| cell | λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc (pad_heldout, 6-cls) |
|---|---:|---:|---|---|---|---|---|---|---|
| two-stream DANN (same recipe) | 0 | 5 | 0.927 ± 0.018 | 0.496 ± 0.016 | 0.181 ± 0.026 | 0.219 ± 0.046 | 0.808 ± 0.027 | 0.717 ± 0.034 | 0.275 ± 0.023 |
| two-stream DANN (same recipe) | 0.5 | 5 | 0.581 ± 0.031 | 0.501 ± 0.012 | 0.191 ± 0.008 | 0.710 ± 0.017 | 0.426 ± 0.008 | 0.411 ± 0.047 | 0.259 ± 0.025 |
| two-stream DANN (same recipe) | 2 | 5 | 0.555 ± 0.046 | 0.473 ± 0.023 | 0.181 ± 0.027 | 0.717 ± 0.022 | 0.403 ± 0.028 | 0.402 ± 0.018 | 0.254 ± 0.015 |

## Results in plain words

At λ = 0.5, Mahalanobis on pad_heldout by shift — shift 100%: 0.361 ± 0.029 [0.338, 0.383], 5/5 seed CIs below 0.5, rule A met (all: 0.361 ± 0.029 [0.338, 0.383], 5/5 seed CIs below 0.5, rule A met); shift 50%: 0.490 ± 0.058 [0.466, 0.512], 2/3 seed CIs below 0.5, rule A not met (all: 0.490 ± 0.058 [0.466, 0.512], 2/3 seed CIs below 0.5, rule A not met); shift 0%: 0.499 ± 0.029 [0.479, 0.519], 0/4 seed CIs below 0.5, rule A not met (all: 0.499 ± 0.029 [0.479, 0.519], 0/4 seed CIs below 0.5, rule A not met). Same shift ordering is tabulated for Fitzpatrick / MED-NODE / SD-198 / SCIN above. Prediction: inversion weakens as shift -> 0 and AUROC returns above 0.5 at 0% (work order). Numerically, Mahalanobis pad_heldout at λ = 2: shift 100% 0.38–0.45 (rule A met), 50% 0.45–0.60, 0% 0.55–0.75 (rule A not met); pad_heldout NV-nearest fraction falls with shift; Table 1 leakage at 0% within 0.05 of 100%.

At λ = 2, Mahalanobis on pad_heldout by shift — shift 100%: 0.368 ± 0.017 [0.344, 0.391], 5/5 seed CIs below 0.5, rule A met (all: 0.368 ± 0.017 [0.344, 0.391], 5/5 seed CIs below 0.5, rule A met); shift 50%: 0.454 ± 0.009 [0.432, 0.476], 3/3 seed CIs below 0.5, rule A met (all: 0.454 ± 0.009 [0.432, 0.476], 3/3 seed CIs below 0.5, rule A met); shift 0%: 0.458 ± 0.003 [0.438, 0.477], 4/4 seed CIs below 0.5, rule A met; excluded degraded [52] (all: 0.514 ± 0.123 [0.497, 0.529], 4/5 seed CIs below 0.5, rule A not met). Same shift ordering is tabulated for Fitzpatrick / MED-NODE / SD-198 / SCIN above. Prediction: inversion weakens as shift -> 0 and AUROC returns above 0.5 at 0% (work order). Numerically, Mahalanobis pad_heldout at λ = 2: shift 100% 0.38–0.45 (rule A met), 50% 0.45–0.60, 0% 0.55–0.75 (rule A not met); pad_heldout NV-nearest fraction falls with shift; Table 1 leakage at 0% within 0.05 of 100%.

λ = 0 baseline (shift 100%): Mahalanobis pad_heldout 0.825 ± 0.022 [0.810, 0.839], 0/5 seed CIs below 0.5, rule A not met, ID bal acc 0.481 ± 0.016.

## LaTeX rows (cell & λ & n & leakage & ID bal acc & ID ECE & OOD ECE & OOD AUROC & Fitz AUROC & X-dom)

```latex
λ=0 baseline (shift 100%) & 0 & 5 & 0.928\,$\pm$\,0.020 & 0.481\,$\pm$\,0.016 & 0.207\,$\pm$\,0.003 & 0.305\,$\pm$\,0.042 & 0.825\,$\pm$\,0.022 & 0.712\,$\pm$\,0.013 & 0.277\,$\pm$\,0.024 \\
shift 100% & 0.5 & 5 & 0.628\,$\pm$\,0.062 & 0.483\,$\pm$\,0.013 & 0.214\,$\pm$\,0.007 & 0.775\,$\pm$\,0.010 & 0.361\,$\pm$\,0.029 & 0.385\,$\pm$\,0.023 & 0.222\,$\pm$\,0.016 \\
shift 50% & 0.5 & 3 & 0.629\,$\pm$\,0.045 & 0.489\,$\pm$\,0.002 & 0.204\,$\pm$\,0.005 & 0.641\,$\pm$\,0.033 & 0.490\,$\pm$\,0.058 & 0.482\,$\pm$\,0.143 & 0.293\,$\pm$\,0.012 \\
shift 0% & 0.5 & 4 & 0.790\,$\pm$\,0.007 & 0.468\,$\pm$\,0.013 & 0.212\,$\pm$\,0.003 & 0.442\,$\pm$\,0.020 & 0.499\,$\pm$\,0.029 & 0.674\,$\pm$\,0.053 & 0.380\,$\pm$\,0.025 \\
shift 100% & 2 & 5 & 0.634\,$\pm$\,0.054 & 0.486\,$\pm$\,0.012 & 0.205\,$\pm$\,0.007 & 0.757\,$\pm$\,0.018 & 0.368\,$\pm$\,0.017 & 0.340\,$\pm$\,0.065 & 0.231\,$\pm$\,0.029 \\
shift 50% & 2 & 3 & 0.658\,$\pm$\,0.020 & 0.491\,$\pm$\,0.003 & 0.208\,$\pm$\,0.004 & 0.646\,$\pm$\,0.019 & 0.454\,$\pm$\,0.009 & 0.441\,$\pm$\,0.076 & 0.311\,$\pm$\,0.014 \\
shift 0% & 2 & 5 & 0.782\,$\pm$\,0.019 & 0.414\,$\pm$\,0.154 & 0.188\,$\pm$\,0.057 & 0.361\,$\pm$\,0.144 & 0.514\,$\pm$\,0.123 & 0.719\,$\pm$\,0.069 & 0.366\,$\pm$\,0.103 \\
% two-stream reference
two-stream DANN (same recipe) & 0 & 5 & 0.927\,$\pm$\,0.018 & 0.496\,$\pm$\,0.016 & 0.181\,$\pm$\,0.026 & 0.219\,$\pm$\,0.046 & 0.808\,$\pm$\,0.027 & 0.717\,$\pm$\,0.034 & 0.275\,$\pm$\,0.023 \\
two-stream DANN (same recipe) & 0.5 & 5 & 0.581\,$\pm$\,0.031 & 0.501\,$\pm$\,0.012 & 0.191\,$\pm$\,0.008 & 0.710\,$\pm$\,0.017 & 0.426\,$\pm$\,0.008 & 0.411\,$\pm$\,0.047 & 0.259\,$\pm$\,0.025 \\
two-stream DANN (same recipe) & 2 & 5 & 0.555\,$\pm$\,0.046 & 0.473\,$\pm$\,0.023 & 0.181\,$\pm$\,0.027 & 0.717\,$\pm$\,0.022 & 0.403\,$\pm$\,0.028 & 0.402\,$\pm$\,0.018 & 0.254\,$\pm$\,0.015 \\
```

