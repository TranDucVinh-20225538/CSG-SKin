# Paper B R6 R1 — causal dose test (recipe: cur)

Design: `decisions/paperB_r6_r1.md`. Commit `dba39c7`. Three streams per step: ISIC classifier stream (full ISIC train, natural order), ISIC adversary stream resampled to the shift level, PAD adversary stream unchanged. PAD labels set only the resampling proportions and enter no loss. Primary checkpoint = `last.ckpt`. Every shift cell is scored on pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN (same detectors + cluster/patient bootstrap).

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
| shift 50% | 0.5 | 72 | 0.472 | 0.476 | — |
| shift 50% | 0.5 | 82 | 0.478 | 0.482 | — |
| shift 0% | 0.5 | 42 | 0.469 | 0.509 | — |
| shift 0% | 0.5 | 52 | 0.475 | 0.469 | — |
| shift 0% | 0.5 | 62 | 0.450 | 0.471 | — |
| shift 0% | 0.5 | 72 | 0.479 | 0.476 | — |
| shift 0% | 0.5 | 82 | 0.480 | 0.482 | — |
| shift 100% | 2 | 42 | 0.499 | 0.509 | — |
| shift 100% | 2 | 52 | 0.479 | 0.469 | — |
| shift 100% | 2 | 62 | 0.480 | 0.471 | — |
| shift 100% | 2 | 72 | 0.473 | 0.476 | — |
| shift 100% | 2 | 82 | 0.500 | 0.482 | — |
| shift 50% | 2 | 42 | 0.493 | 0.509 | — |
| shift 50% | 2 | 52 | 0.488 | 0.469 | — |
| shift 50% | 2 | 62 | 0.491 | 0.471 | — |
| shift 50% | 2 | 72 | 0.471 | 0.476 | — |
| shift 50% | 2 | 82 | 0.491 | 0.482 | — |
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
| shift 100% | 5 | 0.368 ± 0.017 | 0.371 ± 0.016 | 0.368 ± 0.016 | 0.392 ± 0.028 | 0.404 ± 0.045 | — | — |
| shift 50% | 5 | 0.469 ± 0.027 | 0.470 ± 0.031 | 0.440 ± 0.034 | 0.448 ± 0.019 | 0.429 ± 0.028 | — | — |
| shift 0% | 5 | 0.458 ± 0.003 | 0.581 ± 0.010 | 0.479 ± 0.002 | 0.535 ± 0.018 | 0.510 ± 0.041 | — | — |

### Fitzpatrick17k (λ = 2)

| shift | n seeds | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha [rule A excl. degraded] | Maha [all seeds] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| shift 100% | 5 | 0.340 ± 0.065 | 0.360 ± 0.054 | 0.364 ± 0.048 | 0.396 ± 0.056 | 0.462 ± 0.079 | — | — |
| shift 50% | 5 | 0.457 ± 0.062 | 0.442 ± 0.057 | 0.444 ± 0.051 | 0.451 ± 0.039 | 0.487 ± 0.038 | — | — |
| shift 0% | 5 | 0.691 ± 0.031 | 0.677 ± 0.045 | 0.631 ± 0.034 | 0.610 ± 0.053 | 0.609 ± 0.086 | — | — |

### MED-NODE (λ = 2)

| shift | n seeds | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha [rule A excl. degraded] | Maha [all seeds] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| shift 100% | 5 | 0.298 ± 0.032 | 0.327 ± 0.035 | 0.362 ± 0.035 | 0.388 ± 0.058 | 0.433 ± 0.060 | — | — |
| shift 50% | 5 | 0.322 ± 0.054 | 0.339 ± 0.058 | 0.381 ± 0.053 | 0.426 ± 0.047 | 0.459 ± 0.037 | — | — |
| shift 0% | 5 | 0.541 ± 0.027 | 0.531 ± 0.028 | 0.542 ± 0.025 | 0.533 ± 0.035 | 0.497 ± 0.076 | — | — |

### SD-198 (λ = 2)

| shift | n seeds | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha [rule A excl. degraded] | Maha [all seeds] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| shift 100% | 5 | 0.325 ± 0.069 | 0.348 ± 0.052 | 0.355 ± 0.052 | 0.381 ± 0.052 | 0.448 ± 0.072 | — | — |
| shift 50% | 5 | 0.432 ± 0.047 | 0.421 ± 0.045 | 0.430 ± 0.039 | 0.437 ± 0.045 | 0.475 ± 0.042 | — | — |
| shift 0% | 5 | 0.694 ± 0.031 | 0.675 ± 0.046 | 0.633 ± 0.032 | 0.604 ± 0.058 | 0.599 ± 0.094 | — | — |

### SCIN (λ = 2)

| shift | n seeds | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha [rule A excl. degraded] | Maha [all seeds] |
|---|---:|---:|---:|---:|---:|---:|---|---|
| shift 100% | 5 | 0.519 ± 0.073 | 0.528 ± 0.058 | 0.522 ± 0.054 | 0.499 ± 0.051 | 0.550 ± 0.072 | — | — |
| shift 50% | 5 | 0.583 ± 0.043 | 0.557 ± 0.030 | 0.557 ± 0.023 | 0.537 ± 0.037 | 0.568 ± 0.046 | — | — |
| shift 0% | 5 | 0.732 ± 0.035 | 0.677 ± 0.050 | 0.652 ± 0.037 | 0.609 ± 0.069 | 0.619 ± 0.097 | — | — |

### λ = 2 Mahalanobis point Δ (shift 0% − shift 100%), excl. degraded

| set | mean Δ | shift 0% | shift 100% | shift 50% |
|---|---:|---:|---:|---:|
| pad_heldout | +0.090 | 0.458 ± 0.003 | 0.368 ± 0.017 | 0.469 ± 0.027 |
| Fitzpatrick17k | +0.351 | 0.691 ± 0.031 | 0.340 ± 0.065 | 0.457 ± 0.062 |
| MED-NODE | +0.243 | 0.541 ± 0.027 | 0.298 ± 0.032 | 0.322 ± 0.054 |
| SD-198 | +0.368 | 0.694 ± 0.031 | 0.325 ± 0.069 | 0.432 ± 0.047 |
| SCIN | +0.212 | 0.732 ± 0.035 | 0.519 ± 0.073 | 0.583 ± 0.043 |

### Flags (Table 1 leakage vs the λ = 0 cell; Mahalanobis pad_heldout headroom)

| cell | λ | leakage | Δ vs λ = 0 | flag |
|---|---:|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.928 ± 0.020 | — | — |
| shift 100% | 0.5 | 0.628 ± 0.062 | -0.299 | — |
| shift 50% | 0.5 | 0.637 ± 0.039 | -0.290 | — |
| shift 0% | 0.5 | 0.783 ± 0.017 | -0.144 | — |
| shift 100% | 2 | 0.634 ± 0.054 | -0.293 | — |
| shift 50% | 2 | 0.647 ± 0.021 | -0.280 | — |
| shift 0% | 2 | 0.782 ± 0.019 | -0.145 | — |

## Table 1 columns

Leakage = logistic domain probe on z_lesion_norm, ISIC test vs all of PAD (Table 1 definition). Mean ± s.d. over seeds; n = seeds present.

| cell | λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc (pad_heldout, 6-cls) |
|---|---:|---:|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 5 | 0.928 ± 0.020 | 0.481 ± 0.016 | 0.207 ± 0.003 | 0.305 ± 0.042 | 0.825 ± 0.022 | 0.712 ± 0.013 | 0.277 ± 0.024 |
| shift 100% | 0.5 | 5 | 0.628 ± 0.062 | 0.483 ± 0.013 | 0.214 ± 0.007 | 0.775 ± 0.010 | 0.361 ± 0.029 | 0.385 ± 0.023 | 0.222 ± 0.016 |
| shift 50% | 0.5 | 5 | 0.637 ± 0.039 | 0.484 ± 0.008 | 0.206 ± 0.005 | 0.641 ± 0.029 | 0.486 ± 0.047 | 0.455 ± 0.115 | 0.296 ± 0.028 |
| shift 0% | 0.5 | 5 | 0.783 ± 0.017 | 0.471 ± 0.012 | 0.212 ± 0.003 | 0.437 ± 0.021 | 0.498 ± 0.025 | 0.672 ± 0.046 | 0.379 ± 0.022 |
| shift 100% | 2 | 5 | 0.634 ± 0.054 | 0.486 ± 0.012 | 0.205 ± 0.007 | 0.757 ± 0.018 | 0.368 ± 0.017 | 0.340 ± 0.065 | 0.231 ± 0.029 |
| shift 50% | 2 | 5 | 0.647 ± 0.021 | 0.487 ± 0.009 | 0.208 ± 0.003 | 0.643 ± 0.017 | 0.469 ± 0.027 | 0.457 ± 0.062 | 0.308 ± 0.011 |
| shift 0% | 2 | 5 | 0.782 ± 0.019 | 0.414 ± 0.154 | 0.188 ± 0.057 | 0.361 ± 0.144 | 0.514 ± 0.123 | 0.719 ± 0.069 | 0.366 ± 0.103 |

### Detectors on pad_heldout (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.825 ± 0.022 | 0.892 ± 0.013 | 0.844 ± 0.015 | 0.928 ± 0.008 | 0.959 ± 0.008 | — | — | 0.365 ± 0.070 |
| shift 100% | 0.5 | 0.361 ± 0.029 | 0.376 ± 0.026 | 0.381 ± 0.014 | 0.407 ± 0.035 | 0.444 ± 0.048 | — | — | 0.653 ± 0.041 |
| shift 50% | 0.5 | 0.486 ± 0.047 | 0.491 ± 0.032 | 0.461 ± 0.026 | 0.461 ± 0.015 | 0.422 ± 0.039 | — | — | 0.432 ± 0.051 |
| shift 0% | 0.5 | 0.498 ± 0.025 | 0.604 ± 0.015 | 0.508 ± 0.027 | 0.527 ± 0.024 | 0.493 ± 0.044 | — | — | 0.134 ± 0.027 |
| shift 100% | 2 | 0.368 ± 0.017 | 0.371 ± 0.016 | 0.368 ± 0.016 | 0.392 ± 0.028 | 0.404 ± 0.045 | — | — | 0.628 ± 0.039 |
| shift 50% | 2 | 0.469 ± 0.027 | 0.470 ± 0.031 | 0.440 ± 0.034 | 0.448 ± 0.019 | 0.429 ± 0.028 | — | — | 0.447 ± 0.035 |
| shift 0% | 2 | 0.514 ± 0.123 | 0.583 ± 0.010 | 0.495 ± 0.035 | 0.512 ± 0.056 | 0.504 ± 0.038 | — | — | 0.110 ± 0.031 |

### Detectors on fitzpatrick17k (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.712 ± 0.013 | 0.770 ± 0.015 | 0.731 ± 0.015 | 0.795 ± 0.034 | 0.875 ± 0.041 | — | — | 0.601 ± 0.085 |
| shift 100% | 0.5 | 0.385 ± 0.023 | 0.418 ± 0.030 | 0.423 ± 0.021 | 0.450 ± 0.046 | 0.524 ± 0.062 | — | — | 0.777 ± 0.054 |
| shift 50% | 0.5 | 0.455 ± 0.115 | 0.450 ± 0.095 | 0.450 ± 0.087 | 0.459 ± 0.049 | 0.497 ± 0.045 | — | — | 0.714 ± 0.118 |
| shift 0% | 0.5 | 0.672 ± 0.046 | 0.656 ± 0.046 | 0.620 ± 0.037 | 0.560 ± 0.056 | 0.556 ± 0.066 | — | — | 0.483 ± 0.092 |
| shift 100% | 2 | 0.340 ± 0.065 | 0.360 ± 0.054 | 0.364 ± 0.048 | 0.396 ± 0.056 | 0.462 ± 0.079 | — | — | 0.821 ± 0.055 |
| shift 50% | 2 | 0.457 ± 0.062 | 0.442 ± 0.057 | 0.444 ± 0.051 | 0.451 ± 0.039 | 0.487 ± 0.038 | — | — | 0.710 ± 0.070 |
| shift 0% | 2 | 0.719 ± 0.069 | 0.663 ± 0.051 | 0.634 ± 0.030 | 0.566 ± 0.109 | 0.569 ± 0.115 | — | — | 0.335 ± 0.138 |

### Detectors on mednode (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.559 ± 0.019 | 0.603 ± 0.010 | 0.602 ± 0.005 | 0.674 ± 0.016 | 0.732 ± 0.015 | — | — | 0.548 ± 0.069 |
| shift 100% | 0.5 | 0.310 ± 0.055 | 0.355 ± 0.056 | 0.389 ± 0.048 | 0.430 ± 0.050 | 0.490 ± 0.043 | — | — | 0.762 ± 0.067 |
| shift 50% | 0.5 | 0.312 ± 0.051 | 0.331 ± 0.038 | 0.370 ± 0.042 | 0.412 ± 0.041 | 0.426 ± 0.050 | — | — | 0.722 ± 0.082 |
| shift 0% | 0.5 | 0.525 ± 0.050 | 0.516 ± 0.041 | 0.527 ± 0.040 | 0.511 ± 0.043 | 0.480 ± 0.047 | — | — | 0.548 ± 0.086 |
| shift 100% | 2 | 0.298 ± 0.032 | 0.327 ± 0.035 | 0.362 ± 0.035 | 0.388 ± 0.058 | 0.433 ± 0.060 | — | — | 0.754 ± 0.063 |
| shift 50% | 2 | 0.322 ± 0.054 | 0.339 ± 0.058 | 0.381 ± 0.053 | 0.426 ± 0.047 | 0.459 ± 0.037 | — | — | 0.718 ± 0.091 |
| shift 0% | 2 | 0.591 ± 0.114 | 0.561 ± 0.071 | 0.564 ± 0.053 | 0.531 ± 0.031 | 0.479 ± 0.077 | — | — | 0.429 ± 0.136 |

### Detectors on sd198 (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.708 ± 0.019 | 0.771 ± 0.007 | 0.733 ± 0.007 | 0.806 ± 0.026 | 0.885 ± 0.035 | — | — | 0.605 ± 0.083 |
| shift 100% | 0.5 | 0.365 ± 0.029 | 0.399 ± 0.037 | 0.412 ± 0.032 | 0.441 ± 0.056 | 0.519 ± 0.077 | — | — | 0.808 ± 0.052 |
| shift 50% | 0.5 | 0.428 ± 0.104 | 0.422 ± 0.086 | 0.429 ± 0.081 | 0.433 ± 0.043 | 0.471 ± 0.056 | — | — | 0.769 ± 0.098 |
| shift 0% | 0.5 | 0.664 ± 0.058 | 0.642 ± 0.051 | 0.611 ± 0.043 | 0.542 ± 0.055 | 0.537 ± 0.058 | — | — | 0.519 ± 0.103 |
| shift 100% | 2 | 0.325 ± 0.069 | 0.348 ± 0.052 | 0.355 ± 0.052 | 0.381 ± 0.052 | 0.448 ± 0.072 | — | — | 0.851 ± 0.062 |
| shift 50% | 2 | 0.432 ± 0.047 | 0.421 ± 0.045 | 0.430 ± 0.039 | 0.437 ± 0.045 | 0.475 ± 0.042 | — | — | 0.752 ± 0.049 |
| shift 0% | 2 | 0.720 ± 0.065 | 0.657 ± 0.057 | 0.637 ± 0.029 | 0.556 ± 0.118 | 0.560 ± 0.119 | — | — | 0.366 ± 0.155 |

### Detectors on scin (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.723 ± 0.018 | 0.770 ± 0.023 | 0.735 ± 0.023 | 0.767 ± 0.050 | 0.850 ± 0.048 | — | — | 0.585 ± 0.096 |
| shift 100% | 0.5 | 0.534 ± 0.015 | 0.560 ± 0.026 | 0.560 ± 0.022 | 0.530 ± 0.031 | 0.579 ± 0.055 | — | — | 0.769 ± 0.028 |
| shift 50% | 0.5 | 0.567 ± 0.080 | 0.545 ± 0.063 | 0.544 ± 0.062 | 0.514 ± 0.074 | 0.562 ± 0.091 | — | — | 0.743 ± 0.090 |
| shift 0% | 0.5 | 0.718 ± 0.036 | 0.664 ± 0.036 | 0.643 ± 0.035 | 0.579 ± 0.066 | 0.589 ± 0.072 | — | — | 0.659 ± 0.061 |
| shift 100% | 2 | 0.519 ± 0.073 | 0.528 ± 0.058 | 0.522 ± 0.054 | 0.499 ± 0.051 | 0.550 ± 0.072 | — | — | 0.786 ± 0.071 |
| shift 50% | 2 | 0.583 ± 0.043 | 0.557 ± 0.030 | 0.557 ± 0.023 | 0.537 ± 0.037 | 0.568 ± 0.046 | — | — | 0.714 ± 0.065 |
| shift 0% | 2 | 0.719 ± 0.041 | 0.670 ± 0.046 | 0.643 ± 0.039 | 0.597 ± 0.065 | 0.582 ± 0.117 | — | — | 0.505 ± 0.224 |

### Per-class Mahalanobis AUROC and decomposition check on pad_heldout

ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs the OOD images of that class group.

| cell | λ | ID MEL | ID NV | ID BCC | ID AK | ID BKL | ID DF | ID VASC | ID SCC | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.844 ± 0.028 | 0.926 ± 0.012 | 0.695 ± 0.042 | 0.525 ± 0.049 | 0.727 ± 0.026 | 0.358 ± 0.041 | 0.455 ± 0.072 | 0.456 ± 0.066 | 0.659 ± 0.039 | 0.659 ± 0.039 | 0.688 ± 0.040 | 0.844 ± 0.020 |
| shift 100% | 0.5 | 0.312 ± 0.025 | 0.487 ± 0.038 | 0.197 ± 0.020 | 0.123 ± 0.025 | 0.214 ± 0.026 | 0.124 ± 0.012 | 0.137 ± 0.039 | 0.106 ± 0.017 | 0.207 ± 0.022 | 0.207 ± 0.022 | 0.260 ± 0.026 | 0.375 ± 0.032 |
| shift 50% | 0.5 | 0.446 ± 0.047 | 0.602 ± 0.047 | 0.351 ± 0.047 | 0.267 ± 0.059 | 0.340 ± 0.057 | 0.193 ± 0.019 | 0.199 ± 0.066 | 0.226 ± 0.036 | 0.347 ± 0.049 | 0.347 ± 0.049 | 0.325 ± 0.017 | 0.508 ± 0.051 |
| shift 0% | 0.5 | 0.449 ± 0.039 | 0.613 ± 0.019 | 0.381 ± 0.053 | 0.305 ± 0.047 | 0.355 ± 0.035 | 0.176 ± 0.042 | 0.155 ± 0.020 | 0.262 ± 0.025 | 0.375 ± 0.040 | 0.375 ± 0.040 | 0.528 ± 0.058 | 0.494 ± 0.025 |
| shift 100% | 2 | 0.324 ± 0.021 | 0.487 ± 0.026 | 0.219 ± 0.017 | 0.139 ± 0.010 | 0.231 ± 0.012 | 0.140 ± 0.012 | 0.118 ± 0.026 | 0.118 ± 0.015 | 0.222 ± 0.007 | 0.222 ± 0.007 | 0.288 ± 0.043 | 0.379 ± 0.017 |
| shift 50% | 2 | 0.424 ± 0.023 | 0.596 ± 0.029 | 0.318 ± 0.038 | 0.253 ± 0.033 | 0.307 ± 0.022 | 0.165 ± 0.022 | 0.169 ± 0.033 | 0.196 ± 0.031 | 0.323 ± 0.031 | 0.323 ± 0.031 | 0.342 ± 0.030 | 0.486 ± 0.029 |
| shift 0% | 2 | 0.451 ± 0.123 | 0.607 ± 0.089 | 0.445 ± 0.175 | 0.366 ± 0.193 | 0.394 ± 0.171 | 0.256 ± 0.250 | 0.262 ± 0.233 | 0.326 ± 0.217 | 0.426 ± 0.172 | 0.426 ± 0.172 | 0.546 ± 0.029 | 0.509 ± 0.138 |

### Per-class Mahalanobis AUROC and decomposition check on pad_full

ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs the OOD images of that class group.

| cell | λ | ID MEL | ID NV | ID BCC | ID AK | ID BKL | ID DF | ID VASC | ID SCC | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.867 ± 0.025 | 0.939 ± 0.010 | 0.724 ± 0.045 | 0.555 ± 0.054 | 0.754 ± 0.026 | 0.375 ± 0.042 | 0.476 ± 0.074 | 0.480 ± 0.071 | 0.679 ± 0.043 | 0.679 ± 0.043 | 0.759 ± 0.040 | 0.854 ± 0.020 |
| shift 100% | 0.5 | 0.291 ± 0.021 | 0.467 ± 0.038 | 0.182 ± 0.017 | 0.114 ± 0.021 | 0.197 ± 0.021 | 0.117 ± 0.015 | 0.127 ± 0.033 | 0.099 ± 0.014 | 0.188 ± 0.018 | 0.188 ± 0.018 | 0.244 ± 0.029 | 0.354 ± 0.029 |
| shift 50% | 0.5 | 0.423 ± 0.035 | 0.588 ± 0.039 | 0.330 ± 0.034 | 0.246 ± 0.047 | 0.317 ± 0.042 | 0.181 ± 0.010 | 0.186 ± 0.059 | 0.208 ± 0.026 | 0.321 ± 0.037 | 0.321 ± 0.037 | 0.298 ± 0.027 | 0.488 ± 0.038 |
| shift 0% | 0.5 | 0.404 ± 0.037 | 0.571 ± 0.019 | 0.339 ± 0.049 | 0.266 ± 0.043 | 0.312 ± 0.034 | 0.153 ± 0.039 | 0.134 ± 0.016 | 0.227 ± 0.023 | 0.330 ± 0.037 | 0.330 ± 0.037 | 0.473 ± 0.053 | 0.455 ± 0.022 |
| shift 100% | 2 | 0.303 ± 0.015 | 0.464 ± 0.021 | 0.204 ± 0.021 | 0.131 ± 0.012 | 0.214 ± 0.014 | 0.133 ± 0.011 | 0.114 ± 0.025 | 0.113 ± 0.017 | 0.204 ± 0.012 | 0.204 ± 0.012 | 0.265 ± 0.050 | 0.358 ± 0.016 |
| shift 50% | 2 | 0.401 ± 0.025 | 0.577 ± 0.036 | 0.298 ± 0.037 | 0.234 ± 0.032 | 0.286 ± 0.020 | 0.156 ± 0.020 | 0.159 ± 0.034 | 0.183 ± 0.030 | 0.299 ± 0.031 | 0.299 ± 0.031 | 0.308 ± 0.041 | 0.466 ± 0.031 |
| shift 0% | 2 | 0.405 ± 0.150 | 0.564 ± 0.112 | 0.402 ± 0.198 | 0.326 ± 0.214 | 0.352 ± 0.194 | 0.233 ± 0.261 | 0.239 ± 0.245 | 0.292 ± 0.236 | 0.381 ± 0.195 | 0.381 ± 0.195 | 0.480 ± 0.078 | 0.470 ± 0.155 |

### Rule A with degraded seeds included (secondary; all unseen sets)

| cell | λ | pad_heldout | Fitzpatrick17k | MED-NODE | SD-198 | SCIN |
|---|---:|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | — | — | — | — | — |
| shift 100% | 0.5 | — | — | — | — | — |
| shift 50% | 0.5 | — | — | — | — | — |
| shift 0% | 0.5 | — | — | — | — | — |
| shift 100% | 2 | — | — | — | — | — |
| shift 50% | 2 | — | — | — | — | — |
| shift 0% | 2 | — | — | — | — | — |

### Cross-domain behaviour on pad_heldout

| cell | λ | AK recall (6-cls) | NV recall | NV predicted share | ID NV-nearest | pad_heldout NV-nearest | leakage (class-matched ISIC subsample) |
|---|---:|---|---|---|---|---|---|
| λ=0 baseline (shift 100%) | 0 | 0.145 ± 0.056 | 0.523 ± 0.084 | 0.196 ± 0.038 | 0.543 ± 0.008 | 0.365 ± 0.070 | 0.908 ± 0.021 |
| shift 100% | 0.5 | 0.008 ± 0.003 | 0.895 ± 0.036 | 0.644 ± 0.041 | 0.557 ± 0.012 | 0.653 ± 0.041 | 0.722 ± 0.035 |
| shift 50% | 0.5 | 0.124 ± 0.034 | 0.857 ± 0.038 | 0.429 ± 0.048 | 0.543 ± 0.007 | 0.432 ± 0.051 | 0.603 ± 0.040 |
| shift 0% | 0.5 | 0.454 ± 0.027 | 0.634 ± 0.056 | 0.143 ± 0.027 | 0.538 ± 0.015 | 0.134 ± 0.027 | 0.553 ± 0.026 |
| shift 100% | 2 | 0.010 ± 0.011 | 0.882 ± 0.027 | 0.617 ± 0.041 | 0.551 ± 0.009 | 0.628 ± 0.039 | 0.728 ± 0.021 |
| shift 50% | 2 | 0.134 ± 0.029 | 0.841 ± 0.028 | 0.441 ± 0.040 | 0.544 ± 0.008 | 0.447 ± 0.035 | 0.643 ± 0.034 |
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
| shift 50% | 0.5 | Δ vs λ=0 pad_heldout | -0.238 | +0.068 | +0.153 | +0.076 | -0.095 | +0.000 | +0.000 | +0.037 |
| shift 50% | 0.5 | Δ vs ID test (same cell) | -0.029 | -0.111 | +0.076 | +0.048 | +0.001 | -0.003 | -0.007 | +0.024 |
| shift 50% | 0.5 | raw pad_heldout | 0.138 | 0.432 | 0.219 | 0.077 | 0.094 | 0.000 | 0.001 | 0.039 |
| shift 50% | 0.5 | raw ID test | 0.167 | 0.543 | 0.143 | 0.029 | 0.092 | 0.003 | 0.008 | 0.015 |
| shift 0% | 0.5 | Δ vs λ=0 pad_heldout | -0.322 | -0.230 | +0.348 | +0.249 | -0.105 | +0.000 | -0.001 | +0.061 |
| shift 0% | 0.5 | Δ vs ID test (same cell) | -0.108 | -0.404 | +0.269 | +0.214 | -0.006 | -0.003 | -0.006 | +0.044 |
| shift 0% | 0.5 | raw pad_heldout | 0.054 | 0.134 | 0.414 | 0.250 | 0.084 | 0.000 | 0.000 | 0.063 |
| shift 0% | 0.5 | raw ID test | 0.163 | 0.538 | 0.145 | 0.036 | 0.090 | 0.003 | 0.007 | 0.019 |
| shift 100% | 2 | Δ vs λ=0 pad_heldout | -0.205 | +0.264 | +0.031 | +0.009 | -0.108 | +0.001 | +0.001 | +0.006 |
| shift 100% | 2 | Δ vs ID test (same cell) | +0.007 | +0.078 | -0.045 | -0.010 | -0.016 | -0.003 | -0.006 | -0.004 |
| shift 100% | 2 | raw pad_heldout | 0.172 | 0.628 | 0.097 | 0.010 | 0.082 | 0.001 | 0.002 | 0.009 |
| shift 100% | 2 | raw ID test | 0.165 | 0.551 | 0.142 | 0.020 | 0.098 | 0.004 | 0.008 | 0.013 |
| shift 50% | 2 | Δ vs λ=0 pad_heldout | -0.241 | +0.082 | +0.132 | +0.095 | -0.097 | +0.001 | +0.000 | +0.028 |
| shift 50% | 2 | Δ vs ID test (same cell) | -0.030 | -0.097 | +0.059 | +0.065 | -0.002 | -0.003 | -0.007 | +0.015 |
| shift 50% | 2 | raw pad_heldout | 0.135 | 0.447 | 0.198 | 0.096 | 0.092 | 0.001 | 0.001 | 0.030 |
| shift 50% | 2 | raw ID test | 0.165 | 0.544 | 0.139 | 0.031 | 0.093 | 0.004 | 0.008 | 0.015 |
| shift 0% | 2 | Δ vs λ=0 pad_heldout | -0.329 | -0.242 | +0.330 | +0.258 | -0.090 | +0.000 | -0.001 | +0.075 |
| shift 0% | 2 | Δ vs ID test (same cell) | -0.113 | -0.413 | +0.252 | +0.220 | +0.004 | -0.003 | -0.007 | +0.058 |
| shift 0% | 2 | raw pad_heldout | 0.047 | 0.123 | 0.396 | 0.259 | 0.099 | 0.000 | 0.000 | 0.077 |
| shift 0% | 2 | raw ID test | 0.160 | 0.535 | 0.143 | 0.038 | 0.095 | 0.003 | 0.007 | 0.019 |

## Dose tables (Mahalanobis by shift; every unseen set)

Primary rule A excludes degraded seeds; all-seeds column secondary. Point detectors for each set are in the detector sections above.

### Dose — pad_heldout

| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |
|---:|---|---|---|---|---|
| 0.5 | shift 100% | — | — | 0.628 ± 0.062 | 0.653 ± 0.041 |
| 0.5 | shift 50% | — | — | 0.637 ± 0.039 | 0.432 ± 0.051 |
| 0.5 | shift 0% | — | — | 0.783 ± 0.017 | 0.134 ± 0.027 |
| 2 | shift 100% | — | — | 0.634 ± 0.054 | 0.628 ± 0.039 |
| 2 | shift 50% | — | — | 0.647 ± 0.021 | 0.447 ± 0.035 |
| 2 | shift 0% | — | — | 0.791 ± 0.008 | 0.123 ± 0.016 |

### Dose — Fitzpatrick17k

| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |
|---:|---|---|---|---|---|
| 0.5 | shift 100% | — | — | 0.628 ± 0.062 | 0.777 ± 0.054 |
| 0.5 | shift 50% | — | — | 0.637 ± 0.039 | 0.714 ± 0.118 |
| 0.5 | shift 0% | — | — | 0.783 ± 0.017 | 0.483 ± 0.092 |
| 2 | shift 100% | — | — | 0.634 ± 0.054 | 0.821 ± 0.055 |
| 2 | shift 50% | — | — | 0.647 ± 0.021 | 0.710 ± 0.070 |
| 2 | shift 0% | — | — | 0.791 ± 0.008 | 0.389 ± 0.079 |

### Dose — MED-NODE

| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |
|---:|---|---|---|---|---|
| 0.5 | shift 100% | — | — | 0.628 ± 0.062 | 0.762 ± 0.067 |
| 0.5 | shift 50% | — | — | 0.637 ± 0.039 | 0.722 ± 0.082 |
| 0.5 | shift 0% | — | — | 0.783 ± 0.017 | 0.548 ± 0.086 |
| 2 | shift 100% | — | — | 0.634 ± 0.054 | 0.754 ± 0.063 |
| 2 | shift 50% | — | — | 0.647 ± 0.021 | 0.718 ± 0.091 |
| 2 | shift 0% | — | — | 0.791 ± 0.008 | 0.490 ± 0.019 |

### Dose — SD-198

| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |
|---:|---|---|---|---|---|
| 0.5 | shift 100% | — | — | 0.628 ± 0.062 | 0.808 ± 0.052 |
| 0.5 | shift 50% | — | — | 0.637 ± 0.039 | 0.769 ± 0.098 |
| 0.5 | shift 0% | — | — | 0.783 ± 0.017 | 0.519 ± 0.103 |
| 2 | shift 100% | — | — | 0.634 ± 0.054 | 0.851 ± 0.062 |
| 2 | shift 50% | — | — | 0.647 ± 0.021 | 0.752 ± 0.049 |
| 2 | shift 0% | — | — | 0.791 ± 0.008 | 0.423 ± 0.100 |

### Dose — SCIN

| λ | shift | AUROC [excl. degraded] | AUROC [all seeds] | leakage (pad probe) | NV-nearest (this set) |
|---:|---|---|---|---|---|
| 0.5 | shift 100% | — | — | 0.628 ± 0.062 | 0.769 ± 0.028 |
| 0.5 | shift 50% | — | — | 0.637 ± 0.039 | 0.743 ± 0.090 |
| 0.5 | shift 0% | — | — | 0.783 ± 0.017 | 0.659 ± 0.061 |
| 2 | shift 100% | — | — | 0.634 ± 0.054 | 0.786 ± 0.071 |
| 2 | shift 50% | — | — | 0.647 ± 0.021 | 0.714 ± 0.065 |
| 2 | shift 0% | — | — | 0.791 ± 0.008 | 0.595 ± 0.110 |

## Continuity reference: two-stream runs of the same recipe (separate table)

| cell | λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc (pad_heldout, 6-cls) |
|---|---:|---:|---|---|---|---|---|---|---|
| two-stream DANN (same recipe) | 0 | 5 | 0.927 ± 0.018 | 0.496 ± 0.016 | 0.181 ± 0.026 | 0.219 ± 0.046 | 0.808 ± 0.027 | 0.717 ± 0.034 | 0.275 ± 0.023 |
| two-stream DANN (same recipe) | 0.5 | 5 | 0.581 ± 0.031 | 0.501 ± 0.012 | 0.191 ± 0.008 | 0.710 ± 0.017 | 0.426 ± 0.008 | 0.411 ± 0.047 | 0.259 ± 0.025 |
| two-stream DANN (same recipe) | 2 | 5 | 0.555 ± 0.046 | 0.473 ± 0.023 | 0.181 ± 0.027 | 0.717 ± 0.022 | 0.403 ± 0.028 | 0.402 ± 0.018 | 0.254 ± 0.015 |

## Results in plain words

At λ = 0.5, Mahalanobis on pad_heldout by shift — shift 100%: — (all: —); shift 50%: — (all: —); shift 0%: — (all: —). Same shift ordering is tabulated for Fitzpatrick / MED-NODE / SD-198 / SCIN above. Prediction: inversion weakens as shift -> 0 and AUROC returns above 0.5 at 0% (work order). Numerically, Mahalanobis pad_heldout at λ = 2: shift 100% 0.38–0.45 (rule A met), 50% 0.45–0.60, 0% 0.55–0.75 (rule A not met); pad_heldout NV-nearest fraction falls with shift; Table 1 leakage at 0% within 0.05 of 100%.

At λ = 2, Mahalanobis on pad_heldout by shift — shift 100%: — (all: —); shift 50%: — (all: —); shift 0%: — (all: —). Same shift ordering is tabulated for Fitzpatrick / MED-NODE / SD-198 / SCIN above. Prediction: inversion weakens as shift -> 0 and AUROC returns above 0.5 at 0% (work order). Numerically, Mahalanobis pad_heldout at λ = 2: shift 100% 0.38–0.45 (rule A met), 50% 0.45–0.60, 0% 0.55–0.75 (rule A not met); pad_heldout NV-nearest fraction falls with shift; Table 1 leakage at 0% within 0.05 of 100%.

λ = 0 baseline (shift 100%): Mahalanobis pad_heldout —, ID bal acc 0.481 ± 0.016.

## LaTeX rows (cell & λ & n & leakage & ID bal acc & ID ECE & OOD ECE & OOD AUROC & Fitz AUROC & X-dom)

```latex
λ=0 baseline (shift 100%) & 0 & 5 & 0.928\,$\pm$\,0.020 & 0.481\,$\pm$\,0.016 & 0.207\,$\pm$\,0.003 & 0.305\,$\pm$\,0.042 & 0.825\,$\pm$\,0.022 & 0.712\,$\pm$\,0.013 & 0.277\,$\pm$\,0.024 \\
shift 100% & 0.5 & 5 & 0.628\,$\pm$\,0.062 & 0.483\,$\pm$\,0.013 & 0.214\,$\pm$\,0.007 & 0.775\,$\pm$\,0.010 & 0.361\,$\pm$\,0.029 & 0.385\,$\pm$\,0.023 & 0.222\,$\pm$\,0.016 \\
shift 50% & 0.5 & 5 & 0.637\,$\pm$\,0.039 & 0.484\,$\pm$\,0.008 & 0.206\,$\pm$\,0.005 & 0.641\,$\pm$\,0.029 & 0.486\,$\pm$\,0.047 & 0.455\,$\pm$\,0.115 & 0.296\,$\pm$\,0.028 \\
shift 0% & 0.5 & 5 & 0.783\,$\pm$\,0.017 & 0.471\,$\pm$\,0.012 & 0.212\,$\pm$\,0.003 & 0.437\,$\pm$\,0.021 & 0.498\,$\pm$\,0.025 & 0.672\,$\pm$\,0.046 & 0.379\,$\pm$\,0.022 \\
shift 100% & 2 & 5 & 0.634\,$\pm$\,0.054 & 0.486\,$\pm$\,0.012 & 0.205\,$\pm$\,0.007 & 0.757\,$\pm$\,0.018 & 0.368\,$\pm$\,0.017 & 0.340\,$\pm$\,0.065 & 0.231\,$\pm$\,0.029 \\
shift 50% & 2 & 5 & 0.647\,$\pm$\,0.021 & 0.487\,$\pm$\,0.009 & 0.208\,$\pm$\,0.003 & 0.643\,$\pm$\,0.017 & 0.469\,$\pm$\,0.027 & 0.457\,$\pm$\,0.062 & 0.308\,$\pm$\,0.011 \\
shift 0% & 2 & 5 & 0.782\,$\pm$\,0.019 & 0.414\,$\pm$\,0.154 & 0.188\,$\pm$\,0.057 & 0.361\,$\pm$\,0.144 & 0.514\,$\pm$\,0.123 & 0.719\,$\pm$\,0.069 & 0.366\,$\pm$\,0.103 \\
% two-stream reference
two-stream DANN (same recipe) & 0 & 5 & 0.927\,$\pm$\,0.018 & 0.496\,$\pm$\,0.016 & 0.181\,$\pm$\,0.026 & 0.219\,$\pm$\,0.046 & 0.808\,$\pm$\,0.027 & 0.717\,$\pm$\,0.034 & 0.275\,$\pm$\,0.023 \\
two-stream DANN (same recipe) & 0.5 & 5 & 0.581\,$\pm$\,0.031 & 0.501\,$\pm$\,0.012 & 0.191\,$\pm$\,0.008 & 0.710\,$\pm$\,0.017 & 0.426\,$\pm$\,0.008 & 0.411\,$\pm$\,0.047 & 0.259\,$\pm$\,0.025 \\
two-stream DANN (same recipe) & 2 & 5 & 0.555\,$\pm$\,0.046 & 0.473\,$\pm$\,0.023 & 0.181\,$\pm$\,0.027 & 0.717\,$\pm$\,0.022 & 0.403\,$\pm$\,0.028 & 0.402\,$\pm$\,0.018 & 0.254\,$\pm$\,0.015 \\
```

## Leakage-matched test (94e2399)

See `leakage_matched/REPORT.md` (commit dba39c7). Verdict (a): supported (≥2/3 primary sets: shift coef negative, CI excludes 0). Verdict (b): supported (≥2/3: R1 shift0 λ2 > A4 λ2, CI excludes 0). label shift lowers Mahalanobis AUROC at matched leakage on the unseen primary sets (prediction).
