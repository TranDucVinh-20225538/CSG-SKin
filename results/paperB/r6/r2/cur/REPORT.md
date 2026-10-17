# Paper B R6 R2 — label-shift-aware alignment (recipe: cur)

**On origin/paperb/reviewer-r1:** `5723254` (baseline correction; full cluster-CI Δ/gap tables when R2 report job runs without `--no_boot`).
Design: `decisions/paperB_r6_r2.md`. Commit `9877217`.

## Baseline and pairing (fixed 2026-10-17)

- **DANN reference** and **λ = 0 reference** are **R1 shift 100%, RECIPE=`current`, primary=`last.ckpt`** (`results/paperB/r6/r1/cur/`), **not** Table 1 / item4 best-val.
- Headline DANN λ = 2 cell means (R1 shift100 last.ckpt): Maha pad_heldout **0.368**, Fitzpatrick **0.340**, leakage **0.634** (verify against the DANN row below).
- **Paired-by-seed deltas are valid:** R2 and R1 use the **same seeds** `[42, 52, 62, 72, 82]` and the **same data split** (lesion-level ISIC train/val/test from `train_r2_item4_lesion_split` / Phase 3; Phase 2 PAD patient split pad_adv / pad_heldout). R2 `default_mode(current)` is the two-stream Table 1 path; R1 shift 100% is three-stream with `adv_shift=1` (ISIC mix), so the **classifier stream and split match**; the DANN reference is the R1 shift-100% last.ckpt cell at the same λ/seed.
- Pre-committed fix reading (estimated vs oracle on Fitz/MED/SD) is **unchanged**; interpret only after oracle, CDAN, and λ = 0.5 cells are complete.

### Flags (Table 1 leakage vs the λ = 0 cell; Mahalanobis pad_heldout headroom)

| cell | λ | leakage | Δ vs λ = 0 | flag |
|---|---:|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.928 ± 0.020 | — | — |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.628 ± 0.062 | -0.299 | — |
| DANN (R1 shift100 last.ckpt) | 2 | 0.634 ± 0.054 | -0.293 | — |
| IWDAN | 2 | 0.642 ± 0.042 | -0.285 | — |
| IWCDAN | 2 | 0.661 ± 0.069 | -0.267 | — |

## Table 1 columns

Leakage = logistic domain probe on z_lesion_norm, ISIC test vs all of PAD (Table 1 definition). Mean ± s.d. over seeds; n = seeds present.

| cell | λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc (pad_heldout, 6-cls) |
|---|---:|---:|---|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 5 | 0.928 ± 0.020 | 0.481 ± 0.016 | 0.207 ± 0.003 | 0.305 ± 0.042 | 0.825 ± 0.022 | 0.712 ± 0.013 | 0.277 ± 0.024 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 5 | 0.628 ± 0.062 | 0.483 ± 0.013 | 0.214 ± 0.007 | 0.775 ± 0.010 | 0.361 ± 0.029 | 0.385 ± 0.023 | 0.222 ± 0.016 |
| DANN (R1 shift100 last.ckpt) | 2 | 5 | 0.634 ± 0.054 | 0.486 ± 0.012 | 0.205 ± 0.007 | 0.757 ± 0.018 | 0.368 ± 0.017 | 0.340 ± 0.065 | 0.231 ± 0.029 |
| IWDAN | 2 | 5 | 0.642 ± 0.042 | 0.484 ± 0.013 | 0.193 ± 0.012 | 0.670 ± 0.018 | 0.513 ± 0.014 | 0.508 ± 0.078 | 0.305 ± 0.013 |
| IWCDAN | 2 | 3 | 0.661 ± 0.069 | 0.443 ± 0.014 | 0.205 ± 0.009 | 0.771 ± 0.011 | 0.371 ± 0.037 | 0.376 ± 0.048 | 0.249 ± 0.019 |

### Detectors on pad_heldout (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.825 ± 0.022 | 0.892 ± 0.013 | 0.844 ± 0.015 | 0.928 ± 0.008 | 0.959 ± 0.008 | — | — | 0.365 ± 0.070 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.361 ± 0.029 | 0.376 ± 0.026 | 0.381 ± 0.014 | 0.407 ± 0.035 | 0.444 ± 0.048 | — | — | 0.653 ± 0.041 |
| DANN (R1 shift100 last.ckpt) | 2 | 0.368 ± 0.017 | 0.371 ± 0.016 | 0.368 ± 0.016 | 0.392 ± 0.028 | 0.404 ± 0.045 | — | — | 0.628 ± 0.039 |
| IWDAN | 2 | 0.513 ± 0.014 | 0.492 ± 0.014 | 0.451 ± 0.021 | 0.423 ± 0.020 | 0.404 ± 0.048 | — | — | 0.366 ± 0.032 |
| IWCDAN | 2 | 0.371 ± 0.037 | 0.353 ± 0.028 | 0.373 ± 0.032 | 0.374 ± 0.005 | 0.368 ± 0.006 | — | — | 0.560 ± 0.021 |

### Detectors on fitzpatrick17k (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.712 ± 0.013 | 0.770 ± 0.015 | 0.731 ± 0.015 | 0.795 ± 0.034 | 0.875 ± 0.041 | — | — | 0.601 ± 0.085 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.385 ± 0.023 | 0.418 ± 0.030 | 0.423 ± 0.021 | 0.450 ± 0.046 | 0.524 ± 0.062 | — | — | 0.777 ± 0.054 |
| DANN (R1 shift100 last.ckpt) | 2 | 0.340 ± 0.065 | 0.360 ± 0.054 | 0.364 ± 0.048 | 0.396 ± 0.056 | 0.462 ± 0.079 | — | — | 0.821 ± 0.055 |
| IWDAN | 2 | 0.508 ± 0.078 | 0.489 ± 0.062 | 0.472 ± 0.050 | 0.475 ± 0.043 | 0.499 ± 0.048 | — | — | 0.599 ± 0.080 |
| IWCDAN | 2 | 0.376 ± 0.048 | 0.399 ± 0.031 | 0.418 ± 0.038 | 0.429 ± 0.025 | 0.487 ± 0.023 | — | — | 0.756 ± 0.049 |

### Detectors on mednode (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.559 ± 0.019 | 0.603 ± 0.010 | 0.602 ± 0.005 | 0.674 ± 0.016 | 0.732 ± 0.015 | — | — | 0.548 ± 0.069 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.310 ± 0.055 | 0.355 ± 0.056 | 0.389 ± 0.048 | 0.430 ± 0.050 | 0.490 ± 0.043 | — | — | 0.762 ± 0.067 |
| DANN (R1 shift100 last.ckpt) | 2 | 0.298 ± 0.032 | 0.327 ± 0.035 | 0.362 ± 0.035 | 0.388 ± 0.058 | 0.433 ± 0.060 | — | — | 0.754 ± 0.063 |
| IWDAN | 2 | 0.350 ± 0.046 | 0.380 ± 0.042 | 0.424 ± 0.040 | 0.434 ± 0.022 | 0.447 ± 0.035 | — | — | 0.644 ± 0.053 |
| IWCDAN | 2 | 0.287 ± 0.013 | 0.312 ± 0.022 | 0.377 ± 0.052 | 0.404 ± 0.074 | 0.414 ± 0.073 | — | — | 0.698 ± 0.078 |

### Detectors on sd198 (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.708 ± 0.019 | 0.771 ± 0.007 | 0.733 ± 0.007 | 0.806 ± 0.026 | 0.885 ± 0.035 | — | — | 0.605 ± 0.083 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.365 ± 0.029 | 0.399 ± 0.037 | 0.412 ± 0.032 | 0.441 ± 0.056 | 0.519 ± 0.077 | — | — | 0.808 ± 0.052 |
| DANN (R1 shift100 last.ckpt) | 2 | 0.325 ± 0.069 | 0.348 ± 0.052 | 0.355 ± 0.052 | 0.381 ± 0.052 | 0.448 ± 0.072 | — | — | 0.851 ± 0.062 |
| IWDAN | 2 | 0.488 ± 0.075 | 0.473 ± 0.058 | 0.466 ± 0.042 | 0.476 ± 0.049 | 0.495 ± 0.069 | — | — | 0.631 ± 0.064 |
| IWCDAN | 2 | 0.369 ± 0.057 | 0.394 ± 0.038 | 0.419 ± 0.042 | 0.431 ± 0.025 | 0.481 ± 0.029 | — | — | 0.770 ± 0.058 |

### Detectors on scin (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.723 ± 0.018 | 0.770 ± 0.023 | 0.735 ± 0.023 | 0.767 ± 0.050 | 0.850 ± 0.048 | — | — | 0.585 ± 0.096 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.534 ± 0.015 | 0.560 ± 0.026 | 0.560 ± 0.022 | 0.530 ± 0.031 | 0.579 ± 0.055 | — | — | 0.769 ± 0.028 |
| DANN (R1 shift100 last.ckpt) | 2 | 0.519 ± 0.073 | 0.528 ± 0.058 | 0.522 ± 0.054 | 0.499 ± 0.051 | 0.550 ± 0.072 | — | — | 0.786 ± 0.071 |
| IWDAN | 2 | 0.586 ± 0.071 | 0.567 ± 0.057 | 0.542 ± 0.050 | 0.542 ± 0.057 | 0.571 ± 0.085 | — | — | 0.700 ± 0.074 |
| IWCDAN | 2 | 0.467 ± 0.027 | 0.494 ± 0.025 | 0.492 ± 0.018 | 0.486 ± 0.026 | 0.542 ± 0.026 | — | — | 0.821 ± 0.022 |

### Per-class Mahalanobis AUROC and decomposition check on pad_heldout

ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs the OOD images of that class group.

| cell | λ | ID MEL | ID NV | ID BCC | ID AK | ID BKL | ID DF | ID VASC | ID SCC | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.844 ± 0.028 | 0.926 ± 0.012 | 0.695 ± 0.042 | 0.525 ± 0.049 | 0.727 ± 0.026 | 0.358 ± 0.041 | 0.455 ± 0.072 | 0.456 ± 0.066 | 0.659 ± 0.039 | 0.659 ± 0.039 | 0.688 ± 0.040 | 0.844 ± 0.020 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.312 ± 0.025 | 0.487 ± 0.038 | 0.197 ± 0.020 | 0.123 ± 0.025 | 0.214 ± 0.026 | 0.124 ± 0.012 | 0.137 ± 0.039 | 0.106 ± 0.017 | 0.207 ± 0.022 | 0.207 ± 0.022 | 0.260 ± 0.026 | 0.375 ± 0.032 |
| DANN (R1 shift100 last.ckpt) | 2 | 0.324 ± 0.021 | 0.487 ± 0.026 | 0.219 ± 0.017 | 0.139 ± 0.010 | 0.231 ± 0.012 | 0.140 ± 0.012 | 0.118 ± 0.026 | 0.118 ± 0.015 | 0.222 ± 0.007 | 0.222 ± 0.007 | 0.288 ± 0.043 | 0.379 ± 0.017 |
| IWDAN | 2 | 0.485 ± 0.013 | 0.643 ± 0.013 | 0.313 ± 0.027 | 0.243 ± 0.023 | 0.378 ± 0.019 | 0.290 ± 0.043 | 0.259 ± 0.009 | 0.240 ± 0.034 | 0.336 ± 0.019 | 0.336 ± 0.019 | 0.353 ± 0.046 | 0.536 ± 0.011 |
| IWCDAN | 2 | 0.303 ± 0.042 | 0.527 ± 0.044 | 0.159 ± 0.027 | 0.077 ± 0.017 | 0.196 ± 0.028 | 0.144 ± 0.022 | 0.124 ± 0.017 | 0.079 ± 0.011 | 0.180 ± 0.021 | 0.180 ± 0.021 | 0.263 ± 0.026 | 0.386 ± 0.039 |

### Per-class Mahalanobis AUROC and decomposition check on pad_full

ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs the OOD images of that class group.

| cell | λ | ID MEL | ID NV | ID BCC | ID AK | ID BKL | ID DF | ID VASC | ID SCC | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.867 ± 0.025 | 0.939 ± 0.010 | 0.724 ± 0.045 | 0.555 ± 0.054 | 0.754 ± 0.026 | 0.375 ± 0.042 | 0.476 ± 0.074 | 0.480 ± 0.071 | 0.679 ± 0.043 | 0.679 ± 0.043 | 0.759 ± 0.040 | 0.854 ± 0.020 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.291 ± 0.021 | 0.467 ± 0.038 | 0.182 ± 0.017 | 0.114 ± 0.021 | 0.197 ± 0.021 | 0.117 ± 0.015 | 0.127 ± 0.033 | 0.099 ± 0.014 | 0.188 ± 0.018 | 0.188 ± 0.018 | 0.244 ± 0.029 | 0.354 ± 0.029 |
| DANN (R1 shift100 last.ckpt) | 2 | 0.303 ± 0.015 | 0.464 ± 0.021 | 0.204 ± 0.021 | 0.131 ± 0.012 | 0.214 ± 0.014 | 0.133 ± 0.011 | 0.114 ± 0.025 | 0.113 ± 0.017 | 0.204 ± 0.012 | 0.204 ± 0.012 | 0.265 ± 0.050 | 0.358 ± 0.016 |
| IWDAN | 2 | 0.464 ± 0.018 | 0.627 ± 0.009 | 0.294 ± 0.018 | 0.227 ± 0.019 | 0.358 ± 0.024 | 0.275 ± 0.035 | 0.246 ± 0.010 | 0.226 ± 0.036 | 0.313 ± 0.015 | 0.313 ± 0.015 | 0.360 ± 0.031 | 0.512 ± 0.012 |
| IWCDAN | 2 | 0.278 ± 0.042 | 0.508 ± 0.044 | 0.139 ± 0.029 | 0.064 ± 0.014 | 0.174 ± 0.029 | 0.128 ± 0.021 | 0.109 ± 0.014 | 0.067 ± 0.012 | 0.155 ± 0.022 | 0.155 ± 0.022 | 0.236 ± 0.024 | 0.364 ± 0.039 |

### Cross-domain behaviour on pad_heldout

| cell | λ | AK recall (6-cls) | NV recall | NV predicted share | ID NV-nearest | pad_heldout NV-nearest | leakage (class-matched ISIC subsample) |
|---|---:|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.145 ± 0.056 | 0.523 ± 0.084 | 0.196 ± 0.038 | 0.543 ± 0.008 | 0.365 ± 0.070 | 0.908 ± 0.021 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.008 ± 0.003 | 0.895 ± 0.036 | 0.644 ± 0.041 | 0.557 ± 0.012 | 0.653 ± 0.041 | 0.722 ± 0.035 |
| DANN (R1 shift100 last.ckpt) | 2 | 0.010 ± 0.011 | 0.882 ± 0.027 | 0.617 ± 0.041 | 0.551 ± 0.009 | 0.628 ± 0.039 | 0.728 ± 0.021 |
| IWDAN | 2 | 0.035 ± 0.037 | 0.768 ± 0.071 | 0.347 ± 0.037 | 0.545 ± 0.008 | 0.366 ± 0.032 | 0.629 ± 0.052 |
| IWCDAN | 2 | 0.007 ± 0.008 | 0.864 ± 0.030 | 0.532 ± 0.029 | 0.569 ± 0.007 | 0.560 ± 0.021 | 0.716 ± 0.034 |

## Paired-by-seed Δ vs DANN (R1 shift100 last.ckpt) and gap recovered

Δ = method − DANN at the same λ and seed (Mahalanobis unless noted), with **cluster/patient bootstrap CI** of the seed-mean (shared draws). Gap recovered = (IW − DANN) / (λ0 − DANN) using R1 shift100 last.ckpt λ = 0 as the λ0 reference; **seed bootstrap** CI of the mean ratio. Denominator near zero → seed dropped.

| method | λ | set | Δ Maha vs DANN [cluster CI] | gap recovered (IW−DANN)/(λ0−DANN) [seed boot CI] |
|---|---:|---|---|---|
| IWDAN | 2 | pad_heldout | — | — |
| IWDAN | 2 | fitzpatrick17k | — | — |
| IWDAN | 2 | mednode | — | — |
| IWDAN | 2 | sd198 | — | — |
| IWCDAN | 2 | pad_heldout | — | — |
| IWCDAN | 2 | fitzpatrick17k | — | — |
| IWCDAN | 2 | mednode | — | — |
| IWCDAN | 2 | sd198 | — | — |

### Leakage paired Δ vs DANN (pad_full probe; seed bootstrap of seed-mean)

| method | λ | Δ leakage (method − DANN) |
|---|---:|---|
| IWDAN | 2 | +0.008 ± 0.086 [-0.058, 0.074] (n=5) |
| IWCDAN | 2 | +0.035 ± 0.132 [-0.073, 0.182] (n=3) |

## Estimated vs true PAD class proportions

IW = final-epoch training estimate (w ⊙ p_S, renormalised; target = pad_adv, no target labels). BBSE = post-hoc estimate from the final checkpoint (confusion on ISIC val, predictions on pad_heldout). True mixes from PAD labels, evaluation only. TV = total variation distance.

| cell | λ | IW est NV | IW est AK | IW TV vs pad_adv | BBSE est NV (pad_heldout) | BBSE est AK | BBSE TV vs pad_heldout |
|---|---:|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | — | — | — | 0.009 ± 0.009 | 0.169 ± 0.140 | 0.671 ± 0.128 |
| DANN (R1 shift100 last.ckpt) | 0.5 | — | — | — | 0.629 ± 0.083 | 0.000 ± 0.000 | 0.741 ± 0.042 |
| DANN (R1 shift100 last.ckpt) | 2 | — | — | — | 0.608 ± 0.069 | 0.004 ± 0.009 | 0.683 ± 0.035 |
| IWDAN | 2 | 0.251 ± 0.025 | 0.063 ± 0.058 | 0.515 ± 0.052 | 0.248 ± 0.077 | 0.012 ± 0.026 | 0.652 ± 0.039 |
| IWCDAN | 2 | 0.464 ± 0.004 | 0.030 ± 0.013 | 0.625 ± 0.015 | 0.453 ± 0.072 | 0.000 ± 0.000 | 0.767 ± 0.088 |

True mixes: pad_adv NV 0.099, AK 0.334; pad_heldout NV 0.123, AK 0.282.

## Results in plain words

DANN reference (R1 shift 100% last.ckpt) at λ = 2: leakage 0.634 ± 0.054, Mahalanobis pad_heldout —, Fitzpatrick — — baseline for all R2 comparisons (not Table 1 best-val).

λ = 0 reference (R1 shift 100% last.ckpt): leakage 0.928 ± 0.020, Mahalanobis pad_heldout —.

DANN (R1 shift100 last.ckpt) at λ = 0.5: leakage 0.628 ± 0.062, Mahalanobis pad_heldout —, Fitzpatrick —, X-dom bal acc 0.222 ± 0.016, AK recall 0.008 ± 0.003, NV predicted share 0.644 ± 0.041.

DANN (R1 shift100 last.ckpt) at λ = 2: leakage 0.634 ± 0.054, Mahalanobis pad_heldout —, Fitzpatrick —, X-dom bal acc 0.231 ± 0.029, AK recall 0.010 ± 0.011, NV predicted share 0.617 ± 0.041.

IWDAN at λ = 2: leakage 0.642 ± 0.042, Mahalanobis pad_heldout —, Fitzpatrick —, X-dom bal acc 0.305 ± 0.013, AK recall 0.035 ± 0.037, NV predicted share 0.347 ± 0.037.

IWCDAN at λ = 2: leakage 0.661 ± 0.069, Mahalanobis pad_heldout —, Fitzpatrick —, X-dom bal acc 0.249 ± 0.019, AK recall 0.007 ± 0.008, NV predicted share 0.532 ± 0.029.

Interpretation deferred until oracle, CDAN, and λ = 0.5 cells finish. Pre-committed reading unchanged. Prediction (decisions): CDAN: inversion persists at λ = 2 (Mahalanobis pad_heldout 0.38–0.50). IWDAN / IWCDAN: Mahalanobis pad_heldout 0.55–0.80 at λ = 2, leakage below the λ = 0 value by ≥ 0.10 but above DANN's, AK recall above DANN's at the same λ, estimated PAD NV share 0.08–0.25 (true pad_adv 0.099), AK share underestimated.

## LaTeX rows (cell & λ & n & leakage & ID bal acc & ID ECE & OOD ECE & OOD AUROC & Fitz AUROC & X-dom)

```latex
λ=0 (R1 shift100 last.ckpt) & 0 & 5 & 0.928\,$\pm$\,0.020 & 0.481\,$\pm$\,0.016 & 0.207\,$\pm$\,0.003 & 0.305\,$\pm$\,0.042 & 0.825\,$\pm$\,0.022 & 0.712\,$\pm$\,0.013 & 0.277\,$\pm$\,0.024 \\
DANN (R1 shift100 last.ckpt) & 0.5 & 5 & 0.628\,$\pm$\,0.062 & 0.483\,$\pm$\,0.013 & 0.214\,$\pm$\,0.007 & 0.775\,$\pm$\,0.010 & 0.361\,$\pm$\,0.029 & 0.385\,$\pm$\,0.023 & 0.222\,$\pm$\,0.016 \\
DANN (R1 shift100 last.ckpt) & 2 & 5 & 0.634\,$\pm$\,0.054 & 0.486\,$\pm$\,0.012 & 0.205\,$\pm$\,0.007 & 0.757\,$\pm$\,0.018 & 0.368\,$\pm$\,0.017 & 0.340\,$\pm$\,0.065 & 0.231\,$\pm$\,0.029 \\
IWDAN & 2 & 5 & 0.642\,$\pm$\,0.042 & 0.484\,$\pm$\,0.013 & 0.193\,$\pm$\,0.012 & 0.670\,$\pm$\,0.018 & 0.513\,$\pm$\,0.014 & 0.508\,$\pm$\,0.078 & 0.305\,$\pm$\,0.013 \\
IWCDAN & 2 & 3 & 0.661\,$\pm$\,0.069 & 0.443\,$\pm$\,0.014 & 0.205\,$\pm$\,0.009 & 0.771\,$\pm$\,0.011 & 0.371\,$\pm$\,0.037 & 0.376\,$\pm$\,0.048 & 0.249\,$\pm$\,0.019 \\
```

