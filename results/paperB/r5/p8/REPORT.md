# Paper B R5 P8 — stronger dual-encoder base model

Design: `decisions/paperB_r5_p8.md`. Commit `e94be83`. Code: `scripts/train_r5_p8.py`, `scripts/r6_common.py`, this report `scripts/r6_report.py --item p8`.

## Recipe search (λ = 0, lesion-level validation split only)

| recipe | latent | cls LR × | balance | epochs | seeds | val bal acc | val acc | val ECE | min class recall |
|---|---:|---:|---|---:|---|---|---|---|---|
| A0_e40 | 16 | 0.2 | none | 40 | 42/52 | 0.474 ± 0.006 | 0.698 ± 0.018 | 0.104 ± 0.079 | 0.219 ± 0.057 |
| A1_e40 | 64 | 1.0 | none | 40 | 42/52 | 0.468 ± 0.004 | 0.688 ± 0.000 | 0.210 ± 0.016 | 0.241 ± 0.051 |
| A2_e40 | 128 | 1.0 | none | 40 | 42/52 | 0.466 ± 0.008 | 0.680 ± 0.004 | 0.207 ± 0.011 | 0.228 ± 0.070 |
| A3_e40 | 64 | 5.0 | none | 40 | 42/52 | 0.477 ± 0.002 | 0.691 ± 0.006 | 0.174 ± 0.012 | 0.220 ± 0.016 |
| A4_e40 | 64 | 1.0 | loss | 40 | 42/52 | 0.543 ± 0.005 | 0.565 ± 0.009 | 0.026 ± 0.008 | 0.390 ± 0.008 |
| A5_e40 | 64 | 1.0 | sampling | 40 | 42/52 | 0.513 ± 0.007 | 0.541 ± 0.073 | 0.036 ± 0.001 | 0.378 ± 0.026 |
| A6_e40 | 16 | 1.0 | none | 40 | 42/52 | 0.481 ± 0.021 | 0.695 ± 0.003 | 0.174 ± 0.019 | 0.248 ± 0.046 |
| A7_e40 | 128 | 1.0 | loss | 40 | 42/52 | 0.542 ± 0.007 | 0.603 ± 0.007 | 0.019 ± 0.008 | 0.390 ± 0.019 |

Lesion-level single-encoder ERM reference (ID test bal acc, last.ckpt, seeds 42/52): 0.556 ± 0.016. Image-level single-encoder ERM in the paper: 0.779 (leakage-inflated on the image-level split; the 0.70 competence threshold was set from that figure; original ≥ 0.70 verdicts stay visible; this lesion-level mean ± SD is the competence reference). Full detector AUROCs: `results/paperB/r5/p8/erm_ref/REPORT.md`.

Stage A winner: A4 (within-tie set ['A4', 'A7']); rule: highest mean val bal acc (seeds 42, 52); among recipes within 0.01 of the top: fewest changes from the current recipe, then smaller latent, then grid order.
Selected recipe: {'latent': 64, 'cls_lr_mult': 1.0, 'balance': 'loss', 'max_epochs': 40, 'grid_name': 'A4_e40', 'grid_id': 'A4'} (40 epochs fixed by the user (2026-10-15); stage B not run).

Main-recipe rule (λ = 0, lesion-level val, last.ckpt, 5 seeds): D = +0.0304, 95% paired t CI [-0.0186, +0.0794]; outcome **old** (RECIPE=auto → current); P8 ext: skip.

### Flags (Table 1 leakage vs the λ = 0 cell; Mahalanobis pad_heldout headroom)

| cell | λ | leakage | Δ vs λ = 0 | flag |
|---|---:|---|---|---|
| A4 (last.ckpt) | 0 | 0.903 ± 0.012 | — | — |
| A4 (last.ckpt) | 0.5 | 0.780 ± 0.010 | -0.123 | — |
| A4 (last.ckpt) | 2 | 0.786 ± 0.034 | -0.117 | — |

## Table 1 columns — A4 last.ckpt (primary)

Leakage = logistic domain probe on z_lesion_norm, ISIC test vs all of PAD (Table 1 definition). Mean ± s.d. over seeds; n = seeds present.

| cell | λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc (pad_heldout, 6-cls) |
|---|---:|---:|---|---|---|---|---|---|---|
| A4 (last.ckpt) | 0 | 5 | 0.903 ± 0.012 | 0.520 ± 0.017 | 0.193 ± 0.008 | 0.454 ± 0.036 | 0.786 ± 0.019 | 0.821 ± 0.023 | 0.324 ± 0.022 |
| A4 (last.ckpt) | 0.5 | 5 | 0.780 ± 0.010 | 0.520 ± 0.023 | 0.203 ± 0.017 | 0.692 ± 0.013 | 0.468 ± 0.028 | 0.444 ± 0.049 | 0.276 ± 0.022 |
| A4 (last.ckpt) | 2 | 5 | 0.786 ± 0.034 | 0.510 ± 0.014 | 0.198 ± 0.010 | 0.693 ± 0.009 | 0.452 ± 0.014 | 0.458 ± 0.061 | 0.286 ± 0.012 |

### Detectors on pad_heldout (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 last.ckpt (primary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (last.ckpt) | 0 | 0.786 ± 0.019 [0.770, 0.802] | 0.891 ± 0.014 [0.877, 0.902] | 0.816 ± 0.020 [0.802, 0.828] | 0.767 ± 0.019 [0.754, 0.780] | 0.861 ± 0.017 [0.848, 0.873] | 0/5 | not met | 0.282 ± 0.029 |
| A4 (last.ckpt) | 0.5 | 0.468 ± 0.028 [0.440, 0.495] | 0.472 ± 0.027 [0.451, 0.494] | 0.470 ± 0.011 [0.452, 0.488] | 0.477 ± 0.009 [0.460, 0.493] | 0.473 ± 0.037 [0.456, 0.489] | 3/5 | met | 0.574 ± 0.041 |
| A4 (last.ckpt) | 2 | 0.452 ± 0.014 [0.425, 0.478] | 0.475 ± 0.028 [0.453, 0.496] | 0.464 ± 0.021 [0.446, 0.482] | 0.473 ± 0.031 [0.455, 0.489] | 0.465 ± 0.048 [0.449, 0.482] | 4/5 | met | 0.588 ± 0.052 |

### Detectors on fitzpatrick17k (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 last.ckpt (primary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (last.ckpt) | 0 | 0.821 ± 0.023 [0.811, 0.831] | 0.863 ± 0.036 [0.856, 0.871] | 0.785 ± 0.030 [0.777, 0.794] | 0.741 ± 0.028 [0.733, 0.749] | 0.822 ± 0.054 [0.815, 0.830] | 0/5 | not met | 0.562 ± 0.102 |
| A4 (last.ckpt) | 0.5 | 0.444 ± 0.049 [0.431, 0.457] | 0.506 ± 0.038 [0.493, 0.520] | 0.463 ± 0.024 [0.452, 0.474] | 0.501 ± 0.029 [0.490, 0.512] | 0.558 ± 0.026 [0.547, 0.569] | 4/5 | met | 0.805 ± 0.046 |
| A4 (last.ckpt) | 2 | 0.458 ± 0.061 [0.445, 0.471] | 0.521 ± 0.047 [0.508, 0.535] | 0.464 ± 0.053 [0.453, 0.474] | 0.478 ± 0.062 [0.467, 0.489] | 0.516 ± 0.098 [0.505, 0.526] | 3/5 | met | 0.811 ± 0.030 |

### Detectors on pad_full (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 last.ckpt (primary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (last.ckpt) | 0 | 0.796 ± 0.020 | 0.907 ± 0.011 | 0.831 ± 0.013 | 0.781 ± 0.013 | 0.876 ± 0.020 | — | — | 0.269 ± 0.033 |
| A4 (last.ckpt) | 0.5 | 0.460 ± 0.029 | 0.438 ± 0.028 | 0.437 ± 0.018 | 0.447 ± 0.016 | 0.443 ± 0.038 | — | — | 0.558 ± 0.036 |
| A4 (last.ckpt) | 2 | 0.446 ± 0.014 | 0.437 ± 0.023 | 0.421 ± 0.011 | 0.435 ± 0.026 | 0.425 ± 0.041 | — | — | 0.568 ± 0.050 |

### Detectors on mednode (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 last.ckpt (primary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (last.ckpt) | 0 | 0.560 ± 0.021 | 0.699 ± 0.038 | 0.651 ± 0.039 | 0.655 ± 0.022 | 0.746 ± 0.025 | — | — | 0.578 ± 0.061 |
| A4 (last.ckpt) | 0.5 | 0.292 ± 0.069 | 0.441 ± 0.040 | 0.445 ± 0.036 | 0.506 ± 0.023 | 0.552 ± 0.028 | — | — | 0.726 ± 0.038 |
| A4 (last.ckpt) | 2 | 0.298 ± 0.038 | 0.380 ± 0.056 | 0.391 ± 0.032 | 0.447 ± 0.035 | 0.451 ± 0.075 | — | — | 0.735 ± 0.045 |

### Detectors on sd198 (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 last.ckpt (primary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (last.ckpt) | 0 | 0.799 ± 0.026 | 0.850 ± 0.036 | 0.777 ± 0.032 | 0.733 ± 0.029 | 0.815 ± 0.048 | — | — | 0.604 ± 0.125 |
| A4 (last.ckpt) | 0.5 | 0.431 ± 0.047 | 0.480 ± 0.037 | 0.454 ± 0.025 | 0.489 ± 0.041 | 0.527 ± 0.038 | — | — | 0.813 ± 0.080 |
| A4 (last.ckpt) | 2 | 0.430 ± 0.061 | 0.499 ± 0.042 | 0.449 ± 0.054 | 0.465 ± 0.063 | 0.502 ± 0.098 | — | — | 0.836 ± 0.027 |

### Detectors on scin (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 last.ckpt (primary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (last.ckpt) | 0 | 0.822 ± 0.022 | 0.814 ± 0.027 | 0.742 ± 0.021 | 0.698 ± 0.034 | 0.778 ± 0.055 | — | — | 0.655 ± 0.084 |
| A4 (last.ckpt) | 0.5 | 0.649 ± 0.034 | 0.659 ± 0.037 | 0.604 ± 0.027 | 0.577 ± 0.034 | 0.597 ± 0.067 | — | — | 0.745 ± 0.043 |
| A4 (last.ckpt) | 2 | 0.659 ± 0.073 | 0.661 ± 0.045 | 0.599 ± 0.051 | 0.544 ± 0.046 | 0.546 ± 0.084 | — | — | 0.766 ± 0.033 |

### Per-class Mahalanobis AUROC and decomposition check on pad_heldout — A4 last.ckpt (primary)

ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs the OOD images of that class group.

| cell | λ | ID MEL | ID NV | ID BCC | ID AK | ID BKL | ID DF | ID VASC | ID SCC | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| A4 (last.ckpt) | 0 | 0.793 ± 0.027 | 0.882 ± 0.014 | 0.655 ± 0.018 | 0.526 ± 0.051 | 0.675 ± 0.035 | 0.458 ± 0.073 | 0.473 ± 0.058 | 0.527 ± 0.048 | 0.639 ± 0.028 | 0.639 ± 0.028 | 0.620 ± 0.027 | 0.809 ± 0.020 |
| A4 (last.ckpt) | 0.5 | 0.462 ± 0.029 | 0.570 ± 0.036 | 0.304 ± 0.018 | 0.249 ± 0.014 | 0.351 ± 0.017 | 0.248 ± 0.043 | 0.254 ± 0.052 | 0.250 ± 0.018 | 0.323 ± 0.016 | 0.323 ± 0.016 | 0.380 ± 0.018 | 0.480 ± 0.032 |
| A4 (last.ckpt) | 2 | 0.442 ± 0.018 | 0.556 ± 0.023 | 0.291 ± 0.014 | 0.216 ± 0.028 | 0.335 ± 0.022 | 0.231 ± 0.036 | 0.238 ± 0.031 | 0.228 ± 0.026 | 0.303 ± 0.016 | 0.303 ± 0.016 | 0.325 ± 0.039 | 0.470 ± 0.014 |

### Per-class Mahalanobis AUROC and decomposition check on pad_full — A4 last.ckpt (primary)

ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs the OOD images of that class group.

| cell | λ | ID MEL | ID NV | ID BCC | ID AK | ID BKL | ID DF | ID VASC | ID SCC | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| A4 (last.ckpt) | 0 | 0.804 ± 0.027 | 0.889 ± 0.014 | 0.671 ± 0.019 | 0.543 ± 0.055 | 0.690 ± 0.037 | 0.472 ± 0.076 | 0.488 ± 0.060 | 0.544 ± 0.051 | 0.648 ± 0.032 | 0.648 ± 0.032 | 0.649 ± 0.030 | 0.814 ± 0.021 |
| A4 (last.ckpt) | 0.5 | 0.454 ± 0.029 | 0.559 ± 0.038 | 0.301 ± 0.017 | 0.248 ± 0.008 | 0.346 ± 0.020 | 0.248 ± 0.042 | 0.254 ± 0.050 | 0.250 ± 0.018 | 0.315 ± 0.014 | 0.315 ± 0.014 | 0.380 ± 0.023 | 0.470 ± 0.034 |
| A4 (last.ckpt) | 2 | 0.435 ± 0.017 | 0.548 ± 0.024 | 0.288 ± 0.014 | 0.216 ± 0.029 | 0.330 ± 0.020 | 0.231 ± 0.037 | 0.239 ± 0.030 | 0.228 ± 0.027 | 0.296 ± 0.016 | 0.296 ± 0.016 | 0.329 ± 0.031 | 0.460 ± 0.014 |

### Cross-domain behaviour on pad_heldout — A4 last.ckpt (primary)

| cell | λ | AK recall (6-cls) | NV recall | NV predicted share | ID NV-nearest | pad_heldout NV-nearest | leakage (class-matched ISIC subsample) |
|---|---:|---|---|---|---|---|---|
| A4 (last.ckpt) | 0 | 0.228 ± 0.046 | 0.534 ± 0.053 | 0.123 ± 0.017 | 0.554 ± 0.004 | 0.282 ± 0.029 | 0.871 ± 0.021 |
| A4 (last.ckpt) | 0.5 | 0.010 ± 0.012 | 0.820 ± 0.065 | 0.527 ± 0.039 | 0.574 ± 0.010 | 0.574 ± 0.041 | 0.769 ± 0.025 |
| A4 (last.ckpt) | 2 | 0.011 ± 0.009 | 0.855 ± 0.025 | 0.537 ± 0.049 | 0.568 ± 0.007 | 0.588 ± 0.052 | 0.785 ± 0.018 |

### Bootstrap units — A4 last.ckpt (primary)

ID: lesion. pad_heldout: patient. Fitzpatrick17k: P13 cluster_union.

A4 λ = 0.5, last.ckpt, pad_heldout: Mahalanobis 0.468 ± 0.028 [0.440, 0.495], 3/5 seed CIs below 0.5, rule A met; rule A met; units: ID lesion, OOD patient.
A4 λ = 0.5, last.ckpt, fitzpatrick17k: Mahalanobis 0.444 ± 0.049 [0.431, 0.457], 4/5 seed CIs below 0.5, rule A met; rule A met; units: ID lesion, OOD P13 cluster_union.
A4 λ = 2, last.ckpt, pad_heldout: Mahalanobis 0.452 ± 0.014 [0.425, 0.478], 4/5 seed CIs below 0.5, rule A met; rule A met; units: ID lesion, OOD patient.
A4 λ = 2, last.ckpt, fitzpatrick17k: Mahalanobis 0.458 ± 0.061 [0.445, 0.471], 3/5 seed CIs below 0.5, rule A met; rule A met; units: ID lesion, OOD P13 cluster_union.

### Flags (Table 1 leakage vs the λ = 0 cell; Mahalanobis pad_heldout headroom)

| cell | λ | leakage | Δ vs λ = 0 | flag |
|---|---:|---|---|---|
| A4 (best-val) | 0 | 0.908 ± 0.007 | — | — |
| A4 (best-val) | 0.5 | 0.812 ± 0.024 | -0.096 | — |
| A4 (best-val) | 2 | 0.800 ± 0.027 | -0.108 | — |

## Table 1 columns — A4 best-val (secondary)

Leakage = logistic domain probe on z_lesion_norm, ISIC test vs all of PAD (Table 1 definition). Mean ± s.d. over seeds; n = seeds present.

| cell | λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc (pad_heldout, 6-cls) |
|---|---:|---:|---|---|---|---|---|---|---|
| A4 (best-val) | 0 | 5 | 0.908 ± 0.007 | 0.554 ± 0.020 | 0.032 ± 0.014 | 0.161 ± 0.060 | 0.781 ± 0.025 | 0.892 ± 0.033 | 0.343 ± 0.027 |
| A4 (best-val) | 0.5 | 5 | 0.812 ± 0.024 | 0.569 ± 0.011 | 0.049 ± 0.019 | 0.455 ± 0.075 | 0.672 ± 0.069 | 0.787 ± 0.049 | 0.307 ± 0.027 |
| A4 (best-val) | 2 | 5 | 0.800 ± 0.027 | 0.555 ± 0.015 | 0.071 ± 0.039 | 0.508 ± 0.099 | 0.627 ± 0.144 | 0.750 ± 0.121 | 0.274 ± 0.034 |

### Detectors on pad_heldout (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 best-val (secondary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (best-val) | 0 | 0.781 ± 0.025 [0.764, 0.796] | 0.922 ± 0.024 [0.909, 0.934] | 0.887 ± 0.021 [0.872, 0.900] | 0.725 ± 0.036 [0.709, 0.741] | 0.789 ± 0.021 [0.773, 0.805] | 0/5 | not met | 0.064 ± 0.018 |
| A4 (best-val) | 0.5 | 0.672 ± 0.069 [0.650, 0.694] | 0.668 ± 0.031 [0.648, 0.689] | 0.625 ± 0.025 [0.606, 0.644] | 0.560 ± 0.023 [0.542, 0.578] | 0.570 ± 0.033 [0.551, 0.588] | 0/5 | not met | 0.470 ± 0.052 |
| A4 (best-val) | 2 | 0.627 ± 0.144 [0.606, 0.647] | 0.633 ± 0.090 [0.612, 0.652] | 0.592 ± 0.095 [0.574, 0.609] | 0.509 ± 0.028 [0.492, 0.527] | 0.500 ± 0.044 [0.482, 0.519] | 1/5 | not met | 0.522 ± 0.083 |

### Detectors on fitzpatrick17k (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 best-val (secondary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (best-val) | 0 | 0.892 ± 0.033 [0.885, 0.899] | 0.963 ± 0.011 [0.958, 0.967] | 0.913 ± 0.030 [0.907, 0.919] | 0.744 ± 0.043 [0.734, 0.754] | 0.803 ± 0.050 [0.793, 0.813] | 0/5 | not met | 0.115 ± 0.056 |
| A4 (best-val) | 0.5 | 0.787 ± 0.049 [0.776, 0.798] | 0.735 ± 0.042 [0.722, 0.748] | 0.639 ± 0.064 [0.627, 0.652] | 0.578 ± 0.036 [0.566, 0.590] | 0.537 ± 0.048 [0.524, 0.551] | 0/5 | not met | 0.635 ± 0.075 |
| A4 (best-val) | 2 | 0.750 ± 0.121 [0.740, 0.761] | 0.733 ± 0.093 [0.722, 0.744] | 0.634 ± 0.121 [0.623, 0.645] | 0.514 ± 0.065 [0.502, 0.526] | 0.490 ± 0.081 [0.478, 0.503] | 0/5 | not met | 0.653 ± 0.109 |

### Detectors on pad_full (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 best-val (secondary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (best-val) | 0 | 0.787 ± 0.023 | 0.924 ± 0.027 | 0.888 ± 0.022 | 0.721 ± 0.039 | 0.782 ± 0.025 | — | — | 0.066 ± 0.016 |
| A4 (best-val) | 0.5 | 0.669 ± 0.070 | 0.661 ± 0.032 | 0.614 ± 0.026 | 0.550 ± 0.026 | 0.562 ± 0.043 | — | — | 0.481 ± 0.055 |
| A4 (best-val) | 2 | 0.626 ± 0.143 | 0.625 ± 0.095 | 0.578 ± 0.102 | 0.502 ± 0.028 | 0.489 ± 0.041 | — | — | 0.528 ± 0.085 |

### Detectors on mednode (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 best-val (secondary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (best-val) | 0 | 0.733 ± 0.064 | 0.825 ± 0.023 | 0.798 ± 0.018 | 0.689 ± 0.014 | 0.721 ± 0.039 | — | — | 0.295 ± 0.088 |
| A4 (best-val) | 0.5 | 0.482 ± 0.074 | 0.547 ± 0.031 | 0.482 ± 0.024 | 0.488 ± 0.049 | 0.503 ± 0.109 | — | — | 0.724 ± 0.032 |
| A4 (best-val) | 2 | 0.494 ± 0.158 | 0.554 ± 0.090 | 0.507 ± 0.104 | 0.413 ± 0.030 | 0.428 ± 0.059 | — | — | 0.731 ± 0.069 |

### Detectors on sd198 (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 best-val (secondary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (best-val) | 0 | 0.880 ± 0.028 | 0.945 ± 0.014 | 0.894 ± 0.026 | 0.737 ± 0.045 | 0.798 ± 0.037 | — | — | 0.119 ± 0.050 |
| A4 (best-val) | 0.5 | 0.765 ± 0.044 | 0.732 ± 0.044 | 0.646 ± 0.061 | 0.598 ± 0.033 | 0.556 ± 0.059 | — | — | 0.646 ± 0.071 |
| A4 (best-val) | 2 | 0.727 ± 0.132 | 0.743 ± 0.093 | 0.646 ± 0.130 | 0.523 ± 0.053 | 0.513 ± 0.067 | — | — | 0.664 ± 0.142 |

### Detectors on scin (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap) — A4 best-val (secondary)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| A4 (best-val) | 0 | 0.872 ± 0.051 | 0.956 ± 0.013 | 0.913 ± 0.023 | 0.815 ± 0.043 | 0.845 ± 0.059 | — | — | 0.190 ± 0.051 |
| A4 (best-val) | 0.5 | 0.847 ± 0.034 | 0.830 ± 0.025 | 0.733 ± 0.034 | 0.667 ± 0.031 | 0.628 ± 0.045 | — | — | 0.514 ± 0.066 |
| A4 (best-val) | 2 | 0.813 ± 0.058 | 0.838 ± 0.077 | 0.735 ± 0.115 | 0.612 ± 0.090 | 0.595 ± 0.086 | — | — | 0.581 ± 0.132 |

### Per-class Mahalanobis AUROC and decomposition check on pad_heldout — A4 best-val (secondary)

ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs the OOD images of that class group.

| cell | λ | ID MEL | ID NV | ID BCC | ID AK | ID BKL | ID DF | ID VASC | ID SCC | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| A4 (best-val) | 0 | 0.759 ± 0.040 | 0.845 ± 0.025 | 0.714 ± 0.055 | 0.710 ± 0.069 | 0.663 ± 0.044 | 0.633 ± 0.103 | 0.572 ± 0.087 | 0.686 ± 0.046 | 0.722 ± 0.039 | 0.722 ± 0.039 | 0.721 ± 0.021 | 0.789 ± 0.027 |
| A4 (best-val) | 0.5 | 0.668 ± 0.064 | 0.779 ± 0.058 | 0.507 ± 0.102 | 0.478 ± 0.149 | 0.535 ± 0.086 | 0.423 ± 0.080 | 0.331 ± 0.069 | 0.465 ± 0.093 | 0.533 ± 0.104 | 0.533 ± 0.104 | 0.604 ± 0.058 | 0.682 ± 0.073 |
| A4 (best-val) | 2 | 0.634 ± 0.142 | 0.727 ± 0.123 | 0.457 ± 0.191 | 0.437 ± 0.202 | 0.505 ± 0.172 | 0.386 ± 0.200 | 0.294 ± 0.171 | 0.428 ± 0.189 | 0.489 ± 0.182 | 0.489 ± 0.182 | 0.583 ± 0.130 | 0.633 ± 0.146 |

### Per-class Mahalanobis AUROC and decomposition check on pad_full — A4 best-val (secondary)

ID class c = ISIC test images of class c vs all OOD images; reweighted = ID side reweighted to the OOD class mix (PAD labels used for evaluation only); identity = Σ_c π_OOD(c)·AUROC_c (equals the reweighted AUROC by construction; difference reported in cells.json). OOD NV / non-NV = all ID vs the OOD images of that class group.

| cell | λ | ID MEL | ID NV | ID BCC | ID AK | ID BKL | ID DF | ID VASC | ID SCC | reweighted | identity Σπ·AUROC_c | OOD NV | OOD non-NV |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| A4 (best-val) | 0 | 0.766 ± 0.038 | 0.851 ± 0.026 | 0.723 ± 0.050 | 0.718 ± 0.069 | 0.671 ± 0.041 | 0.642 ± 0.103 | 0.580 ± 0.083 | 0.694 ± 0.044 | 0.728 ± 0.038 | 0.728 ± 0.038 | 0.737 ± 0.029 | 0.793 ± 0.024 |
| A4 (best-val) | 0.5 | 0.664 ± 0.064 | 0.776 ± 0.059 | 0.503 ± 0.102 | 0.474 ± 0.149 | 0.531 ± 0.086 | 0.420 ± 0.081 | 0.327 ± 0.070 | 0.461 ± 0.093 | 0.526 ± 0.106 | 0.526 ± 0.106 | 0.616 ± 0.060 | 0.675 ± 0.072 |
| A4 (best-val) | 2 | 0.633 ± 0.141 | 0.725 ± 0.124 | 0.458 ± 0.187 | 0.438 ± 0.198 | 0.506 ± 0.169 | 0.387 ± 0.195 | 0.294 ± 0.166 | 0.429 ± 0.184 | 0.487 ± 0.179 | 0.487 ± 0.179 | 0.598 ± 0.131 | 0.630 ± 0.145 |

### Cross-domain behaviour on pad_heldout — A4 best-val (secondary)

| cell | λ | AK recall (6-cls) | NV recall | NV predicted share | ID NV-nearest | pad_heldout NV-nearest | leakage (class-matched ISIC subsample) |
|---|---:|---|---|---|---|---|---|
| A4 (best-val) | 0 | 0.306 ± 0.138 | 0.455 ± 0.055 | 0.103 ± 0.020 | 0.430 ± 0.025 | 0.064 ± 0.018 | 0.874 ± 0.005 |
| A4 (best-val) | 0.5 | 0.021 ± 0.011 | 0.730 ± 0.052 | 0.458 ± 0.039 | 0.459 ± 0.023 | 0.470 ± 0.052 | 0.797 ± 0.020 |
| A4 (best-val) | 2 | 0.008 ± 0.010 | 0.768 ± 0.073 | 0.510 ± 0.044 | 0.469 ± 0.034 | 0.522 ± 0.083 | 0.798 ± 0.023 |

### Bootstrap units — A4 best-val (secondary)

ID: lesion. pad_heldout: patient. Fitzpatrick17k: P13 cluster_union.

A4 λ = 0.5, best-val, pad_heldout: Mahalanobis 0.672 ± 0.069 [0.650, 0.694], 0/5 seed CIs below 0.5, rule A not met; rule A not met; units: ID lesion, OOD patient.
A4 λ = 0.5, best-val, fitzpatrick17k: Mahalanobis 0.787 ± 0.049 [0.776, 0.798], 0/5 seed CIs below 0.5, rule A not met; rule A not met; units: ID lesion, OOD P13 cluster_union.
A4 λ = 2, best-val, pad_heldout: Mahalanobis 0.627 ± 0.144 [0.606, 0.647], 1/5 seed CIs below 0.5, rule A not met; rule A not met; units: ID lesion, OOD patient.
A4 λ = 2, best-val, fitzpatrick17k: Mahalanobis 0.750 ± 0.121 [0.740, 0.761], 0/5 seed CIs below 0.5, rule A not met; rule A not met; units: ID lesion, OOD P13 cluster_union.

## Reference: current recipe, lesion-level (separate table; recipes never mixed)

| cell | λ | n | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc (pad_heldout, 6-cls) |
|---|---:|---:|---|---|---|---|---|---|---|
| current recipe (R4 Table 1) | 0 | 5 | 0.927 ± 0.018 | 0.496 ± 0.016 | 0.181 ± 0.026 | 0.219 ± 0.046 | 0.808 ± 0.027 | 0.717 ± 0.034 | 0.275 ± 0.023 |
| current recipe (R4 Table 1) | 0.5 | 5 | 0.581 ± 0.031 | 0.501 ± 0.012 | 0.191 ± 0.008 | 0.710 ± 0.017 | 0.426 ± 0.008 | 0.411 ± 0.047 | 0.259 ± 0.025 |
| current recipe (R4 Table 1) | 2 | 5 | 0.555 ± 0.046 | 0.473 ± 0.023 | 0.181 ± 0.027 | 0.717 ± 0.022 | 0.403 ± 0.028 | 0.402 ± 0.018 | 0.254 ± 0.015 |

## Results in plain words

Competence. Under A4 (last.ckpt) the λ = 0 model reaches ID test balanced accuracy 0.520 ± 0.017 (n = 5), against the target 0.70 (set from the image-level single-encoder ERM 0.779, which is leakage-inflated on the image-level split; the lesion-level single-encoder ERM reference, last.ckpt, is 0.556 ± 0.016). Competence flag (≥ 0.70): no.

Main-recipe rule. D = +0.0304 [-0.0186, +0.0794]; outcome old; A4 reported as a robustness check at 3 λ; P8 ext skip.

Inversion under A4 last.ckpt. Mahalanobis on pad_heldout at λ = 0: 0.786 ± 0.019 [0.770, 0.802], 0/5 seed CIs below 0.5, rule A not met; at λ = 2: 0.452 ± 0.014 [0.425, 0.478], 4/5 seed CIs below 0.5, rule A met. Fitzpatrick at λ = 2: 0.458 ± 0.061 [0.445, 0.471], 3/5 seed CIs below 0.5, rule A met. Prediction: selected recipe raises λ = 0 val bal acc to 0.60–0.70 (current 0.47 on val); ID test bal acc at λ = 0 0.60–0.72; Mahalanobis pad_heldout at λ = 0 0.75–0.85; at λ = 2 0.38–0.48 with rule A met (inversion survives a stronger classifier); Fitzpatrick at λ = 2 below 0.5.

## LaTeX rows (cell & λ & n & leakage & ID bal acc & ID ECE & OOD ECE & OOD AUROC & Fitz AUROC & X-dom)

```latex
A4 (last.ckpt) & 0 & 5 & 0.903\,$\pm$\,0.012 & 0.520\,$\pm$\,0.017 & 0.193\,$\pm$\,0.008 & 0.454\,$\pm$\,0.036 & 0.786\,$\pm$\,0.019 & 0.821\,$\pm$\,0.023 & 0.324\,$\pm$\,0.022 \\
A4 (last.ckpt) & 0.5 & 5 & 0.780\,$\pm$\,0.010 & 0.520\,$\pm$\,0.023 & 0.203\,$\pm$\,0.017 & 0.692\,$\pm$\,0.013 & 0.468\,$\pm$\,0.028 & 0.444\,$\pm$\,0.049 & 0.276\,$\pm$\,0.022 \\
A4 (last.ckpt) & 2 & 5 & 0.786\,$\pm$\,0.034 & 0.510\,$\pm$\,0.014 & 0.198\,$\pm$\,0.010 & 0.693\,$\pm$\,0.009 & 0.452\,$\pm$\,0.014 & 0.458\,$\pm$\,0.061 & 0.286\,$\pm$\,0.012 \\
A4 (best-val) & 0 & 5 & 0.908\,$\pm$\,0.007 & 0.554\,$\pm$\,0.020 & 0.032\,$\pm$\,0.014 & 0.161\,$\pm$\,0.060 & 0.781\,$\pm$\,0.025 & 0.892\,$\pm$\,0.033 & 0.343\,$\pm$\,0.027 \\
A4 (best-val) & 0.5 & 5 & 0.812\,$\pm$\,0.024 & 0.569\,$\pm$\,0.011 & 0.049\,$\pm$\,0.019 & 0.455\,$\pm$\,0.075 & 0.672\,$\pm$\,0.069 & 0.787\,$\pm$\,0.049 & 0.307\,$\pm$\,0.027 \\
A4 (best-val) & 2 & 5 & 0.800\,$\pm$\,0.027 & 0.555\,$\pm$\,0.015 & 0.071\,$\pm$\,0.039 & 0.508\,$\pm$\,0.099 & 0.627\,$\pm$\,0.144 & 0.750\,$\pm$\,0.121 & 0.274\,$\pm$\,0.034 \\
% current recipe reference
current recipe (R4 Table 1) & 0 & 5 & 0.927\,$\pm$\,0.018 & 0.496\,$\pm$\,0.016 & 0.181\,$\pm$\,0.026 & 0.219\,$\pm$\,0.046 & 0.808\,$\pm$\,0.027 & 0.717\,$\pm$\,0.034 & 0.275\,$\pm$\,0.023 \\
current recipe (R4 Table 1) & 0.5 & 5 & 0.581\,$\pm$\,0.031 & 0.501\,$\pm$\,0.012 & 0.191\,$\pm$\,0.008 & 0.710\,$\pm$\,0.017 & 0.426\,$\pm$\,0.008 & 0.411\,$\pm$\,0.047 & 0.259\,$\pm$\,0.025 \\
current recipe (R4 Table 1) & 2 & 5 & 0.555\,$\pm$\,0.046 & 0.473\,$\pm$\,0.023 & 0.181\,$\pm$\,0.027 & 0.717\,$\pm$\,0.022 & 0.403\,$\pm$\,0.028 & 0.402\,$\pm$\,0.018 & 0.254\,$\pm$\,0.015 \\
```

