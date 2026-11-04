# Paper B R6 R2 — SCIN (secondary, separate report)

Design: `decisions/paperB_r6_r2_scin.md`. Commit `dba39c7`.

## Scope

- **Not** in the R2 estimated-vs-oracle fix reading (primary sets remain Fitzpatrick17k / MED-NODE / SD-198).
- Only **0.615%** of SCIN images map to ISIC classes (64/10406 images; 5033 cases). Mapped counts: AK=11, BCC=10, DF=7, NV=7, SCC=16, VASC=13.
- Detectors: same five as R1. Cluster CIs: PREREGISTER 2,000 draws, rng 20261005; **ID = lesion**, **OOD = case** (P7a `group`, unit=`P7a group column`).
- Baselines: R1 shift 100% RECIPE=`current` `last.ckpt`. R2 methods: CDAN / IWDAN / IWCDAN / IWDAN_ORACLE as present on disk; re-run this script when held λ = 0.5 / remaining CDAN seeds finish.
- No new dermatology datasets.

### Detectors on scin (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.723 ± 0.018 | 0.770 ± 0.023 | 0.735 ± 0.023 | 0.767 ± 0.050 | 0.850 ± 0.048 | — | — | 0.585 ± 0.096 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.534 ± 0.015 | 0.560 ± 0.026 | 0.560 ± 0.022 | 0.530 ± 0.031 | 0.579 ± 0.055 | — | — | 0.769 ± 0.028 |
| CDAN | 0.5 | 0.540 ± 0.028 | 0.553 ± 0.037 | 0.546 ± 0.037 | 0.523 ± 0.062 | 0.575 ± 0.057 | — | — | 0.759 ± 0.051 |
| IWDAN | 0.5 | 0.671 ± 0.030 | 0.659 ± 0.013 | 0.631 ± 0.014 | 0.607 ± 0.007 | 0.614 ± 0.020 | — | — | 0.656 ± 0.072 |
| IWCDAN | 0.5 | 0.536 ± 0.035 | 0.531 ± 0.024 | 0.522 ± 0.016 | 0.511 ± 0.029 | 0.526 ± 0.020 | — | — | 0.786 ± 0.025 |
| IWDAN_ORACLE | 0.5 | 0.737 ± 0.016 | 0.690 ± 0.025 | 0.644 ± 0.032 | 0.634 ± 0.056 | 0.584 ± 0.084 | — | — | 0.355 ± 0.114 |
| DANN (R1 shift100 last.ckpt) | 2 | 0.519 ± 0.073 | 0.528 ± 0.058 | 0.522 ± 0.054 | 0.499 ± 0.051 | 0.550 ± 0.072 | — | — | 0.786 ± 0.071 |
| CDAN | 2 | 0.510 ± 0.056 | 0.526 ± 0.051 | 0.519 ± 0.039 | 0.499 ± 0.057 | 0.558 ± 0.059 | — | — | 0.772 ± 0.052 |
| IWDAN | 2 | 0.586 ± 0.071 | 0.567 ± 0.057 | 0.542 ± 0.050 | 0.542 ± 0.057 | 0.571 ± 0.085 | — | — | 0.700 ± 0.074 |
| IWCDAN | 2 | 0.478 ± 0.026 | 0.492 ± 0.025 | 0.489 ± 0.021 | 0.485 ± 0.061 | 0.538 ± 0.046 | — | — | 0.810 ± 0.040 |
| IWDAN_ORACLE | 2 | 0.727 ± 0.039 | 0.676 ± 0.045 | 0.641 ± 0.047 | 0.652 ± 0.039 | 0.627 ± 0.094 | — | — | 0.311 ± 0.126 |

### Mahalanobis per seed (point + case-level CI)

| cell | λ | seed | point | ci_lo | ci_hi | CI entirely < 0.5 |
|---|---:|---:|---:|---:|---:|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 42 | 0.728 | — | — | — |
| λ=0 (R1 shift100 last.ckpt) | 0 | 52 | 0.702 | — | — | — |
| λ=0 (R1 shift100 last.ckpt) | 0 | 62 | 0.725 | — | — | — |
| λ=0 (R1 shift100 last.ckpt) | 0 | 72 | 0.749 | — | — | — |
| λ=0 (R1 shift100 last.ckpt) | 0 | 82 | 0.710 | — | — | — |
| DANN (R1 shift100 last.ckpt) | 0.5 | 42 | 0.539 | — | — | — |
| DANN (R1 shift100 last.ckpt) | 0.5 | 52 | 0.548 | — | — | — |
| DANN (R1 shift100 last.ckpt) | 0.5 | 62 | 0.545 | — | — | — |
| DANN (R1 shift100 last.ckpt) | 0.5 | 72 | 0.513 | — | — | — |
| DANN (R1 shift100 last.ckpt) | 0.5 | 82 | 0.526 | — | — | — |
| CDAN | 0.5 | 42 | 0.521 | — | — | — |
| CDAN | 0.5 | 52 | 0.504 | — | — | — |
| CDAN | 0.5 | 62 | 0.547 | — | — | — |
| CDAN | 0.5 | 72 | 0.576 | — | — | — |
| CDAN | 0.5 | 82 | 0.549 | — | — | — |
| IWDAN | 0.5 | 42 | 0.697 | — | — | — |
| IWDAN | 0.5 | 52 | 0.627 | — | — | — |
| IWDAN | 0.5 | 62 | 0.663 | — | — | — |
| IWDAN | 0.5 | 72 | 0.700 | — | — | — |
| IWDAN | 0.5 | 82 | 0.669 | — | — | — |
| IWCDAN | 0.5 | 42 | 0.555 | — | — | — |
| IWCDAN | 0.5 | 52 | 0.489 | — | — | — |
| IWCDAN | 0.5 | 62 | 0.527 | — | — | — |
| IWCDAN | 0.5 | 72 | 0.528 | — | — | — |
| IWCDAN | 0.5 | 82 | 0.581 | — | — | — |
| IWDAN_ORACLE | 0.5 | 42 | 0.713 | — | — | — |
| IWDAN_ORACLE | 0.5 | 52 | 0.729 | — | — | — |
| IWDAN_ORACLE | 0.5 | 62 | 0.744 | — | — | — |
| IWDAN_ORACLE | 0.5 | 72 | 0.750 | — | — | — |
| IWDAN_ORACLE | 0.5 | 82 | 0.751 | — | — | — |
| DANN (R1 shift100 last.ckpt) | 2 | 42 | 0.546 | — | — | — |
| DANN (R1 shift100 last.ckpt) | 2 | 52 | 0.542 | — | — | — |
| DANN (R1 shift100 last.ckpt) | 2 | 62 | 0.597 | — | — | — |
| DANN (R1 shift100 last.ckpt) | 2 | 72 | 0.513 | — | — | — |
| DANN (R1 shift100 last.ckpt) | 2 | 82 | 0.400 | — | — | — |
| CDAN | 2 | 42 | 0.425 | — | — | — |
| CDAN | 2 | 52 | 0.506 | — | — | — |
| CDAN | 2 | 62 | 0.498 | — | — | — |
| CDAN | 2 | 72 | 0.568 | — | — | — |
| CDAN | 2 | 82 | 0.551 | — | — | — |
| IWDAN | 2 | 42 | 0.521 | — | — | — |
| IWDAN | 2 | 52 | 0.533 | — | — | — |
| IWDAN | 2 | 62 | 0.564 | — | — | — |
| IWDAN | 2 | 72 | 0.691 | — | — | — |
| IWDAN | 2 | 82 | 0.622 | — | — | — |
| IWCDAN | 2 | 42 | 0.495 | — | — | — |
| IWCDAN | 2 | 52 | 0.464 | — | — | — |
| IWCDAN | 2 | 62 | 0.442 | — | — | — |
| IWCDAN | 2 | 72 | 0.479 | — | — | — |
| IWCDAN | 2 | 82 | 0.509 | — | — | — |
| IWDAN_ORACLE | 2 | 42 | 0.676 | — | — | — |
| IWDAN_ORACLE | 2 | 52 | 0.765 | — | — | — |
| IWDAN_ORACLE | 2 | 62 | 0.696 | — | — | — |
| IWDAN_ORACLE | 2 | 72 | 0.758 | — | — | — |
| IWDAN_ORACLE | 2 | 82 | 0.739 | — | — | — |

## Paired-by-seed Δ vs DANN and gap recovered (SCIN only; descriptive)

Same formulas as main R2 report; **does not** enter the fix reading.

| method | λ | Δ Maha vs DANN [case-level cluster CI] | gap recovered (method−DANN)/(λ0−DANN) [seed boot CI] |
|---|---:|---|---|
| CDAN | 0.5 | — | — |
| IWDAN | 0.5 | — | — |
| IWCDAN | 0.5 | — | — |
| IWDAN_ORACLE | 0.5 | — | — |
| CDAN | 2 | — | — |
| IWDAN | 2 | — | — |
| IWCDAN | 2 | — | — |
| IWDAN_ORACLE | 2 | — | — |

Prediction (decisions): descriptive only — SCIN under DANN λ = 2 near chance (~0.52); IWDAN may sit slightly higher; not a gate.

