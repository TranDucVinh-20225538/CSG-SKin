# Paper B R6 R2 — SCIN (secondary, separate report)

Design: `decisions/paperB_r6_r2_scin.md`. Commit `cc216b7`.

## Scope

- **Not** in the R2 estimated-vs-oracle fix reading (primary sets remain Fitzpatrick17k / MED-NODE / SD-198).
- Only **0.615%** of SCIN images map to ISIC classes (64/10406 images; 5033 cases). Mapped counts: AK=11, BCC=10, DF=7, NV=7, SCC=16, VASC=13.
- Detectors: same five as R1. Cluster CIs: PREREGISTER 2,000 draws, rng 20261005; **ID = lesion**, **OOD = case** (P7a `group`, unit=`P7a group column`).
- Baselines: R1 shift 100% RECIPE=`current` `last.ckpt`. R2 methods: CDAN / IWDAN / IWCDAN / IWDAN_ORACLE as present on disk; re-run this script when held λ = 0.5 / remaining CDAN seeds finish.
- No new dermatology datasets.

### Detectors on scin (mean ± s.d.; [95% CI of the seed mean], rule A bootstrap)

| cell | λ | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha k<0.5 | rule A | NV-nearest |
|---|---:|---|---|---|---|---|---|---|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 0.723 ± 0.018 [0.710, 0.736] | 0.770 ± 0.023 [0.760, 0.781] | 0.735 ± 0.023 [0.725, 0.744] | 0.767 ± 0.050 | 0.850 ± 0.048 | 0/5 | not met | 0.585 ± 0.096 |
| DANN (R1 shift100 last.ckpt) | 0.5 | 0.534 ± 0.015 [0.521, 0.547] | 0.560 ± 0.026 [0.547, 0.573] | 0.560 ± 0.022 [0.549, 0.571] | 0.530 ± 0.031 | 0.579 ± 0.055 | 0/5 | not met | 0.769 ± 0.028 |
| CDAN | 0.5 | 0.540 ± 0.028 [0.526, 0.553] | 0.553 ± 0.037 [0.540, 0.566] | 0.546 ± 0.037 [0.536, 0.557] | 0.523 ± 0.062 | 0.575 ± 0.057 | 0/5 | not met | 0.759 ± 0.051 |
| IWDAN | 0.5 | 0.671 ± 0.030 [0.658, 0.684] | 0.659 ± 0.013 [0.646, 0.673] | 0.631 ± 0.014 [0.620, 0.641] | 0.607 ± 0.007 | 0.614 ± 0.020 | 0/5 | not met | 0.656 ± 0.072 |
| IWCDAN | 0.5 | 0.536 ± 0.035 [0.523, 0.550] | 0.531 ± 0.024 [0.518, 0.544] | 0.522 ± 0.016 [0.512, 0.532] | 0.511 ± 0.029 | 0.526 ± 0.020 | 0/5 | not met | 0.786 ± 0.025 |
| IWDAN_ORACLE | 0.5 | 0.737 ± 0.016 [0.726, 0.749] | 0.690 ± 0.025 [0.677, 0.702] | 0.644 ± 0.032 [0.635, 0.654] | 0.634 ± 0.056 | 0.584 ± 0.084 | 0/5 | not met | 0.355 ± 0.114 |
| DANN (R1 shift100 last.ckpt) | 2 | 0.519 ± 0.073 [0.507, 0.532] | 0.528 ± 0.058 [0.515, 0.541] | 0.522 ± 0.054 [0.511, 0.533] | 0.499 ± 0.051 | 0.550 ± 0.072 | 1/5 | not met | 0.786 ± 0.071 |
| CDAN | 2 | 0.510 ± 0.056 [0.497, 0.522] | 0.526 ± 0.051 [0.513, 0.538] | 0.519 ± 0.039 [0.509, 0.530] | 0.499 ± 0.057 | 0.558 ± 0.059 | 1/5 | not met | 0.772 ± 0.052 |
| IWDAN | 2 | 0.586 ± 0.071 [0.573, 0.599] | 0.567 ± 0.057 [0.555, 0.581] | 0.542 ± 0.050 [0.532, 0.552] | 0.542 ± 0.057 | 0.571 ± 0.085 | 0/5 | not met | 0.700 ± 0.074 |
| IWCDAN | 2 | 0.478 ± 0.026 [0.464, 0.491] | 0.492 ± 0.025 [0.479, 0.506] | 0.489 ± 0.021 [0.479, 0.499] | 0.485 ± 0.061 | 0.538 ± 0.046 | 3/5 | met | 0.810 ± 0.040 |
| IWDAN_ORACLE | 2 | 0.727 ± 0.039 [0.716, 0.739] | 0.676 ± 0.045 [0.664, 0.689] | 0.641 ± 0.047 [0.632, 0.650] | 0.652 ± 0.039 | 0.627 ± 0.094 | 0/5 | not met | 0.311 ± 0.126 |

### Mahalanobis per seed (point + case-level CI)

| cell | λ | seed | point | ci_lo | ci_hi | CI entirely < 0.5 |
|---|---:|---:|---:|---:|---:|---|
| λ=0 (R1 shift100 last.ckpt) | 0 | 42 | 0.728 | 0.713 | 0.742 | no |
| λ=0 (R1 shift100 last.ckpt) | 0 | 52 | 0.702 | 0.688 | 0.716 | no |
| λ=0 (R1 shift100 last.ckpt) | 0 | 62 | 0.725 | 0.711 | 0.739 | no |
| λ=0 (R1 shift100 last.ckpt) | 0 | 72 | 0.749 | 0.735 | 0.763 | no |
| λ=0 (R1 shift100 last.ckpt) | 0 | 82 | 0.710 | 0.696 | 0.724 | no |
| DANN (R1 shift100 last.ckpt) | 0.5 | 42 | 0.539 | 0.525 | 0.552 | no |
| DANN (R1 shift100 last.ckpt) | 0.5 | 52 | 0.548 | 0.534 | 0.563 | no |
| DANN (R1 shift100 last.ckpt) | 0.5 | 62 | 0.545 | 0.531 | 0.560 | no |
| DANN (R1 shift100 last.ckpt) | 0.5 | 72 | 0.513 | 0.499 | 0.527 | no |
| DANN (R1 shift100 last.ckpt) | 0.5 | 82 | 0.526 | 0.511 | 0.541 | no |
| CDAN | 0.5 | 42 | 0.521 | 0.506 | 0.536 | no |
| CDAN | 0.5 | 52 | 0.504 | 0.490 | 0.520 | no |
| CDAN | 0.5 | 62 | 0.547 | 0.533 | 0.561 | no |
| CDAN | 0.5 | 72 | 0.576 | 0.561 | 0.592 | no |
| CDAN | 0.5 | 82 | 0.549 | 0.535 | 0.564 | no |
| IWDAN | 0.5 | 42 | 0.697 | 0.682 | 0.712 | no |
| IWDAN | 0.5 | 52 | 0.627 | 0.612 | 0.642 | no |
| IWDAN | 0.5 | 62 | 0.663 | 0.648 | 0.678 | no |
| IWDAN | 0.5 | 72 | 0.700 | 0.686 | 0.714 | no |
| IWDAN | 0.5 | 82 | 0.669 | 0.655 | 0.683 | no |
| IWCDAN | 0.5 | 42 | 0.555 | 0.540 | 0.570 | no |
| IWCDAN | 0.5 | 52 | 0.489 | 0.474 | 0.503 | no |
| IWCDAN | 0.5 | 62 | 0.527 | 0.511 | 0.542 | no |
| IWCDAN | 0.5 | 72 | 0.528 | 0.514 | 0.543 | no |
| IWCDAN | 0.5 | 82 | 0.581 | 0.565 | 0.597 | no |
| IWDAN_ORACLE | 0.5 | 42 | 0.713 | 0.698 | 0.728 | no |
| IWDAN_ORACLE | 0.5 | 52 | 0.729 | 0.715 | 0.743 | no |
| IWDAN_ORACLE | 0.5 | 62 | 0.744 | 0.732 | 0.757 | no |
| IWDAN_ORACLE | 0.5 | 72 | 0.750 | 0.737 | 0.763 | no |
| IWDAN_ORACLE | 0.5 | 82 | 0.751 | 0.738 | 0.764 | no |
| DANN (R1 shift100 last.ckpt) | 2 | 42 | 0.546 | 0.532 | 0.560 | no |
| DANN (R1 shift100 last.ckpt) | 2 | 52 | 0.542 | 0.526 | 0.556 | no |
| DANN (R1 shift100 last.ckpt) | 2 | 62 | 0.597 | 0.583 | 0.611 | no |
| DANN (R1 shift100 last.ckpt) | 2 | 72 | 0.513 | 0.500 | 0.527 | no |
| DANN (R1 shift100 last.ckpt) | 2 | 82 | 0.400 | 0.387 | 0.412 | yes |
| CDAN | 2 | 42 | 0.425 | 0.411 | 0.438 | yes |
| CDAN | 2 | 52 | 0.506 | 0.491 | 0.521 | no |
| CDAN | 2 | 62 | 0.498 | 0.483 | 0.513 | no |
| CDAN | 2 | 72 | 0.568 | 0.554 | 0.582 | no |
| CDAN | 2 | 82 | 0.551 | 0.538 | 0.565 | no |
| IWDAN | 2 | 42 | 0.521 | 0.506 | 0.535 | no |
| IWDAN | 2 | 52 | 0.533 | 0.518 | 0.548 | no |
| IWDAN | 2 | 62 | 0.564 | 0.551 | 0.578 | no |
| IWDAN | 2 | 72 | 0.691 | 0.677 | 0.705 | no |
| IWDAN | 2 | 82 | 0.622 | 0.606 | 0.636 | no |
| IWCDAN | 2 | 42 | 0.495 | 0.481 | 0.509 | no |
| IWCDAN | 2 | 52 | 0.464 | 0.449 | 0.480 | yes |
| IWCDAN | 2 | 62 | 0.442 | 0.427 | 0.456 | yes |
| IWCDAN | 2 | 72 | 0.479 | 0.463 | 0.493 | yes |
| IWCDAN | 2 | 82 | 0.509 | 0.494 | 0.524 | no |
| IWDAN_ORACLE | 2 | 42 | 0.676 | 0.661 | 0.692 | no |
| IWDAN_ORACLE | 2 | 52 | 0.765 | 0.754 | 0.778 | no |
| IWDAN_ORACLE | 2 | 62 | 0.696 | 0.683 | 0.710 | no |
| IWDAN_ORACLE | 2 | 72 | 0.758 | 0.745 | 0.771 | no |
| IWDAN_ORACLE | 2 | 82 | 0.739 | 0.726 | 0.753 | no |

## Paired-by-seed Δ vs DANN and gap recovered (SCIN only; descriptive)

Same formulas as main R2 report; **does not** enter the fix reading.

| method | λ | Δ Maha vs DANN [case-level cluster CI] | gap recovered (method−DANN)/(λ0−DANN) [seed boot CI] |
|---|---:|---|---|
| CDAN | 0.5 | +0.005 ± 0.041 [0.001, 0.009] (n=5) | +0.005 ± 0.211 [-0.165, 0.159] (n=5) · Δabs=+0.005 · mean(λ0−DANN)=+0.188 |
| IWDAN | 0.5 | +0.137 ± 0.041 [0.130, 0.143] (n=5) | +0.714 ± 0.132 [0.618, 0.808] (n=5) · Δabs=+0.137 · mean(λ0−DANN)=+0.188 |
| IWCDAN | 0.5 | +0.002 ± 0.043 [-0.004, 0.007] (n=5) | -0.007 ± 0.256 [-0.202, 0.177] (n=5) · Δabs=+0.002 · mean(λ0−DANN)=+0.188 |
| IWDAN_ORACLE | 0.5 | +0.203 ± 0.027 [0.193, 0.213] (n=5) | +1.087 ± 0.124 [0.992, 1.182] (n=5) · Δabs=+0.203 · mean(λ0−DANN)=+0.188 |
| CDAN | 2 | -0.010 ± 0.113 [-0.014, -0.005] (n=5) | -0.187 ± 0.549 [-0.619, 0.234] (n=5) · Δabs=-0.010 · mean(λ0−DANN)=+0.203 |
| IWDAN | 2 | +0.067 ± 0.123 [0.061, 0.072] (n=5) | +0.204 ± 0.490 [-0.169, 0.569] (n=5) · Δabs=+0.067 · mean(λ0−DANN)=+0.203 |
| IWCDAN | 2 | -0.042 ± 0.096 [-0.047, -0.036] (n=5) | -0.354 ± 0.572 [-0.815, 0.059] (n=5) · Δabs=-0.042 · mean(λ0−DANN)=+0.203 |
| IWDAN_ORACLE | 2 | +0.208 ± 0.096 [0.197, 0.218] (n=5) | +1.004 ± 0.274 [0.804, 1.212] (n=5) · Δabs=+0.208 · mean(λ0−DANN)=+0.203 |

Prediction (decisions): descriptive only — SCIN under DANN λ = 2 near chance (~0.52); IWDAN may sit slightly higher; not a gate.

