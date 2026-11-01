# Paper B R8 consolidated report

Generated at commit `e5e7c20`.

| item | report |
|---|---|
| d1 | yes |
| d2 | yes |
| d4 | yes |

GPU hours (sum under r8): 22.59.


---

## D1

# Paper B R8 D1: single-encoder invariance screen

Design: `decisions/paperB_r8_d1.md`. Commit `2b4ecab`.

GPU hours used: 18.95 / 30.
ERM floors: EffB3 ≥ 0.508; DINOv2 ≥ 0.515.

| backbone | config | λ/w | family | V1 ± sd | ID val | ID test | degraded | pass |
|---|---|---|---|---|---:|---:|---|---|
| dinov2_vits14 | baseline | 3 | align | 0.990 ± 0.002 | 0.320 | 0.319 | False | False |
| dinov2_vits14 | baseline | 3 | align | 0.665 ± 0.025 | 0.530 | 0.573 | False | False |
| dinov2_vits14 | baseline | 10 | align | 0.650 ± 0.013 | 0.125 | 0.125 | True | False |
| dinov2_vits14 | baseline | 10 | align | 0.618 ± 0.016 | 0.489 | 0.553 | False | False |
| dinov2_vits14 | baseline | 30 | align | 0.616 ± 0.004 | 0.125 | 0.125 | True | False |
| dinov2_vits14 | baseline | 30 | align | 0.600 ± 0.014 | 0.518 | 0.555 | False | True |
| dinov2_vits14 | fix_lr30_d2 | 4 | grl | 0.806 ± 0.020 | 0.499 | 0.530 | False | False |
| dinov2_vits14 | gw_lr30_d2 | 4 | grl | 0.789 ± 0.008 | 0.525 | 0.541 | False | False |
| dinov2_vits14 | fix_lr30_d2 | 8 | grl | 0.828 ± 0.014 | 0.537 | 0.570 | False | False |
| dinov2_vits14 | gw_lr30_d2 | 8 | grl | 0.847 ± 0.005 | 0.538 | 0.587 | False | False |
| dinov2_vits14 | gw_lr30_d3_k1 | 8 | grl | 0.821 ± 0.017 | 0.497 | 0.502 | False | False |
| dinov2_vits14 | gw_lr30_d3_k3 | 8 | grl | 0.772 ± 0.021 | 0.527 | 0.540 | False | False |
| dinov2_vits14 | gw_lr30_d3_k5 | 8 | grl | 0.820 ± 0.011 | 0.494 | 0.533 | False | False |
| dinov2_vits14 | fix_lr30_d2 | 16 | grl | 0.930 ± 0.009 | 0.486 | 0.508 | False | False |
| dinov2_vits14 | gw_lr30_d2 | 16 | grl | 0.854 ± 0.019 | 0.527 | 0.582 | False | False |
| effb3 | baseline | 3 | align | 0.981 ± 0.001 | 0.491 | 0.540 | False | False |
| effb3 | baseline | 3 | align | 0.810 ± 0.005 | 0.519 | 0.574 | False | False |
| effb3 | baseline | 10 | align | 0.993 ± 0.004 | 0.449 | 0.465 | False | False |
| effb3 | baseline | 10 | align | 0.839 ± 0.011 | 0.526 | 0.532 | False | False |
| effb3 | baseline | 30 | align | 0.992 ± 0.003 | 0.319 | 0.311 | False | False |
| effb3 | baseline | 30 | align | 0.894 ± 0.008 | 0.527 | 0.547 | False | False |
| effb3 | fix_lr10_d2 | 4 | grl | 0.685 ± 0.015 | 0.492 | 0.546 | False | False |
| effb3 | fix_lr10_d2 | 8 | grl | 0.695 ± 0.007 | 0.487 | 0.504 | False | False |
| effb3 | gw_lr10_d3_k3 | 8 | grl | 0.697 ± 0.030 | 0.525 | 0.566 | False | False |
| effb3 | gw_lr10_d3_k5 | 8 | grl | 0.693 ± 0.032 | 0.533 | 0.539 | False | False |
| effb3 | fix_lr10_d2 | 16 | grl | 0.727 ± 0.017 | 0.483 | 0.504 | False | False |
| effb3 | gw_lr10_d2 | 16 | grl | 0.706 ± 0.014 | 0.457 | 0.541 | False | False |

## Selection

- **effb3**: no run passed. W1 not addressed by R8 for effb3.
- **dinov2_vits14**: selected `r8_d1_single_dinov2_vits14_baseline_ladv0_s42_bb0p1_mkmmd30` (family=align, V1=0.600). D2 tests single-encoder alignment, not DANN.

---

## D2

# Paper B R8 D2: single-encoder detection test (MK-MMD)

Design: `decisions/paperB_r8_d2.md`. Commit `e5e7c20`.

Selected from D1: `r8_d1_single_dinov2_vits14_baseline_ladv0_s42_bb0p1_mkmmd30` (family=align; **not DANN**).

## Reading-rule case (computed first)

**not_tested.** invariance not reached over 5 seeds (seed-42 pass did not hold): the test did not take place.

5-seed mean V1 = 0.604 (invariance ≤0.60: False). Competence gap = 0.018 (ERM 0.565 − align 0.547; ≤0.03: True).

| arm | n | V1 | Leakage | ID bal acc | ID ECE | OOD ECE (pad_heldout) | OOD AUROC (Maha pad_heldout) | Fitz AUROC (Maha) | X-dom bal acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| align MK-MMD w30 | 5 | 0.604 ± 0.018 | 0.691 ± 0.030 | 0.547 ± 0.020 | 0.161 ± 0.018 | 0.623 ± 0.028 | 0.540 ± 0.017 | 0.519 ± 0.028 | 0.301 ± 0.032 |
| ERM DINOv2 | 5 | — | 0.983 ± 0.002 | 0.565 ± 0.015 | 0.195 ± 0.007 | 0.426 ± 0.018 | 0.591 ± 0.060 | 0.704 ± 0.059 | 0.323 ± 0.028 |

pad_heldout Rule A: above (k_below=0/5).

GPU hours (D2 runs): 3.64.


---

## D4

# R8 / D4: matched competence (paired lambda = 2 minus lambda = 0)

paired lambda=2 minus lambda=0 per seed, last checkpoint; seed bootstrap 2000 draws, rng 20261005. Mahalanobis AUROC.

| backbone | quantity | mean Δ | 95% CI | per seed (42, 52, 62, 72, 82) |
|---|---|---:|---|---|
| effb3_dual | id_bal_acc | -0.009 | [-0.026, +0.006] | -0.037, -0.025, +0.001, +0.009, +0.007 |
| effb3_dual | pad_heldout | -0.444 | [-0.465, -0.417] | -0.399, -0.462, -0.459, -0.472, -0.429 |
| effb3_dual | fitzpatrick17k | -0.307 | [-0.334, -0.275] | -0.258, -0.280, -0.321, -0.354, -0.320 |
| dinov2_dual | id_bal_acc | -0.008 | [-0.014, -0.002] | -0.018, -0.004, -0.010, +0.001, -0.010 |
| dinov2_dual | pad_heldout | -0.383 | [-0.428, -0.329] | -0.458, -0.388, -0.288, -0.407, -0.376 |
| dinov2_dual | fitzpatrick17k | -0.306 | [-0.361, -0.244] | -0.378, -0.362, -0.196, -0.272, -0.324 |
| dinov2_dual | mednode | -0.218 | [-0.253, -0.183] | -0.264, -0.172, -0.194, -0.272, -0.188 |
| dinov2_dual | sd198 | -0.329 | [-0.372, -0.270] | -0.382, -0.378, -0.212, -0.339, -0.335 |

EffB3 ID CI here [-0.0260, +0.0064] vs paper source [-0.0260, +0.0064].
