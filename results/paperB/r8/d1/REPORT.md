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
