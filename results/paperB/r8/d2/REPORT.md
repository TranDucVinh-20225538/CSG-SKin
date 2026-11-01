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

