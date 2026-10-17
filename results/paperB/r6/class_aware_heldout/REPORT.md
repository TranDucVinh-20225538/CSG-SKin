# Paper B — class-aware held-out domain probes

Design: `decisions/paperB_r6_class_aware_heldout.md`. Commit `94279f5`.
Pool: ISIC-test ∪ pad_heldout, GroupShuffleSplit 70/30 (lesion/patient). Rep: `z_lesion_norm`.
**(a)** unweighted (P6 V1); **(b)** class-balanced (P6 V3); **(c)** within-class (≥20/side), class-size-weighted mean.

Oracle λ=2 seeds on disk: [42, 52, 62, 72, 82]; complete=True.

| cell | n | (a) unweighted | (b) class-balanced V3 | (c) within-class |
|---|---:|---|---|---|
| R2 IWDAN λ=2 | 5 | 0.534 ± 0.025 [0.515, 0.553] (n=5) | 0.562 ± 0.020 [0.548, 0.577] (n=5) | 0.597 ± 0.011 [0.589, 0.606] (n=5) |
| R2 IWCDAN λ=2 | 5 | 0.521 ± 0.039 [0.502, 0.556] (n=5) | 0.633 ± 0.062 [0.590, 0.686] (n=5) | 0.621 ± 0.010 [0.614, 0.629] (n=5) |
| R2 IWDAN_ORACLE λ=2 | 5 | 0.645 ± 0.041 [0.613, 0.677] (n=5) | 0.549 ± 0.032 [0.526, 0.574] (n=5) | 0.576 ± 0.020 [0.560, 0.592] (n=5) |
| R1 shift0% λ=2 | 5 | 0.629 ± 0.019 [0.614, 0.643] (n=5) | 0.539 ± 0.041 [0.512, 0.575] (n=5) | 0.560 ± 0.013 [0.551, 0.571] (n=5) |
| R1 shift100% λ=2 | 5 | 0.510 ± 0.007 [0.505, 0.515] (n=5) | 0.602 ± 0.054 [0.562, 0.648] (n=5) | 0.618 ± 0.003 [0.616, 0.621] (n=5) |

## Oracle λ=2 per seed

| seed | (a) | (b) | (c) |
|---:|---:|---:|---:|
| 42 | 0.614 | 0.530 | 0.556 |
| 52 | 0.689 | 0.563 | 0.602 |
| 62 | 0.598 | 0.516 | 0.566 |
| 72 | 0.639 | 0.540 | 0.562 |
| 82 | 0.685 | 0.597 | 0.592 |

**On origin/paperb/reviewer-r1:** `a7dbf25`.

