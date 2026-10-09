# R2 Item 5 — exploratory seeds (Camelyon17, two hospitals)

Commit: `7a7333f`. **Exploratory.** Not pre-registered; the Item 5 verdict (negative, leakage gate failed) is unchanged. Same code, recipe and primary assignment (A = hospital 3, B = hospital 0, held-out slides of B for detection).

**Question:** does Mahalanobis drop below 0.5 at λ = 1 on every seed while the probe stays ≈ 0.95?

**Answer:** Mahalanobis < 0.5 at λ=1 on 1 of 3 seeds (s42 0.478, s43 0.577, s44 0.546); hospital probe at λ=1: 0.950, 0.952, 0.953 (λ=0: 0.952, 0.966, 0.961).

## Mean ± s.d. over seeds 42–44

| λ | n | Maha (B held-out) | Cosine (B held-out) | kNN (B held-out) | MSP (B held-out) | Energy (B held-out) | Hospital probe bal acc | Adv CE min | Adv CE last | ID acc | ID acc slide-disjoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 0.651 ± 0.087 | 0.785 ± 0.049 | 0.794 ± 0.078 | 0.798 ± 0.038 | 0.794 ± 0.039 | 0.959 ± 0.007 | 0.101 ± 0.005 | 0.118 ± 0.006 | 0.995 ± 0.000 | 0.987 ± 0.004 |
| 0.3 | 3 | 0.600 ± 0.030 | 0.619 ± 0.038 | 0.607 ± 0.023 | 0.590 ± 0.080 | 0.581 ± 0.088 | 0.950 ± 0.003 | 0.240 ± 0.007 | 0.690 ± 0.001 | 0.995 ± 0.001 | 0.991 ± 0.000 |
| 1 | 3 | 0.534 ± 0.050 | 0.589 ± 0.034 | 0.648 ± 0.027 | 0.589 ± 0.025 | 0.575 ± 0.026 | 0.952 ± 0.001 | 0.371 ± 0.008 | 0.692 ± 0.000 | 0.995 ± 0.000 | 0.990 ± 0.002 |

## Per seed

| λ | seed | Maha (B held-out) | Cosine (B held-out) | kNN (B held-out) | MSP (B held-out) | Energy (B held-out) | Hospital probe bal acc | Adv CE min | Adv CE last | ID acc | ID acc slide-disjoint |
|---:|---:|---|---|---|---|---|---|---|---|---|---|
| 0 | 42 | 0.732 | 0.761 | 0.724 | 0.783 | 0.757 | 0.952 | 0.106 | 0.118 | 0.995 | 0.985 |
| 0 | 43 | 0.662 | 0.841 | 0.879 | 0.842 | 0.834 | 0.966 | 0.097 | 0.111 | 0.995 | 0.985 |
| 0 | 44 | 0.558 | 0.753 | 0.780 | 0.770 | 0.790 | 0.961 | 0.099 | 0.124 | 0.994 | 0.992 |
| 0.3 | 42 | 0.574 | 0.589 | 0.580 | 0.607 | 0.585 | 0.948 | 0.246 | 0.690 | 0.994 | 0.991 |
| 0.3 | 43 | 0.593 | 0.606 | 0.622 | 0.503 | 0.492 | 0.951 | 0.233 | 0.691 | 0.995 | 0.991 |
| 0.3 | 44 | 0.632 | 0.662 | 0.619 | 0.660 | 0.667 | 0.953 | 0.243 | 0.691 | 0.996 | 0.990 |
| 1 | 42 | 0.478 | 0.555 | 0.622 | 0.611 | 0.599 | 0.950 | 0.365 | 0.692 | 0.995 | 0.988 |
| 1 | 43 | 0.577 | 0.623 | 0.676 | 0.562 | 0.548 | 0.952 | 0.369 | 0.691 | 0.995 | 0.989 |
| 1 | 44 | 0.546 | 0.590 | 0.645 | 0.594 | 0.578 | 0.953 | 0.380 | 0.691 | 0.995 | 0.992 |

## Caveats

- Exploratory: added after the pre-registered coarse scan returned a negative; it does not reopen the gates.
- Detectors: ID = A patch-level test, OOD = held-out slides of hospital B. Probe: logistic ID-vs-B_heldout on encoder features, 70/30, 3 probe seeds.
- Adversary CE at chance is ln 2 = 0.693.
