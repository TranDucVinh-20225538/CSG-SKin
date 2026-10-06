# CORAL anomaly — mean distance vs covariance discrepancy

No new training. Features from existing Phase 15.2 checkpoints.
Pair CORAL was trained on: ISIC train vs pad_adv. pad_heldout is reported beside it and was not in the penalty.

Hypothesis: classification can satisfy CORAL by matching covariances while pushing domain means apart.

| objective | weight | n | mean L2 (adv) | cov Frobenius (adv) | CORAL term (adv) | mean L2 (heldout) |
|---|---:|---:|---|---|---|---|
| coral | 1 | 3 | 20.649 ± 1.599 | 112.207 ± 2.208 | 0.001334 ± 0.000052 | 19.825 ± 1.587 |
| coral | 10 | 3 | 61.274 ± 1.041 | 40.475 ± 2.244 | 0.000174 ± 0.000019 | 59.747 ± 0.777 |
| coral | 100 | 3 | 70.086 ± 0.641 | 16.603 ± 1.815 | 0.000029 ± 0.000007 | 68.498 ± 0.523 |
| erm | 0 | 3 | 14.463 ± 0.688 | 158.920 ± 3.165 | 0.002677 ± 0.000107 | 13.937 ± 0.703 |

If mean L2 grows with weight while the CORAL term shrinks, the anomaly is the mean escaping a covariance-only penalty.

