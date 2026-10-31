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
