# Checkpoint audit (P8 deviation 2026-10-15 21:57, items 3 and 4)

Table 1 GRL schedule (csg_lightning, also used by the P8 recipe): α = max(0.2, 2/(1+exp(-20 p^0.5)) - 1), p = global step / total steps; no adversary warm-up epochs (the adversary is trained and the reversed gradient is applied from step 0). `grl_alpha_epoch_mean` is the epoch mean of the logged α. Flag: λ > 0 and α < 0.99 at the selected epoch (criterion fixed after the selected epochs were listed).

## Item 4: Table 1 selected epochs (35 runs)

| run | lambda_adv | seed | selected_epoch | last_epoch | grl_alpha_epoch_mean | adv_train_acc | adv_train_loss | val_acc_logged | before_alpha_on |
|---|---|---|---|---|---|---|---|---|---|
| runB_orth1_ladv0_s42 | 0.000 | 42 | 33 | 39 | 1.000 | 0.560 | 0.670 | 0.470 | False |
| runB_orth1_ladv0_s52 | 0.000 | 52 | 26 | 39 | 1.000 | 0.478 | 0.738 | 0.474 | False |
| runB_orth1_ladv0_s62 | 0.000 | 62 | 19 | 39 | 1.000 | 0.439 | 0.719 | 0.484 | False |
| runB_orth1_ladv0_s72 | 0.000 | 72 | 15 | 39 | 1.000 | 0.547 | 0.745 | 0.481 | False |
| runB_orth1_ladv0_s82 | 0.000 | 82 | 39 | 39 | 1.000 | 0.496 | 0.726 | 0.490 | False |
| runB_orth1_ladv0p25_s42 | 0.250 | 42 | 6 | 39 | 0.999 | 0.785 | 0.479 | 0.467 | False |
| runB_orth1_ladv0p25_s43 | 0.250 | 43 | 17 | 39 | 1.000 | 0.628 | 0.651 | 0.469 | False |
| runB_orth1_ladv0p25_s44 | 0.250 | 44 | 15 | 39 | 1.000 | 0.647 | 0.634 | 0.475 | False |
| runB_orth1_ladv0p25_s45 | 0.250 | 45 | 39 | 39 | 1.000 | 0.551 | 0.685 | 0.463 | False |
| runB_orth1_ladv0p25_s46 | 0.250 | 46 | 39 | 39 | 1.000 | 0.552 | 0.685 | 0.480 | False |
| runB_orth1_ladv0p5_s42 | 0.500 | 42 | 33 | 39 | 1.000 | 0.541 | 0.689 | 0.467 | False |
| runB_orth1_ladv0p5_s43 | 0.500 | 43 | 33 | 39 | 1.000 | 0.543 | 0.688 | 0.461 | False |
| runB_orth1_ladv0p5_s44 | 0.500 | 44 | 29 | 39 | 1.000 | 0.534 | 0.689 | 0.463 | False |
| runB_orth1_ladv0p5_s45 | 0.500 | 45 | 26 | 39 | 1.000 | 0.545 | 0.687 | 0.470 | False |
| runB_orth1_ladv0p5_s46 | 0.500 | 46 | 27 | 39 | 1.000 | 0.540 | 0.688 | 0.468 | False |
| runB_orth1_ladv1_s42 | 1.000 | 42 | 24 | 39 | 1.000 | 0.523 | 0.691 | 0.474 | False |
| runB_orth1_ladv1_s43 | 1.000 | 43 | 27 | 39 | 1.000 | 0.520 | 0.692 | 0.457 | False |
| runB_orth1_ladv1_s44 | 1.000 | 44 | 32 | 39 | 1.000 | 0.518 | 0.691 | 0.479 | False |
| runB_orth1_ladv1_s45 | 1.000 | 45 | 22 | 39 | 1.000 | 0.523 | 0.691 | 0.452 | False |
| runB_orth1_ladv1_s46 | 1.000 | 46 | 21 | 39 | 1.000 | 0.529 | 0.691 | 0.464 | False |
| runB_orth1_ladv2_s42 | 2.000 | 42 | 33 | 39 | 1.000 | 0.514 | 0.693 | 0.466 | False |
| runB_orth1_ladv2_s52 | 2.000 | 52 | 21 | 39 | 1.000 | 0.518 | 0.692 | 0.471 | False |
| runB_orth1_ladv2_s62 | 2.000 | 62 | 20 | 39 | 1.000 | 0.516 | 0.692 | 0.461 | False |
| runB_orth1_ladv2_s72 | 2.000 | 72 | 26 | 39 | 1.000 | 0.514 | 0.692 | 0.470 | False |
| runB_orth1_ladv2_s82 | 2.000 | 82 | 39 | 39 | 1.000 | 0.514 | 0.692 | 0.469 | False |
| runB_orth1_ladv4_s42 | 4.000 | 42 | 18 | 39 | 1.000 | 0.513 | 0.693 | 0.448 | False |
| runB_orth1_ladv4_s43 | 4.000 | 43 | 27 | 39 | 1.000 | 0.509 | 0.693 | 0.461 | False |
| runB_orth1_ladv4_s44 | 4.000 | 44 | 16 | 39 | 1.000 | 0.513 | 0.693 | 0.460 | False |
| runB_orth1_ladv4_s45 | 4.000 | 45 | 37 | 39 | 1.000 | 0.506 | 0.693 | 0.453 | False |
| runB_orth1_ladv4_s46 | 4.000 | 46 | 39 | 39 | 1.000 | 0.503 | 0.693 | 0.469 | False |
| runB_orth1_ladv8_s42 | 8.000 | 42 | 33 | 39 | 1.000 | 0.504 | 0.693 | 0.465 | False |
| runB_orth1_ladv8_s43 | 8.000 | 43 | 24 | 39 | 1.000 | 0.503 | 0.693 | 0.469 | False |
| runB_orth1_ladv8_s44 | 8.000 | 44 | 19 | 39 | 1.000 | 0.503 | 0.693 | 0.468 | False |
| runB_orth1_ladv8_s45 | 8.000 | 45 | 32 | 39 | 1.000 | 0.501 | 0.693 | 0.481 | False |
| runB_orth1_ladv8_s46 | 8.000 | 46 | 32 | 39 | 1.000 | 0.508 | 0.693 | 0.497 | False |

λ > 0 runs flagged: 0

## Item 3: competence, λ = 0, lesion-level split (5 seeds; single-encoder ERM: 5 seeds)

Image-level references (not like-for-like; leakage-inflated): paper single-encoder ERM 0.779 (the 0.70 competence threshold was set from this figure; original ≥ 0.70 verdicts stay visible). Phase 15 image-level single-encoder λ = 0 ID test bal acc 0.768 ± 0.016 (5 seeds, best-val). Lesion-level ERM competence reference: `results/paperB/r5/p8/erm_ref/REPORT.md`.

| recipe | seed | best_epoch | last_epoch | best_val_bal_acc | best_test_bal_acc | last_val_bal_acc | last_test_bal_acc | check_best_test_vs_stored |
|---|---|---|---|---|---|---|---|---|
| single-encoder ERM (lesion-level) | 42 | 31 | 39 | 0.522 | 0.567 | 0.513 | 0.567 | True |
| single-encoder ERM (lesion-level) | 52 | 7 | 39 | 0.524 | 0.545 | 0.529 | 0.578 | True |
| Table 1 (current) | 42 | 33 | 39 | 0.465 | 0.495 | 0.445 | 0.508 | True |
| Table 1 (current) | 52 | 26 | 39 | 0.474 | 0.519 | 0.409 | 0.469 | False |
| Table 1 (current) | 62 | 19 | 39 | 0.484 | 0.497 | 0.440 | 0.470 | True |
| Table 1 (current) | 72 | 15 | 39 | 0.481 | 0.484 | 0.462 | 0.476 | True |
| Table 1 (current) | 82 | 39 | 39 | 0.490 | 0.482 | 0.490 | 0.482 | True |
| A4 (P8) | 42 | 1 | 39 | 0.539 | 0.535 | 0.526 | 0.544 |  |
| A4 (P8) | 52 | 1 | 39 | 0.546 | 0.530 | 0.470 | 0.518 |  |
| A4 (P8) | 62 | 3 | 39 | 0.545 | 0.567 | 0.465 | 0.506 |  |
| A4 (P8) | 72 | 3 | 39 | 0.537 | 0.567 | 0.450 | 0.502 |  |
| A4 (P8) | 82 | 7 | 39 | 0.530 | 0.571 | 0.489 | 0.528 |  |

Mean ± SD (n):

| recipe | best_val_bal_acc_mean | best_val_bal_acc_std | best_val_bal_acc_count | best_test_bal_acc_mean | best_test_bal_acc_std | best_test_bal_acc_count | last_val_bal_acc_mean | last_val_bal_acc_std | last_val_bal_acc_count | last_test_bal_acc_mean | last_test_bal_acc_std | last_test_bal_acc_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A4 (P8) | 0.540 | 0.007 | 5 | 0.554 | 0.020 | 5 | 0.480 | 0.029 | 5 | 0.520 | 0.017 | 5 |
| Table 1 (current) | 0.479 | 0.010 | 5 | 0.495 | 0.015 | 5 | 0.449 | 0.030 | 5 | 0.481 | 0.016 | 5 |
| single-encoder ERM (lesion-level) | 0.523 | 0.002 | 2 | 0.556 | 0.016 | 2 | 0.521 | 0.012 | 2 | 0.572 | 0.008 | 2 |

LaTeX rows:

```
A4 (P8) & 0.540 $\pm$ 0.007 & 0.554 $\pm$ 0.020 & 0.480 $\pm$ 0.029 & 0.520 $\pm$ 0.017 \\
Table 1 (current) & 0.479 $\pm$ 0.010 & 0.495 $\pm$ 0.015 & 0.449 $\pm$ 0.030 & 0.481 $\pm$ 0.016 \\
single-encoder ERM (lesion-level) & 0.523 $\pm$ 0.002 & 0.556 $\pm$ 0.016 & 0.521 $\pm$ 0.012 & 0.572 $\pm$ 0.008 \\
```

