# Phase 15.1 gate, rescored

Existing `summary.json` files were not rewritten. This file is the corrected gate.
Rule: a run is valid when its best-epoch adversary CE falls below ln 2 − 0.02.
Returning to chance by the last epoch, after that drop, means the encoder won.
λ=0 is a valid control: the adversary term is multiplied by zero, so the head is not trained.
A run that never leaves ln 2 is inconclusive (the Camelyon17 failure).

n = 25 / 25. valid+control = 25. inconclusive = 0.
Leakage moved by ≥ 0.10 from λ=0: **True**.
Port verdict: **sound**. pending_or_mixed clears: **True**.

## Per-run gate

| run | λ | status | best CE | best epoch | last CE | last acc |
|---|---:|---|---:|---:|---:|---:|
| single_dann_ladv0_s42 | 0 | valid_control | 0.7763 | 1 | 0.7875 | 0.475 |
| single_dann_ladv0_s52 | 0 | valid_control | 0.7536 | 17 | 0.7813 | 0.474 |
| single_dann_ladv0_s62 | 0 | valid_control | 0.7457 | 16 | 0.7614 | 0.500 |
| single_dann_ladv0_s72 | 0 | valid_control | 0.7854 | 1 | 0.8110 | 0.444 |
| single_dann_ladv0_s82 | 0 | valid_control | 0.7413 | 39 | 0.7473 | 0.511 |
| single_dann_ladv0p25_s42 | 0.25 | valid | 0.1657 | 1 | 0.6431 | 0.618 |
| single_dann_ladv0p25_s52 | 0.25 | valid | 0.1615 | 1 | 0.6465 | 0.621 |
| single_dann_ladv0p25_s62 | 0.25 | valid | 0.1679 | 1 | 0.6470 | 0.613 |
| single_dann_ladv0p5_s42 | 0.5 | valid | 0.2415 | 1 | 0.6728 | 0.574 |
| single_dann_ladv0p5_s52 | 0.5 | valid | 0.2319 | 1 | 0.6745 | 0.570 |
| single_dann_ladv0p5_s62 | 0.5 | valid | 0.2398 | 1 | 0.6736 | 0.576 |
| single_dann_ladv1_s42 | 1 | valid | 0.4346 | 1 | 0.6853 | 0.550 |
| single_dann_ladv1_s52 | 1 | valid | 0.4066 | 1 | 0.6861 | 0.544 |
| single_dann_ladv1_s62 | 1 | valid | 0.4220 | 1 | 0.6855 | 0.548 |
| single_dann_ladv2_s42 | 2 | valid | 0.5636 | 1 | 0.6907 | 0.528 |
| single_dann_ladv2_s52 | 2 | valid | 0.5552 | 1 | 0.6916 | 0.526 |
| single_dann_ladv2_s62 | 2 | valid | 0.5546 | 1 | 0.6908 | 0.526 |
| single_dann_ladv2_s72 | 2 | valid | 0.5362 | 1 | 0.6905 | 0.530 |
| single_dann_ladv2_s82 | 2 | valid | 0.5335 | 1 | 0.6902 | 0.532 |
| single_dann_ladv4_s42 | 4 | valid | 0.6327 | 1 | 0.6920 | 0.520 |
| single_dann_ladv4_s52 | 4 | valid | 0.6228 | 1 | 0.6923 | 0.520 |
| single_dann_ladv4_s62 | 4 | valid | 0.6295 | 1 | 0.6925 | 0.519 |
| single_dann_ladv8_s42 | 8 | valid | 0.6598 | 1 | 0.6930 | 0.511 |
| single_dann_ladv8_s52 | 8 | valid | 0.6638 | 1 | 0.6928 | 0.512 |
| single_dann_ladv8_s62 | 8 | valid | 0.6575 | 1 | 0.6928 | 0.515 |

ln 2 = 0.6931. Margin = 0.02. λ=8 best CE is ≈ 0.66, which is below the margin, so those three runs are valid. Their last epoch is back at ln 2; that is the encoder winning.

## Table (unchanged measurements)

| λ | n | leak bal | Maha | kNN | MSP | cosine | Energy | ID acc | xfer bal | gate |
|---:|---:|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.970 ± 0.001 | 0.700 ± 0.030 | 0.944 ± 0.004 | 0.868 ± 0.013 | 0.884 ± 0.009 | 0.911 ± 0.012 | 0.845 ± 0.003 | 0.390 ± 0.027 | valid_control |
| 0.25 | 3 | 0.763 ± 0.003 | 0.525 ± 0.017 | 0.528 ± 0.025 | 0.515 ± 0.016 | 0.509 ± 0.017 | 0.507 ± 0.008 | 0.845 ± 0.003 | 0.333 ± 0.006 | valid |
| 0.5 | 3 | 0.795 ± 0.006 | 0.510 ± 0.022 | 0.517 ± 0.036 | 0.501 ± 0.026 | 0.493 ± 0.024 | 0.507 ± 0.022 | 0.844 ± 0.003 | 0.328 ± 0.011 | valid |
| 1 | 3 | 0.795 ± 0.014 | 0.548 ± 0.031 | 0.521 ± 0.037 | 0.462 ± 0.029 | 0.482 ± 0.017 | 0.448 ± 0.035 | 0.840 ± 0.001 | 0.334 ± 0.013 | valid |
| 2 | 5 | 0.807 ± 0.014 | 0.509 ± 0.018 | 0.516 ± 0.028 | 0.479 ± 0.013 | 0.468 ± 0.013 | 0.470 ± 0.024 | 0.839 ± 0.004 | 0.325 ± 0.012 | valid |
| 4 | 3 | 0.798 ± 0.010 | 0.502 ± 0.028 | 0.486 ± 0.030 | 0.471 ± 0.019 | 0.454 ± 0.014 | 0.449 ± 0.034 | 0.835 ± 0.005 | 0.320 ± 0.006 | valid |
| 8 | 3 | 0.787 ± 0.007 | 0.520 ± 0.031 | 0.532 ± 0.011 | 0.519 ± 0.026 | 0.490 ± 0.017 | 0.500 ± 0.024 | 0.832 ± 0.004 | 0.321 ± 0.026 | valid |

