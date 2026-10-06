# Phase 15.1 — single-encoder DANN on ISIC↔PAD

Tier: **main text** (fixed before numbers). Leakage is 2-class balanced accuracy; floor **0.5**.
OOD statistics fit on ISIC train only. Transfer is pad_heldout 6-class.

n_summaries = 25 / 25.

Gate rescored from `adversary_log.jsonl` (summary.json on disk may still carry the old label).
Valid when best-epoch adversary CE is below ln 2 − 0.02; at chance at convergence after that drop is success.

## Gate

- rescored: 25 valid/control, 0 inconclusive, 25 total.
- leakage moved toward 0.5 by ≥0.10 from λ=0: **True**.
- port verdict: **sound**.

The port reproduces a leakage drop. Camelyon17 may be retried (15.4) as a dataset-specific question.

## Four-column table plus detectors / ECE / transfer

| λ | n | leak bal | ID bal | Maha | kNN | MSP | cosine | Energy | OOD ECE | OOD conf | xfer bal |
|---:|---:|---|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.970 ± 0.001 | 0.768 ± 0.016 | 0.700 ± 0.030 | 0.944 ± 0.004 | 0.868 ± 0.013 | 0.884 ± 0.009 | 0.911 ± 0.012 | 0.436 ± 0.028 | 0.726 ± 0.020 | 0.390 ± 0.027 |
| 0.25 | 3 | 0.763 ± 0.003 | 0.778 ± 0.003 | 0.525 ± 0.017 | 0.528 ± 0.025 | 0.515 ± 0.016 | 0.509 ± 0.017 | 0.507 ± 0.008 | 0.699 ± 0.019 | 0.925 ± 0.004 | 0.333 ± 0.006 |
| 0.5 | 3 | 0.795 ± 0.006 | 0.766 ± 0.004 | 0.510 ± 0.022 | 0.517 ± 0.036 | 0.501 ± 0.026 | 0.493 ± 0.024 | 0.507 ± 0.022 | 0.710 ± 0.022 | 0.927 ± 0.010 | 0.328 ± 0.011 |
| 1 | 3 | 0.795 ± 0.014 | 0.763 ± 0.004 | 0.548 ± 0.031 | 0.521 ± 0.037 | 0.462 ± 0.029 | 0.482 ± 0.017 | 0.448 ± 0.035 | 0.704 ± 0.013 | 0.924 ± 0.009 | 0.334 ± 0.013 |
| 2 | 5 | 0.807 ± 0.014 | 0.758 ± 0.007 | 0.509 ± 0.018 | 0.516 ± 0.028 | 0.479 ± 0.013 | 0.468 ± 0.013 | 0.470 ± 0.024 | 0.712 ± 0.013 | 0.924 ± 0.007 | 0.325 ± 0.012 |
| 4 | 3 | 0.798 ± 0.010 | 0.750 ± 0.006 | 0.502 ± 0.028 | 0.486 ± 0.030 | 0.471 ± 0.019 | 0.454 ± 0.014 | 0.449 ± 0.034 | 0.698 ± 0.007 | 0.918 ± 0.001 | 0.320 ± 0.006 |
| 8 | 3 | 0.787 ± 0.007 | 0.731 ± 0.010 | 0.520 ± 0.031 | 0.532 ± 0.011 | 0.519 ± 0.026 | 0.490 ± 0.017 | 0.500 ± 0.024 | 0.670 ± 0.025 | 0.900 ± 0.012 | 0.321 ± 0.026 |

Per-run `gate` and `adversary_log.jsonl` sit next to each summary. Do not drop a λ because it is ugly.

