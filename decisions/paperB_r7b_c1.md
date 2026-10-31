# Paper B R7b / C1 — column rename fix (CPU)

Committed before regenerating C1 tables under `results/paperB/r7b/c1/`. Deviations appended at the end.

## Design

Fix to R7 C1 naming only. In `scripts/r7_c1_fpr95.py` and its outputs:

- The quantity previously labelled `fpr95` was `mean(sood ≥ τ)`, the share of OOD flagged at the ID 95th-percentile threshold. Rename to **`ood_flagged_at_95id`**.
- Standard **FPR@95TPR** (ID positive, 95% of ID kept) is the share of OOD accepted as ID = `mean(sood ≤ τ)`, previously labelled `ood_accept_at_95id`. Rename to **`fpr95tpr`**.
- Add the pooled bootstrap CI for `fpr95tpr` (same bootstrap units as R7 C1).
- No other change to scoring, threshold, detectors, or run set (Table 1 last.ckpt Mahalanobis on `z_lesion_norm`).

## Outputs

`results/paperB/r7b/c1/{per_run.csv,summary.csv,BOOTSTRAP.json,REPORT.md}`. The same corrected tables are also written to `results/paperB/r7/c1/` so the R7 C1 path stays consistent.

## Deviations
