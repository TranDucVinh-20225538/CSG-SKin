# Paper B R7 / C1 — FPR at 95% TPR (CPU, cached features)

Committed before the C1 job. Deviations appended at the end.

## Design

- For every Table 1 and Table 2 cell (last.ckpt), add FPR@95TPR of Mahalanobis, with the same bootstrap units as the paper.
- Also report, for pad_heldout at λ = 0 and 2, the share of OOD images accepted at the 95% ID-coverage threshold (threshold from ID test or ID val scores as in the existing S-flip / clinical posthoc convention; document which in REPORT.md).
- Source: Table 1 last.ckpt feature caches under `results/paperB/r5/t1last/` (and matching OOD packs).
- No new training. CPU only.

## Outputs

`results/paperB/r7/c1/{summary.csv,BOOTSTRAP.json,REPORT.md}`.

## Deviations
