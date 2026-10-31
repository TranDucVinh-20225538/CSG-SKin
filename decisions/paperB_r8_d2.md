# Paper B R8 / D2 — single-encoder detection test (selected configs only)

Committed before the first D1 run (reading rules locked with D1). Deviations appended at the end. Run only after D1 selection files exist for a backbone that passed.

## Design

- Selected config × seeds 42, 52, 62, 72, 82 (seed 42 reused from D1).
- λ = 0 reference: DINOv2 = R7 T1 ERM (5 seeds); EfficientNet-B3 = P9 v2 λ = 0 control (seeds 42, 52, 62) plus seeds 72 and 82 (2 new runs).
- Full Table 1 / Table 2 protocol: Mahalanobis, kNN, cosine, MSP, energy on pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN; Rule A (pooled CI upper < 0.5 and ≥ 3 of 5 seeds), FPR@95TPR as in R7b C1 (fpr95tpr = share of OOD accepted at 95% ID kept), ID balanced accuracy, competence gap, leakage V1, class-aware probes as in `r6_class_aware_heldout_probes.py`, nevus-nearest share.
- Outputs under `results/paperB/r8/d2/`.

## Reading rules for D2 (locked before the first D1 run)

Invariance reached = 5-seed mean V1 ≤ 0.60. Competent = gap to the λ = 0 reference ≤ 0.03.

Invariance reached, competent, Rule A met on pad_heldout: W1 is answered for this single-encoder model; the result enters the main text and the contribution list.

Invariance reached, competent, Rule A not met on pad_heldout and on every unseen set: a single-encoder model at comparable invariance does not invert; the result enters the main text and the paper's scope stays the dual encoder, stated as such.

Invariance not reached over 5 seeds (seed-42 pass did not hold): as R7 T1b, the test did not take place.

Anything else (gap > 0.03, inversion on some sets only): reported as found, next to the dual-encoder rows; it settles neither question.

Report which case occurred, computed by the rule, before any comment on it.

## Outputs

`results/paperB/r8/d2/{summary.csv,BOOTSTRAP.json,REPORT.md,OUTCOME.json}` and consolidated `results/paperB/r8/REPORT.md`.

## Deviations
