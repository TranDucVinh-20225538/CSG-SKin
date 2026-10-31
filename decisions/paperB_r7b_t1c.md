# Paper B R7b / T1c — dual DINOv2 with a higher backbone LR

Committed before any T1c job runs. Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

T1 used backbone LR × 0.1 for both the dual model and the ERM reference; the competence gap was 0.095 (Case C). Does a higher backbone LR close the gap? Both arms get the same LR grid, so the comparison stays matched.

## Design

- Backbone DINOv2-S/14; recipe and protocol as T1 (`decisions/paperB_r7_t1.md`). Outputs under `results/paperB/r7b/t1c/` with tag prefix `r7b_t1c`; checkpoints under `checkpoints/r7b_t1c/`.
- **Selection (seed 42 only):** backbone LR × {0.3, 1.0} for (a) ERM DINOv2-S, (b) dual DINOv2-S at λ = 0. 4 runs. For each arm separately, pick the multiplier with the higher ISIC **validation** balanced accuracy (`last.ckpt`); × 0.1 from T1 seed 42 is the third candidate. Ties (absolute difference < 0.005): the smaller multiplier. Test-set numbers of the selection runs are reported but not used for the choice.
- **Main:** at the chosen multipliers, ERM seeds 42, 52, 62, 72, 82 (5 runs) and dual λ ∈ {0, 2} × seeds 42..82 (10 runs). Selection runs at seed 42 are reused if they match the chosen setting. If the dual λ = 0 choice is × 0.1, **T1c stops after selection** (T1 is the result).
- Same protocol and report columns as T1. All LR settings that were run are reported, including the ones not chosen. **T1 stays in the paper whatever T1c shows.**
- ERM: no domain-head gradient reaches the encoder at λ = 0 (same as T1).

## Data / protocol

Same as Table 1 / T1: lesion-disjoint ISIC split (seed 42), pad_adv / pad_heldout, last.ckpt, Fitzpatrick17k + MED-NODE + SD-198 + SCIN, Mahalanobis plus kNN / cosine / MSP / energy, held-out leakage probe V1, ID balanced accuracy, ID ECE, nevus-nearest fraction, per-class AUROC and the decomposition check. Bootstrap units as in the paper. Degraded runs (ID balanced accuracy < 0.25) are flagged, not dropped. Report numbers including nulls. No pass/fail wording.

## Report

Table 1 columns at λ = 0 and 2 for the chosen dual multiplier; ERM at its chosen multiplier; selection table (val bal per arm × multiplier); competence gap and Cases A/B/C. Per-run JSON, summary.csv, BOOTSTRAP.json, REPORT.md with LaTeX rows. One plain paragraph stating which reading-rule case occurred, computed by the rule below, before any comment on it.

## Reading rules for T1c (locked before the first run)

Identical to T1:

Competence gap = ERM-DINOv2 mean ID balanced accuracy − dual-DINOv2 λ = 0 mean ID balanced accuracy. "Near ERM" means gap ≤ 0.03. (The EfficientNet-B3 gap in the paper is 0.04–0.08.)

Case A, near ERM and inverts (Rule A on pad_heldout, and on at least two of the three labelled unseen sets Fitzpatrick17k / MED-NODE / SD-198): the result enters the main results as evidence that the inversion does not depend on a weak model.

Case B, near ERM and does not invert (Rule A fails on pad_heldout and on every unseen set): the result enters the main results table, not an appendix, and the paper's conclusion is narrowed to the EfficientNet-B3 dual encoder.

Case C, anything else (gap > 0.03, or inversion on some sets only, or leakage at λ = 2 above 0.60, i.e. the adversary did not remove the domain): reported as found in the main results next to the EfficientNet-B3 rows, with the gap and leakage stated; it settles neither question.

Rule A for "inverts": pooled CI upper < 0.5 and at least 3 seeds with their own CI upper < 0.5 (an absolute count). With 5 seeds this is the same absolute count as Table 1.

## Outputs

`results/paperB/r7b/t1c/{selection,runs,erm}/`, `results/paperB/r7b/t1c/{summary.csv,BOOTSTRAP.json,REPORT.md,SELECTION.json}`.

## Deviations
