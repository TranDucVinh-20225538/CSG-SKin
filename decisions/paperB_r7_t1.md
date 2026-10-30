# Paper B R7 / T1 — dual encoder with a transformer backbone

Committed before any T1 job runs. Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

Addresses the "weak model" objection (W2) only. It does not address W1: it is still the dual encoder.

## Design

- Dual encoder, backbone DINOv2-S/14 in both branches (`scripts/train_r5_dual_isic.py`, same recipe as Table 1 / `decisions/paperB_r5_q3.md` dual DINOv2 arm only). Fallback: DINOv2-B/14 only if S/14 fails to load or train; log as a deviation.
- Backbone LR × 0.1; everything else as the main recipe (`r5_recipe` current).
- Outputs under `results/paperB/r7/t1/` with tag prefix `r7_t1` (not the old `r5/q3` folder).
- λ ∈ {0, 2} × seeds 42, 52, 62, 72, 82 = 10 dual runs.
- ERM reference: DINOv2-S/14 as a **single-encoder classifier**, no adversarial or context branch (λ = 0), seeds 42, 52, 62, same lesion-disjoint split and last-checkpoint rule. Implemented by extending `train_r5_single_isic.py` with the `r5_models` DINOv2 backbone and `backbone_lr_mult=0.1`. These 3 runs are also the λ = 0 reference for T1b.
- **ERM: no domain-head gradient reaches the encoder.** λ_adv = 0, so the GRL adversary term is not applied to the encoder; the domain head (if present in the toolbox module) does not reverse-gradient into the backbone. There is no context branch.

## Data / protocol

Same as Table 1: lesion-disjoint ISIC split (seed 42), pad_adv / pad_heldout, last.ckpt, Fitzpatrick17k + MED-NODE + SD-198 + SCIN, Mahalanobis plus kNN / cosine / MSP / energy, held-out leakage probe V1, ID balanced accuracy, ID ECE, nevus-nearest fraction, per-class AUROC and the decomposition check. Bootstrap units as in the paper (pad: patient; Fitzpatrick: P13 cluster; SCIN: case; MED-NODE, SD-198: image). Degraded runs (ID balanced accuracy < 0.25) are flagged, not dropped. Report numbers including nulls. No pass/fail wording.

## Report

Table 1 columns at λ = 0 and 2; Table 2 (unseen sets) with CIs; ID balanced accuracy of the λ = 0 dual model vs this ERM reference, and vs the EfficientNet-B3 ERM reference (0.558 ± 0.022); class-aware probes (unweighted, class-balanced, within-class) as in `r6_class_aware_heldout_probes.py`. Per-run JSON, summary.csv, BOOTSTRAP.json, REPORT.md with LaTeX rows. One plain paragraph stating which reading-rule case occurred, computed by the rule below, before any comment on it.

## Reading rules for T1 (locked before the first run)

Competence gap = ERM-DINOv2 mean ID balanced accuracy − dual-DINOv2 λ = 0 mean ID balanced accuracy. "Near ERM" means gap ≤ 0.03. (The EfficientNet-B3 gap in the paper is 0.04–0.08.)

Case A, near ERM and inverts (Rule A on pad_heldout, and on at least two of the three labelled unseen sets Fitzpatrick17k / MED-NODE / SD-198): the result enters the main results as evidence that the inversion does not depend on a weak model.

Case B, near ERM and does not invert (Rule A fails on pad_heldout and on every unseen set): the result enters the main results table, not an appendix, and the paper's conclusion is narrowed to the EfficientNet-B3 dual encoder.

Case C, anything else (gap > 0.03, or inversion on some sets only, or leakage at λ = 2 above 0.60, i.e. the adversary did not remove the domain): reported as found in the main results next to the EfficientNet-B3 rows, with the gap and leakage stated; it settles neither question.

Rule A for "inverts": pooled CI upper < 0.5 and at least 3 seeds with their own CI upper < 0.5 (an absolute count). With 5 seeds this is the same absolute count as Table 1.

## Code note (pre-run)

`train_r5_single_isic.py` gains DINOv2-S/14 and DINOv2-B/14 via `r5_models.build_backbone`; `SingleEncToolboxLit` gains `backbone_lr_mult` (0.1 for ViTs). Logged here before the first ERM run.

## Outputs

`results/paperB/r7/t1/runs/<tag>/`, `results/paperB/r7/t1/{summary.csv,BOOTSTRAP.json,REPORT.md,probes/}`, checkpoints under `checkpoints/r7_t1/`.

## Deviations

### Amendment 2026-10-30 (before T1b; ERM still running)

Work-order update: the T1 ERM reference uses **5 seeds** (42, 52, 62, **72, 82**), not 3. Dual grid unchanged (λ ∈ {0, 2} × 5 seeds). Seeds 72 and 82 for ERM are submitted as additional runs after the original 3-seed array; reading rules above are **not** edited. Competence gap and Cases A/B/C still use the ERM-DINOv2 mean over whatever ERM seeds completed (now five).
