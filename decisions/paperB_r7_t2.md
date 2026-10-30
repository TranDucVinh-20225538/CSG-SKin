# Paper B R7 / T2 — size of the PAD set the adversary sees

Committed before any T2 job runs. Fixed now; not changed after results are seen. Deviations are appended at the end. Run after T1b; **cut first** if GPU time runs out.

## Question

Sensitivity analysis, not a main result: does inversion at λ = 2 depend on how large pad_adv is?

## Design

- Subsample pad_adv by patient to 25% and 50%, stratified so the class mix of pad_adv stays the same (report the class counts of each subsample). 100% is the existing Table 1 run set (not rerun).
- **The pad_adv subsample feeds both the adversarial branch and the context branch** (CombinedTrainDataset / two-stream recipe uses the same pad_adv rows for domain 1 in both paths). **pad_heldout is unchanged.**
- One fixed subsample per fraction (seed 0), not one per run. Option `--pad_adv_frac` on `train_r5_dual_isic.py`; default 1.0 must reproduce Table 1.
- Main recipe (EfficientNet-B3 dual), λ = 2, seeds 42, 52, 62 at 25% and 50% (6 runs), plus λ = 0 at 25%, seeds 42, 52, 62 (3 runs) as the no-adversary control at the smallest size. 9 runs.
- Outputs under `results/paperB/r7/t2/` with tag prefix `r7_t2`.

## Data / protocol

Same as Table 1. Report Table 1 / Table 2 columns per fraction next to the 100% rows (seeds 42, 52, 62 from Table 1).

## Reading rules for T2 (locked before the first run)

With 3 seeds per cell, "inverts" means all 3 seeds have CI upper < 0.5 (Rule A); write this in the decisions file.

Rule A for "inverts": pooled CI upper < 0.5 and at least 3 seeds with their own CI upper < 0.5 (an absolute count). With 3 seeds this means all 3 seeds must have CI upper < 0.5.

## Outputs

`results/paperB/r7/t2/runs/<tag>/`, subsample manifests with class counts, `results/paperB/r7/t2/{summary.csv,BOOTSTRAP.json,REPORT.md}`.

## Deviations
