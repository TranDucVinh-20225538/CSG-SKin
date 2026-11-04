# Paper B R10 — fill the incomplete cells

Committed before any R10 run. Fixed now; not changed after results are seen. Deviations are appended at the end.

These items complete existing tables. They test nothing new: no new reading rules, no pass/fail wording. Report every number as found, including nulls.

## Shared rules

Same split, last.ckpt, Rule A (pooled CI upper < 0.5 and ≥ 3 seeds with own CI upper < 0.5), degraded rule (ID test balanced accuracy < 0.25: report the run, exclude it from every mean, CI, probe and regression).

## C0: degraded-run audit (CPU, first)

R6 class-aware probes (`results/paperB/r6/class_aware_heldout/`) still include R1 shift 0 % λ = 2 seed 52 (ID balanced accuracy 0.139). Without it, the three probes are unweighted 0.630, class-balanced 0.522 and within-class 0.555 (currently 0.629 / 0.539 / 0.560).

Apply the degraded rule in `r6_class_aware_heldout_probes.py` (and the report builder), regenerate.
Then check every report and number store the paper draws on (r5, r6, r7, r7b, r8, r9) for any cell that still includes a degraded run. Write `results/paperB/r10/DEGRADED_AUDIT.md`: file, cell, run, fixed or not affected.

## GPU items

### G1: shift 50 %, seeds 72 and 82 (4 runs, ≈ 6 GPU-h)

R1 design unchanged (`train_r6_dose.py`, mode three, recipe current, shift 0.5), λ ∈ {0.5, 2}. The 50 % cell then has five seeds like the 0 % and 100 % cells. Afterwards rerun, with the new runs included: the R1 report (Table tab:shift, tab:shiftfull), the leakage-matched regression (`r6_r1_leakage_matched.py`; report n runs, n clusters and n runs in the R1 cells), the class-aware probes, the clinical post hoc for the R1 cells (`r5/posthoc/clinical`) and the R9 E1 report (shift references).

### G2: pad_adv 50 % at λ = 0 (3 runs, ≈ 4 GPU-h)

The T2 design, same three seeds as the 25 % λ = 0 cell and the 50 % λ = 2 cell, same 802-image subset. Same columns as Table tab:padadv.

### G3: IWDAN with 50-label priors at λ = 0.5 (5 runs, ≈ 7.6 GPU-h)

As the R6 R2-prior n50 arm at λ = 2: same 50-image draw per seed, seeds 42–82. Same columns as Table tab:prior, plus prior error (TV) and the three class-aware probes.

## CPU / evaluation items

Cached features first; forward pass only where features are missing, ≈ 0.1 GPU-h per run.

### C1: within-class column at λ = 0.5 for tab:prior

Class-aware probes (a, b, c) for DANN λ = 0.5 (R1 shift 100 %), IWDAN λ = 0.5, IWDAN true priors λ = 0.5, and the G3 runs.

### C2: tab:dial: MED-NODE, SD-198, SCIN

Mahalanobis AUROC (seed mean ± sd, pooled CI, k below 0.5, Rule A) for every λ and seed in the table.

### C3: tab:unseen: DINOv2 dual rows and the missing A4 cells

DINOv2 from r7/t1 (λ = 0 and 2); A4 for every set in the table that has no A4 value yet.

### C4: SCIN in tab:shift, tab:sixclass, tab:splits

Same columns as the other sets in each table.

### C5: tab:single-dino: MED-NODE, SD-198, SCIN for the R7 T1b runs.

### C6: Fig. 1b from data

`paper/midl/figures/fig_mechanism.py` hard-codes the panel (b) values. Read them from `results/paperB/r5/posthoc/clinical_t1last/summary.csv` (MEL rows, `share_pred_nv_conf09` with CI) and re-render. Panel (a) already reads the run summaries and picks up the G1 runs on re-render.

## Not in R10

Fitzpatrick17k as a second target, more MK-MMD seeds, patient-level IDs for MED-NODE / SD-198 (not available), a second modality.

## Outputs

`results/paperB/r10/REPORT.md`: one section per item with the table rows in the paper's LaTeX format, the regenerated regression table, the degraded audit, GPU and CPU hours, commit hashes. List every paper number that changed, as old → new with its table or section.


## Deviations

- R1 / E1 report regenerate used `--no_boot` (seed means ± sd only) after full bootstrap stalled >40 min on the finalize CPU job; Rule A CIs for new shift-50% cells can be recomputed later with boots.
- λ=2 n50 `summary.json` files were missing on disk; rebuilt from cached features (`rebuilt_from_features`) so class-aware / inventory stay complete. Locked prior verdicts unchanged.
- R2 SCIN report regenerate used `--no_boot` (same stall risk as R1 boots).
- Clinical post hoc inventory extended to R1 shift ∈ {0,50,100} at λ=2 (was {0,100}).
- R10b (deadline 10 Nov 2026): fill bootstrap CIs skipped under `--no_boot`, plus A4 CIs and SCIN on tab:splits. Each of B1–B4 runs as its own CPU job with wall time ≫ 40 min so a slow bootstrap cannot block the others. B1 restricts boots to shift 50% cells (λ ∈ {0.5, 2}, all R1 OOD sets including SCIN) and keeps other cells' CIs from the last full run; seed lists checked against {42,52,62,72,82}. E1 does not reference shift 50% (only 0%/100%), so no E1 shift-50 boot. B2 = full R2 SCIN report with boots. B3 = A4 rows of tab:unseen (λ as in `r10/C3_unseen.json`, currently {0, 2}). B4 = SCIN on split_stability splits {7,123,2027} × λ ∈ {0,2}; OOD resampling unit = P7a `group` (same as REPORT_SCIN / other SCIN cells), not image-level.
