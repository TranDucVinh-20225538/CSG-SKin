# Paper B R6 R2 — SCIN secondary score (separate report)

Committed before the CPU SCIN score / bootstrap job. Fixed now; deviations appended.

## Why separate

SCIN is already extracted for completed R2 cells, but only **64 / 10,406 images (0.615%)** map into the ISIC 8-class label set (`results/paperB/r5/p7a/sets/scin.csv` / P7a `mapped_counts`). It is **not** a primary set for the R2 estimated-vs-oracle fix reading (locked: Fitzpatrick17k, MED-NODE, SD-198). pad_heldout stays reported separately as before.

**No new dermatology datasets** for this item.

## What is scored

- **Checkpoints:** every R2 `last.ckpt` under `checkpoints/r6_r2/cur/` that exists when the job runs — first wave: IWDAN / IWCDAN / IWDAN_ORACLE at λ = 2 (5 seeds) and any completed CDAN λ = 2 seeds; **later** remaining CDAN λ = 2 and all λ = 0.5 cells when the held array (`68562_[16-39]`) finishes (re-run the same script).
- **Baselines (same as main R2 report):** R1 shift 100% RECIPE=`current` `last.ckpt` at λ ∈ {0, 0.5, 2} (DANN / λ = 0 references).
- **Detectors:** same five as R1 / R2 (`mahalanobis_classcond`, `knn_k50`, `cosine_max`, `MSP`, `Energy_T1`), fit on ISIC train only.
- **Cluster CIs:** PREREGISTER (`results/paperB/PREREGISTER_BOOTSTRAP.json`: 2,000 draws, rng 20261005). ID by **lesion**; SCIN by P7a **`group`** (case-level; 5,033 cases). Rule A reported per cell but **does not** enter the R2 fix reading.

## Outputs

- `results/paperB/r6/r2/cur/REPORT_SCIN.md` — detector table with case-level cluster CIs; paired Δ vs DANN and gap recovered on SCIN only (descriptive).
- `results/paperB/r6/r2/cur/scin_report.json` — machine-readable boots + points.
- Main `REPORT.md` keeps Fitz/MED/SD/pad; SCIN rows there are not the CI source of truth (see deviation).

## Predictions (descriptive only; not a gate)

At λ = 2: SCIN Mahalanobis stays closer to chance than Fitz/MED/SD under DANN (R1 shift100 ≈ 0.52); IWDAN may sit slightly above DANN; oracle similar order of magnitude. No pass/fail on SCIN.

## Deviations

- 2026-10-17 ~22:14 (UTC+7), before CPU SCIN report job. Locked separate SCIN report + case-level CIs; no new derm datasets; remaining R2 cells re-scored with the same script when released.
