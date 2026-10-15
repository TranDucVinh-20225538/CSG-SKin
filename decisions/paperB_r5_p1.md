# Paper B R5 / P1 — the mechanism analyses on the lesion-level runs

Committed before any P1 run (no P1 output exists at this commit). Fixed now; not changed after results are seen.
Deviations are appended at the end.

## Question

The NV-centroid mechanism (R2 item 2, Phase 13, Phase 3.5 distance family) was shown on the image-level runs. Does it
hold on the 35 lesion-level runs that now carry Table 1: nevus-nearest fraction, AUROC on the nevus-nearest subset vs
the rest, distances, per-class AUROC, decomposition check?

## Data, runs, splits

Cached features of the 35 lesion-level runs, `results/paperB/r2/item4/features/runB_orth1_ladv{0,0p25,0p5,1,2,4,8}_s*`
(λ ∈ {0, 2}: seeds 42/52/62/72/82; other λ: 42–46). Fit = ISIC lesion-level train (16,211); ID = ISIC lesion-level test
(5,067); OOD = pad_heldout (716 images, 412 patients; primary) and Fitzpatrick17k (3,887). No retraining, no feature
extraction.

## Design (code: `scripts/r5_p1_mechanism.py`, reusing `r2_item2_nv_centroid.run_block`)

Representations: `z_lesion_norm` (primary, Table 1 space) and `backbone_raw_lesion` (pre-projection, secondary).
Per run, per representation, per OOD set:
- (a) `run_block` unchanged: Mahalanobis AUROC (pooled covariance + 1e-3·I, fit on ISIC train); nevus-nearest fraction
  of OOD and of ID test (nearest class mean under the same metric); AUROC on the NV-nearest OOD subset and on the rest;
  predicted-NV count; per-class median distance ratios OOD→class c / ID class c→class c.
- (b) Phase 3.5 distance family (`eval_final_gpu` definitions): medians of Mahalanobis, Euclidean nearest-centroid,
  kNN k=50 and feature norm for ID test and OOD, and OOD/ID ratios.
- (c) pad_heldout only (labels map to six classes): per-class AUROC both ways (all ID vs OOD class c; ID class c vs all
  OOD); decomposition: AUROC with ID reweighted to the OOD class mix; identity check Σ_c w_c·AUROC(ID_c, OOD) vs the
  reweighted AUROC (residual reported); NV-only counterfactual (only the NV weight moved to the OOD share) and the
  fraction of the reweighting change it explains.

## Seeds, λ grid, OOD sets

All 7 λ, all 35 runs; pad_heldout and Fitzpatrick17k. Reported cells: every λ; λ = 0 vs 2 in the plain paragraph.

## Gates and controls (flag, never drop)

- NV-nearest fraction on ID test is the control for "OOD pulled to NV": the OOD fraction is read against it.
- Identity residual |Σ w_c·AUROC_c − reweighted| > 0.01 → flagged (would indicate a bookkeeping error).
- Classes with < 20 OOD images (MEL 9 in pad_heldout) → reported with n, flagged "small n".

## CI / bootstrap

Mean ± s.d. over seeds per λ (as R2 item 2). The headline Mahalanobis AUROC CIs are those of R3 A2 (unchanged); P1 adds
no new bootstrap.

## PREDICTED OUTCOME (written now)

- z_lesion_norm, pad_heldout: NV-nearest fraction of OOD rises from 0.30–0.40 at λ = 0 to 0.55–0.65 at λ ≥ 0.5, while
  the ID fraction stays at 0.52–0.54 at every λ (image-level: 0.332 → 0.601; ID 0.53).
- At λ ≥ 0.5 the AUROC on the NV-nearest subset is 0.25–0.35, the rest 0.60–0.70 (image-level λ = 2: 0.282 / 0.644).
- OOD→NV distance ratio falls from ≈ 6 at λ = 0 to 2–3 at λ ≥ 0.5; Mahalanobis and kNN medians of OOD fall below ID
  medians at λ ≥ 0.5; norm ratio near 1.
- Per class: OOD NV vs ID lowest AUROC at λ ≥ 0.5 (< 0.35); reweighting the ID side to the OOD mix lowers the AUROC by
  0.02–0.06 at λ = 2; the NV-only counterfactual explains most (≥ 60 %) of that change.
- Fitzpatrick17k: same direction, NV-nearest fraction 0.50–0.65 at λ ≥ 0.5.
- backbone_raw_lesion: same direction, weaker (AUROC near 0.5 at λ = 2, image-level 0.489).
Alternative I would not be surprised by (≈ 25 %): lesion-level NV-nearest fractions shift less than image-level ones
because the lesion-level ID test has fewer near-duplicate NV images close to the centroid.

## Outputs

`results/paperB/r5/p1/`: `per_run/<tag>.json`, `summary.csv`, `REPORT.md` (LaTeX rows, one plain paragraph per result,
commit hash). `--report_only` regenerates the report.

## Cost

GPU: 0. CPU: ≈ 1.5 h wall on 8 CPUs (35 runs × 2 representations × 2 OOD sets; the 1,536-d backbone dominates;
nearest-centroid search chunked). Basis: R2 item 2 on 31 image-level runs took < 1 h on 8 CPUs for z only.

## Stop rules / budget

Wall-time cap 8 h (sbatch). If a run's feature file is missing it is listed in the report, never imputed.

## Deviations

- 2026-10-15. The first run (job 67557) failed at import: `scripts/r2_item2_nv_centroid.py` line 230 had an unescaped
  apostrophe ("PAD's") inside a single-quoted string (SyntaxError since commit 4f8e7ca). Escaped; no logic change.
