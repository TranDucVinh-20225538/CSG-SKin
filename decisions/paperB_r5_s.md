# Paper B R5 / S1–S3 — sign-flipped Mahalanobis, threshold view, score conventions

Committed before the S1/S2 features are extracted or any S1/S2 number is computed. Fixed now; not changed after
results are seen. Deviations are appended at the end. CPU analysis on cached features; the two missing feature sets
(ISIC validation, BCN↔HAM) need a short GPU extraction (user decision 2026-10-15: GPU, ahead of the P8 queue).

## Questions

- S1. Does flipping the sign of the Mahalanobis score rescue the detector? Per OOD set and λ; per class (nevus vs
  others); and with one flip direction chosen on pad_heldout at λ = 2, then applied unchanged to every other set and λ.
- S2. At a threshold that accepts 95% of ID validation images, what fraction of OOD images is accepted (FPR@95TPR),
  for the original and the flipped score, per set and λ; and with the threshold calibrated at λ = 0 and carried to
  λ > 0 (a monitor calibrated before DA is enabled)?
- S3. One line in the methods notes on the AUROC and score sign conventions.

## Runs and sets

Lesion-level: the 35 Table 1 runs (`results/paperB/r2/item4/features/runB_orth1_ladv*`, 7 λ × 5 seeds). OOD sets and
resample units: pad_heldout (716, patient), Fitzpatrick17k (3,887, P13 `cluster_union`), MED-NODE (170, image), SD-198
(6,584, image; anti-conservative, no patient IDs), SCIN (10,406, case). ID test = lesion-level ISIC test (5,067, lesion);
ID validation = lesion-level ISIC val (4,053), new feature file `isic_val.npz` per run (same extraction path as R2 item 4).

BCN↔HAM (user decision 2026-10-15: existing runs as they are): phase16 B1 (ID = BCN, OOD = HAM held-out) and B1-rev
(ID = HAM, OOD = BCN held-out), EfficientNet-B3 single-encoder DANN at 448 px, λ ∈ {0, 0.1, 0.25, 0.5, 1, 2, 4, 8}; 3
seeds (42/52/62) at λ ∈ {0, 0.25, 1}, 1 seed (42) elsewhere, 14 runs per direction. This is not the 7 λ × 5 seed
lesion design; single-seed cells carry no seed s.d. Features (z = the run's L2-normalised embedding, logits, labels,
lesion_id) re-extracted for id_train / id_val / id_test / ood with the phase16 evaluation path (`collect_all`,
`build_eval_tf(448)`); the stored `ood_auroc_maha.unrestricted` of each run is recomputed as a pipeline check and the
absolute difference reported. Resample unit lesion on both sides.

## Score and AUROC (unchanged from Table 1)

Mahalanobis = minimum over the 8 classes of the squared Mahalanobis distance, class means and shared (pooled
within-class) covariance + 1e-3·I fit on ID train only (`bootstrap_ood_ci.maha_scores`); on `z_lesion_norm` for the
lesion runs. AUROC = `sklearn.metrics.roc_auc_score` (rank form for bootstrap draws), ID = 0, OOD = 1, larger score =
more OOD. Flipped score = −score. With ties counted ½, AUROC(−s) = 1 − AUROC(s) exactly; the script asserts this per
run, so "per-set flip" is the oracle direction max(A, 1 − A) chosen on the test data itself (an upper bound, reported
as such).

## S1 outputs

Per (set, λ): original AUROC, flipped AUROC (mean ± s.d. over seeds, PREREGISTER pooled CI of the seed mean; flipped
CI = 1 − reversed original CI); oracle per-set direction; transferred direction.

Transferred direction: d = flip if the pad_heldout λ = 2 Mahalanobis mean AUROC < 0.5, else keep; d is applied to every
set and λ including BCN↔HAM (a different model family; reported in a separate block). Reported: AUROC under d per cell,
and the count of cells where d gives AUROC < 0.5.

Per class (nevus vs others): class-matched — ID NV vs OOD NV, and ID non-NV vs OOD non-NV (OOD class from the mapped
label: pad_heldout ISIC-index labels; Fitzpatrick17k `congenital nevus` → NV and the P7(a) skin-type map for others;
MED-NODE / SD-198 / SCIN the P7(a) `mapped` column; unmapped OOD images excluded from both and counted); also the
OOD-only split (full ID test vs OOD NV; full ID test vs OOD mapped non-NV). Original and flipped, mean ± s.d.; n per
cell; cells with n_OOD < 20 are shown with their n and flagged. BCN↔HAM: both sides labelled (8 classes).

## S2 outputs

Threshold t = `np.quantile(s_val, 0.95)` on the ID validation scores of the same run (accept if s ≤ t); flipped:
t' = `np.quantile(−s_val, 0.95)`, accept if −s ≤ t'. Reported per (set, λ): ID test acceptance at t (achieved TPR),
OOD acceptance at t (FPR@95TPR), both scores; mean ± s.d. over seeds; 95% CI of the seed-mean OOD acceptance with the
PREREGISTER draws over OOD groups (threshold held fixed; validation not resampled). Also, in the CSV only, FPR@95TPR
with t calibrated on ID test (the textbook definition).

Transferred threshold: t0 = mean over the λ = 0 seeds of the per-run validation thresholds (5 seeds lesion-level, 3
seeds BCN↔HAM), applied unchanged to every λ > 0 run; flipped likewise. Reported: ID test acceptance and OOD acceptance
at t0, and the λ = 0 per-seed thresholds (spread). Mahalanobis distances are not on a common scale across runs (the
covariance is refit per run), which is the point of this check.

## Predictions (written before any S1/S2 number exists)

Known inputs: Table 1 / R4 lesion Mahalanobis pad_heldout ≈ 0.81 at λ = 0 and ≈ 0.39–0.45 for λ ≥ 0.5; Fitzpatrick
≈ 0.72 at λ = 0, ≈ 0.38–0.41 for λ ≥ 0.5; P7(a) MED-NODE 0.54 → ≈ 0.29–0.31, SD-198 0.71 → ≈ 0.34–0.38, SCIN 0.73 →
≈ 0.53–0.56; phase16 stored BCN→HAM 0.33–0.43 at every λ, HAM→BCN 0.85 at λ = 0 → ≈ 0.65 at λ ≥ 2.

1. Transferred direction d = flip (pad_heldout λ = 2 mean < 0.5).
2. Under d, λ = 0 is below 0.5 on all five lesion sets and on HAM→BCN (flip destroys the working λ = 0 detector).
3. Under d, λ ≥ 0.5: pad_heldout, Fitzpatrick, MED-NODE, SD-198 between 0.55 and 0.72; SCIN stays below 0.5
   (≈ 0.44–0.48); HAM→BCN below 0.5 (≈ 0.30–0.35); BCN→HAM above 0.5 (≈ 0.57–0.68) at every λ including 0.
   So no single direction is above 0.5 on every set at a fixed λ ≥ 0.5; count of λ ≥ 0.5 cells below 0.5 under d ≥ 2 sets.
4. Per class, class-matched NV vs NV: original below 0.5 at λ ≥ 0.5 on pad_heldout, MED-NODE and SD-198 (inversion
   is not only a class-mix effect); non-NV vs non-NV within 0.5 ± 0.1 at λ ≥ 0.5 on pad_heldout and SD-198. Low
   confidence on the non-NV prediction.
5. S2, validation threshold, λ ≥ 0.5: OOD acceptance > 0.85 for both the original and the flipped score on
   pad_heldout, Fitzpatrick, MED-NODE, SD-198 (a flipped score with AUROC ≈ 0.6 still accepts most OOD at 95% ID
   acceptance). At λ = 0, original pad_heldout OOD acceptance between 0.4 and 0.8. ID test acceptance within
   0.95 ± 0.02 for the lesion runs.
6. S2, transferred threshold: |ID test acceptance − 0.95| > 0.05 in at least half of the lesion λ > 0 cells
   (direction not predicted).

## S3

Add to `results/paperB/reviewer_r1/MANUSCRIPT_NOTES.md` (not the .tex): AUROC = `sklearn.metrics.roc_auc_score`, ID = 0,
OOD = 1; Mahalanobis OOD score = minimum over classes of the squared Mahalanobis distance (class means, shared
covariance, fit on ISIC train); MSP score = −max softmax probability; Energy score = −logsumexp(logits) (T = 1); cosine
score = −max cosine similarity to the L2-normalised class-mean prototypes; kNN score = mean Euclidean distance from the
L2-normalised feature to its 50 nearest L2-normalised ISIC-train features. For every detector larger = more OOD.

## Code and outputs

`scripts/r5_s_extract.py` + `slurm/r5_s_extract.sbatch` (2 GPU workers), `scripts/r5_s_flip.py` (CPU). Outputs in
`results/paperB/r5/s_flip/`: per-run JSON (`runs/*.json`), `summary.csv`, `REPORT.md` with LaTeX rows.

## Deviations

(none yet)
