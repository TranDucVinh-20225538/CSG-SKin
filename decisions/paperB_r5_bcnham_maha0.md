# Paper B — post hoc: why BCN→HAM Mahalanobis inverts at λ = 0

Committed before `scripts/r5_bcnham_maha0_explain.py`. CPU only; labelled (uses HAM/BCN class labels for
evaluation). Does not change any GPU order or detector fit.

## Question

On BCN→HAM at λ = 0 (3 seeds), Mahalanobis AUROC ≈ 0.369 while kNN ≈ 0.658. Is that label shift (class mix),
within-class dispersion, or both?

## Runs and features

- Phase16 BCN→HAM λ = 0 seeds 42/52/62.
- Features: `results/paperB/r5/s_flip/features/bcn_ham_ladv0_s{42,52,62}/`
  (`id_train` = BCN train, `id_test` = BCN-test, `ood` = HAM-heldout; `z`, `labels`, `group` = lesion_id).
- Detectors fit on BCN train only (same as A1 / phase16): Mahalanobis class-conditional + 1e-3·I, kNN k=50.

## Analyses (fixed now)

1. **Class mix** of HAM-heldout vs BCN-test on this split (same split R6-R5 refers to for BCN→HAM outcomes);
   TV, majority shares. Corpus-level BCN/HAM counts stay in R6-R5.
2. **Per-class AUROC (class-matched):** for each class c, HAM images of class c vs BCN-test images of class c;
   Mahalanobis and kNN. Point mean ± s.d. over 3 seeds; lesion-clustered 95% CI of the seed mean
   (PREREGISTER 2,000 draws, rng 20261005, lesion on both sides).
3. **Reweighted AUROC:** reweight the **HAM-heldout (OOD)** side to the BCN-test class mix; report Mahalanobis
   AUROC with the same lesion-clustered CI. Unweighted Mahalanobis and kNN reported alongside.
4. **Per-class dispersion:** median class-conditional Mahalanobis distance of HAM vs BCN-test images to their
   **own** class centroid (means + shared precision from BCN train).
5. **n per class** on both sides. Classes with **n < 20 on either side** still get a CI but are **excluded from
   the reading**.

## Reading (report both; do not pick one)

- If reweighted Mahalanobis AUROC is ≥ 0.5 **and** class-matched AUROCs (eligible classes) are ≥ 0.5 → label shift.
- If class-matched AUROCs stay < 0.5 after matching → within-class dispersion.
- **Mixed:** reweighting moves AUROC toward 0.5 but not above it, and some eligible class-matched AUROCs are < 0.5
  → **both contribute**; report the share of the gap to 0.5 that reweighting recovers:
  `(AUROC_rw − AUROC_unw) / (0.5 − AUROC_unw)` when AUROC_unw < 0.5.

## Outputs

`results/paperB/r5/audit/bcnham_maha0/`: `class_mix.json`, `per_class.csv`, `dispersion.csv`, `reweight.json`,
`summary.csv`, `REPORT.md`.

## Cost

GPU 0. CPU ≈ 0.5–1 h on 8 CPUs (bootstrap).

## Deviations
