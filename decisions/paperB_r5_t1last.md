# Paper B R5 — Table 1 at last.ckpt (robustness of the inversion to the checkpoint rule)

Written and committed 2026-10-15 22:15 (UTC+7), before any last.ckpt number of a Table 1 run exists.
User request 2026-10-15 22:05: CPU, low priority.

## Question

Does the Table 1 result (Mahalanobis AUROC below 0.5 at λ > 0, leakage, ID competence) depend on the best-val
checkpoint rule? Every one of the 35 Table 1 runs (R2 item 4, lesion-level split; λ ∈ {0, 0.25, 0.5, 1, 2, 4, 8}, five
seeds per λ) is evaluated at `last.ckpt` (end of epoch 40) and reported next to its best-val checkpoint.

## Protocol (fixed)

- Checkpoints: `checkpoints/r2_item4/<run>/last.ckpt` (new CPU re-forward) and the best-val checkpoint (existing
  features in `results/paperB/r2/item4/features/<run>/`, extracted from `best-NN.ckpt`). Same loader
  (`eval_final_gpu.load_csg`), same frames as `train_r2_item4_lesion_split.extract_features` (`isic_train`,
  `isic_test`, `pad_full`, `pad_heldout`, `fitzpatrick17k`), and the same metric code for both checkpoints
  (`r6_common.run_metrics`, detectors fit on ISIC train in z_lesion_norm).
- Columns: leakage on the held-out probe (P6 V1: ISIC test ∪ pad_heldout, 70/30 GroupShuffleSplit by lesion / patient,
  probe seeds 42/52/62, z_lesion_norm), ID balanced accuracy (ISIC test), Mahalanobis AUROC on pad_heldout and on
  Fitzpatrick17k. Also reported: Table 1 pad_full leakage, the other four detectors.
- Bootstrap: PREREGISTER_BOOTSTRAP.json (2000 draws, seed 20261005), `r6_common.bootstrap_cell`, draws shared across
  seeds within a λ cell; ID resampled by lesion; pad_heldout by patient; Fitzpatrick by P13 cluster (`cluster_union`
  in `results/paperB/r5/p13/fitz_clusters.csv`). Rule A as in the code (pooled CI upper bound < 0.5 and ≥ 3 seeds with
  their own CI upper bound < 0.5), for each λ at each checkpoint.
- Consistency check: the best-val pad_heldout Mahalanobis AUROC recomputed here vs the stored item-4 summary value
  (tolerance 2e-3); mismatches are reported, not fixed silently.
- Outputs: `results/paperB/r5/t1last/{per_run/<run>.json, summary.csv, BOOTSTRAP.json, REPORT.md}` with LaTeX rows
  (best-val and last.ckpt side by side, mean ± SD over seeds per λ, plus paired last − best differences).

## Predictions

- P1: at λ ≥ 1, last.ckpt pad_heldout Mahalanobis mean AUROC < 0.5 at every λ, and rule A holds at λ = 2 at both
  checkpoints.
- P2: per λ, |mean(last) − mean(best)| < 0.03 for Mahalanobis pad_heldout and Fitzpatrick.
- P3: per λ, |mean(last) − mean(best)| < 0.02 for held-out leakage and for ID balanced accuracy.
- P4: λ = 0.25 (best-val epochs 6–39, seed 42 at epoch 6) shows the largest best-vs-last difference of the λ > 0 cells.

## Budget

35 CPU re-forwards (≈ 26k images each), estimated 1–1.5 h per run at 8 CPUs, array throttled to 2 concurrent tasks and
not prioritised by the CPU governor: ≈ 20–25 h wall. No GPU.
