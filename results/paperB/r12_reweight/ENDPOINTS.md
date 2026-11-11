# R12 endpoints (frozen before outcome inspection)

Primary detector: class-conditional Mahalanobis on `z_lesion_norm` (fit ISIC train only).
Primary unseen panels: Fitzpatrick17k, MED-NODE, SD-198.
Aligned-domain panel (reported separately): pad_heldout.
Secondary: SCIN and any other eligible panels already in the extract path.

Per eligible cell:
1. Pooled AUROC (ID test vs OOD panel).
2. ID-class-stratified AUROC: each of the fixed 8 ISIC labels vs the **same** OOD panel; unweighted macro over classes present in ID test.
3. Shared-label-subset analyses only with a verified mapping (reported separately if run).
4. FPR@95TPR = fraction of OOD with score ≤ τ where τ = 95th percentile of ID scores (higher score = more OOD) — i.e. OOD accepted as ID.
5. ID balanced accuracy (ISIC test).
6. Held-out domain leakage (V1 GroupShuffleSplit probe) and class-matched leakage (`classmatched_leakage`).
7. Seed mean ± sd; cluster bootstrap CI with documented units (lesion / patient / P13 / P7a group). Bootstrap ≠ between-seed uncertainty.

Checkpoint selection: `last.ckpt` only (never OOD AUROC).
Paired contrast: prior0 − prior100 within seed for Block A λ∈{0.5,2}.

Far-OOD non-skin: unavailable unless already on disk (marked unavailable).
