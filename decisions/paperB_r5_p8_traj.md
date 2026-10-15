# Paper B R5 P8 — per-epoch trajectory (claim (c)): when does the Mahalanobis collapse form?

Written and committed 2026-10-15 23:30 (UTC+7), before any trajectory run or per-epoch number exists. User request
2026-10-15 23:14. Descriptive analysis; no pass/fail threshold.

## Availability check (done before writing this file)

- P8 runs trained so far (grid, main λ = 0 s42/s52 reused from the grid, main λ = 2 s42, main λ = 0 s62 running) keep
  only `best-NN.ckpt` and `last.ckpt`: no per-epoch checkpoints.
- Table 1 (R2 item 4, 35 runs): only `best-NN.ckpt` and `last.ckpt`. No old-recipe run can be traced per epoch, so the
  old-recipe arm is not run.
- Main λ = 2 seeds 52/62/72/82 (P8 main tasks 11–14, not yet started) will save per-epoch weights
  (`save_every_epoch`, weights only; training unchanged). They are an optional extra, evaluated only if the user asks
  (≈ 1–1.5 GPU-h each).

## Runs

Two extra runs after P8 main, A4 recipe (latent 64, cls LR × 1, loss balancing, 40 epochs), seed 42, identical to the
P8 main runs except for per-epoch weight saving: `traj_ladv0_s42` (λ = 0) and `traj_ladv2_s42` (λ = 2). P8 stage
`traj` (`scripts/train_r5_p8.py`), `slurm/r5_p8_traj.sbatch`. Outputs in `results/paperB/r5/p8/traj/`, checkpoints
in `checkpoints/r5_p8/traj/`. They do not replace the P8 main seed-42 runs (primary results stay with P8 main,
last.ckpt policy unchanged).

## Per-epoch quantities (epochs 0–39, every epoch)

- Held-out-probe leakage: P6 V1 (ISIC test ∪ pad_heldout, 70/30 GroupShuffleSplit by lesion / patient, probe seeds
  42/52/62, logistic probe of P6) on z_lesion_norm.
- Adversary train accuracy: epoch mean of `train/acc_adv` from the run's lightning log (also GRL α, `train/grl_a`).
- ID balanced accuracy: ISIC test (lesion-level split).
- Mahalanobis AUROC (class-conditional, fit on ISIC train z_lesion_norm, same code as Table 1) on pad_heldout and on
  Fitzpatrick17k.

## Analysis (fixed)

- Plot all curves against epoch for each run (left: Mahalanobis pad_heldout, Mahalanobis Fitzpatrick, held-out leakage;
  right: adversary train accuracy, ID bal acc). Raw per-epoch values, 0.5 reference line.
- Timing on a 3-epoch centred running mean: t10, t50, t90 = first epoch at which a curve has covered 10 / 50 / 90 % of its
  change from epoch 0 to epoch 39.
- Label for each Mahalanobis curve relative to held-out leakage, by t50: Mahalanobis t50 − leakage t50 < −1 epoch →
  "precedes"; > +1 → "follows"; otherwise "tracks". This is a description, not a test.
- Also reported: Pearson correlation of first differences between leakage and each Mahalanobis curve at lags −5..+5
  (lag > 0 = Mahalanobis later), and the lag with the largest |r|.
- λ = 0 is the reference trajectory (same seed, no reversed gradient); its curves are plotted and timed the same way.
- One seed per λ: no CI; the result is stated as a single-seed trajectory.

## Predictions (written before any number)

- λ = 2: held-out leakage falls mostly in the first ~5 epochs (GRL α ≥ 0.99 from epoch 2–3); the pad_heldout
  Mahalanobis AUROC falls below 0.5 later, with t50 more than 1 epoch after the leakage t50 ("follows").
- λ = 2: Fitzpatrick follows the same pattern as pad_heldout but ends higher (P8 main λ = 2 s42 last.ckpt: 0.525 vs
  0.452).
- λ = 0: Mahalanobis on both sets stays above 0.7 at every epoch after epoch 2; leakage stays above 0.8.

## Budget

2 × (≈ 70 min training + ≈ 40 × 1.5 min per-epoch evaluation + ≈ 5 min end-of-run evaluation) ≈ 4.5 GPU-h, more than
the ~3 GPU-h in the request because the per-epoch evaluation forwards ISIC train (16,211 images) for the detector fit
at every epoch. Starts after P8 main (dependency on job 67543).
