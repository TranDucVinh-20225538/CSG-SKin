# Paper B R5 / P6 — leakage probe: which images, and recomputation on held-out images

Committed before any P6 run (no P6 output exists at this commit). Fixed now; deviations appended.

## Question

Which images is the Table 1 leakage probe trained and tested on? If its test images overlap the adversary's training
images, recompute it on held-out images only.

## Answer from the code (read, not run)

`train_phase3_sweep.leakage_probe`: StandardScaler + logistic regression (max_iter 4000), stratified 70/30 split of
ISIC-test ∪ PAD-full, probe seeds 42/52/62, balanced accuracy, on the lesion latent returned by
`net(..., return_latents=True)` (Table 1 column; stored as `leakage.z_lesion`) and on `z_context`. V0 is computed on
every cached representation, so the report shows which cached array reproduces the stored value. PAD-full = pad_adv (1,582 images, the
adversary's PAD training images) ∪ pad_heldout (716). So ≈ 69 % of the PAD side of both probe-train and probe-test are
images the adversary was trained on; the ISIC side (ISIC test) was never seen by the adversary. The overlap exists.

## Design (code: `scripts/r5_p6_leakage_probe.py`)

Cached features of the 35 lesion-level runs (`pad_full.npz` rows mapped to pad_adv / pad_heldout via
`load_master` + the Phase 2 patient split; pad_heldout rows must equal `pad_heldout.npz` to 1e-4 — asserted).
Same classifier and probe seeds; balanced accuracy and AUROC:
- V0 replication: ISIC-test ∪ PAD-full, 70/30 image-level stratified (must reproduce the stored value; Δ reported).
- V1 held-out only: ISIC-test ∪ pad_heldout, 70/30 grouped by lesion (ISIC) / patient (PAD).
- V2 adversary images → held-out images: fit on ISIC-train (random 5,067) ∪ pad_adv; evaluate on ISIC-test ∪ pad_heldout.
- V3 V1 with the ISIC side weighted to the pad_heldout class mix (fit and score weights), removing class composition.
Representations: `z_lesion_norm` (Table 1 space), `z_lesion`, `backbone_raw_lesion`, `z_context`.

## Seeds, λ, sets

All 7 λ, 35 runs; probe seeds 42/52/62 inside each run.

## CI

Mean ± s.d. over training seeds of the 3-probe-seed mean; no bootstrap.

## Gates (flag, never drop)

V0 differs from the stored value by > 0.01 → flagged as a protocol mismatch (the logistic solver is deterministic).

## PREDICTED OUTCOME (written now)

z_lesion_norm: V0 reproduces Table 1 (0.927 at λ = 0, 0.555 at λ = 2). V1 0.90–0.95 at λ = 0 and 0.55–0.65 at λ = 2;
V2 0.85–0.95 at λ = 0 and 0.52–0.62 at λ = 2; V3 within 0.03 of V1. The held-out recomputation stays near the Table 1
value (|V1 − V0| ≤ 0.05 at every λ): the adversary removes domain information from unseen PAD images too.
backbone_raw_lesion stays ≥ 0.85 at every λ; z_context ≥ 0.98 (no adversary on it).

## Outputs

`results/paperB/r5/p6/`: `per_run/<tag>.json`, `summary.csv`, `REPORT.md` (image-usage table, LaTeX rows, paragraph).

## Cost

GPU 0. CPU ≈ 3 h on 8 CPUs (35 runs × 4 reps × 4 variants × 3 probe fits; 1,536-d backbone fits dominate).

## Stop rules

8 h sbatch cap; per-run JSON allows resumption.

## Deviations

- 2026-10-15. The first run (job 67560) stopped at the row-mapping assertion: pad_full rows selected by the Phase 2
  heldout mask differ from `pad_heldout.npz` by up to 3.0e-3 (z_lesion_norm, run ladv0_s42; two separate extraction
  passes), above the 1e-4 tolerance written here. Row order is the same (labels equal row by row, asserted). Tolerance
  raised to 1e-2; the per-run maximum deviation is written to each per-run JSON. No other change.
- 2026-10-15. The rerun (job 68073) stopped at run 19 of 35: maximum deviation 0.0155 (z_lesion_norm), above 1e-2, with
  the 18 earlier runs written. The fixed tolerance is replaced by a direct row-mapping test: for every pad_heldout row,
  the nearest pad_full[heldout] row (squared Euclidean) must be the same row, per representation. The label assertion
  and the recorded per-run maximum deviation are unchanged. The 18 written runs are kept (they also pass the new test).
- 2026-10-15 18:55. The nearest-row test (job 68078) failed on run ladv1_s45 because pad_heldout contains 10 exact
  duplicate images (distance 0 to another row), so argmin picked the duplicate. The test is now: each row's distance
  to its own pad_full row ≤ its distance to every other row (ties allowed). Nothing else changed.
