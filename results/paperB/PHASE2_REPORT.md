# Paper B — Phase 2 report

Retrain `runB_orth1` (`λ_adv=2`, `λ_orth=1`) with PAD held out of `L_ctx`/`L_adv` at patient level. Five seeds. Job `60446`, COMPLETED.

PAD split (fixed, seed 42): pad_adv 1582 images / 961 patients; pad_heldout 716 / 412. Zero patient overlap, zero `(patient_id, lesion_id)` overlap.

## P2 verdict: FAILED

Prediction: `z_lesion` Mahalanobis on pad_heldout ≈ 0.50, vs pad_adv ≈ 0.41.

| OOD set | `z_lesion` Maha AUROC | `z_context` Maha AUROC |
|---|---|---|
| pad_adv (seen by adversary) | 0.407 ± 0.021 | > 0.9999 |
| pad_heldout (unseen PAD) | 0.427 ± 0.025 | > 0.9996 |
| pad_full | 0.413 ± 0.021 | > 0.9998 |

Holding PAD out moved AUROC by ~0.02. Nothing. ID balanced acc 0.707 ± 0.009.

The phenomenon is confirmed and stronger than predicted: domain-invariance inverts covariate-shift OOD detection, robustly, on PAD the adversary never saw. The proposed mechanism (the encoder memorised these specific PAD images) is refuted. The Phase 2 design could not distinguish image-level memorisation from a general PAD-like mapping — if the latter, held-out PAD behaves identically, which is what we observe.

No mechanism claim is written here. Phase 2.5 is the follow-up (unseen Fitzpatrick17k + class-composition control). Detectors, splits and hyperparameters were not adjusted toward the predicted 0.50.

## Files

- `results/paperB/phase2_pad_holdout/phase2_aggregate.json`
- `results/paperB/phase2_pad_holdout/phase2_per_seed.csv`
