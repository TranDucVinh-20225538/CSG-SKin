# Paper B R5 / P3 — frozen ImageNet monitor vs the lesion-level model

Committed before any P3 run (no P3 output exists at this commit). Fixed now; deviations appended at the end.

## Question

R2 item 1 compared a frozen ImageNet ResNet-50 monitor with the model on the image-level split. On the lesion-level
split that now carries Table 1: what are the frozen monitor's Mahalanobis AUROCs on pad_heldout and Fitzpatrick17k, next
to the model's at λ = 0 and λ = 2, on the same ID test?

## Data

- Frozen side, source `mapped` (no extraction; primary at first run): the R2 item-1 features
  (`results/paperB/r2/features/imagenet_resnet50/{isic_train,isic_test,pad_heldout,fitzpatrick17k}.npz`, torchvision
  IMAGENET1K_V1, fc removed, 2,048-d) re-indexed by image path (`isic_image_split_with_lesion.csv`: split, row) onto the
  lesion-level frames. Those features exist only for image-level train and test; lesion-level images that were in the
  image-level validation split have no feature and are dropped from the frozen fit and the frozen ID test. Coverage is
  reported; the model is scored on the full ID test and on the same covered subset (matched comparison).
- Frozen side, source `full` (after the GPU-light extraction `slurm/r5_frozen_extract.sbatch`, index 0): the same
  encoder on the complete lesion-level frames (`scripts/r5_frozen_frames.py` → `scripts/r5_frozen_extract.py`). When it
  exists it is the primary frozen row; `mapped` stays as a check (the two must agree within ±0.01 on the covered set).
- Model side: `z_lesion_norm` of the 35 lesion-level runs; λ = 0 and 2 reported, all λ listed.

## Detectors

Mahalanobis class-conditional (pooled covariance + 1e-3·I, `r2_common.Maha`; pre-registered), Ledoit–Wolf Mahalanobis
(sensitivity; 2,048-d), kNN k=50 on L2-normalised features (`r2_common.KNN`). Fit on ISIC lesion-level train only.

## CI / bootstrap

Frozen monitor (one deterministic encoder): percentile 95% CI, 2,000 draws (rng 20261005), ID resampled by lesion
group, pad_heldout by patient, Fitzpatrick17k by image and — once P13 has run — by near-duplicate cluster
(`cluster_phash`, `cluster_union`). Model: PREREGISTER pooled bootstrap (shared draws across seeds) at λ = 0 and 2.

## Target (carried over from R2, not changed)

Mitigation target: frozen-monitor Mahalanobis ≥ 0.90 on pad_heldout and on Fitzpatrick17k. Reported as numbers next to
the target; no pass/fail wording.

## PREDICTED OUTCOME (written now)

- Frozen ResNet-50, Mahalanobis: pad_heldout 0.81–0.86, Fitzpatrick17k 0.97–0.99 (image-level: 0.833 / 0.988);
  kNN: pad_heldout 0.89–0.93. Coverage of the lesion-level ID test by the mapped features ≈ 0.80 (image-level test +
  train share of the lesion-level test).
- Model z_lesion_norm: λ = 0 0.81 / 0.72; λ = 2 0.40 / 0.40 (Table 1). Frozen > model at λ = 2 by ≥ 0.35 on both sets;
  frozen ≈ model at λ = 0 on pad_heldout (|Δ| ≤ 0.05), frozen > model on Fitzpatrick at λ = 0 by ≥ 0.2.
- Ledoit–Wolf changes frozen Mahalanobis by ≤ 0.02.

## Outputs

`results/paperB/r5/p3/`: `results.json` (frozen cells with CIs, model bootstrap), `summary.csv`, `REPORT.md` (LaTeX rows,
one paragraph per frozen source).

## Cost

GPU: 0 for `mapped`; `full` needs ≈ 0.2 GPU-h (ResNet-50 inference on ≈ 28k images; shared with P12/Q5). CPU ≈ 0.5 h on
8 CPUs (2,048-d Mahalanobis/LW/kNN; 2,000-draw CIs).

## Stop rules

If coverage of the mapped ID test is < 0.6, the mapped row is reported but the `full` extraction is requested before any
conclusion.

## Deviations
