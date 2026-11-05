# Paper B R11 X2 — OOD monitor on features the adversary did not touch

Locked before first run. Append deviations only; never edit rules after a run starts.

## Question

Can a DANN model be monitored safely by scoring on a separate feature extractor?

## Design

Classifier: the Table 1 dual model at λ ∈ {0, 0.5, 2}, five seeds. Monitors, each fitted on ISIC train: (M1) the model's own latent (as in the paper, reference); (M2) single-encoder ERM EfficientNet-B3, same split, the five seeds of r5/p8/erm_ref (same seed as the classifier); (M3) ImageNet-pretrained EfficientNet-B3, frozen, penultimate features, class-conditional Mahalanobis with ISIC train labels. If the M2 checkpoints are not on disk, retrain them (5 runs, ≈ 7 GPU-h) and say so in the estimate. Per monitor, λ and set: AUROC, pooled CI, k, Rule A, FPR@95TPR; and on Fitzpatrick17k and SD-198 MEL images, the share of melanomas predicted NV with confidence > 0.9 by the λ = 2 classifier that the monitor flags at the 95 %-TPR threshold (as r5/posthoc/clinical).

## Reading rule

Report as found. A monitor "avoids the inversion" at λ = 2 if Rule A is met on no labelled set and its AUROC is within 0.03 of its own λ = 0 value on each labelled set (M3 has no λ dependence by construction; then only the first condition applies).

## Shared R7–R10 rules (copied)

Lesion-disjoint ISIC split (seed 42), pad_adv / pad_heldout, last.ckpt, Mahalanobis class-conditional (primary) plus kNN, cosine, MSP, energy. Held-out leakage probe V1 as Table 1.
Bootstrap: 2,000 draws, rng 20261005; resampling unit patient (pad_heldout), P13 cluster (Fitzpatrick17k), cluster_union (SD-198), image (MED-NODE), case (SCIN).
Rule A = pooled CI upper < 0.5 and ≥ 3 seeds with their own CI upper < 0.5.
Degraded = ID test balanced accuracy < 0.25: report the run, exclude it from every mean, CI and verdict.
Report every run and every number as found, nulls included. No wording beyond the reading rules.

## Pre-run inventory

- M2 checkpoints **present**: `checkpoints/r5_p8/erm_ref/single_dann_ladv0_s{42,52,62,72,82}/last.ckpt` — no retrain planned.
- M2 features missing mednode/sd198/scin (forward required).
- Table 1 λ ∈ {0, 2} seeds {42,52,62,72,82}; λ = 0.5 seeds {42,43,44,45,46}. M2 erm_ref seeds {42,52,62,72,82}. **Only seed 42 overlaps at λ = 0.5** for “same seed as the classifier”.

## Deviations

_(append only)_
- Seed pairing at λ = 0.5: only seed 42 has a matching M2 checkpoint; report n for that cell as found.
- M3 has no skin classifier head: MSP / Energy not scored (null); Mahalanobis / kNN / cosine only.
- SD-198 bootstrap unit = P7a `group` column (image-level, as r5/p7a); decision text "cluster_union (SD-198)" follows shared R11 boilerplate (Fitz uses P13 `cluster_union`).
