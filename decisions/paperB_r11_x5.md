# Paper B R11 X5 — semi-synthetic domain with controlled label shift

Locked before first run. Append deviations only; never edit rules after a run starts.

## Question

What is the smallest setting that produces the inversion? (The Gaussian toy did not.)

## Design

Frozen ERM features (X3 cache). Synthetic target domain = ISIC train images under one fixed "smartphone" transform (Gaussian blur σ = 1.5, colour cast +10 % R / −10 % B, JPEG quality 40), forward-passed through the frozen encoder. Its class mix is set by resampling to five levels between ISIC's and PAD's mix, t ∈ {0, 0.25, 0.5, 0.75, 1} (t = 0: ISIC's own mix, no label shift between source and synthetic domain; t = 1: PAD's mix, the full ISIC/PAD label shift). Head and adversary as X3, λ ∈ {0, 2}, three seeds per cell. Evaluate on the real OOD sets and on the transformed ISIC test set.

## Reading rule

Report the dose–response (AUROC vs t at λ = 2, with the λ = 0 reference) as found; "reproduces the inversion" if Rule A is met on ≥ 2 of the 3 labelled unseen sets at t = 1 and on none at t = 0.

## Shared R7–R10 rules (copied)

Lesion-disjoint ISIC split (seed 42), pad_adv / pad_heldout, last.ckpt, Mahalanobis class-conditional (primary) plus kNN, cosine, MSP, energy. Held-out leakage probe V1 as Table 1.
Bootstrap: 2,000 draws, rng 20261005; resampling unit patient (pad_heldout), P13 cluster (Fitzpatrick17k), cluster_union (SD-198), image (MED-NODE), case (SCIN).
Rule A = pooled CI upper < 0.5 and ≥ 3 seeds with their own CI upper < 0.5.
Degraded = ID test balanced accuracy < 0.25: report the run, exclude it from every mean, CI and verdict.
Report every run and every number as found, nulls included. No wording beyond the reading rules.

## Deviations

_(append only)_
