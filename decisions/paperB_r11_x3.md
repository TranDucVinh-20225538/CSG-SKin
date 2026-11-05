# Paper B R11 X3 — adversary on a head over a frozen ERM encoder

Locked before first run. Append deviations only; never edit rules after a run starts.

## Question

Does invariance under label shift invert detection in a model as competent as ERM? (The dual encoder is 0.04–0.08 below ERM.)

## Design

Frozen ERM EfficientNet-B3 (the X2 M2 checkpoints, five seeds). Cache features for ISIC train/val/test, pad_adv, pad_heldout and the four OOD sets once (no augmentation; state this). Trainable head: 2-layer MLP to the dual model's latent size, classifier on the head output, domain adversary with gradient reversal on the head output, same adversary architecture and schedule as the main recipe, 40 epochs.

- λ ∈ {0, 0.5, 2, 8} × seeds 42–82 at shift 100 %.
- Label-shift control: λ = 2 with the adversary's ISIC stream resampled to PAD's mix (shift 0 %, as R1 dose_mix).
- Per run: ID balanced accuracy, V1 leakage, all detectors and sets, Rule A.

## Reading rules (λ = 2 at shift 100 % decides)

Competent = five-seed ID balanced accuracy ≥ ERM reference (0.558) − 0.03. Invariant = V1 ≤ 0.60.

- **Not invariant at any λ:** invariance not reached; settles nothing (Case C). Stop after the λ grid.
- **Competent, invariant, Rule A met on pad_heldout and ≥ 2 of the 3 labelled unseen sets:** inversion at ERM competence (Case A).
- **Competent, invariant, Rule A met on none of the labelled unseen sets:** no inversion at ERM competence (Case B).
- **Not competent:** reported, settles neither.
- In Case A, the shift 0 % control is read as R1: label shift necessary if Rule A is met on none of the labelled unseen sets at shift 0 %.

## Shared R7–R10 rules (copied)

Lesion-disjoint ISIC split (seed 42), pad_adv / pad_heldout, last.ckpt, Mahalanobis class-conditional (primary) plus kNN, cosine, MSP, energy. Held-out leakage probe V1 as Table 1.
Bootstrap: 2,000 draws, rng 20261005; resampling unit patient (pad_heldout), P13 cluster (Fitzpatrick17k), cluster_union (SD-198), image (MED-NODE), case (SCIN).
Rule A = pooled CI upper < 0.5 and ≥ 3 seeds with their own CI upper < 0.5.
Degraded = ID test balanced accuracy < 0.25: report the run, exclude it from every mean, CI and verdict.
Report every run and every number as found, nulls included. No wording beyond the reading rules.

## Deviations

_(append only)_
