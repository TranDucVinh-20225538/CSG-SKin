# Paper B R11 X4 — second domain pair, BCN → HAM

Locked before first run. Append deviations only; never edit rules after a run starts.

## Question

Does the inversion appear outside ISIC/PAD?

## Design

Source = BCN_20000 (labelled), adversary target = HAM10000 (unlabelled), both from the ISIC 2019 data used in Appendix E (`scripts/phase16_bcn_ham_data.py`). Lesion-disjoint splits on both (lesion IDs); HAM split by lesion into ham_adv (adversary) and ham_heldout (≥ 20 % of lesions; plays the role of pad_heldout). Main recipe (dual encoder, two-stream), 40 epochs, last.ckpt.

- λ ∈ {0, 2} × seeds 42–82 at shift 100 % (BCN's own mix in the adversary stream).
- λ = 2 × seeds 42–82 at shift 0 % (adversary BCN stream resampled to HAM's class mix; record the TV and any class-support change, as R1).
- OOD sets: ham_heldout, pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN.
- Report TV(BCN, HAM) class priors first.

## Reading rules (λ = 2)

Invariant = V1 ≤ 0.60 (else Case C, settles nothing).

- Rule A met on ≥ 2 of the 3 labelled unseen sets at shift 100 % and on none at shift 0 %: inversion reproduced and label shift necessary in this pair.
- Met at shift 100 % and also at shift 0 %: inversion reproduced, label shift not necessary here.
- Met on fewer than 2 at shift 100 %: not reproduced in this pair.
- ham_heldout and SCIN are reported, not counted.

## Shared R7–R10 rules (copied)

Lesion-disjoint ISIC split (seed 42), pad_adv / pad_heldout, last.ckpt, Mahalanobis class-conditional (primary) plus kNN, cosine, MSP, energy. Held-out leakage probe V1 as Table 1.
Bootstrap: 2,000 draws, rng 20261005; resampling unit patient (pad_heldout), P13 cluster (Fitzpatrick17k), cluster_union (SD-198), image (MED-NODE), case (SCIN).
Rule A = pooled CI upper < 0.5 and ≥ 3 seeds with their own CI upper < 0.5.
Degraded = ID test balanced accuracy < 0.25: report the run, exclude it from every mean, CI and verdict.
Report every run and every number as found, nulls included. No wording beyond the reading rules.

## Deviations

_(append only)_
