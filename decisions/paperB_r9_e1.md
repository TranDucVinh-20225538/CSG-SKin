# Paper B R9 / E1 — six-class control

Committed before any E1 run. Fixed now; not changed after results are seen. Deviations are appended at the end.

## Why

At shift 0 % (R1) the ISIC adversary stream has PAD's class mix, which gives DF and VASC weight 0. The intervention therefore changes class support as well as class priors. E1 changes only the support.

## Design

As R1 (`scripts/train_r6_dose.py`, mode three, recipe current), with one new adversary mix: π_6cls = π_ISIC with DF and VASC set to 0, renormalised over the six shared classes (MEL 0.182, NV 0.518, BCC 0.134, AK 0.035, BKL 0.106, SCC 0.025; check against the run config). TV to π_ISIC 0.019, TV to π_PAD 0.579. Classifier stream unchanged (all eight classes).

λ ∈ {0.5, 2} × seeds 42, 52, 62, 72, 82 = 10 runs (≈ 21 GPU-h).
References, not rerun: R1 shift 100 % and shift 0 % at the same λ and seeds (`results/paperB/r6/r1/cur/`).
Store π_6cls in each run's config. PAD labels are not used at all in E1.

Full R1 protocol: Mahalanobis (primary), kNN, cosine, MSP, energy on pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN; Rule A; leakage V1 and unweighted leakage; ID balanced accuracy; within-class probe as in `r6_class_aware_heldout_probes.py`; nevus-nearest share.
Paired by seed vs shift 100 % and vs shift 0 % on each set: mean difference, seed-bootstrap 95 % CI (2,000 draws, rng 20261005).

## Reading rules

(λ = 2 decides; λ = 0.5 is reported next to it). Labelled unseen sets = Fitzpatrick17k, MED-NODE, SD-198. For each set, recovery = (six-class − shift 100 %) / (shift 0 % − shift 100 %), from the seed means.

- **Support not the driver:** Rule A met on ≥ 2 of the 3 labelled unseen sets and recovery < 0.25 on each of those sets.
- **Support explains part:** recovery ≥ 0.25 on ≥ 2 of the 3 sets, with Rule A still met on at least one of them.
- **Support explains the effect:** recovery ≥ 0.5 on ≥ 2 of the 3 sets and Rule A met on none.
- **Anything else:** reported as found; it settles neither reading.

Report the case computed by the rule first, then the table. pad_heldout is reported but not counted (as R1).

## Shared rules (R7 / R8)

Lesion-disjoint ISIC split (seed 42), pad_adv / pad_heldout, last.ckpt, held-out leakage probe V1 as Table 1.
Degraded = ID test balanced accuracy < 0.25: report the run, exclude it from means (as R1).
Report every run, including failed or degraded ones. No pass/fail wording beyond the rules above.

## Outputs

`results/paperB/r9/e1/`: runs, summary, REPORT.md with case first; also folded into `results/paperB/r9/REPORT.md`.

## Deviations

(none yet)
