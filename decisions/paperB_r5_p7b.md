# Paper B R5 / P7(b) — PAD fully held out of training (adversary on ISIC sites BCN vs HAM)

Committed before any P7(b) run (no P7(b) training, feature extraction or scoring has run at this commit).
Fixed now; not changed after results are seen. Deviations are appended at the end. Evaluation-only unseen sets are in
`decisions/paperB_r5_p7a.md`.

## Question

The main OOD claim uses PAD-UFES-20, which is the adversary's own target domain. If the adversary aligns two ISIC sites
instead and PAD never enters training, does detection on PAD (and Fitzpatrick17k) still fall below chance as λ grows?

## Design (code: `scripts/train_r5_p7b.py`, `scripts/r6_common.py` mode `p7b`)

Three streams of 32 images per step:

1. ISIC classifier stream: full lesion-level ISIC train split (16,211), natural order; classification loss only.
2. BCN stream (domain 0) and 3. HAM stream (domain 1): images of the ISIC train split from each site, drawn uniformly;
   they feed the adversarial head (GRL) and the context-domain head.

Site: the master CSV (`master_metadata_lesion_only_soft.csv`: path, label, domain) has no site column. Site is taken from
the `lesion_id` prefix in `ISIC_2019_Training_Metadata.csv` (`BCN_` / `HAM_` / `MSK*`), as in
`scripts/phase16_bcn_ham_data.py`. All ISIC: BCN 12,413, HAM 10,015, MSK 819, null lesion_id 2,084. MSK and
null-lesion images stay in the classifier stream and are not in the adversary streams.

No PAD image enters any loss, BatchNorm statistics, or checkpoint selection (ISIC val). Therefore the existing λ = 0
runs are not identical (there PAD passes through the encoders' BatchNorm and trains the context-domain head), and
P7(b) has its own λ = 0 runs.

Label shift between the adversary's domains: BCN NV 33.9%, HAM NV 67.0% (all ISIC; total variation 0.355 over the 8
classes). This is reported next to the results (the adversary aligns two domains with different class mixes, but PAD is
not one of them).

Runs: λ ∈ {0, 0.5, 2} × seeds 42/52/62/72/82 = 15 runs. Recipe via `--recipe` (current / p8 / auto); own directory per
recipe.

## OOD sets

All of PAD (2,298 images; primary, bootstrap by patient), pad_heldout (716, for comparison with Table 1),
Fitzpatrick17k (3,887, by image), P7a sets (MED-NODE, SD-198, SCIN) when their CSVs exist.

## Reported

Table 1 columns with OOD AUROC = Mahalanobis on all of PAD (also pad_heldout); leakage as in Table 1 (ISIC test vs PAD
full; here PAD is unseen) and the BCN-vs-HAM probe on ISIC test (pressure on the adversary's own domains); five
detectors per set; rule A bootstrap; per-class AUROC and decomposition check on pad_full and pad_heldout; NV-nearest
fractions; AK recall and NV-predicted share.

## CI / bootstrap, gates

`PREREGISTER_BOOTSTRAP.json` unchanged (2,000 draws, percentile, shared draws across seeds; fresh generator seeded
20261005 per cell and set). ID by lesion group; PAD by patient; Fitzpatrick by image. Rule A applied to pad_full
(and reported for pad_heldout, Fitzpatrick). Flags: adversary not engaged (BCN-vs-HAM probe at λ > 0 not ≥ 0.05 below
λ = 0), no headroom at λ = 0 (Mahalanobis in [0.45, 0.55]).

## Predicted outcome (written now)

- BCN-vs-HAM probe falls with λ (λ = 0 ≈ 0.80–0.90 → λ = 2 ≈ 0.60–0.70).
- Mahalanobis on all of PAD ≥ 0.60 at every λ, drop from λ = 0 to λ = 2 ≤ 0.15, rule A not met; pad_heldout the same.
- Fitzpatrick17k above 0.5 at every λ.
- NV-nearest fraction of PAD changes by < 0.10 between λ = 0 and λ = 2.
Reasoning: the inversion needs the OOD images to be the adversary's target, pulled towards the source mix; here PAD is
nobody's target.

## Outputs

`results/paperB/r5/p7b/<recipe>/`: `runs/p7b_<recipe>_ladv<λ>_s<seed>/`, `features/...`, `summary.csv`, `cells.json`,
`REPORT.md`. Checkpoints `checkpoints/r5_p7b/<recipe>/`.

## GPU-hour estimate and basis

Three-stream batches (96 images per step) assumed ×1.5 the measured 1.6 GPU-h = 2.4 GPU-h per 40-epoch run;
15 runs = **36 GPU-h** (72 GPU-h if P8 selects 80 epochs). Includes evaluation on PAD, Fitzpatrick and the P7a sets.
CPU: report ≈ 1 CPU-h.

## Stop rules and budget caps

Cap 40 GPU-h (75 with an 80-epoch recipe). Out-of-memory at 96 images per step stops the run and is reported.

## Pre-run amendment (user decisions 2026-10-15, before any P7(b) run)

The user first chose to reuse the same-recipe λ = 0 checkpoints, then reversed this after the caveat was raised: those
checkpoints were trained with PAD as the second domain, so PAD images reached their BatchNorm statistics and
context-domain branch, and a reused λ = 0 row would not be "PAD fully held out". λ = 0 is therefore trained inside
this design, as originally specified (15 runs, ≈ 36 GPU-h, cap 40). `--reuse_l0` exists but is not used.

## Deviations

