# Paper B R5 / P7(a) — truly unseen clinical sets, evaluation only

Committed before any feature of these sets is extracted or scored (no compute has run on them at this commit).
Fixed now; not changed after results are seen. Deviations are appended at the end.

## Question

The OOD claim rests on PAD-UFES-20 (the adversary's own target domain) and Fitzpatrick17k. Do the existing lesion-level
checkpoints show the same λ pattern on clinical-photo sets that no part of training (classifier or adversary) has seen?

## Sets (obtained / not obtained, licences)

| Set | Status | Images | Groups | Licence / terms |
|---|---|---:|---|---|
| MED-NODE (UMCG; Giotis et al. 2015) | obtained, official page | 170 (70 melanoma, 100 naevus) | none (image = unit) | no licence stated; "kindly invited to use", citation requested |
| SD-198 (Sun et al. ECCV 2016) | obtained, third-party Hugging Face mirror `resyhgerwshshgdfghsdfgh/SD-198` (count 6,584 = published) | 6,584, 198 classes | none (image = unit) | mirror says "license: other / unknown"; original images from DermQuest; research use. Provenance flagged in every table |
| SCIN (Google, Ward et al. 2024) | obtained, public bucket `dx-scin-public-data` | 10,379 images, 5,033 cases | case_id | CC-BY 4.0. Not in the work-order list; added as an extra clinical set and reported separately |
| DDI (Stanford) | not obtained | 656 | — | individual registration under the Stanford Research Use Agreement; needs the user to register |
| Derm7pt clinical (SFU) | not obtained | 1,011 clinical | — | CC BY-NC-ND 4.0; password request form with personal details; needs the user |

If DDI / Derm7pt are relayed later, they are run with exactly this file's protocol and appended as a deviation.

## Checkpoints and features

All 35 lesion-level runs (`results/paperB/r2/item4/features/runB_orth1_ladv{0,0p25,0p5,1,2,4,8}_s*`, 7 λ × 5 seeds;
the Table 1 runs). Features with the unchanged R2 item-4 extraction (`extract_features` path:
`fin.load_csg(best_checkpoint)`, `build_val_transform_robust`, `x13.collect_depths`, same REQUIRED keys), one npz per
set per run under the same feature directory (`mednode.npz`, `sd198.npz`, `scin.npz`). No retraining, no refit of
anything except the detectors (fit on ISIC train only, as in Table 1).

## Detectors and ID side

The five Table 1 / R4 detectors: Mahalanobis class-conditional (Table 1 "OOD AUROC" column), kNN k=50, cosine max on
`z_lesion_norm` (the pre-registered scoring space), MSP and Energy (T=1) on logits. ID = ISIC lesion-level test split
(5,067 images), fit = ISIC train. OOD positive; AUROC as in Table 1.

## Uncertainty and rule A

PREREGISTER_BOOTSTRAP.json unchanged (2,000 draws, percentile 95%, shared draws across seeds within a draw, rng seed
20261005). ID resampled by lesion group (as R3 A2). OOD resample unit: MED-NODE image, SD-198 image (no patient IDs;
anti-conservative, labelled), SCIN case_id. Rule A per set: at λ = 2 the pooled (mean over seeds) CI entirely below
0.5 and ≥ 3 of 5 per-seed CIs entirely below 0.5, Mahalanobis. Reported for every λ, not only λ = 2.

## Per-class AUROC (where labels map) and decomposition check

ISIC classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC. Per-class AUROC = ID test (all classes) vs the OOD images of one
mapped class. Mapping fixed now:

- MED-NODE: melanoma → MEL (70), naevus → NV (100).
- SD-198 (1,057 images map; the other 175 classes are unmapped and enter only the all-images AUROC):
  NV = Blue, Compound, Congenital, Dysplastic, Halo, Junction, Nail, Nevus Incipiens, Nevus Spilus (435);
  MEL = Malignant Melanoma, Lentigo Maligna Melanoma (115); BCC (61); AK = Actinic solar damage (Actinic Keratosis) (61);
  BKL = Seborrheic Keratosis, Solar Lentigo (74); DF (61); VASC = Angioma, Pyogenic Granuloma, Strawberry Hemangioma
  (129); SCC = Bowen's Disease, Keratoacanthoma (121). Non-melanocytic "nevi" (epidermal, sebaceous, comedonicus,
  Becker's) are not NV.
- SCIN: a case maps when its top weighted dermatologist label has weight ≥ 0.5 and is one of: Melanocytic Nevus → NV,
  Melanoma → MEL, Basal Cell Carcinoma → BCC, Actinic Keratosis → AK, Seborrheic Keratosis → BKL, SCC/SCCIS → SCC,
  Dermatofibroma → DF, Hemangioma / Pyogenic granuloma → VASC. Counts reported; classes with < 20 images are reported
  with n and no CI. From the label file alone (no image read): mapped cases are SCC 8, AK 7, BCC 5, VASC 6, DF 3, NV 3,
  MEL 0, so SCIN contributes the all-images AUROC and per-class n only; no SCIN decomposition check.

Decomposition check as in R4: reweight the OOD side to the ID class mix over mapped classes and report the all-class
AUROC next to the reweighted one; also the nevus-nearest fraction (share of OOD features whose nearest ISIC class mean
under the Mahalanobis metric is NV) per λ.

## Gates and flags (flag, never drop)

Near chance: Mahalanobis AUROC in [0.45, 0.55] at λ = 0 on a set → that set is flagged "no headroom at λ = 0" (an
inversion cannot be read there). n < 200 OOD images (MED-NODE) → flagged "small n".

## Predictions (written before any score exists)

- λ = 0: Mahalanobis AUROC above chance on all three sets, in the range of Fitzpatrick17k (0.717): MED-NODE 0.60–0.80,
  SD-198 0.65–0.80, SCIN 0.70–0.85 (SCIN is mostly non-lesion inflammatory conditions, far from ISIC).
- λ ≥ 0.5: Mahalanobis mean AUROC below 0.5 on SD-198 and SCIN (0.35–0.48), as on Fitzpatrick (0.375–0.412).
  MED-NODE below 0.5 in mean, CI may straddle 0.5 (n = 170).
- Rule A at λ = 2: met on SD-198 and SCIN, not met on MED-NODE (n).
- Per class: the NV subset has the lowest AUROC at λ ≥ 0.5 (OOD pulled to the NV centroid); nevus-nearest fraction
  rises with λ on every set.
- The five detectors move together (range at λ = 2 < 0.10), as on pad_heldout.

## Outputs

`results/paperB/r5/p7a/`: per-run-per-set JSON, `summary.csv`, `REPORT.md` (one paragraph per set, LaTeX rows),
commit hash. Compute starts only after the R7 A5 jobs have finished (user instruction).

## Deviations

(none yet)
