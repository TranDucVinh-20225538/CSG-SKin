# Paper B R6 / R5 — explain HAM↔BCN by label shift

Committed before any R5 script run (no R5 output exists at this commit). Fixed now; deviations appended.

## Question

Report the class distributions of HAM and BCN and the label shift between them (TV, KL) next to ISIC↔PAD and
ISIC↔Fitzpatrick17k (where labels map). Does the size of the shift order the outcomes (HAM/BCN: collapse without
inversion vs ISIC→PAD: inversion)?

## Data (metadata only; code: `scripts/r6_r5_label_shift.py`)

- HAM (10,015) and BCN (12,413): `results/paperB/phase16/bcn_ham_class_counts.json` (ISIC 2019 metadata join).
  HAM: MEL 1113, NV 6705, BCC 514, AK 130, BKL 1099, DF 115, VASC 142, SCC 197.
  BCN: MEL 2857, NV 4206, BCC 2809, AK 737, BKL 1138, DF 124, VASC 111, SCC 431.
- ISIC lesion-level train (source of the Table 1 runs) and all ISIC: master CSV via `load_master` and the lesion split.
- PAD-UFES-20 full / pad_adv / pad_heldout: master CSV + Phase 2 patient split (6 classes: no DF, VASC).
- Fitzpatrick17k: the 3,887 OOD images, strict map melanoma→MEL (83), basal cell carcinoma→BCC (257), squamous cell
  carcinoma→SCC (212), actinic keratosis→AK (24), dermatofibroma→DF (8), congenital nevus→NV (2) — 586 of 3,887 images;
  no BKL counterpart; sensitivity adds granuloma pyogenic→VASC (26). Unmapped images are counted, not used.

## Metrics

On the target's label support: TV = ½Σ|p − q|; KL(q‖p) and KL(p‖q) with ε = 1e-4 added before renormalising; source
majority share p_maj, target share q_maj, ratio. Outcomes copied from existing results (no recomputation):
Mahalanobis OOD AUROC at λ = 0, λ = 2 and the minimum over λ — HAM→BCN (`phase16/b1_rev`, ID = HAM), BCN→HAM
(`phase16/b1`, ID = BCN), ISIC→pad_heldout and ISIC→Fitzpatrick (`r4/r4_results.json`, lesion-level Table 1).

## Pairs reported

HAM→BCN, BCN→HAM, ISIC→PAD (heldout), ISIC→PAD (adversary target), ISIC→Fitzpatrick (mapped), ISIC→Fitzpatrick
(mapped + VASC).

## Design problems stated now

- n = 4 outcome pairs (two directions per family): the ordering question is descriptive; no correlation or test.
- BCN→HAM (ID = BCN) is already below chance at λ = 0 (seed 42: 0.379), so it cannot show an inversion; it is reported
  but it is not an inversion axis.
- The Fitzpatrick mapped subset is 15 % of the OOD images and is dominated by carcinomas; its class mix is not the
  mix the detector sees.

## PREDICTED OUTCOME (written now)

TV(HAM, BCN) ≈ 0.355 (NV 0.67 → 0.34); TV(ISIC lesion train, pad_heldout) ≈ 0.55–0.60 (NV ≈ 0.51 → 0.11);
TV(ISIC, Fitzpatrick mapped) ≈ 0.75–0.85 (NV ≈ 0.5 → < 0.01). Ordering by TV: HAM↔BCN < ISIC→PAD < ISIC→Fitzpatrick, and
the majority ratio q_maj / p_maj: HAM→BCN ≈ 0.5, ISIC→PAD ≈ 0.2, ISIC→Fitzpatrick ≈ 0.01. Outcomes: HAM→BCN collapse
without inversion (0.85 → ≈ 0.68), ISIC→PAD / Fitzpatrick inversion (0.81 / 0.72 → 0.39–0.40). The shift size orders the
outcomes in the predicted direction (larger majority-share drop ↔ inversion), with the n = 4 caveat.

## Outputs

`results/paperB/r6/r5/`: `results.json` (counts, shift metrics, outcomes, mapping), `summary.csv`, `REPORT.md`.

## Cost

GPU 0. CPU < 5 min (metadata).

## Deviations
