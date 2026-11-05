# Paper B R11 X1 — OOD split by label space

Locked before first scoring run. Append deviations only; never edit rules after a run starts.

## Question

Does the inversion concern lesions of ISIC classes only? SCIN (0.6 % ISIC-mappable) does not invert.

## Design

Fitzpatrick17k frame `r5/frozen/frames/fitzpatrick17k.csv` (3,887 images), condition names from `phase11_third_domain/fitzpatrick17k.csv`. Mapping, fixed here:

- **in-ISIC:** melanoma, malignant melanoma, superficial spreading melanoma ssm, lentigo maligna (MEL); nevocytic nevus, congenital nevus, halo nevus (NV); basal cell carcinoma, basal cell carcinoma morpheiform, solid cystic basal cell carcinoma (BCC); squamous cell carcinoma (SCC); actinic keratosis (AK); seborrheic keratosis (BKL); dermatofibroma (DF); pyogenic granuloma (VASC);
- **ambiguous**, reported separately and in neither subset: epidermal nevus, naevus comedonicus, becker nevus, nevus sebaceous of jadassohn, kaposi sarcoma;
- **out-of-ISIC:** every other condition.

Report the counts of each subset in the frame first. Repeat on SD-198 with the same rule applied to its class names (write the SD-198 mapping into the decisions file before scoring, from class names only). For λ ∈ {0, 0.25, 0.5, 1, 2, 4, 8}: Mahalanobis AUROC per subset (ID = ISIC test), seed mean, pooled CI, k, Rule A; and the paired (by seed) difference in-ISIC − out-of-ISIC with seed-bootstrap CI.

## Reading rules (λ = 2 decides)

On each of Fitzpatrick17k and SD-198:

- **Label space carries it:** Rule A met on in-ISIC and not met on out-of-ISIC, and the paired difference CI lies below 0.
- **Not label space:** Rule A met on both subsets.
- **Anything else:** reported as found.

## Shared R7–R10 rules (copied)

Lesion-disjoint ISIC split (seed 42), pad_adv / pad_heldout, last.ckpt, Mahalanobis class-conditional (primary) plus kNN, cosine, MSP, energy. Held-out leakage probe V1 as Table 1.
Bootstrap: 2,000 draws, rng 20261005; resampling unit patient (pad_heldout), P13 cluster (Fitzpatrick17k), cluster_union (SD-198), image (MED-NODE), case (SCIN).
Rule A = pooled CI upper < 0.5 and ≥ 3 seeds with their own CI upper < 0.5.
Degraded = ID test balanced accuracy < 0.25: report the run, exclude it from every mean, CI and verdict.
Report every run and every number as found, nulls included. No wording beyond the reading rules.

## Fitzpatrick17k frame counts (from mapping above, join on md5hash)

| subset | n images |
|---|---:|
| in-ISIC | 586 |
| out-of-ISIC | 3241 |
| ambiguous | 60 |
| total | 3887 |

in-ISIC labels present in frame: actinic keratosis, basal cell carcinoma, congenital nevus, dermatofibroma, melanoma, squamous cell carcinoma.
(Other listed in-ISIC names have zero images in this 3,887 frame.)
ambiguous present: epidermal nevus, kaposi sarcoma, naevus comedonicus.

## SD-198 mapping (from `source_label` names only; locked before scoring)

Same rule applied to SD-198 class names (`results/paperB/r5/p7a/sets/sd198.csv` `source_label`).

### in-ISIC

| source_label | ISIC code |
|---|---|
| Actinic_solar_Damage(Actinic_Keratosis) | AK |
| Basal_Cell_Carcinoma | BCC |
| Congenital_Nevus | NV |
| Dermatofibroma | DF |
| Halo_Nevus | NV |
| Lentigo_Maligna_Melanoma | MEL |
| Malignant_Melanoma | MEL |
| Pyogenic_Granuloma | VASC |
| Seborrheic_Keratosis | BKL |

No SD-198 class mapped to SCC under this rule.

### ambiguous

| source_label |
|---|
| Becker's_Nevus |
| Epidermal_Nevus |
| Linear_Epidermal_Nevus |
| Nevus_Comedonicus |
| Nevus_Sebaceous_of_Jadassohn |

### out-of-ISIC

Every other `source_label` among the 198 classes.

### Image counts (SD-198, 6584 images)

| subset | n images | n classes |
|---|---:|---:|
| in-ISIC | 525 | 9 |
| out-of-ISIC | 5950 | 184 |
| ambiguous | 109 | 5 |

## Deviations

_(append only)_
