# Data

No image data is committed to this repository. Everything under `data/` is
gitignored except this file. Obtain each dataset from its original provider under
that provider's terms.

## Datasets

### ISIC 2019 — in-distribution, dermoscopy

25,331 dermoscopic images, eight diagnostic classes (MEL, NV, BCC, AK, BKL, DF, VASC,
SCC). Used for training and for the in-distribution test split.

- Source: <https://challenge.isic-archive.com/data/>
- Licence: **CC BY-NC 4.0**
- Cite: Tschandl et al., *Scientific Data* 5, 2018 (HAM10000); Combalia et al.,
  arXiv:1908.02288, 2019 (BCN20000); ISIC 2019 challenge.

### PAD-UFES-20 — out-of-domain, smartphone clinical photography

2,298 clinical photographs, six diagnostic classes mapping into the same label space
(DF and VASC are absent). Used in the context and adversarial branches, and — via a
disjoint patient-level partition — as the out-of-domain evaluation set.

- Source: <https://data.mendeley.com/datasets/zr7vgbcyr2>
- Licence: **CC BY 4.0**
- Cite: Pacheco et al., *Data in Brief* 32, 2020.

### Fitzpatrick17k — third domain, never seen by the adversary

Used for evaluation only, to test whether the effect holds on a domain absent from
training entirely. 3,887 of 16,577 images were obtained for the study.

- Source: <https://github.com/mattgroh/fitzpatrick17k>
- Licence: annotations **CC BY-NC-SA 3.0**; the images are web-scraped from
  dermatology atlases under **mixed copyright**.
- **Evaluation only. Not redistributed.** If you use these images, obtain them
  yourself and cite both Fitzpatrick17k and the source atlases.
- Cite: Groh et al., *CVPR Workshops*, 2021.

## Label mapping

PAD diagnostic codes are mapped into the ISIC eight-class space:

| PAD | ISIC |
|-----|------|
| BCC | BCC |
| MEL | MEL |
| NEV | NV |
| ACK | AK |
| SEK | BKL |
| SCC | SCC |

DF and VASC have no PAD counterpart. This asymmetry matters: the domain label is
therefore not independent of the diagnostic label, and any domain probe can exploit
that correlation. The study measures the size of that effect separately (a label-only
domain probe reaches 0.797 against a majority baseline of 0.688) and reports
class-restricted and class-reweighted variants of every affected number.

## Expected layout

```
data/
├── ISIC_2019_Training_Input/        # ISIC images
├── ISIC_2019_Training_GroundTruth.csv
├── ISIC_2019_Training_Metadata.csv  # has lesion_id, used for grouped splits
├── pad_ufes20/
│   ├── images/
│   └── metadata.csv                 # has patient_id, lesion_id, fitspatrick
├── fitzpatrick17k/
│   ├── images/
│   └── fitzpatrick17k.csv
└── master_metadata_lesion_only_soft.csv   # generated, see below
```

Build the last file with:

```bash
python scripts/build_lesion_only_metadata.py --preset soft
```

It applies an Otsu-threshold lesion crop (0.30 margin, 0.70 minimum crop ratio),
resizes to 224×224, and writes the combined metadata table that every training script
reads.

## Splits

- **ISIC** is split 60/20/20 train/val/test, stratified by label, `random_state=42`.
  These splits are image-level, not patient-level; `lesion_id` is present in the ISIC
  metadata if you want to enforce grouped splits.
- **PAD** is partitioned **at patient level** into `pad_adv` (1,582 images, 961
  patients) and `pad_heldout` (716 images, 412 patients). No patient and no lesion
  appears in both. Only `pad_adv` is visible to the context and adversarial branches
  during training; `pad_heldout` is the out-of-domain evaluation set.

PAD labels never enter the classification loss (`ignore_index=-1`), so cross-domain
diagnostic evaluation on `pad_heldout` is zero-shot.
