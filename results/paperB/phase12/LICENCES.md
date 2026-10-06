# Phase 12 — licences and citations

Derm entries are copied from `results/paperB/phase11_third_domain/LICENCES.md` so this file stands alone. That Phase 11 file was not modified.

## WILDS (benchmark / code)

- **used_in**: Phase 12 data access (`wilds==2.0.0` in torch-env)
- **licence**: MIT (Python package)
- **citation**: Koh, Sagawa, et al., WILDS: A Benchmark of in-the-Wild Distribution Shifts, ICML 2021.
- **url**: https://wilds.stanford.edu
- **status**: installed; no download of the package required

## Camelyon17-WILDS

- **used_in**: Phase 12.1 (primary). Histopathology, hospital domain, stain shift.
- **licence**: Public domain, CC0 1.0. https://creativecommons.org/publicdomain/zero/1.0/
- **challenge terms**: CAMELYON17 grand-challenge (https://camelyon17.grand-challenge.org/). WILDS redistributes 96×96 patches, not whole-slide images.
- **citation (dataset)**: Bandi et al., From detection of individual metastases to classification of lymph node status at the patient level: the CAMELYON17 challenge, IEEE TMI 38(2):550–560, 2018.
- **citation (WILDS variant)**: Koh, Sagawa, et al., ICML 2021.
- **url**: https://wilds.stanford.edu/datasets/  and  https://camelyon17.grand-challenge.org/
- **local**: `/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds/camelyon17_v1.0` (already extracted)

## iWildCam-WILDS (v2.0)

- **used_in**: Phase 12.2 (gated on 12.1). Camera-trap locations as domains.
- **licence**: Community Data License Agreement – Permissive – Version 1.0 (CDLA-Permissive 1.0). https://cdla.io/permissive-1-0/
- **citation (dataset)**: Beery, Cole, Gjoka, The iWildCam 2020 Competition Dataset, arXiv:2004.10340, 2020.
- **citation (WILDS variant)**: Koh, Sagawa, et al., ICML 2021.
- **url**: https://wilds.stanford.edu/datasets/  and  https://www.kaggle.com/c/iwildcam-2020-fgvc7
- **note**: WILDS v2.0 is the height-448 JPEG redistribution (~12 GB on disk), not the original competition dump (~90 GB). Phase 12 uses the WILDS copy.
- **local**: `/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds/iwildcam_v2.0` (already extracted)

## Existing dermatology datasets (Paper B, unchanged)

### ISIC 2019
- **used_in**: ID training / ID test / Paper B
- **licence**: ISIC Archive / challenge release: CC BY-NC 4.0 (non-commercial). Individual contributing centres may impose additional terms.
- **citation**: Tschandl et al., The HAM10000 dataset, Sci. Data 2018; Combalia et al., BCN20000, arXiv:1908.02288; ISIC 2019 Challenge.
- **url**: https://challenge.isic-archive.com/data/#2019

### PAD-UFES-20
- **used_in**: domain-adversary stream + covariate-shift OOD (Paper B)
- **licence**: CC BY 4.0
- **citation**: Pacheco et al., PAD-UFES-20, Data in Brief 32:106221, 2020.
- **url**: https://data.mendeley.com/datasets/zr7vgbcyr2

### Fitzpatrick17k
- **used_in**: Paper B third domain (eval only, 3887 / 16577 images on disk, not redistributed)
- **licence_annotations**: CC BY-NC-SA 3.0 (Groh et al.)
- **licence_images**: Mixed web-scraped photograph copyright; non-commercial research only; do not redistribute image files.
- **citation**: Groh et al., Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset, CVPRW 2021.
- **url**: https://github.com/mattgroh/fitzpatrick17k
