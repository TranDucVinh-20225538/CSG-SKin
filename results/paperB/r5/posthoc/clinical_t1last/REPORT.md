# Paper B post hoc — clinical consequence of 95% ID-acceptance

Design: `decisions/paperB_r5_posthoc_scoring.md`. Commit `57e080b`. **Label: post hoc.**

Threshold τ = 95th percentile of each run’s ISIC **validation** standard Mahalanobis scores (λ = 0 and λ = 2 separately). Accept as ID if score ≤ τ. Malignant = MEL, BCC, SCC, AK.

## Label maps

- **Fitzpatrick17k:** `label` → melanoma→MEL, basal cell carcinoma→BCC, squamous cell carcinoma→SCC, actinic keratosis→AK, dermatofibroma→DF, congenital nevus→NV.
- **SD-198:** `source_label` folder map in `scripts/r5_p7a_sets.py` (`SD198_MAP`).
- **pad_heldout:** native PAD labels; **n only** (MEL = 9).

## main_last λ=0

τ (mean ± s.d.): 58.567 ± 1.668. pad_heldout n: {'MEL': 9, 'NV': 88, 'BCC': 276, 'AK': 202, 'BKL': 75, 'DF': 0, 'VASC': 0, 'SCC': 66}.

### fitzpatrick17k

mapped / unmapped images: 586 / 3301.

| class | n | share accepted as ID [CI] | share pred NV & conf>0.9 [CI] |
|---|---:|---|---|
| MEL | 83 | 0.961 ± 0.035 [0.938, 0.981] | 0.125 ± 0.041 [0.075, 0.176] |
| BCC | 257 | 0.932 ± 0.052 [0.915, 0.948] | 0.068 ± 0.015 [0.047, 0.092] |
| SCC | 212 | 0.896 ± 0.080 [0.870, 0.921] | 0.069 ± 0.022 [0.048, 0.093] |
| AK | 24 | 0.958 ± 0.029 [0.925, 0.992] | 0.092 ± 0.068 [0.033, 0.167] |

### sd198

mapped / unmapped images: 1036 / 5548.

| class | n | share accepted as ID [CI] | share pred NV & conf>0.9 [CI] |
|---|---:|---|---|
| MEL | 113 | 0.963 ± 0.024 [0.940, 0.981] | 0.129 ± 0.022 [0.088, 0.172] |
| BCC | 60 | 0.933 ± 0.061 [0.903, 0.960] | 0.043 ± 0.038 [0.013, 0.077] |
| SCC | 119 | 0.921 ± 0.060 [0.887, 0.950] | 0.089 ± 0.024 [0.061, 0.121] |
| AK | 60 | 0.950 ± 0.061 [0.927, 0.973] | 0.033 ± 0.017 [0.013, 0.053] |

## main_last λ=2

τ (mean ± s.d.): 57.516 ± 2.506. pad_heldout n: {'MEL': 9, 'NV': 88, 'BCC': 276, 'AK': 202, 'BKL': 75, 'DF': 0, 'VASC': 0, 'SCC': 66}.

### fitzpatrick17k

mapped / unmapped images: 586 / 3301.

| class | n | share accepted as ID [CI] | share pred NV & conf>0.9 [CI] |
|---|---:|---|---|
| MEL | 83 | 0.990 ± 0.010 [0.978, 1.000] | 0.511 ± 0.041 [0.414, 0.610] |
| BCC | 257 | 0.985 ± 0.007 [0.977, 0.992] | 0.742 ± 0.039 [0.702, 0.781] |
| SCC | 212 | 0.946 ± 0.024 [0.925, 0.965] | 0.498 ± 0.088 [0.446, 0.551] |
| AK | 24 | 1.000 ± 0.000 [1.000, 1.000] | 0.725 ± 0.070 [0.583, 0.858] |

### sd198

mapped / unmapped images: 1036 / 5548.

| class | n | share accepted as ID [CI] | share pred NV & conf>0.9 [CI] |
|---|---:|---|---|
| MEL | 113 | 0.979 ± 0.010 [0.958, 0.996] | 0.814 ± 0.048 [0.759, 0.865] |
| BCC | 60 | 0.983 ± 0.020 [0.967, 0.997] | 0.857 ± 0.056 [0.793, 0.913] |
| SCC | 119 | 0.976 ± 0.019 [0.958, 0.992] | 0.721 ± 0.043 [0.659, 0.778] |
| AK | 60 | 0.997 ± 0.007 [0.990, 1.000] | 0.867 ± 0.066 [0.813, 0.917] |
