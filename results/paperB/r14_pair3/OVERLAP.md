# OVERLAP — R14 ISIC 2019 → MILK10k dermoscopic

Generated: 2026-11-18T04:40:04.083579+00:00
Commit: `b48d29d`

## ISIC API lesion overlap (§2.3)

- Overlap lesions removed: **859** (expected 859)
- Linked ISIC 2019 images: **1895** (expected 1,895)
- Remaining mapped lesions: **4278**
- Remaining counts: `{'MEL': 230, 'NV': 347, 'BCC': 2429, 'AK': 293, 'BKL': 442, 'DF': 33, 'VASC': 31, 'SCC': 473}`
- Diff from box check: `{'n_overlap_lesions': 0, 'n_overlap_isic2019_images': 0, 'remaining_counts': {'MEL': 0, 'NV': 0, 'BCC': 0, 'AK': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'SCC': 0}}`

Patient overlap note: Vienna/HAM10000 patient sharing cannot be removed (no patient id); only shared lesions removed. milk_heldout not in verdict.

## pHash near-duplicates (64-bit, Hamming ≤ 6)

| Panel | Matches | Action |
|---|---:|---|
| isic2019 | 227 | DROP from MILK |
| pad_ufes20 | 7 | list only |
| fitzpatrick17k | 0 | list only |
| mednode | 0 | list only |
| sd198 | 6 | list only |

MILK images dropped by pHash vs ISIC 2019: **67**

Files: `overlap_lesions.csv` (API), `overlap_phash.csv` (pHash).

