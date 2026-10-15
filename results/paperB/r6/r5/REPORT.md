# Paper B R6 R5 — label shift: HAM↔BCN vs ISIC↔PAD vs ISIC↔Fitzpatrick17k

Design: `decisions/paperB_r6_r5.md`. Commit `7cbfde4`. Metadata only; outcomes copied from existing results.

## Class distributions (counts)

| set | n | MEL | NV | BCC | AK | BKL | DF | VASC | SCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HAM | 10015 | 1113 | 6705 | 514 | 130 | 1099 | 115 | 142 | 197 |
| BCN | 12413 | 2857 | 4206 | 2809 | 737 | 1138 | 124 | 111 | 431 |
| ISIC_lesion_train | 16211 | 2895 | 8240 | 2128 | 554 | 1679 | 152 | 162 | 401 |
| ISIC_all | 25331 | 4522 | 12875 | 3323 | 867 | 2624 | 239 | 253 | 628 |
| PAD_full | 2298 | 52 | 244 | 845 | 730 | 235 | 0 | 0 | 192 |
| PAD_adv | 1582 | 43 | 156 | 569 | 528 | 160 | 0 | 0 | 126 |
| PAD_heldout | 716 | 9 | 88 | 276 | 202 | 75 | 0 | 0 | 66 |
| Fitz_mapped | 586 | 83 | 2 | 257 | 24 | 0 | 8 | 0 | 212 |
| Fitz_mapped_sens | 612 | 83 | 2 | 257 | 24 | 0 | 8 | 26 | 212 |
| MED-NODE_mapped | 170 | 70 | 100 | 0 | 0 | 0 | 0 | 0 | 0 |
| SD-198_mapped | 1036 | 113 | 426 | 60 | 60 | 72 | 60 | 126 | 119 |
| SCIN_mapped | 64 | 0 | 7 | 10 | 11 | 0 | 7 | 13 | 16 |

Fitzpatrick17k: 586 of 3887 images map to an ISIC class (strict map); BKL has no counterpart.

## Share of the P7(a) unseen clinical sets that maps to the 8 ISIC classes

Maps as in `scripts/r5_p7a_sets.py`. The outcome columns below are AUROCs on the whole set (mapped and unmapped images); the shift rows use the mapped images only. SCIN: a case maps if its top weighted dermatologist label is an ISIC class with weight ≥ 0.5 (used in P7(a)); sensitivity rows count a case if its top label maps at any weight, or if any label in its differential maps.

| set | images | groups | mapped images | mapped groups | mapped share (images) |
|---|---:|---:|---:|---:|---:|
| MED-NODE | 170 | 170 | 170 | 170 | 100.0% |
| SD-198 | 6584 | 6584 | 1036 | 1036 | 15.7% |
| SCIN | 10406 | 5033 | 64 | 32 | 0.6% |

SCIN cases: 5033 total, 1972 without any dermatologist label; top label maps at weight ≥ 0.5: 32 (0.6%); top label maps at any weight: 49 (1.0%); any label in the differential maps: 95 (1.9%). Images in cases whose top label maps at any weight: 101 (1.0%).
Most frequent SCIN top labels: Eczema 488; Allergic Contact Dermatitis 270; Urticaria 214; Insect Bite 185; Folliculitis 142; Psoriasis 109; Tinea 93; Impetigo 69; Herpes Zoster 68; Pigmented purpuric eruption 62.

## Shift and outcome

| pair (source→target) | k classes | TV | KL(q‖p) | KL(p‖q) | source majority p → q | Maha λ=0 | Maha λ=2 | min over λ (λ) |
|---|---:|---:|---:|---:|---|---|---|---|
| HAM→BCN | 8 | 0.355 | 0.360 | 0.295 | NV 0.669 → 0.339 | 0.846 | 0.649 | 0.646 (4) |
| BCN→HAM | 8 | 0.355 | 0.295 | 0.360 | NV 0.339 → 0.669 | 0.369 | 0.383 | 0.342 (0.25) |
| ISIC→PAD (heldout) | 6 | 0.566 | 0.904 | 0.984 | NV 0.518 → 0.123 | 0.808 | 0.403 | 0.387 (1) |
| ISIC→PAD (adv target) | 6 | 0.579 | 0.980 | 0.970 | NV 0.518 → 0.099 | — | — | — |
| ISIC→Fitzpatrick (mapped) | 6 | 0.630 | 1.339 | 2.754 | NV 0.573 → 0.003 | 0.717 | 0.402 | 0.375 (1) |
| ISIC→Fitzpatrick (mapped + VASC) | 7 | 0.627 | 1.308 | 2.739 | NV 0.567 → 0.003 | — | — | — |
| ISIC→MED-NODE (mapped) | 2 | 0.152 | 0.054 | 0.050 | NV 0.740 → 0.588 | 0.536 | 0.306 | 0.291 (1) |
| ISIC→SD-198 (mapped) | 8 | 0.274 | 0.398 | 0.246 | NV 0.508 → 0.411 | 0.713 | 0.375 | 0.342 (4) |
| ISIC→SCIN (mapped) | 6 | 0.625 | 1.261 | 1.155 | NV 0.708 → 0.109 | 0.726 | 0.559 | 0.525 (1) |

## Plain reading

Ordered by TV: ISIC→MED-NODE (mapped) TV 0.152 (majority 0.74→0.59), Mahalanobis 0.536→0.291; ISIC→SD-198 (mapped) TV 0.274 (majority 0.51→0.41), Mahalanobis 0.713→0.342; HAM→BCN TV 0.355 (majority 0.67→0.34), Mahalanobis 0.846→0.646; BCN→HAM TV 0.355 (majority 0.34→0.67), Mahalanobis 0.369→0.342; ISIC→PAD (heldout) TV 0.566 (majority 0.52→0.12), Mahalanobis 0.808→0.387; ISIC→SCIN (mapped) TV 0.625 (majority 0.71→0.11), Mahalanobis 0.726→0.525; ISIC→Fitzpatrick (mapped) TV 0.630 (majority 0.57→0.00), Mahalanobis 0.717→0.375.

## LaTeX rows (pair & TV & KL(q‖p) & majority p→q & Maha λ=0 & min Maha)

```latex
HAM$\to$BCN & 0.36 & 0.36 & 0.67$\to$0.34 & 0.846 & 0.646 \\
BCN$\to$HAM & 0.36 & 0.30 & 0.34$\to$0.67 & 0.369 & 0.342 \\
ISIC$\to$PAD (heldout) & 0.57 & 0.90 & 0.52$\to$0.12 & 0.808 & 0.387 \\
ISIC$\to$PAD (adv target) & 0.58 & 0.98 & 0.52$\to$0.10 & — & — \\
ISIC$\to$Fitzpatrick (mapped) & 0.63 & 1.34 & 0.57$\to$0.00 & 0.717 & 0.375 \\
ISIC$\to$Fitzpatrick (mapped + VASC) & 0.63 & 1.31 & 0.57$\to$0.00 & — & — \\
ISIC$\to$MED-NODE (mapped) & 0.15 & 0.05 & 0.74$\to$0.59 & 0.536 & 0.291 \\
ISIC$\to$SD-198 (mapped) & 0.27 & 0.40 & 0.51$\to$0.41 & 0.713 & 0.342 \\
ISIC$\to$SCIN (mapped) & 0.63 & 1.26 & 0.71$\to$0.11 & 0.726 & 0.525 \\
```

