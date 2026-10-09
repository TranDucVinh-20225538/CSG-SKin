# R3 A2 — pre-registered bootstrap on the lesion-level split

Commit: `c5d2129`. Protocol: `PREREGISTER_BOOTSTRAP.json` unmodified (2000 resamples, percentile 95%, both sides resampled, draws shared across seeds). ID side resampled by lesion group (2787 groups in the lesion-level ISIC test, null `lesion_id` = own group); pad_heldout by patient (412 patients, 716 images); Fitzpatrick17k by image.

**Verdict (Mahalanobis, pad_heldout, λ = 2):** Interval entirely below 0.5 -> the word 'inverts' survives the move to the lesion-level split (pre-registered rule A: 3/3 seeds' own CI below 0.5).

## pad_heldout

| λ | seeds | Mahalanobis mean [95% CI] | k below 0.5 | kNN k=50 mean [95% CI] | k below 0.5 | Cosine mean [95% CI] | k below 0.5 |
|---:|---|---|---|---|---|---|---|
| 0 | 42/52/62 | 0.815 [0.800, 0.830] | 0/3 | 0.915 [0.903, 0.926] | 0/3 | 0.869 [0.856, 0.881] | 0/3 |
| 0.25 | 42/43/44 | 0.580 [0.561, 0.600] | 1/3 | 0.650 [0.632, 0.670] | 0/3 | 0.626 [0.609, 0.643] | 0/3 |
| 0.5 | 42/43/44 | 0.428 [0.404, 0.454] | 3/3 | 0.439 [0.418, 0.463] | 3/3 | 0.442 [0.421, 0.464] | 3/3 |
| 1 | 42/43/44 | 0.389 [0.364, 0.412] | 3/3 | 0.408 [0.383, 0.431] | 3/3 | 0.415 [0.393, 0.436] | 3/3 |
| 2 | 42/52/62 | 0.400 [0.375, 0.423] | 3/3 | 0.420 [0.397, 0.442] | 3/3 | 0.416 [0.396, 0.436] | 3/3 |
| 4 | 42 | 0.432 [0.405, 0.460] | 1/1 | 0.441 [0.413, 0.468] | 1/1 | 0.445 [0.419, 0.472] | 1/1 |

## fitzpatrick17k

| λ | seeds | Mahalanobis mean [95% CI] | k below 0.5 | kNN k=50 mean [95% CI] | k below 0.5 | Cosine mean [95% CI] | k below 0.5 |
|---:|---|---|---|---|---|---|---|
| 0 | 42/52/62 | 0.727 [0.714, 0.740] | 0/3 | 0.815 [0.805, 0.824] | 0/3 | 0.772 [0.763, 0.781] | 0/3 |
| 0.25 | 42/43/44 | 0.582 [0.570, 0.595] | 1/3 | 0.585 [0.572, 0.598] | 1/3 | 0.561 [0.549, 0.572] | 1/3 |
| 0.5 | 42/43/44 | 0.437 [0.424, 0.450] | 3/3 | 0.457 [0.445, 0.470] | 2/3 | 0.461 [0.449, 0.472] | 2/3 |
| 1 | 42/43/44 | 0.386 [0.372, 0.399] | 3/3 | 0.414 [0.401, 0.428] | 3/3 | 0.421 [0.409, 0.433] | 3/3 |
| 2 | 42/52/62 | 0.407 [0.395, 0.421] | 3/3 | 0.436 [0.423, 0.450] | 3/3 | 0.436 [0.425, 0.449] | 3/3 |
| 4 | 42 | 0.340 [0.326, 0.354] | 1/1 | 0.341 [0.326, 0.355] | 1/1 | 0.357 [0.344, 0.371] | 1/1 |

## Caveats

- Fitzpatrick17k has no patient identifiers; its bootstrap is over images and the interval is anti-conservative.
- The decision applies to Mahalanobis, the abstract's number; kNN and cosine are reported alongside.
- Seeds per λ are listed in the table; λ ∈ {0, 2} reach n = 5 only once B1 (seeds 72, 82) lands.
