# R3 A2 — pre-registered bootstrap on the lesion-level split

Commit: `6c1f516`. Protocol: `PREREGISTER_BOOTSTRAP.json` unmodified (2000 resamples, percentile 95%, both sides resampled, draws shared across seeds). ID side resampled by lesion group (2787 groups in the lesion-level ISIC test, null `lesion_id` = own group); pad_heldout by patient (412 patients, 716 images); Fitzpatrick17k by image.

**Verdict (Mahalanobis, pad_heldout, λ = 2):** Interval entirely below 0.5 -> the word 'inverts' survives the move to the lesion-level split (pre-registered rule A: 5/5 seeds' own CI below 0.5).

## pad_heldout

| λ | seeds | Mahalanobis mean [95% CI] | k below 0.5 | kNN k=50 mean [95% CI] | k below 0.5 | Cosine mean [95% CI] | k below 0.5 |
|---:|---|---|---|---|---|---|---|
| 0 | 42/52/62/72/82 | 0.808 [0.792, 0.822] | 0/5 | 0.912 [0.900, 0.923] | 0/5 | 0.866 [0.854, 0.877] | 0/5 |
| 0.25 | 42/43/44/45/46 | 0.525 [0.506, 0.546] | 3/5 | 0.574 [0.554, 0.594] | 2/5 | 0.559 [0.542, 0.575] | 2/5 |
| 0.5 | 42/43/44/45/46 | 0.426 [0.403, 0.452] | 5/5 | 0.440 [0.418, 0.464] | 5/5 | 0.442 [0.422, 0.464] | 5/5 |
| 1 | 42/43/44/45/46 | 0.387 [0.363, 0.410] | 5/5 | 0.407 [0.383, 0.428] | 5/5 | 0.412 [0.391, 0.432] | 5/5 |
| 2 | 42/52/62/72/82 | 0.403 [0.380, 0.425] | 5/5 | 0.427 [0.406, 0.448] | 5/5 | 0.420 [0.401, 0.439] | 5/5 |
| 4 | 42/43/44/45/46 | 0.433 [0.410, 0.456] | 5/5 | 0.424 [0.402, 0.447] | 5/5 | 0.428 [0.409, 0.447] | 5/5 |
| 8 | 42/43/44/45/46 | 0.454 [0.432, 0.477] | 4/5 | 0.440 [0.420, 0.462] | 5/5 | 0.440 [0.421, 0.459] | 4/5 |

## fitzpatrick17k

| λ | seeds | Mahalanobis mean [95% CI] | k below 0.5 | kNN k=50 mean [95% CI] | k below 0.5 | Cosine mean [95% CI] | k below 0.5 |
|---:|---|---|---|---|---|---|---|
| 0 | 42/52/62/72/82 | 0.717 [0.704, 0.730] | 0/5 | 0.801 [0.791, 0.811] | 0/5 | 0.758 [0.749, 0.767] | 0/5 |
| 0.25 | 42/43/44/45/46 | 0.496 [0.484, 0.509] | 3/5 | 0.515 [0.502, 0.528] | 3/5 | 0.505 [0.495, 0.516] | 3/5 |
| 0.5 | 42/43/44/45/46 | 0.411 [0.399, 0.423] | 5/5 | 0.439 [0.426, 0.452] | 4/5 | 0.445 [0.435, 0.457] | 4/5 |
| 1 | 42/43/44/45/46 | 0.375 [0.362, 0.388] | 5/5 | 0.409 [0.395, 0.421] | 5/5 | 0.415 [0.403, 0.427] | 5/5 |
| 2 | 42/52/62/72/82 | 0.402 [0.390, 0.415] | 5/5 | 0.432 [0.419, 0.445] | 5/5 | 0.425 [0.415, 0.437] | 5/5 |
| 4 | 42/43/44/45/46 | 0.376 [0.364, 0.389] | 5/5 | 0.386 [0.373, 0.399] | 5/5 | 0.390 [0.380, 0.401] | 5/5 |
| 8 | 42/43/44/45/46 | 0.412 [0.399, 0.425] | 5/5 | 0.415 [0.402, 0.428] | 5/5 | 0.414 [0.402, 0.426] | 5/5 |

## Caveats

- Fitzpatrick17k has no patient identifiers; its bootstrap is over images and the interval is anti-conservative.
- The decision applies to Mahalanobis, the abstract's number; kNN and cosine are reported alongside.
- Seeds per λ are listed in the table: λ ∈ {0, 2} seeds 42/52/62/72/82 (72, 82 from B1); other λ 42/43/44/45/46 (45, 46 from R4).
