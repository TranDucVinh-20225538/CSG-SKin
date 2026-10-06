# Bootstrap AUROC report (reviewer item 1)

Preregistration: `PREREGISTER_BOOTSTRAP.json` (2000 resamples, percentile 95% CI).

## Primary decision (@ λ=2.0, pad_heldout, Mahalanobis)

- Mean AUROC across seeds (point): **0.4267**
- Bootstrap CI on **mean across seeds**: **[0.4141, 0.4392]**
- Seeds with entire per-seed 95% CI below 0.5: **5 / 5**
- **Decision rule A** applies (see preregistration).

## Manuscript action

Keep **inverts below chance** at λ=2; report uncertainty as in `MANUSCRIPT_NOTES.md`:

- Bootstrap **95% CI on the mean** [0.414, 0.439] (test-set resampling only; shared resample across seeds → **not** seed uncertainty).
- **Seed s.d.** ±0.025 on the five seed point AUROCs.
- **5/5** seeds with entire per-seed bootstrap CI below 0.5.

## λ-curve wording

PAD Mahalanobis vs λ is **U-shaped** (0.513 @ 0.25 → **0.427** @ 2 → 0.494 @ 8), not monotone. Avoid “higher λ always worsens OOD.” At λ=0.25 use **drops sharply to chance**, not **cliff**, until Item 4 adds seeds (CI currently straddles 0.5).

## λ grid (pad_heldout, Mahalanobis mean CI)

| λ | mean AUROC | 95% CI | k below 0.5 |
|---:|---:|---|---:|
| 0.0 | 0.843 | [0.837, 0.849] | 0/5 |
| 0.25 | 0.513 | [0.498, 0.529] | 0/3 |
| 0.5 | 0.465 | [0.449, 0.481] | 2/3 |
| 1.0 | 0.455 | [0.437, 0.470] | 2/3 |
| 2.0 | 0.427 | [0.414, 0.439] | 5/5 |
| 4.0 | 0.455 | [0.440, 0.471] | 2/3 |
| 8.0 | 0.494 | [0.482, 0.506] | 1/5 |

Full detector grid (kNN, cosine) and Fitzpatrick17k (image bootstrap) are in `bootstrap_results.json`.
