# Phase 16B — morning acceptance bundle

Generated: 2026-10-03T01:00:38.331091+00:00

## Checklist

- [x] B0 probe + label-only fix: `/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase16/b0/summary.json`
- [x] B1 dense 3-seed (BCN→HAM): `/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase16/b1/dense3seed_aggregate.json`
  - λ=0 leakage 0.963±0.006, kNN 0.658±0.041
  - λ=0.25 leakage 0.821±0.030, kNN 0.486±0.009
  - λ=1 leakage 0.825±0.023, kNN 0.472±0.018
- [x] B1 coarse 8λ CSV: `/data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase16/b1/coarse_scan_seed42_maha_columns.csv`
- [x] B1-rev λ=0 gate: Maha **0.849**, pass=True
- [x] B1-rev coarse 8λ: **8/8** summaries

## B1-rev aggregate script
```
coarse rows 8 dense rows 3 missing λ []
wrote /data2/cmdir/home/toandq/CSG-Skin-paperB/results/paperB/phase16/b1_rev/b1_rev_aggregate.json
```

# B1-rev aggregate (auto-generated)

## Coarse scan (seed 42, ID=HAM, OOD=BCN heldout)

| λ | leakage | Maha | Maha rew | kNN-50 | cosine |
|---|--------:|-----:|---------:|-------:|-------:|
| 0 | 0.954 | 0.849 | 0.740 | 0.796 | 0.754 |
| 0.1 | 0.782 | 0.741 | 0.608 | 0.627 | 0.576 |
| 0.25 | 0.777 | 0.705 | 0.574 | 0.569 | 0.545 |
| 0.5 | 0.777 | 0.691 | 0.564 | 0.614 | 0.577 |
| 1 | 0.768 | 0.666 | 0.563 | 0.559 | 0.546 |
| 2 | 0.764 | 0.649 | 0.545 | 0.566 | 0.518 |
| 4 | 0.772 | 0.646 | 0.551 | 0.596 | 0.563 |
| 8 | 0.799 | 0.646 | 0.541 | 0.586 | 0.565 |

## Dense 3-seed @ λ∈{0, 0.25, 1}

| λ | leakage | kNN-50 | cosine | Maha |
|---|---------|--------|--------|------|
| 0 | 0.951±0.003 | 0.780±0.017 | 0.740±0.022 | 0.846±0.003 |
| 0.25 | 0.788±0.024 | 0.600±0.027 | 0.562±0.023 | 0.719±0.019 |
| 1 | 0.775±0.038 | 0.586±0.032 | 0.552±0.006 | 0.683±0.015 |


## Narrative hooks (locked)

1. **BCN→HAM:** site probe ~0.96; adv drops leakage ~0.17 (plateau ~0.80); kNN collapses with small multi-seed σ; Maha below chance → not inversion axis.
2. **HAM→BCN:** λ=0 Maha ~0.85 → headroom; test whether adv inverts (coarse scan + dense).
3. Label-only cohort reference **~0.668** (not 0.5).
