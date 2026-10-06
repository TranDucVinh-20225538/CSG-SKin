# BCN → HAM (B1 forward) — locked coarse + dense

## Coarse Mahalanobis columns (seed 42)

See `b1/coarse_scan_seed42_maha_columns.csv`. Shared = unrestricted (8/8 classes).

## Dense 3-seed (leakage, kNN, cosine — no Maha)

See `b1/dense3seed_aggregate.json` and interim `b1/DENSE3SEED_INTERIM_seed42_only.md`.

## Decision

- Do **not** extend λ beyond coarse grid (plateau at 0.25–1).
- Manuscript label-only baseline: cohort **bal ≈ 0.668** (`b0/summary.json`).
