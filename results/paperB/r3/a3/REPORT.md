# R3 A3 — JSON for markdown-only manuscript numbers

Commit: `1168c5c`.

**Verdict:** All values written to JSON match their published reports; no recomputation needed.

| Manuscript number | JSON file | key | value |
|---|---|---|---|
| 4.879 | phase13/participation_ratio.json | by_lambda.0.pr_z_lesion_norm.mean | 4.8787 |
| 4.500 | phase13/participation_ratio.json | by_lambda.0.25.pr_z_lesion_norm.mean | 4.4999 |
| 84.5% | phase1_6/concentration.json | z_context_pc1_var_explained | 0.84476 |
| 2 discordant pairs | phase1_6/concentration.json | z_context_maha_classcond_pad_full_by_seed[seed 42].discordant_pairs | 2.000 |
| 11.6M | phase1_6/concentration.json | z_context_maha_classcond_pad_full_by_seed[seed 42].n_pairs | 11643966 |
| 0.55–0.70 | phase1_6/concentration.json | pca_k1_auroc_other_representations_range | 0.545–0.698 |
| 3,041 | r2/item4/image_level_overlap.json | n_test_sharing_lesion_with_train_or_val | 3041 |
| 5,067 | r2/item4/image_level_overlap.json | n_test | 5067 |
| 60.0% | r2/item4/image_level_overlap.json | pct_sharing | 60.02 |
| 2,026 | r2/item4/image_level_overlap.json | n_lesion_disjoint | 2026 |

## Participation ratio table (phase 13, ISIC-train covariance), mean ± s.d.

| λ | n | z_lesion^norm | z_lesion pre-BN | z_context | backbone (1536-d) |
|---:|---:|---|---|---|---|
| 0 | 5 | 4.879 ± 0.148 | 4.145 ± 0.220 | 1.559 ± 0.403 | 57.374 ± 2.950 |
| 0.25 | 3 | 4.500 ± 0.154 | 3.660 ± 0.029 | 1.457 ± 0.084 | 48.186 ± 4.884 |
| 0.5 | 3 | 4.453 ± 0.154 | 3.610 ± 0.196 | 1.470 ± 0.156 | 45.912 ± 2.524 |
| 1 | 3 | 4.525 ± 0.135 | 3.718 ± 0.229 | 1.502 ± 0.241 | 47.971 ± 2.872 |
| 2 | 5 | 4.737 ± 0.203 | 4.003 ± 0.194 | 1.315 ± 0.176 | 45.728 ± 5.729 |
| 4 | 3 | 4.680 ± 0.098 | 3.997 ± 0.186 | 1.369 ± 0.200 | 43.709 ± 7.132 |
| 8 | 5 | 4.763 ± 0.183 | 4.306 ± 0.284 | 1.362 ± 0.087 | 36.395 ± 4.154 |

## Findings

- None: every value equals its published report at the printed precision.
- Wording, not a value: the manuscript's 'participation ratio 4.879 → 4.500 while leakage falls 0.914 → 0.553' is λ = 0 → λ = 0.25 (PHASE13_REPORT.md 13.1). At λ = 2 the ratio is 4.737 ± 0.203. The sentence sits after a λ = 2 statement and does not name the λ pair.
