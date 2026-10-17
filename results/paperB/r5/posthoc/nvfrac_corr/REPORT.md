# Paper B post hoc — nv_frac vs Mahalanobis AUROC (descriptive)

Design: `decisions/paperB_r5_posthoc_scoring.md`. Commit `1ea63f0`. **Label: post hoc.** Correlation / variance explained only — **not** mediation. Not the pre-committed R1 dose test (a).

| OOD set | Spearman(nv_frac true, Maha) | p | n | Spearman(nv_frac nearest, Maha) | R²(nv_frac) | R²(leakage) |
|---|---:|---:|---:|---:|---:|---:|
| pad_heldout | nan | nan | 68 | -0.706 | 0.000 | 0.717 |
| fitzpatrick17k | nan | nan | 68 | -0.817 | 0.000 | 0.568 |
| mednode | nan | nan | 68 | -0.781 | 0.000 | 0.355 |
| sd198 | nan | nan | 68 | -0.824 | -0.000 | 0.535 |
| scin | nan | nan | 68 | -0.713 | 0.000 | 0.663 |

Panel: `panel.csv` (one row per run × OOD set).
