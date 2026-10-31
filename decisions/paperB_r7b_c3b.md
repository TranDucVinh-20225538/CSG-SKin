# Paper B R7b / C3b — Fitzpatrick17k AUROC by skin type (CPU)

Committed before the C3b job. Deviations appended at the end.

## Question

The 3,887 kept Fitzpatrick17k images differ from the full 16,577 in Fitzpatrick type mix (TV 0.257, R7 C3). Does detection AUROC on the kept set change when restricted to skin-type groups, or when the kept set is reweighted to the full csv's type mix?

## Design

- Table 1 last.ckpt runs, all λ, 5 seeds each. Features from `results/paperB/r5/t1last/features_last/`.
- Detectors: Mahalanobis (class-conditional on `z_lesion_norm`) plus kNN; ID side = full ISIC test set (same as Table 1).
- AUROC per Fitzpatrick group on the kept images: types **1–2**, **3–4**, **5–6**, and **unknown (−1)**. Each group's OOD images against the full ISIC test set.
- Report **n images** and **n P13 clusters** (`cluster_union` from `results/paperB/r5/p13/fitz_clusters.csv`) per group.
- **Reweighted AUROC** (all kept images): weight each kept image so the kept set's Fitzpatrick type mix equals the full csv's (weight = full share / kept share per type). Second version: reweight on **type × ISIC-mapped class** (mapping from `fitzpatrick17k_label_map.json`; unmapped class = `UNMAPPED`).
- Bootstrap by P13 cluster as in the paper; Rule A columns as in Table 1. No image download.

## Outputs

`results/paperB/r7b/c3b/{summary.csv,BOOTSTRAP.json,REPORT.md}`.

## Deviations
