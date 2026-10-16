# Paper B R5 — per-detector mechanism (why Maha/cosine invert, Energy/kNN/MSP only collapse on Fitz)

Committed before any run of `scripts/r5_detmech.py`. Fixed now; deviations appended. CPU only; no GPU order change.

## Question

On pad_heldout and Fitzpatrick, at λ = 0 and λ = 2, for the old recipe and for A4 (last.ckpt): do per-class
AUROC and the R3 A4 / R4 decomposition for **each of the five detectors** explain why Mahalanobis and cosine
invert while Energy, kNN and MSP only collapse on Fitzpatrick?

## Runs

- Old recipe: `results/paperB/r2/item4/features/runB_orth1_ladv{0,2}_s{42,52,62,72,82}/`
- A4: `results/paperB/r5/p8/main/features/main_ladv{0,2}_s{42,52,62,72,82}/` (last.ckpt primary)

## Statistics (fixed now)

For each (recipe, λ ∈ {0, 2}, seed, OOD ∈ {pad_heldout, fitzpatrick17k}, detector ∈
{mahalanobis_classcond, cosine_max, knn_k50, MSP, Energy_T1}):
- Overall AUROC (ID = ISIC test scores vs OOD).
- Per-class AUROC: ID class c vs all OOD; OOD class c vs all ID (PAD classes only where labels exist;
  Fitzpatrick has no class labels → OOD-side per-class omitted, ID-side retained).
- Decomposition (where OOD labels exist): ID reweighted to the OOD class mix, identity
  Σ_c π_OOD(c)·AUROC_c, absolute gap, NV vs non-NV OOD-side AUROC — same definitions as
  `r6_common.decomposition`, applied to that detector's scores (not only Mahalanobis).

Detectors fit on ISIC train only (`r6_common.Scorer`).

## Reported cells

Mean ± SD over 5 seeds of overall AUROC per (recipe, λ, OOD, detector); per-class and decomposition
tables for λ = 0 vs λ = 2 side by side; one plain paragraph on the Maha/cosine vs Energy/kNN/MSP contrast.

## Predicted outcome (written before scores)

- Mahalanobis and cosine: large drop from λ = 0 to λ = 2 on pad_heldout (into inversion) driven by the
  NV-nearest / majority-centroid pull; decomposition identity gap grows; Fitzpatrick likewise inverts or
  collapses hard.
- Energy, kNN, MSP: mild drop on pad_heldout (collapse toward 0.5, not inversion); on Fitzpatrick they
  collapse (AUROC near 0.5) without a reliable inversion margin.
- Per-class: the pad_heldout drop concentrates on NV-nearest OOD and on ID minority classes under
  Mahalanobis/cosine; Energy/MSP show flatter per-class profiles.

## Outputs

`results/paperB/r5/detmech/`: `per_run/*.json`, `summary.csv`, `REPORT.md` (LaTeX rows).

## Cost

GPU 0. CPU ≈ 1–2 h on 8–16 CPUs (Fitzpatrick + five detectors × two recipes × two λ × five seeds).

## Stop rules

Detector list, OOD sets, and decomposition definitions fixed now.

## Deviations

- 2026-10-16 (+07). Decomposition table columns renamed to “measured AUROC, ID reweighted to PAD mix”
  and “Σ w_c·AUROC_c (identity)”; added unweighted pooled AUROC beside them so |Δ| ≈ 0 is read as the
  identity check, not a failed reweight. Numbers unchanged (report labelling only).
