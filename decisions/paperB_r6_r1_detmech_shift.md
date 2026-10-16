# Paper B R6 R1 — post hoc detmech at shift 0% vs 100% (λ = 2)

Committed before `scripts/r6_r1_detmech_shift.py`. CPU only; labelled where labels exist.
Does not change R1 training, thresholds, or GPU order.

## Question

At λ = 2, shift 0% (no adversary label shift) vs shift 100% (full ISIC→PAD mix): does the
**within-class pull** (non-nevus-nearest OOD AUROC and class-matched AUROC &lt; 0.5) disappear at
shift 0%, or only the pull toward nevus? Baseline: A4 / R1 λ = 0.

## Cells

| cell | source | seeds |
|---|---|---|
| baseline λ = 0 | A4 `main_ladv0_s*` (current recipe); R1 λ = 0 features if present | healthy only |
| R1 λ = 2, shift 0% | `r1_cur_shift0_ladv2_s*` | healthy only |
| R1 λ = 2, shift 100% | `r1_cur_shift100_ladv2_s*` | healthy only |

**Healthy:** ID test bal acc ≥ 0.25 (same R1 degraded rule). Degraded seeds listed, excluded from pooled
stats.

## OOD sets

pad_heldout, fitzpatrick17k, mednode, sd198.

## Statistics (Mahalanobis, z_lesion_norm, fit ISIC train)

1. **Nevus-nearest vs non-nevus-nearest OOD AUROC:** partition OOD by Mahalanobis nearest class
   (NV vs rest); AUROC of each subset vs all ID test. Cluster CI of the seed mean.
2. **Class-matched OOD-vs-ID AUROC per class:** ID class c vs OOD class c. pad_heldout: true labels.
   MED-NODE / SD-198: p7a `mapped` (unmapped SD-198 excluded). Fitzpatrick: ISIC map where available;
   unmapped excluded. Classes with n &lt; 20 on either side: CI shown, excluded from reading.
3. **nv_frac** of OOD (nearest-class) vs ID-test NV nearest base rate.
4. **Mahalanobis AUROC with OOD reweighted to the ID-test class mix** (true / mapped labels; Fitzpatrick
   mapped only). Unweighted Mahalanobis alongside.

## Bootstrap

PREREGISTER 2,000 draws, rng 20261005; shared across seeds within a cell. Units: pad_heldout = patient;
Fitzpatrick = P13 `cluster_union`; MED-NODE / SD-198 = `group` from p7a sets; ID = lesion.

## Reading (fixed now)

Let AUROC_rw(s) = OOD-reweighted Mahalanobis on a set at shift s.

- If AUROC_rw(0%) ≥ 0.5 **and** AUROC_rw(100%) &lt; 0.5 → label shift drives both 3a and 3b.
- If AUROC_rw ≥ 0.5 at **both** shifts → 3b needs label shift only through 3a (within-class / other
  mechanisms remain at shift 0%).
- Separately state whether non-nevus-nearest and eligible class-matched AUROCs recover to ≥ 0.5 at
  shift 0%.

## Outputs

`results/paperB/r6/r1/cur/detmech_shift/`: per-cell JSON, `summary.csv`, `REPORT.md`.

## Cost

GPU 0. CPU ≈ 1–2 h on 8–16 CPUs.

## Deviations

- 2026-10-16 23:05 (+07). NV / non-NV OOD AUROC originally **nearest-centroid** only. Added the
  **true-class** split (where labels / mapped labels exist) alongside, both labelled — aligned with
  `r5/detmech`.
