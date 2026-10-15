# Paper B R5 / second external audit (on 40571d5): A1–A6

Committed before A1, A2, A4 are computed (A1 needs the S1/S2 BCN↔HAM features, job 68070, not yet run; A2 reads P1/P6
outputs, not yet produced; A4 not run). Fixed now; deviations appended at the end. CPU only (the BCN↔HAM features come
from the S1/S2 GPU extraction already queued).

## A1. BCN↔HAM bootstrap by lesion on both sides

Replaces the image-level `bootstrap_phase16_reverse.py` interval [0.666, 0.699] (HAM→BCN, λ = 1, Mahalanobis, seeds
42/52/62). Code `scripts/r5_a1_bcnham_lesion_boot.py`. Both directions, every λ present (8 λ; 3 seeds at 0/0.25/1, 1
seed elsewhere), Mahalanobis / kNN k=50 / cosine (phase16 definitions, fit on ID train). PREREGISTER draws (2,000, seed
20261005), shared across seeds, ID test and OOD both resampled by `lesion_id`; image-level on both sides alongside.
Features: `results/paperB/r5/s_flip/features` (recomputed Mahalanobis AUROC vs stored reported per run).

Prediction: HAM→BCN λ = 1 lesion CI contains 0.683 and is wider than the image-level one by 10–60 %; lower bound
≥ 0.64, all three seeds' CIs above 0.5. BCN→HAM Mahalanobis CIs below 0.5 at every λ.

## A2. Backbone (pre-projection, 1536-d) on the lesion-level runs, pad_heldout

Replaces the Phase 13 image-level numbers (leakage 0.982 → 0.941; backbone Mahalanobis 0.749 → 0.475; nevus-nearest
fraction 0.484 → 0.603; nevus-nearest AUROC 0.726 → 0.330). Sources, no new model code:
- Mahalanobis AUROC, nevus-nearest fraction (OOD and ID test), AUROC on the nevus-nearest OOD subset and on the rest:
  P1 (`scripts/r5_p1_mechanism.py`, `backbone_raw_lesion`, unchanged `r2_item2_nv_centroid.run_block`).
- Leakage probe on `backbone_raw_lesion`: P6, both the Table 1 protocol (V0, ISIC test ∪ pad_full) and held-out (V1,
  ISIC test ∪ pad_heldout, grouped split).
Collated per λ × 5 seeds by `scripts/r5_a2_backbone_report.py` (mean ± s.d.; per-seed values in the CSV).
No GPU extraction is needed: `backbone_raw_lesion` (716 × 1536) is already in every lesion-level feature file.

Prediction: lesion-level backbone Mahalanobis 0.70–0.80 at λ = 0 and 0.42–0.52 at λ = 2; backbone leakage V0 ≥ 0.93
at every λ; V1 within 0.03 of V0; nevus-nearest fraction rises by ≥ 0.08 from λ = 0 to λ = 2; nevus-nearest AUROC
falls below 0.4 at λ = 2.

## A3. Leakage on held-out images (P6)

P6 as designed (`decisions/paperB_r5_p6.md`); Table 1 leakage column reported both ways: V0 (current, pad_full probe)
and V1 (held-out ISIC test ∪ pad_heldout, probe trained on separate images, split grouped by lesion / patient). P6
predictions stand.

## A4. Standard kNN

Code `scripts/r5_a4_knn_standard.py`. Sun et al. 2022: score = Euclidean distance from the L2-normalised feature to its
k-th nearest L2-normalised ISIC-train feature, k ∈ {10, 50, 200}; next to our mean distance over the k nearest (Table 1
"kNN k=50" is the mean-of-50, asserted equal to `bootstrap_ood_ci.knn50_scores`). z_lesion_norm, 35 Table 1 runs,
pad_heldout, Fitzpatrick17k, MED-NODE, SD-198, SCIN. PREREGISTER pooled CI, resample units as in S1.

Prediction: k-th and mean-of-k differ by ≤ 0.02 in every cell; both below 0.5 at λ = 2 on pad_heldout, Fitzpatrick,
MED-NODE, SD-198 for all k; the λ pattern is the same for all k.

## A5. Paired λ2 − λ0 interval

Already produced by the committed `scripts/r3_b1_report.py` → `results/paperB/r3/b1/b1_results.json`
(`id_bal_paired_diff`: mean −0.0227, s.d. 0.0322, n = 5, t-interval df = 4 [−0.0628, +0.0173]). This is a seed-level
t-interval, not a bootstrap. Three manifest entries added (`id_bal_paired_diff_{mean,lo,hi}`).

## A6. Free-scan mismatches

`scripts/verify_manuscript_numbers.py` with manifest + allowlist on the current tree; every free-scan mismatch traced to
its actual source and classified: (a) correct number, the free-scan found a near-collision instead of the source
(source named); (b) correct but source missing from results/ or the manifest; (c) real problem (wrong, stale or
mislabelled number). Output `results/paperB/r5/audit/A6_mismatches.csv` + section in the audit report. Prediction: no
class (c) beyond the items above (A1 interval label, A2 Phase 13 backbone numbers, A3 leakage column source).

## Deviations

(none yet)
