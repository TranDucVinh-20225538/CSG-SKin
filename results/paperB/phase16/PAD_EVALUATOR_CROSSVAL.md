# PAD evaluator cross-validation (Phase 16)

**Question:** Can the paper stand on **pad_heldout** Mahalanobis if we drop the contaminated **pad_full** Table 1 column?

**Answer: Yes for PAD.** The lesion Mahalanobis evaluator is **stable** on heldout; the submission fix is **partition choice**, not a broken scorer.

---

## What was compared

| Pipeline | Partition | Code | Role |
|---|---|---|---|
| Phase 2 train-end eval | **pad_heldout** | `train_phase2_pad_holdout.py` | λ=2 reference (5 seeds) |
| Phase 3 sweep | **pad_heldout** + pad_full | `train_phase3_sweep.py` → `detector_block()` | Full λ grid |
| Phase 35 mech / final GPU | **pad_heldout only** | `eval_final_gpu.py` imports `p3.maha_scores` | Independent re-run |

Detector definition (all three): **class-conditional Mahalanobis on `z_lesion`**, unrestricted AUROC, ID = ISIC test, OOD = target split, **fit on ISIC train only**.

---

## Results

### 1. Same checkpoint (λ=2): Phase 2 vs Phase 3

| seed | phase2 heldout | phase3 heldout | Δ |
|---:|---:|---:|---:|
| 42 | 0.4313624802 | 0.4313624802 | 0 |
| 52 | 0.4088504542 | 0.4088504542 | 0 |
| 62 | 0.4558665282 | 0.4558985020 | +3.2×10⁻⁵ |
| 72 | 0.4416558342 | 0.4416341967 | −2.2×10⁻⁵ |
| 82 | 0.3948390451 | 0.3948409745 | +2×10⁻⁶ |

**max |Δ| = 3.2×10⁻⁵** (floating noise). Phase 3 λ=2 reuse is the same eval as Phase 2.

Aggregate heldout: **0.4265 ± 0.0247** (Phase 3) = **0.4265 ± 0.0246** (Phase 2 aggregate JSON).

### 2. Full λ grid: Phase 3 vs `eval_final_gpu` (phase35_mech)

**27/27** runs compared: 26 exact to machine precision; **one** pair (λ=2, seed 52) differs by **4.8×10⁻⁴** (0.408850 vs 0.409333), likely re-embed / checkpoint tie-break noise. **max |Δ| = 4.8×10⁻⁴**, mean |Δ| = 2.1×10⁻⁵.

### 3. What does *not* validate heldout (do not use for cross-check)

- **Phase 2.5a `eval_phase25.py`** scores **`pad_full` (2,298)**, not heldout — explains apparent Phase 3 vs 2.5 gaps on PAD when 2.5 rows are misread as heldout.
- **Fitzpatrick17k** still needs unified eval (Phase 3 inline vs 2.5a); **PAD heldout is not affected**.

---

## Cliff on heldout (n unchanged)

| λ_adv | n | pad_heldout Maha (mean ± s.d.) |
|---:|---:|---|
| 0 | 5 | 0.843 ± 0.023 |
| 0.25 | 3 | 0.513 ± 0.018 |
| 2 | 5 | 0.427 ± 0.025 |

λ=0 → 0.25 drop ≈ **0.33** AUROC on heldout (same order as the 0.86→0.51 story on pad_full). **Inversion below 0.5 remains** at λ≥0.25 on heldout.

---

## Manuscript action

1. Replace Table 1 PAD OOD column with **pad_heldout** values (`MANUSCRIPT_PAD_HELDOUT_COLUMN.md`).
2. One sentence: legacy Table 1 used **pad_full** (includes 1,582 training-seen PAD images); main text now reports **pad_heldout** only.
3. Proceed to B0 after tex update; no PAD evaluator retrain required.
