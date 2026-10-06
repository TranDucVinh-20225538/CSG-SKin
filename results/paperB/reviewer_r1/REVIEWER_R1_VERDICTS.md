# Reviewer R1 — verdict summaries (items 2–5)

## Item 1 — bootstrap (primary)

**Status:** Primary pad_heldout @ λ=2 **complete** (`bootstrap_results.json`, 2000 resamples).  
**Reverse HAM→BCN @ λ=1:** running via `scripts/bootstrap_phase16_reverse.py` → `bootstrap_phase16_ham_bcn_ladv1.json`.

**Manuscript decision (primary): Rule A** — keep **“inverts below chance”** for Mahalanobis @ λ=2 on `pad_heldout`.

---

## Item 2 — Mahalanobis numerical stability

**Verdict:** AUROC is **stable** under implementation choices. On `runB_orth1_ladv2_s42` / `pad_heldout`, max |ΔAUROC| vs baseline **≈ 9.4×10⁻⁴** across float32/float64, Ledoit–Wolf, and BLAS thread counts 1/4/8 (`maha_stability.json`). Not a plausible explanation for sub-chance scores.

---

## Item 3 — adversary LR ×10 / ×100

**λ=0.25 (retrained ×10/×100):** Mahalanobis stays at chance (×30 **0.513±0.018**; ×10 **0.509±0.037**; ×100 **0.528±0.029**). No sensitivity — **that part stands**.

**λ=2 (headline claim):** Earlier table cells at ×10/×100 were **invalid** — `train_phase3_sweep.py` reused Phase-2 ckpts whenever λ=2, so AUROC matched ×30 by construction (`reused_phase2_ckpt: true`). **No manuscript sensitivity claim at λ=2 until retrain completes.**

**In flight:** 6 fresh GPU trains — adv_lr ∈ {10, 100}, λ=2, seeds {42,52,62}; `slurm/reviewer_r1_item3_lam2_advlr_retrain.sbatch`; reuse disabled unless ×30. Report: `scripts/report_item3_lam2_advlr.py` → `ITEM3_LAM2_ADVLR_REPORT.md`.

---

## Item 4 — extra seeds @ λ=0.25 (Table 1 cell)

**Question:** Does the λ=0.25 Mahalanobis `pad_heldout` cell shift when n=3 → n=5?

| n | Seeds | Mean Mahalanobis | Per-seed |
|---|-------|------------------|----------|
| 3 | 42, 52, 62 | **0.513** ± 0.018 | 0.512, 0.496, 0.532 |
| 5 | +72, 82 | **0.509** ± 0.023 | …, 0.528, **0.478** |

**Verdict:** The Table 1 λ=0.25 cell moves **marginally** (Δmean ≈ **−0.004**); seed 82 is lower (0.478) but the **story is unchanged** — λ=0.25 remains **at chance**, not a second sub-chance inversion. Wording should stay **“drops sharply toward chance”** (U-shaped curve), **not** “cliff at 0.25” until/unless denser λ grid says otherwise.

**λ=1 (context):** n=3 mean **0.455** → n=5 **0.457** (stable sub-chance band, not the headline λ=2 claim).

---

## Item 5 — ResNet-50 single-encoder DANN

**Status:** **Submitted** SLURM array **63214** — 9 runs: λ ∈ {0, 0.5, 2} × seeds {42, 52, 62}, script `train_reviewer_r1_item5_resnet50.py`, outputs under `reviewer_r1/resnet50_single_dann/`.

**Pending:** Mahalanobis @ `pad_heldout` per run; compare to EffNet-B3 single-encoder / dual-encoder Table 3 narrative.
