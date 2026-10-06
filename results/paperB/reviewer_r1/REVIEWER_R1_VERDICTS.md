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

**Question:** Is sub-chance / cliff behaviour an artefact of adv_lr×30 being “too strong”?

| Setting | λ=0.25 Mahalanobis (3 seeds) | λ=2 Mahalanobis (3 seeds) |
|--------|------------------------------|---------------------------|
| ×30 (baseline) | 0.511, 0.496, 0.532 → **0.513±0.018** | 0.431, 0.409, 0.456 → **0.432±0.024** |
| ×10 | 0.548, 0.506, 0.474 → **0.509±0.037** | **Same as ×30** (λ=2 reuses Phase-2 ckpts) |
| ×100 | 0.562, 0.508, 0.514 → **0.528±0.029** | **Same as ×30** |

**Verdict:**

- **λ=2 sub-chance inversion persists** at ×10 and ×100 (identical held-out AUROC to ×30 for all three seeds). The “below chance” claim is **not** explained by adversary LR being too high; if anything, weakening the adversary at λ=0.25 **raises** AUROC toward or above 0.5.
- **λ=0.25 does not show a unique “collapse”** under weaker adversaries — scores stay at chance (~0.51) or move **up** with ×10/×100. The sharp drop in the main λ sweep is tied to the **×30 protocol**, not a generic “over-strong adversary” story.

**Placement:** Put **λ=2 sensitivity** (sub-chance robust to adv LR) in **main text** or a short sensitivity paragraph. The full ×10/×100 grid is suitable for **appendix Table** (12 runs under `phase3_sweep/*_advlr10|100/`).

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
