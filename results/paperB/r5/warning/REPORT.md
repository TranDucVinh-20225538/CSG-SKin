# Paper B R5 — two-tier warning signal

Design: `decisions/paperB_r5_warning.md`. **Frozen** thresholds from `thresholds_lambda0.json` (n_lambda0 = 10); **not refit** on the expanded corpus.

## Thresholds (95th percentile, λ = 0) — frozen

- `tier1_val_centroid_collapse_excess`: 0.059727
- `tier1_test_centroid_collapse_excess`: 0.049272
- `tier2_pred_maj_frac_excess`: -0.315839
- `tier2_bbse_tv`: 0.872479

## Pre-committed primary verdict (unchanged)

**tier 1 failed.** Original primary on old_recipe+A4 only; specificity used all non-inverting runs (incl. λ = 0).

| statistic | tier | threshold | sens | spec | n_pos | n_neg |
|---|---:|---:|---:|---:|---:|---:|
| tier1_val_centroid_collapse_excess | 1 | 0.0597 | 0.194 | 0.857 | 36 | 14 |
| tier1_test_centroid_collapse_excess | 1 | 0.0493 | 0.389 | 0.857 | 36 | 14 |
| tier2_pred_maj_frac_excess | 2 | -0.3158 | 1.000 | 0.643 | 36 | 14 |
| tier2_bbse_tv | 2 | 0.8725 | 0.000 | 0.929 | 36 | 14 |

## Updated primary (fairer specificity; expanded corpus)

Corpus ready runs: n = 64. Families: a4_p8=15, old_recipe=35, r1_shift0=9, r1_shift100=5.
R1 shift 0%/100% included when features exist; R2/P9/P11/P10 added as directories arrive.

### Invert label: pad_heldout (Mahalanobis AUROC < 0.5)

Specificity = among **λ > 0 non-inverting** runs only. Sensitivity = among all inverting runs. `n_pos` / `n_neg` are the denominators for sens / spec.

| statistic | tier | threshold | sens (n_pos) | spec (n_neg) |
|---|---:|---:|---|---|
| tier1_val_centroid_collapse_excess | 1 | 0.0597 | 0.146 (n=48) | 0.833 (n=6) |
| tier1_test_centroid_collapse_excess | 1 | 0.0493 | 0.354 (n=48) | 0.833 (n=6) |
| tier2_pred_maj_frac_excess | 2 | -0.3158 | 0.854 (n=48) | 0.333 (n=6) |
| tier2_bbse_tv | 2 | 0.8725 | 0.000 (n=48) | 1.000 (n=6) |

### Invert label: fitzpatrick17k (Mahalanobis AUROC < 0.5)

Specificity = among **λ > 0 non-inverting** runs only. Sensitivity = among all inverting runs. `n_pos` / `n_neg` are the denominators for sens / spec.

| statistic | tier | threshold | sens (n_pos) | spec (n_neg) |
|---|---:|---:|---|---|
| tier1_val_centroid_collapse_excess | 1 | 0.0597 | 0.167 (n=42) | 0.917 (n=12) |
| tier1_test_centroid_collapse_excess | 1 | 0.0493 | 0.381 (n=42) | 0.833 (n=12) |
| tier2_pred_maj_frac_excess | 2 | -0.3158 | 1.000 (n=42) | 0.750 (n=12) |
| tier2_bbse_tv | 2 | 0.8725 | 0.000 (n=42) | 1.000 (n=12) |

## Spearman with Mahalanobis AUROC

| statistic | OOD | ρ | 95% CI | n |
|---|---|---:|---|---:|
| tier1_val_centroid_collapse_excess | pad_heldout | 0.025 | [-0.221, 0.274] | 64 |
| tier1_test_centroid_collapse_excess | pad_heldout | -0.138 | [-0.358, 0.103] | 64 |
| tier2_pred_maj_frac_excess | pad_heldout | -0.770 | [-0.851, -0.631] | 64 |
| tier2_bbse_tv | pad_heldout | 0.718 | [0.538, 0.836] | 64 |
| tier1_val_centroid_collapse_excess | fitzpatrick17k | -0.084 | [-0.333, 0.180] | 64 |
| tier1_test_centroid_collapse_excess | fitzpatrick17k | -0.246 | [-0.458, -0.011] | 64 |
| tier2_pred_maj_frac_excess | fitzpatrick17k | -0.754 | [-0.854, -0.574] | 64 |
| tier2_bbse_tv | fitzpatrick17k | 0.728 | [0.543, 0.846] | 64 |

## BBSE TV audit (2026-10-16)

Checked against prediction that `bbse_tv` should track inversion (higher when AUROC falls).

- TV values: min=0.0753, max=0.8771 (all ≥ 0 → unsigned distance; no sign flip possible).
- Spearman(bbse_tv, Maha pad_heldout AUROC): ρ = 0.718 CI [0.538, 0.836] (n = 64).
- Mean by λ (bbse_tv, auroc): λ=0→(0.812,0.797), λ=0.25→(0.371,0.525), λ=0.5→(0.330,0.462), λ=1→(0.149,0.387), λ=2→(0.314,0.434), λ=4→(0.157,0.433), λ=8→(0.162,0.454)
- Among labelled pad_heldout runs (n = 64): Spearman(bbse_tv, true_tv) = —; Spearman(bbse_abs_err_tv, auroc) = -0.372.

**Reading:** positive ρ is not a sign bug. At λ = 0, PAD has real class-mix shift so BBSE TV is large while AUROC is high; under strong λ the confusion matrix C collapses, BBSE TV shrinks even as AUROC falls — so TV covaries with AUROC rather than with inversion. `pred_maj_frac_excess` keeps the predicted negative ρ. Implementation matches R6 R4 (`q = clip(w)·p_src`, TV = ½‖q−p_src‖₁). **No code change.**

## Leave-one-setting-out (secondary, labelled **weak**)

Settings: a4_p8, old_recipe, r1_shift0, r1_shift100. Few settings — not the primary claim. LOSO still refits λ = 0 p95 on the complement; primary flags use the frozen thresholds above.

