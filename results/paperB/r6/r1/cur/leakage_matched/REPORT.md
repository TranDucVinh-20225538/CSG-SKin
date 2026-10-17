# Paper B R6 R1 — leakage-matched test (label shift vs invariance depth)

Design: `decisions/paperB_r6_r1.md` (committed `94e2399`). Analysis commit `ae25a0f`. Pool: R1 (shift×λ∈{0.5,2}) + Table 1 + A4 (λ∈{0,0.5,2}). Leakage = pad_full probe **bal_acc_mean**. Degraded R1 (ID bal < 0.25) excluded from (a)/(b); marked on (c) plots.

The shift coefficient is identified from variation within R1; A4 and Table 1 only extend the leakage range at shift = 100%.

## (a) Regression — Mahalanobis AUROC ~ leakage + shift_level + recipe

Wild cluster bootstrap (Webb weights), cluster = (recipe, λ, shift), 2000 draws, rng 20261005. VIF gate: if shift vs recipe or shift vs leakage VIF > 5 → not identifiable.

**Verdict (a):** supported (≥2/3 primary sets: shift coef negative, CI excludes 0)

| set | n | shift coef | 95% CI | excludes 0 | direction (−) | VIF shift|leak | VIF shift|recipe | identifiable |
|---|---:|---:|---|:---:|:---:|---:|---:|:---:|
| fitzpatrick17k | 75 | -0.166 | [-0.217, -0.115] | yes | yes | 1.04 | 1.03 | yes |
| mednode | 75 | -0.107 | [-0.148, -0.065] | yes | yes | 1.04 | 1.03 | yes |
| sd198 | 75 | -0.172 | [-0.221, -0.123] | yes | yes | 1.04 | 1.03 | yes |
| pad_heldout (not counted) | 75 | 0.041 | [-0.033, 0.110] | no | no | 1.04 | 1.03 | yes |

## (b) Matched comparison (cross-recipe) — R1 shift 0% λ=2 vs A4 λ=2

Difference = mean Maha(R1 shift0 λ2) − mean Maha(A4 λ2). Seed bootstrap 2000 draws. Supported if diff > 0 with CI excluding 0 on ≥ 2/3 primary sets.

**Verdict (b):** supported (≥2/3: R1 shift0 λ2 > A4 λ2, CI excludes 0)

| set | n R1 / A4 | leakage R1 / A4 | Maha R1 / A4 | diff | 95% CI | supported |
|---|---|---|---|---:|---|:---:|
| fitzpatrick17k | 4 / 5 | 0.791 / 0.786 | 0.691 / 0.458 | 0.233 | [0.182, 0.287] | yes |
| mednode | 4 / 5 | 0.791 / 0.786 | 0.541 / 0.298 | 0.243 | [0.205, 0.278] | yes |
| sd198 | 4 / 5 | 0.791 / 0.786 | 0.694 / 0.430 | 0.264 | [0.212, 0.317] | yes |
| pad_heldout (not counted) | 4 / 5 | 0.791 / 0.786 | 0.458 / 0.452 | 0.007 | [-0.005, 0.017] | no |

## (c) Plots

Mahalanobis AUROC vs pad_full leakage; colour = shift level; marker = recipe; red edge = degraded (excluded from a/b).

- `plot_fitzpatrick17k.png`
- `plot_mednode.png`
- `plot_sd198.png`
- `plot_pad_heldout.png`

## Reading

label shift lowers Mahalanobis AUROC at matched leakage on the unseen primary sets (prediction).
