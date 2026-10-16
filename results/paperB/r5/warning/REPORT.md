# Paper B R5 — two-tier warning signal

Design: `decisions/paperB_r5_warning.md`. Thresholds from λ = 0 only (n = 10), written before flagging.

## Thresholds (95th percentile, λ = 0)

- `tier1_val_centroid_collapse_excess`: 0.059727
- `tier1_test_centroid_collapse_excess`: 0.049272
- `tier2_pred_maj_frac_excess`: -0.315839
- `tier2_bbse_tv`: 0.872479

## Primary: sensitivity / specificity vs invert (Mahalanobis AUROC < 0.5 proxy)

| statistic | tier | threshold | sens | spec | n_pos | n_neg |
|---|---:|---:|---:|---:|---:|---:|
| tier1_val_centroid_collapse_excess | 1 | 0.0597 | 0.194 | 0.857 | 36 | 14 |
| tier1_test_centroid_collapse_excess | 1 | 0.0493 | 0.389 | 0.857 | 36 | 14 |
| tier2_pred_maj_frac_excess | 2 | -0.3158 | 1.000 | 0.643 | 36 | 14 |
| tier2_bbse_tv | 2 | 0.8725 | 0.000 | 0.929 | 36 | 14 |

## Spearman with Mahalanobis AUROC

| statistic | ρ | 95% CI | n |
|---|---:|---|---:|
| tier1_val_centroid_collapse_excess | 0.134 | [-0.158, 0.408] | 50 |
| tier1_test_centroid_collapse_excess | -0.052 | [-0.310, 0.256] | 50 |
| tier2_pred_maj_frac_excess | -0.725 | [-0.855, -0.512] | 50 |
| tier2_bbse_tv | 0.662 | [0.430, 0.815] | 50 |

## Leave-one-setting-out (secondary, labelled **weak**)

Settings: a4_p8, old_recipe. Few settings — not the primary claim.

