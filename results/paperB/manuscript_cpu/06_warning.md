# (6) Warning signal — majority-prediction-rate Spearman (n = 64)

Source: `results/paperB/r5/warning/REPORT.md`.

Corpus: **n = 64** ready runs (a4_p8=15, old_recipe=35, r1_shift0=9, r1_shift100=5).

Feature: `tier2_pred_maj_frac_excess` (majority-prediction-rate excess).

| OOD set | Spearman ρ with Mahalanobis AUROC | 95% CI | n |
|---|---:|---|---:|
| pad_heldout | **−0.770** | [−0.851, −0.631] | 64 |
| Fitzpatrick17k | **−0.754** | [−0.854, −0.574] | 64 |

(Report also notes this keeps the predicted negative ρ; BBSE TV does not.)
