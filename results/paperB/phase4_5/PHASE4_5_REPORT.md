# Paper B — Phase 4.5 report

The clean 2×2 double dissociation is **dead**. It is not rescued. 4a and 4b are reported with equal prominence.
Detectors, splits and hyperparameters were not adjusted toward a predicted value.

## Framing (locked)

- Domain OOD: `z_context` > 0.9999 vs `z_lesion` 0.41 — a large, robust separation between branches.
- Semantic OOD: both branches land in 0.6–0.8; ordering depends on which class is held out.
- The split is **graded**, not categorical: some diagnostic classes carry low-level appearance signatures.
- **4b is not a near-OOD claim.** 4.5c ran; the near/far index does not organise the class-wise advantage. 4a and 4b differ; no systematic pattern is claimed.

## 4.5a — Variance (blocking)

Mahalanobis class-conditional AUROC. Mean ± std over seeds, per-seed values, bootstrap 95% CI over test samples (1000 resamples of ID and OOD).

### 4a hold {DF, VASC} — n_id=4968, n_ood=492

| Model | space | mean ± std [s42, s52, s62] | per-seed bootstrap CI |
|---|---|---|---|
| baseline | backbone_raw | 0.687 ± 0.007  [0.695, 0.681, 0.686] | s42 [0.672, 0.717]; s52 [0.659, 0.706]; s62 [0.665, 0.708] |
| effb3 | backbone_raw | 0.612 ± 0.002  [0.613, 0.613, 0.610] | s42 [0.589, 0.638]; s52 [0.591, 0.637]; s62 [0.587, 0.635] |
| runB_orth1 | z_lesion | 0.648 ± 0.009  [0.638, 0.654, 0.653] | s42 [0.613, 0.663]; s52 [0.629, 0.678]; s62 [0.626, 0.677] |
| runB_orth1 | z_context | 0.641 ± 0.024  [0.614, 0.650, 0.659] | s42 [0.591, 0.639]; s52 [0.627, 0.673]; s62 [0.634, 0.682] |

Paired seed-wise `z_lesion − baseline`: Δ=-0.039, Cohen's d=-2.49, seed-bootstrap CI [-0.057, -0.027], includes 0: **False**.
Paired `z_lesion − z_context`: Δ=0.007, d=0.48, CI [-0.005, 0.024], includes 0: **True**.

### 4b hold {SCC} — n_id=4941, n_ood=628

| Model | space | mean ± std [s42, s52, s62] | per-seed bootstrap CI |
|---|---|---|---|
| baseline | backbone_raw | 0.684 ± 0.052  [0.701, 0.725, 0.626] | s42 [0.678, 0.723]; s52 [0.701, 0.745]; s62 [0.603, 0.646] |
| effb3 | backbone_raw | 0.790 ± 0.027  [0.760, 0.812, 0.797] | s42 [0.742, 0.778]; s52 [0.797, 0.827]; s62 [0.780, 0.812] |
| runB_orth1 | z_lesion | 0.806 ± 0.011  [0.814, 0.809, 0.794] | s42 [0.799, 0.828]; s52 [0.794, 0.825]; s62 [0.778, 0.809] |
| runB_orth1 | z_context | 0.607 ± 0.118  [0.739, 0.572, 0.509] | s42 [0.719, 0.759]; s52 [0.549, 0.595]; s62 [0.483, 0.534] |

Paired seed-wise `z_lesion − baseline`: Δ=0.122, Cohen's d=2.88, seed-bootstrap CI [0.085, 0.168], includes 0: **False**.
Paired `z_lesion − z_context`: Δ=0.199, d=1.82, CI [0.076, 0.284], includes 0: **False**.

## 4.5b — 4a decomposed by class

Hypothesis: `z_context` 0.64 on 4a is VASC (colour) not DF. Prediction: high on VASC, ≈ 0.50 on DF.

| Model / space | DF-only Maha | VASC-only Maha | pooled |
|---|---|---|---|
| runB_orth1 z_context | 0.606 ± 0.009  [0.616, 0.603, 0.598] | 0.675 ± 0.055  [0.612, 0.695, 0.716] | 0.641 ± 0.024  [0.614, 0.650, 0.659] |
| runB_orth1 z_lesion | 0.718 ± 0.024  [0.692, 0.724, 0.738] | 0.582 ± 0.008  [0.587, 0.587, 0.573] | 0.648 ± 0.009  [0.638, 0.654, 0.653] |
| baseline backbone_raw | 0.705 ± 0.011  [0.694, 0.714, 0.709] | 0.670 ± 0.023  [0.696, 0.650, 0.665] | 0.687 ± 0.007  [0.695, 0.681, 0.686] |
| effb3 backbone_raw | 0.687 ± 0.011  [0.699, 0.681, 0.681] | 0.542 ± 0.010  [0.531, 0.550, 0.544] | 0.612 ± 0.002  [0.613, 0.613, 0.610] |

Other detectors (MSP, Energy, cosine, kNN): `phase4_5/phase45_aggregate.json`.

**VASC/DF hypothesis:** not confirmed on 4a: DF 0.61, VASC 0.67 (both above 0.50). On 4c (hold all three), `z_context` does prefer VASC (0.72) over DF (0.57) and SCC (0.53) — a colour-class hint, not the pre-registered ≈0.50/high split.

## 4.5c — Near/far (9/9 complete)

Hold {DF, VASC, SCC} together, 5-class train, score each held-out class separately. 3 models × 3 seeds. Job `60547` COMPLETED. Near/far index = cosine between the held-out class centroid and the nearest kept-class centroid in 8-class `baseline_soft` s42, **ISIC train only**.

| Held-out | nearest keep | cosine (higher = nearer) |
|---|---|---:|
| DF | BKL | 0.570 |
| VASC | NV | 0.566 |
| SCC | AK | 0.565 |

The three classes are **not** near vs far on this index. The cosine range is 0.565–0.570.

### 4c Maha AUROC (mean ± std, n=3)

| Model | DF | VASC | SCC | pooled three |
|---|---|---|---|---|
| baseline | 0.676 ± 0.031 | 0.707 ± 0.054 | 0.678 ± 0.035 | 0.684 ± 0.017 |
| EffB3 16-d | 0.663 ± 0.009 | 0.563 ± 0.011 | 0.785 ± 0.013 | 0.709 ± 0.007 |
| z_lesion | 0.719 ± 0.015 | 0.631 ± 0.039 | 0.807 ± 0.008 | 0.748 ± 0.008 |
| z_context | 0.567 ± 0.062 | 0.723 ± 0.014 | 0.534 ± 0.048 | 0.584 ± 0.038 |

Paired `z_lesion − baseline` on 4c: DF **+0.043**, VASC **−0.077**, SCC **+0.129**. Same sign pattern as 4a/4b (DF ~tie/small win, VASC loss, SCC win).

### Correlation with near/far

| Set | Pearson | Spearman |
|---|---:|---:|
| 4c class means (n=3) | −0.06 | −0.50 |
| 4c class × seed (n=9) | −0.06 | — |
| Pooled 4a DF/VASC + 4b SCC + 4c three (n=6 means) | −0.12 | −0.37 |

**The correlation does not hold.** The x-axis has no spread, so it cannot explain why SCC wins and VASC loses. 4a and 4b simply differ. No near-OOD story is written. Plot: `results/paperB/figures/phase45_nearfar.pdf`.

`z_lesion` still beats EffB3 on every 4c class (DF +0.06, VASC +0.07, SCC +0.02), which is a different comparison than the ResNet-50 baseline.

## Files

- `results/paperB/phase4_5/per_run/`
- `results/paperB/phase4_5/hold3/`
- `results/paperB/phase4_5/phase45_aggregate.json`
- `results/paperB/phase4_5/nearfar_correlation.json`
- `results/paperB/figures/phase45_nearfar.pdf`


