# R2 Item 1 — unsupervised OOD monitor on an unseen domain

Commit: `4c62482`. Pre-commit: `results/paperB/r2/PRECOMMIT_ITEM1_ITEM2.json` (`e131a42`).

**Verdict (pre-registered decision table):** Frozen ImageNet degrades -> there is no cheap fix (Mahalanobis 0.833 on pad_heldout < 0.90). Reading: partial mitigation, see Headline.

<!-- HEADLINE BEGIN -->
## Headline (matched sets)

All entries on the same ID set within a row group (full ISIC test, n = 5067, unless marked lesion-disjoint, n = 2026) and the same OOD set. The model columns are z_lesion^norm, mean ± s.d. over 5 seeds; frozen ImageNet is one deterministic encoder.

### Pre-registered rule vs post-hoc detector

**Pre-registered:** Mahalanobis on frozen ImageNet ResNet-50 ≥ 0.90 in all four cells (2 OOD sets × full / lesion-disjoint ID). **Not met:** pad_heldout gives 0.833 (full) and 0.848 (disjoint). kNN k=50 on the same features gives 0.913 / 0.915 on pad_heldout; kNN was not the pre-registered score and is reported for information only. The bar is not cleared.

| OOD | ID set | Detector | Frozen ImageNet | Model λ=0 | Model λ=2 | Pre-registered |
|---|---|---|---|---|---|---|
| pad_heldout | full | Mahalanobis | 0.833 | 0.843 ± 0.023 | 0.427 ± 0.025 | yes (≥ 0.90) |
| pad_heldout | full | kNN k=50 | 0.913 | 0.918 ± 0.009 | 0.440 ± 0.012 | no |
| pad_heldout | lesion-disjoint | Mahalanobis | 0.847 | 0.890 ± 0.015 | 0.515 ± 0.026 | yes (≥ 0.90) |
| pad_heldout | lesion-disjoint | kNN k=50 | 0.915 | 0.935 ± 0.008 | 0.521 ± 0.012 | no |
| fitzpatrick17k | full | Mahalanobis | 0.988 | 0.764 ± 0.038 | 0.391 ± 0.068 | yes (≥ 0.90) |
| fitzpatrick17k | full | kNN k=50 | 0.982 | 0.837 ± 0.030 | 0.423 ± 0.059 | no |
| fitzpatrick17k | lesion-disjoint | Mahalanobis | 0.990 | 0.830 ± 0.027 | 0.490 ± 0.072 | yes (≥ 0.90) |
| fitzpatrick17k | lesion-disjoint | kNN k=50 | 0.983 | 0.872 ± 0.024 | 0.509 ± 0.058 | no |

**Reading — partial mitigation, below the pre-registered bar.** The frozen encoder never inverts (≥ 0.83 everywhere) and beats the model at λ=2 on both sets (pad_heldout 0.833 vs 0.427; Fitzpatrick 0.988 vs 0.391). Against the model before adversarial training (λ=0) it is higher on Fitzpatrick (0.988 vs 0.764) but not on pad_heldout (0.833 vs 0.843; within one s.d.). It is a partial mitigation, not a fix.

### Per-class AUROC (ID class c vs OOD), Mahalanobis

Within one ID class the score does not depend on the class mix of the ID set, unlike the pooled AUROC.

**pad_heldout**

| ID class | Model λ=2, full | Model λ=2, lesion-disjoint | Model λ=0, full | Frozen ImageNet, full |
|---|---|---|---|---|
| MEL | 0.388 ± 0.025 | 0.376 ± 0.024 | 0.885 ± 0.013 | 0.782 |
| NV | 0.559 ± 0.029 | 0.610 ± 0.028 | 0.932 ± 0.010 | 0.839 |
| BCC | 0.257 ± 0.031 | 0.216 ± 0.028 | 0.733 ± 0.063 | 0.872 |
| AK | 0.158 ± 0.018 | 0.150 ± 0.022 | 0.499 ± 0.047 | 0.895 |
| BKL | 0.270 ± 0.024 | 0.288 ± 0.029 | 0.764 ± 0.039 | 0.823 |
| SCC | 0.142 ± 0.015 | 0.153 ± 0.015 | 0.444 ± 0.064 | 0.868 |
| pooled (all 8) | 0.427 ± 0.025 | 0.515 ± 0.026 | 0.843 ± 0.023 | 0.833 |

Non-NV classes (MEL, BCC, AK, BKL, SCC) at λ=2, both ID sets: 0.14–0.39.

**fitzpatrick17k**

| ID class | Model λ=2, full | Model λ=2, lesion-disjoint | Model λ=0, full | Frozen ImageNet, full |
|---|---|---|---|---|
| MEL | 0.343 ± 0.077 | 0.329 ± 0.078 | 0.799 ± 0.032 | 0.979 |
| NV | 0.536 ± 0.075 | 0.594 ± 0.078 | 0.886 ± 0.022 | 0.989 |
| BCC | 0.202 ± 0.059 | 0.162 ± 0.047 | 0.609 ± 0.087 | 0.993 |
| AK | 0.109 ± 0.039 | 0.101 ± 0.035 | 0.360 ± 0.060 | 0.994 |
| BKL | 0.218 ± 0.066 | 0.237 ± 0.071 | 0.627 ± 0.063 | 0.986 |
| SCC | 0.095 ± 0.032 | 0.105 ± 0.039 | 0.318 ± 0.070 | 0.995 |
| pooled (all 8) | 0.391 ± 0.068 | 0.490 ± 0.072 | 0.764 ± 0.038 | 0.988 |

Non-NV classes (MEL, BCC, AK, BKL, SCC) at λ=2, both ID sets: 0.10–0.34.

<!-- HEADLINE END -->

Statistics fit on ISIC train only; no domain labels in the scoring path. Mahalanobis: class-conditional, pooled covariance + 1e-3 I. kNN: k=50, L2-normalised, mean distance. ID full = ISIC test (n=5067); ID lesion-disjoint = ISIC test rows whose `lesion_id` is absent from ISIC train and val (n=2026, of which 436 have null `lesion_id`, kept as their own group).

## Primary: frozen ImageNet ResNet-50 (λ-independent baseline)

Decision rule: Mahalanobis AUROC ≥ 0.90 in all four cells → stays high.

| Detector | OOD set | ID full | 95% CI | ID lesion-disjoint | 95% CI |
|---|---|---|---|---|---|
| Mahalanobis | pad_heldout | 0.8326 | [0.8140, 0.8512] | 0.8475 | [0.8294, 0.8658] |
| Mahalanobis | fitzpatrick17k | 0.9878 | [0.9858, 0.9896] | 0.9902 | [0.9880, 0.9923] |
| kNN k=50 | pad_heldout | 0.9133 | [0.9014, 0.9245] | 0.9154 | [0.9019, 0.9283] |
| kNN k=50 | fitzpatrick17k | 0.9819 | [0.9795, 0.9842] | 0.9831 | [0.9804, 0.9859] |

CI: cluster bootstrap (2000). ID resampled by lesion group; pad_heldout by patient; Fitzpatrick17k by image.

## Mahalanobis — pad_heldout, by representation and λ (mean ± s.d. across seeds)

| Representation | Adversarial gradient | λ | n | ID full | ID lesion-disjoint |
|---|---|---:|---:|---|---|
| Frozen ImageNet ResNet-50 (2048-d) | no | all | — | 0.833 | 0.847 |
| Context branch z_c (64-d) | no | 0 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 0.25 | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 0.5 | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 1 | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 2 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 4 | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 8 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Lesion backbone, pre-projection (1536-d) | yes | 0 | 5 | 0.749 ± 0.052 | 0.841 ± 0.030 |
| Lesion backbone, pre-projection (1536-d) | yes | 0.25 | 3 | 0.629 ± 0.011 | 0.721 ± 0.010 |
| Lesion backbone, pre-projection (1536-d) | yes | 0.5 | 3 | 0.557 ± 0.015 | 0.659 ± 0.014 |
| Lesion backbone, pre-projection (1536-d) | yes | 1 | 3 | 0.548 ± 0.049 | 0.645 ± 0.047 |
| Lesion backbone, pre-projection (1536-d) | yes | 2 | 5 | 0.489 ± 0.040 | 0.595 ± 0.038 |
| Lesion backbone, pre-projection (1536-d) | yes | 4 | 3 | 0.497 ± 0.041 | 0.594 ± 0.036 |
| Lesion backbone, pre-projection (1536-d) | yes | 8 | 5 | 0.525 ± 0.035 | 0.623 ± 0.031 |
| z_lesion^norm (16-d) | yes | 0 | 5 | 0.843 ± 0.023 | 0.890 ± 0.015 |
| z_lesion^norm (16-d) | yes | 0.25 | 3 | 0.513 ± 0.018 | 0.597 ± 0.032 |
| z_lesion^norm (16-d) | yes | 0.5 | 3 | 0.465 ± 0.015 | 0.553 ± 0.012 |
| z_lesion^norm (16-d) | yes | 1 | 3 | 0.455 ± 0.036 | 0.535 ± 0.036 |
| z_lesion^norm (16-d) | yes | 2 | 5 | 0.427 ± 0.025 | 0.515 ± 0.026 |
| z_lesion^norm (16-d) | yes | 4 | 3 | 0.455 ± 0.039 | 0.545 ± 0.040 |
| z_lesion^norm (16-d) | yes | 8 | 5 | 0.494 ± 0.027 | 0.584 ± 0.016 |

## Mahalanobis — fitzpatrick17k, by representation and λ (mean ± s.d. across seeds)

| Representation | Adversarial gradient | λ | n | ID full | ID lesion-disjoint |
|---|---|---:|---:|---|---|
| Frozen ImageNet ResNet-50 (2048-d) | no | all | — | 0.988 | 0.990 |
| Context branch z_c (64-d) | no | 0 | 5 | 0.999 ± 0.001 | 0.999 ± 0.001 |
| Context branch z_c (64-d) | no | 0.25 | 3 | 0.999 ± 0.001 | 0.999 ± 0.001 |
| Context branch z_c (64-d) | no | 0.5 | 3 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| Context branch z_c (64-d) | no | 1 | 3 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| Context branch z_c (64-d) | no | 2 | 5 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| Context branch z_c (64-d) | no | 4 | 3 | 0.998 ± 0.002 | 0.998 ± 0.002 |
| Context branch z_c (64-d) | no | 8 | 5 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| Lesion backbone, pre-projection (1536-d) | yes | 0 | 5 | 0.813 ± 0.051 | 0.883 ± 0.028 |
| Lesion backbone, pre-projection (1536-d) | yes | 0.25 | 3 | 0.603 ± 0.076 | 0.712 ± 0.055 |
| Lesion backbone, pre-projection (1536-d) | yes | 0.5 | 3 | 0.587 ± 0.031 | 0.697 ± 0.027 |
| Lesion backbone, pre-projection (1536-d) | yes | 1 | 3 | 0.612 ± 0.102 | 0.707 ± 0.089 |
| Lesion backbone, pre-projection (1536-d) | yes | 2 | 5 | 0.516 ± 0.064 | 0.631 ± 0.056 |
| Lesion backbone, pre-projection (1536-d) | yes | 4 | 3 | 0.532 ± 0.048 | 0.634 ± 0.047 |
| Lesion backbone, pre-projection (1536-d) | yes | 8 | 5 | 0.595 ± 0.099 | 0.695 ± 0.082 |
| z_lesion^norm (16-d) | yes | 0 | 5 | 0.764 ± 0.038 | 0.830 ± 0.027 |
| z_lesion^norm (16-d) | yes | 0.25 | 3 | 0.439 ± 0.061 | 0.536 ± 0.052 |
| z_lesion^norm (16-d) | yes | 0.5 | 3 | 0.449 ± 0.025 | 0.547 ± 0.029 |
| z_lesion^norm (16-d) | yes | 1 | 3 | 0.457 ± 0.088 | 0.548 ± 0.086 |
| z_lesion^norm (16-d) | yes | 2 | 5 | 0.391 ± 0.068 | 0.490 ± 0.072 |
| z_lesion^norm (16-d) | yes | 4 | 3 | 0.431 ± 0.036 | 0.533 ± 0.036 |
| z_lesion^norm (16-d) | yes | 8 | 5 | 0.474 ± 0.060 | 0.578 ± 0.046 |

## kNN k=50 — pad_heldout, by representation and λ (mean ± s.d. across seeds)

| Representation | Adversarial gradient | λ | n | ID full | ID lesion-disjoint |
|---|---|---:|---:|---|---|
| Frozen ImageNet ResNet-50 (2048-d) | no | all | — | 0.913 | 0.915 |
| Context branch z_c (64-d) | no | 0 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 0.25 | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 0.5 | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 1 | 3 | 1.000 ± 0.000 | 0.999 ± 0.000 |
| Context branch z_c (64-d) | no | 2 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 4 | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Context branch z_c (64-d) | no | 8 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Lesion backbone, pre-projection (1536-d) | yes | 0 | 5 | 0.957 ± 0.008 | 0.966 ± 0.006 |
| Lesion backbone, pre-projection (1536-d) | yes | 0.25 | 3 | 0.652 ± 0.014 | 0.709 ± 0.018 |
| Lesion backbone, pre-projection (1536-d) | yes | 0.5 | 3 | 0.592 ± 0.014 | 0.661 ± 0.012 |
| Lesion backbone, pre-projection (1536-d) | yes | 1 | 3 | 0.567 ± 0.031 | 0.626 ± 0.034 |
| Lesion backbone, pre-projection (1536-d) | yes | 2 | 5 | 0.514 ± 0.043 | 0.585 ± 0.040 |
| Lesion backbone, pre-projection (1536-d) | yes | 4 | 3 | 0.518 ± 0.031 | 0.577 ± 0.026 |
| Lesion backbone, pre-projection (1536-d) | yes | 8 | 5 | 0.536 ± 0.052 | 0.587 ± 0.037 |
| z_lesion^norm (16-d) | yes | 0 | 5 | 0.918 ± 0.009 | 0.935 ± 0.008 |
| z_lesion^norm (16-d) | yes | 0.25 | 3 | 0.541 ± 0.020 | 0.609 ± 0.036 |
| z_lesion^norm (16-d) | yes | 0.5 | 3 | 0.498 ± 0.018 | 0.580 ± 0.016 |
| z_lesion^norm (16-d) | yes | 1 | 3 | 0.471 ± 0.027 | 0.545 ± 0.022 |
| z_lesion^norm (16-d) | yes | 2 | 5 | 0.440 ± 0.012 | 0.521 ± 0.012 |
| z_lesion^norm (16-d) | yes | 4 | 3 | 0.483 ± 0.033 | 0.565 ± 0.031 |
| z_lesion^norm (16-d) | yes | 8 | 5 | 0.507 ± 0.038 | 0.588 ± 0.034 |

## kNN k=50 — fitzpatrick17k, by representation and λ (mean ± s.d. across seeds)

| Representation | Adversarial gradient | λ | n | ID full | ID lesion-disjoint |
|---|---|---:|---:|---|---|
| Frozen ImageNet ResNet-50 (2048-d) | no | all | — | 0.982 | 0.983 |
| Context branch z_c (64-d) | no | 0 | 5 | 0.999 ± 0.000 | 0.999 ± 0.001 |
| Context branch z_c (64-d) | no | 0.25 | 3 | 0.999 ± 0.001 | 0.998 ± 0.000 |
| Context branch z_c (64-d) | no | 0.5 | 3 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| Context branch z_c (64-d) | no | 1 | 3 | 0.999 ± 0.000 | 0.999 ± 0.001 |
| Context branch z_c (64-d) | no | 2 | 5 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| Context branch z_c (64-d) | no | 4 | 3 | 0.999 ± 0.001 | 0.998 ± 0.002 |
| Context branch z_c (64-d) | no | 8 | 5 | 0.999 ± 0.000 | 0.999 ± 0.000 |
| Lesion backbone, pre-projection (1536-d) | yes | 0 | 5 | 0.913 ± 0.025 | 0.933 ± 0.019 |
| Lesion backbone, pre-projection (1536-d) | yes | 0.25 | 3 | 0.636 ± 0.091 | 0.702 ± 0.071 |
| Lesion backbone, pre-projection (1536-d) | yes | 0.5 | 3 | 0.634 ± 0.054 | 0.702 ± 0.047 |
| Lesion backbone, pre-projection (1536-d) | yes | 1 | 3 | 0.672 ± 0.073 | 0.720 ± 0.068 |
| Lesion backbone, pre-projection (1536-d) | yes | 2 | 5 | 0.570 ± 0.086 | 0.639 ± 0.073 |
| Lesion backbone, pre-projection (1536-d) | yes | 4 | 3 | 0.595 ± 0.084 | 0.648 ± 0.072 |
| Lesion backbone, pre-projection (1536-d) | yes | 8 | 5 | 0.658 ± 0.120 | 0.703 ± 0.098 |
| z_lesion^norm (16-d) | yes | 0 | 5 | 0.837 ± 0.030 | 0.872 ± 0.024 |
| z_lesion^norm (16-d) | yes | 0.25 | 3 | 0.476 ± 0.054 | 0.556 ± 0.045 |
| z_lesion^norm (16-d) | yes | 0.5 | 3 | 0.486 ± 0.034 | 0.574 ± 0.035 |
| z_lesion^norm (16-d) | yes | 1 | 3 | 0.499 ± 0.063 | 0.575 ± 0.054 |
| z_lesion^norm (16-d) | yes | 2 | 5 | 0.423 ± 0.059 | 0.509 ± 0.058 |
| z_lesion^norm (16-d) | yes | 4 | 3 | 0.467 ± 0.033 | 0.555 ± 0.025 |
| z_lesion^norm (16-d) | yes | 8 | 5 | 0.504 ± 0.073 | 0.593 ± 0.060 |

## Caveats required by the work order

1. Frozen ImageNet is the primary test; it has never seen any of this data. z_c is secondary: Fitzpatrick17k is clinical photography, the same modality as PAD, and z_c was trained to separate clinical photography from dermoscopy, so a high z_c score may reflect learned PAD-likeness rather than generic monitoring.
2. The ISIC test split is image-level and lesions recur between train and test (3041 of 5067 test images share a `lesion_id` with train or val). The ID side is therefore reported on the lesion-disjoint subset as well as on the full split.
3. Fitzpatrick17k has no patient identifiers; its bootstrap is over images and the interval is anti-conservative.
4. The lesion backbone (pre-projection) received adversarial gradient; it is not a remedy whatever this item returns.
