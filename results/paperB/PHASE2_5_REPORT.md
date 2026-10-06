# Paper B — Phase 2.5 report

Inference only. Detectors, splits and hyperparameters were not adjusted toward a predicted value.
Phase 2 prediction (held-out PAD restores z_lesion Maha ≈ 0.50) **FAILED** and is left as a finding.

## Which explanation the data support

**structural: inversion is not PAD-image memorisation and is not class composition**

- 2.5a Fitzpatrick17k z_lesion Maha: **0.399 ± 0.039** — ≈0.41 (structural).
- 2.5b PAD unrestricted: **0.409 ± 0.022**; 6-class restricted: **0.414 ± 0.022**; reweighted: **0.240 ± 0.014**.
- 2.5b reading: class composition is not the driver (restriction moved AUROC by <0.02).

## 2.5a — Unseen-domain test (Fitzpatrick17k, n=3887)

ID = ISIC test. Statistics fit on ISIC train only. OOD = Fitzpatrick17k (never in any training branch, any seed, any phase). 5 `runB_orth1` seeds.

| Detector | z_lesion / 16-d z / backbone AUROC |
|---|---|
| Mahalanobis class-cond | 0.399 ± 0.039 |
| MSP | 0.413 ± 0.043 |
| Energy_T1 | 0.478 ± 0.035 |
| cosine_max | 0.425 ± 0.039 |
| knn_k50 | 0.406 ± 0.041 |

Pre-registered reading: AUROC ≈ 0.50 → domain-specific mapping that generalises across PAD but not beyond it; AUROC ≈ 0.41 → structural / class-composition. Observed: **0.399 ± 0.039**.

Same Fitzpatrick suite on the two controls:

| Model | Maha | MSP | Energy | cosine | kNN |
|---|---|---|---|---|---|
| z_lesion (CSG runB_orth1) | 0.399 ± 0.039 | 0.413 ± 0.043 | 0.478 ± 0.035 | 0.425 ± 0.039 | 0.406 ± 0.041 |
| ResNet-50 backbone_raw | 0.994 ± 0.001 | 0.707 ± 0.035 | 0.694 ± 0.051 | 0.844 ± 0.011 | 0.956 ± 0.005 |
| EffNet-B3 16-d z | 0.649 ± 0.032 | 0.716 ± 0.024 | 0.761 ± 0.026 | 0.704 ± 0.029 | 0.730 ± 0.014 |

## 2.5b — Class-composition control

ISIC test has 8 classes; PAD has 6 (no DF, no VASC) and is concentrated in BCC / AK / NEV. Class-conditional Mahalanobis takes distance to the nearest class centroid. Rare ISIC-only classes can drive AUROC below 0.5 with no adversarial explanation.

| Model | Unrestricted (Phase 1 headline protocol) | ID restricted to PAD's 6 classes | ID reweighted to PAD class mix |
|---|---|---|---|
| z_lesion (CSG runB_orth1) | 0.409 ± 0.022 | 0.414 ± 0.022 | 0.240 ± 0.014 |
| ResNet-50 backbone_raw | 0.868 ± 0.021 | 0.875 ± 0.020 | 0.852 ± 0.024 |
| EffNet-B3 16-d z | 0.739 ± 0.023 | 0.749 ± 0.023 | 0.523 ± 0.029 |

Phase 1 headlines under the 6-class restriction: if a number moves, the published inversion was partly a class-mix artifact. If it does not, the inversion survives the confound.

### Per-class Mahalanobis (median, IQR, n) — CSG `z_lesion`, mean across seeds

| Class | in PAD | n ID | n PAD | ID median (IQR) | PAD median (IQR) |
|---|---|---:|---:|---|---|
| MEL | yes | 904 | 52 | 15.0 ± 0.3 (15.0 ± 0.9) | 11.0 ± 1.8 (14.8 ± 1.4) |
| NV | yes | 2575 | 244 | 7.2 ± 0.6 (11.0 ± 0.8) | 4.5 ± 0.6 (6.5 ± 1.0) |
| BCC | yes | 665 | 845 | 26.0 ± 2.2 (23.9 ± 3.0) | 10.0 ± 0.7 (19.9 ± 2.1) |
| AK | yes | 173 | 730 | 37.3 ± 1.8 (25.7 ± 3.5) | 9.1 ± 1.8 (18.6 ± 3.7) |
| BKL | yes | 525 | 235 | 23.2 ± 1.0 (18.5 ± 2.1) | 8.1 ± 1.1 (18.0 ± 2.4) |
| DF | **no** | 48 | 0 | 46.6 ± 3.8 (35.8 ± 5.8) | — (—) |
| VASC | **no** | 51 | 0 | 39.4 ± 6.2 (37.9 ± 8.4) | — (—) |
| SCC | yes | 126 | 192 | 40.0 ± 2.9 (27.2 ± 4.3) | 13.3 ± 1.4 (26.6 ± 5.2) |

Drivers: classes whose ID median Maha exceeds the PAD median are the ones that make ID look more OOD than PAD.

Same table for baseline and EffB3: `phase2_5/per_class_maha.json`.

## 2.5c — Fitzpatrick17k axis: distributions, not means

Shared k=1 axis of `z_context` (ISIC-train PCA, PAD-positive).

| Dataset | n | mean | std | median | IQR |
|---|---:|---:|---:|---:|---:|
| ISIC test | 5067 | 0.07 | 2.58 | -0.00 | 2.86 |
| PAD | 2298 | 44.80 | 2.85 | 44.82 | 3.79 |
| Fitzpatrick17k | 3887 | 35.56 | 10.71 | 38.80 | 13.66 |

| Pair | overlap coefficient | Cohen's d |
|---|---:|---:|
| ISIC ↔ Fitz | 0.028 | -4.56 |
| Fitz ↔ PAD | 0.472 | -1.18 |
| ISIC ↔ PAD | 0.000 | -16.45 |

- Fraction of Fitzpatrick17k outside the ISIC–PAD **mean** interval [0.07, 44.80]: **0.176**.
- Fraction outside the ISIC–PAD **minmax** interval: **0.000**.

**Supported claim:** not on the ISIC side (Fitz overlaps PAD substantially; cannot claim a tight between-mode).

Density plot: `results/paperB/figures/domain_axis_distributions.pdf`.

Means alone (ISIC 0.07, Fitz 36, PAD 45) overstated tightness. Fitz std is 10.71 against an ISIC–PAD gap of 44.7. Fitz spread overlaps PAD (std 10.7 vs ISIC–PAD gap 44.7); means-only 'between' overstates tightness.

## Reframing locked from Phase 1.6a

A frozen ImageNet ResNet-50 that has never seen this data reaches **linear-head domain AUROC 0.998**. A domain monitor is cheap. The ISIC/PAD distinction is linearly decodable from generic visual features.

`z_context` is **not** a better detector. Factorization does not make domain detectable — it already is, from any backbone. Factorization concentrates domain into a single interpretable axis: k=1 gives AUROC 1.00 at 84.5% variance for `z_context`, versus 0.55–0.70 at k=1 for every other representation tested.

No mechanism claim is written beyond what 2.5a/b distinguish. Phase 3 (`λ_adv` sweep) is the controlled intervention.

## Files

- `results/paperB/phase2_5/per_seed/`
- `results/paperB/phase2_5/phase25_aggregate.json`
- `results/paperB/phase2_5/axis_distributions.json`
- `results/paperB/figures/domain_axis_distributions.pdf`

