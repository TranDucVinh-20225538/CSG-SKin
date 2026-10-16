# Paper B R5 P13 — Fitzpatrick17k bootstrap over near-duplicate clusters

Design: `decisions/paperB_r5_p13.md`. Commit `a7195ca`. ID side: lesion-level ISIC test by lesion group; Fitzpatrick side resampled by the cluster column; PREREGISTER draws (2,000, seed 20261005), z_lesion_norm.

| Resample unit | clusters | singletons | largest | effective n |
|---|---:|---:|---:|---:|
| image | 3887 | 3887 | 1 | 3887 |
| cluster_phash | 3876 | 3867 | 4 | 3859 |
| cluster_embed | 3752 | 3646 | 16 | 3422 |
| cluster_union | 3747 | 3639 | 16 | 3408 |
| cluster_imageid_run | 222 | 17 | 175 | 68 |
| cluster_diagnosis | 90 | 2 | 257 | 38 |

## Mahalanobis, pooled mean and 95% CI by resample unit

| λ | n | image | cluster_phash | cluster_embed | cluster_union | cluster_imageid_run | cluster_diagnosis |
|---:|---:|---|---|---|---|---|---|
| 0 | 5 | 0.717 [0.703, 0.730] k=0 | 0.717 [0.703, 0.730] k=0 | 0.717 [0.703, 0.730] k=0 | 0.717 [0.703, 0.730] k=0 | 0.717 [0.700, 0.734] k=0 | 0.717 [0.698, 0.734] k=0 |
| 0.25 | 5 | 0.496 [0.484, 0.509] k=3 | 0.496 [0.484, 0.508] k=3 | 0.496 [0.484, 0.508] k=3 | 0.496 [0.484, 0.509] k=3 | 0.496 [0.475, 0.521] k=3 | 0.496 [0.472, 0.523] k=3 |
| 0.5 | 5 | 0.411 [0.399, 0.423] k=5 | 0.411 [0.399, 0.423] k=5 | 0.411 [0.398, 0.424] k=5 | 0.411 [0.398, 0.424] k=5 | 0.411 [0.386, 0.437] k=4 | 0.411 [0.383, 0.442] k=4 |
| 1 | 5 | 0.375 [0.363, 0.387] k=5 | 0.375 [0.363, 0.387] k=5 | 0.375 [0.363, 0.387] k=5 | 0.375 [0.362, 0.388] k=5 | 0.375 [0.349, 0.404] k=5 | 0.375 [0.346, 0.407] k=5 |
| 2 | 5 | 0.402 [0.389, 0.414] k=5 | 0.402 [0.389, 0.414] k=5 | 0.402 [0.389, 0.415] k=5 | 0.402 [0.389, 0.415] k=5 | 0.402 [0.374, 0.430] k=5 | 0.402 [0.373, 0.435] k=5 |
| 4 | 5 | 0.376 [0.364, 0.389] k=5 | 0.376 [0.365, 0.389] k=5 | 0.376 [0.364, 0.389] k=5 | 0.376 [0.364, 0.389] k=5 | 0.376 [0.351, 0.404] k=5 | 0.376 [0.349, 0.408] k=5 |
| 8 | 5 | 0.412 [0.399, 0.424] k=5 | 0.412 [0.399, 0.424] k=5 | 0.412 [0.398, 0.425] k=5 | 0.412 [0.398, 0.426] k=5 | 0.412 [0.385, 0.442] k=4 | 0.412 [0.382, 0.446] k=4 |

## CI width ratio vs image resampling (all detectors, λ = 2)

| Detector | cluster_phash | cluster_embed | cluster_union | cluster_imageid_run | cluster_diagnosis |
|---|---|---|---|---|---|
| Mahalanobis | 1.00 | 1.05 | 1.07 | 2.24 | 2.49 |
| kNN k=50 | 0.99 | 1.04 | 1.06 | 2.07 | 2.26 |
| Cosine | 1.01 | 1.05 | 1.06 | 2.18 | 2.36 |

## Plain reading

At λ = 2 the Fitzpatrick17k Mahalanobis CI is [0.389, 0.414] resampling images, [0.389, 0.415] resampling near-duplicate clusters (union) and [0.373, 0.435] resampling whole diagnoses; seeds with CI < 0.5: 5, 5, 5 of 5.

## LaTeX rows (λ & image CI & cluster-union CI & diagnosis CI), Mahalanobis

```latex
0 & [0.703, 0.730] & [0.703, 0.730] & [0.698, 0.734] \\
0.25 & [0.484, 0.509] & [0.484, 0.509] & [0.472, 0.523] \\
0.5 & [0.399, 0.423] & [0.398, 0.424] & [0.383, 0.442] \\
1 & [0.363, 0.387] & [0.362, 0.388] & [0.346, 0.407] \\
2 & [0.389, 0.414] & [0.389, 0.415] & [0.373, 0.435] \\
4 & [0.364, 0.389] & [0.364, 0.389] & [0.349, 0.408] \\
8 & [0.399, 0.424] & [0.398, 0.426] & [0.382, 0.446] \\
```

