# SD-198 image vs P13-style cluster_union bootstrap

Clusters: `results/paperB/r5/p13/sd198_clusters.csv` (copied from `r5/d1_domain/`; method = DCT pHash Hamming ≤ 8 ∪ ImageNet-R50 cosine ≥ 0.95, connected components of the union — same as Fitzpatrick P13). Commit `e8ddcc2`.

## Cluster count

| | |
|---|---:|
| images | 6584 |
| cluster_union | 6016 |
| singletons | 5717 |
| largest cluster | 142 |
| effective n | 1517.9 |
| n_edges_phash | 449 |
| n_edges_embed | 363 |

PREREGISTER 2,000 draws, rng 20261005; ID = lesion; Mahalanobis class-conditional on `z_lesion_norm`. Rule A = pooled CI hi < 0.5 and k_below ≥ 3.

## P7(a) — Table 1 runs

| cell | image (old) | cluster_union (new) | verdict flip |
|---|---|---|---|
| P7a λ=0 | 0.713 [0.700, 0.727] k=0/5 ruleA=not | 0.713 [0.700, 0.727] k=0/5 ruleA=not | — |
| P7a λ=0.25 | 0.483 [0.472, 0.495] k=3/5 ruleA=met | 0.483 [0.469, 0.496] k=3/5 ruleA=met | — |
| P7a λ=0.5 | 0.390 [0.378, 0.401] k=5/5 ruleA=met | 0.390 [0.376, 0.403] k=5/5 ruleA=met | — |
| P7a λ=1 | 0.357 [0.346, 0.369] k=5/5 ruleA=met | 0.357 [0.344, 0.371] k=5/5 ruleA=met | — |
| P7a λ=2 | 0.375 [0.363, 0.386] k=5/5 ruleA=met | 0.375 [0.362, 0.387] k=5/5 ruleA=met | — |
| P7a λ=4 | 0.342 [0.331, 0.353] k=5/5 ruleA=met | 0.342 [0.330, 0.354] k=5/5 ruleA=met | — |
| P7a λ=8 | 0.379 [0.367, 0.391] k=5/5 ruleA=met | 0.379 [0.365, 0.391] k=5/5 ruleA=met | — |

## R1 — dose cells

| cell | image (old) | cluster_union (new) | verdict flip |
|---|---|---|---|
| R1 shift 0% λ=0.5 [excl degraded] | 0.664 [0.654, 0.675] k=0/5 ruleA=not | 0.664 [0.651, 0.677] k=0/5 ruleA=not | — |
| R1 shift 0% λ=0.5 [all seeds] | 0.664 [0.654, 0.675] k=0/5 ruleA=not | 0.664 [0.651, 0.677] k=0/5 ruleA=not | — |
| R1 shift 0% λ=2 [excl degraded] | 0.694 [0.684, 0.704] k=0/4 ruleA=not | 0.694 [0.680, 0.706] k=0/4 ruleA=not | — |
| R1 shift 0% λ=2 [all seeds] | 0.720 [0.712, 0.729] k=0/5 ruleA=not | 0.720 [0.710, 0.730] k=0/5 ruleA=not | — |
| R1 shift 50% λ=0.5 [excl degraded] | 0.451 [0.439, 0.462] k=2/3 ruleA=not | 0.451 [0.436, 0.464] k=2/3 ruleA=not | — |
| R1 shift 50% λ=0.5 [all seeds] | 0.451 [0.439, 0.462] k=2/3 ruleA=not | 0.451 [0.436, 0.464] k=2/3 ruleA=not | — |
| R1 shift 50% λ=2 [excl degraded] | 0.414 [0.402, 0.426] k=3/3 ruleA=met | 0.414 [0.399, 0.427] k=3/3 ruleA=met | — |
| R1 shift 50% λ=2 [all seeds] | 0.414 [0.402, 0.426] k=3/3 ruleA=met | 0.414 [0.399, 0.427] k=3/3 ruleA=met | — |
| R1 shift 100% λ=0 [excl degraded] | 0.708 [0.695, 0.722] k=0/5 ruleA=not | 0.708 [0.694, 0.722] k=0/5 ruleA=not | — |
| R1 shift 100% λ=0 [all seeds] | 0.708 [0.695, 0.722] k=0/5 ruleA=not | 0.708 [0.694, 0.722] k=0/5 ruleA=not | — |
| R1 shift 100% λ=0.5 [excl degraded] | 0.365 [0.354, 0.377] k=5/5 ruleA=met | 0.365 [0.353, 0.378] k=5/5 ruleA=met | — |
| R1 shift 100% λ=0.5 [all seeds] | 0.365 [0.354, 0.377] k=5/5 ruleA=met | 0.365 [0.353, 0.378] k=5/5 ruleA=met | — |
| R1 shift 100% λ=2 [excl degraded] | 0.325 [0.315, 0.336] k=5/5 ruleA=met | 0.325 [0.314, 0.337] k=5/5 ruleA=met | — |
| R1 shift 100% λ=2 [all seeds] | 0.325 [0.315, 0.336] k=5/5 ruleA=met | 0.325 [0.314, 0.337] k=5/5 ruleA=met | — |

## detmech_shift

| cell | image (old) | cluster_union (new) | verdict flip |
|---|---|---|---|
| detmech A4 λ=0 (baseline) | 0.799 [0.789, 0.810] k=0/5 ruleA=not | 0.799 [0.786, 0.812] k=0/5 ruleA=not | — |
| detmech R1 λ=2 shift 0% | 0.694 [0.684, 0.704] k=0/4 ruleA=not | 0.694 [0.680, 0.706] k=0/4 ruleA=not | — |
| detmech R1 λ=2 shift 100% | 0.325 [0.315, 0.336] k=5/5 ruleA=met | 0.325 [0.314, 0.337] k=5/5 ruleA=met | — |

## R2

| cell | image (old) | cluster_union (new) | verdict flip |
|---|---|---|---|
| λ=0 (R1 shift100) λ=0 | 0.708 [0.695, 0.722] k=0/5 ruleA=not | 0.708 [0.694, 0.722] k=0/5 ruleA=not | — |
| DANN (R1 shift100) λ=0.5 | 0.365 [0.354, 0.377] k=5/5 ruleA=met | 0.365 [0.353, 0.378] k=5/5 ruleA=met | — |
| R2 CDAN λ=0.5 | 0.358 [0.346, 0.370] k=2/2 ruleA=not | 0.358 [0.345, 0.371] k=2/2 ruleA=not | — |
| R2 IWCDAN λ=0.5 | 0.396 [0.384, 0.409] k=5/5 ruleA=met | 0.396 [0.380, 0.412] k=5/5 ruleA=met | — |
| R2 IWDAN λ=0.5 | 0.607 [0.594, 0.620] k=0/5 ruleA=not | 0.607 [0.590, 0.622] k=0/5 ruleA=not | — |
| DANN (R1 shift100) λ=2 | 0.325 [0.315, 0.336] k=5/5 ruleA=met | 0.325 [0.314, 0.337] k=5/5 ruleA=met | — |
| R2 CDAN λ=2 | 0.401 [0.389, 0.413] k=5/5 ruleA=met | 0.401 [0.388, 0.414] k=5/5 ruleA=met | — |
| R2 IWCDAN λ=2 | 0.381 [0.369, 0.393] k=5/5 ruleA=met | 0.381 [0.367, 0.394] k=5/5 ruleA=met | — |
| R2 IWDAN λ=2 | 0.488 [0.476, 0.500] k=2/5 ruleA=not | 0.488 [0.473, 0.502] k=2/5 ruleA=not | — |
| R2 IWDAN_ORACLE λ=2 | 0.693 [0.682, 0.705] k=0/5 ruleA=not | 0.693 [0.677, 0.708] k=0/5 ruleA=not | — |

## S1/S2 (lesion Table-1 block)

| cell | image (old) | cluster_union (new) | verdict flip |
|---|---|---|---|
| S1/S2 (Table1) λ=0 | 0.713 [0.700, 0.727] k=0/5 ruleA=not | 0.713 [0.700, 0.727] k=0/5 ruleA=not | — |
| S1/S2 (Table1) λ=0.25 | 0.483 [0.472, 0.495] k=3/5 ruleA=met | 0.483 [0.469, 0.496] k=3/5 ruleA=met | — |
| S1/S2 (Table1) λ=0.5 | 0.390 [0.378, 0.401] k=5/5 ruleA=met | 0.390 [0.376, 0.403] k=5/5 ruleA=met | — |
| S1/S2 (Table1) λ=1 | 0.357 [0.346, 0.369] k=5/5 ruleA=met | 0.357 [0.344, 0.371] k=5/5 ruleA=met | — |
| S1/S2 (Table1) λ=2 | 0.375 [0.363, 0.386] k=5/5 ruleA=met | 0.375 [0.362, 0.387] k=5/5 ruleA=met | — |
| S1/S2 (Table1) λ=4 | 0.342 [0.331, 0.353] k=5/5 ruleA=met | 0.342 [0.330, 0.354] k=5/5 ruleA=met | — |
| S1/S2 (Table1) λ=8 | 0.379 [0.367, 0.391] k=5/5 ruleA=met | 0.379 [0.365, 0.391] k=5/5 ruleA=met | — |

## Clinical (acceptance rates; no rule A)

| cell | class | n | point | image CI | cluster CI |
|---|---|---:|---:|---|---|
| R1 shift 0% λ=2 | MEL | 113 | 0.929 | [0.896, 0.958] | [0.897, 0.961] |
| R1 shift 0% λ=2 | BCC | 60 | 0.892 | [0.838, 0.942] | [0.835, 0.942] |
| R1 shift 0% λ=2 | SCC | 119 | 0.855 | [0.821, 0.887] | [0.823, 0.887] |
| R1 shift 0% λ=2 | AK | 60 | 0.946 | [0.912, 0.975] | [0.914, 0.977] |
| R1 shift 100% λ=2 | MEL | 113 | 0.995 | [0.988, 1.000] | [0.989, 1.000] |
| R1 shift 100% λ=2 | BCC | 60 | 0.990 | [0.977, 1.000] | [0.977, 1.000] |
| R1 shift 100% λ=2 | SCC | 119 | 0.982 | [0.963, 0.995] | [0.964, 0.993] |
| R1 shift 100% λ=2 | AK | 60 | 0.997 | [0.990, 1.000] | [0.989, 1.000] |
| a4 λ=0 | MEL | 113 | 0.919 | [0.887, 0.945] | [0.885, 0.947] |
| a4 λ=0 | BCC | 60 | 0.760 | [0.697, 0.813] | [0.697, 0.820] |
| a4 λ=0 | SCC | 119 | 0.773 | [0.724, 0.820] | [0.726, 0.819] |
| a4 λ=0 | AK | 60 | 0.787 | [0.733, 0.840] | [0.731, 0.833] |
| a4 λ=2 | MEL | 113 | 0.979 | [0.956, 0.995] | [0.958, 0.994] |
| a4 λ=2 | BCC | 60 | 1.000 | [1.000, 1.000] | [1.000, 1.000] |
| a4 λ=2 | SCC | 119 | 0.976 | [0.960, 0.990] | [0.960, 0.990] |
| a4 λ=2 | AK | 60 | 0.997 | [0.990, 1.000] | [0.991, 1.000] |
| old_recipe λ=0 | MEL | 113 | 0.977 | [0.958, 0.991] | [0.960, 0.992] |
| old_recipe λ=0 | BCC | 60 | 0.983 | [0.963, 0.997] | [0.963, 0.997] |
| old_recipe λ=0 | SCC | 119 | 0.953 | [0.933, 0.971] | [0.931, 0.972] |
| old_recipe λ=0 | AK | 60 | 0.990 | [0.977, 1.000] | [0.977, 1.000] |
| old_recipe λ=2 | MEL | 113 | 0.986 | [0.973, 0.995] | [0.975, 0.996] |
| old_recipe λ=2 | BCC | 60 | 0.990 | [0.980, 1.000] | [0.977, 1.000] |
| old_recipe λ=2 | SCC | 119 | 0.966 | [0.948, 0.983] | [0.947, 0.983] |
| old_recipe λ=2 | AK | 60 | 0.990 | [0.973, 1.000] | [0.972, 1.000] |

## R1 (a) / (b)

R1 (a) wild-cluster bootstrap clusters = setting cells (recipe|λ|shift_level), not SD-198 images/near-duplicates; y = seed-level Mahalanobis point AUROC from summary.json. R1 (b) is a seed bootstrap of those same point AUROCs. Neither path resamples SD-198, so switching SD-198 from image to cluster_union does not change (a) or (b) CIs or verdicts.

- (a) code: `scripts/r6_r1_leakage_matched.py:244-247 cluster = recipe|lambda|shift_level`
- (b) code: `scripts/r6_r1_leakage_matched.py:283-296 seed bootstrap of point AUROCs`
- AUROC inputs: `maha_of(summary.json) ← metrics.ood.sd198.mahalanobis_classcond (point, no resample)`

## Verdict flips (rule A image → cluster)

- none

