# Paper B R5 D1: domain direction vs per-image pull

Decisions: `decisions/paperB_r5_d1.md`. Commit 0d57cb7. Bootstrap 2000 draws, rng seed 20261005; CI = percentile 95% of the seed mean (lesion-level: 5 matched seeds; BCN↔HAM: seed 42 only).

Pass criterion unchanged from `f0bb6ee` (committed before job 68092): ρ_dom > 0 and the CI of Δ = ρ_dom − |ρ_orth| entirely above 0; hypothesis supported if met in ≥ 4 of 7 sets (latent). The three D1 jobs differ only in load/IO fixes: job 68092 @ `97faa14` (FAILED — missing pyarrow/fastparquet for to_parquet); job 68190 @ `c4cc443` (FAILED — object group arrays need allow_pickle=True); job 68243 @ `54d5cd6` (COMPLETED (includes 5f7e4fd object-array load + lesion cache; csv.gz per-image)).

Domain classifier validity (balanced accuracy, ISIC test vs pad_heldout, λ = 0): latent 0.91351 ± 0.01698, backbone 0.96903 ± 0.00256; BCN↔HAM site classifiers (ID test vs OOD held-out): BCN→HAM 0.955 (adversary pool n = 6975), HAM→BCN 0.943 (adversary pool n = 8749).

## Per-image test (primary pull = d₀/med₀ − d₂/med₂; Spearman, mean over seeds)

Bootstrap unit per set is in the table; within a draw the same resample is applied to all 5 seed pairs (lesion-level) and the statistic is the seed mean (pooled over the 5 pairs). BCN↔HAM / HAM→BCN: n = 1 seed pair per direction; verdicts reported as is.

| set | unit | n (groups) | rep | ρ_dom [CI] | ρ_orth [CI] | Δ = ρ_dom − abs(ρ_orth) [CI] | criterion | partial ρ_dom given P₀ [CI] | partial ρ_orth [CI] |
|---|---|---|---|---|---|---|---|---|---|
| pad_heldout | patient | 716 (412) | latent | +0.344 [+0.303, +0.383] | -0.068 [-0.116, -0.019] | +0.272 [+0.202, +0.325] | met | +0.208 [+0.167, +0.246] | -0.255 [-0.294, -0.214] |
| pad_heldout | patient | 716 (412) | backbone | +0.439 [+0.400, +0.475] | -0.168 [-0.211, -0.118] | +0.272 [+0.219, +0.326] | met | +0.375 [+0.333, +0.412] | -0.246 [-0.289, -0.200] |
| Fitzpatrick17k | P13 cluster (pHash ∪ ImageNet-R50 cosine) | 3887 (3747) | latent | +0.419 [+0.404, +0.435] | -0.066 [-0.084, -0.047] | +0.354 [+0.329, +0.376] | met | +0.257 [+0.238, +0.277] | -0.239 [-0.255, -0.221] |
| Fitzpatrick17k | P13 cluster (pHash ∪ ImageNet-R50 cosine) | 3887 (3747) | backbone | +0.475 [+0.460, +0.490] | -0.120 [-0.140, -0.099] | +0.355 [+0.337, +0.374] | met | +0.398 [+0.378, +0.418] | -0.312 [-0.331, -0.292] |
| MED-NODE | image (no grouping exists) | 170 (170) | latent | +0.433 [+0.348, +0.510] | -0.144 [-0.228, -0.058] | +0.289 [+0.180, +0.374] | met | +0.167 [+0.082, +0.253] | -0.081 [-0.163, +0.003] |
| MED-NODE | image (no grouping exists) | 170 (170) | backbone | +0.443 [+0.354, +0.521] | +0.191 [+0.111, +0.272] | +0.251 [+0.130, +0.348] | met | +0.187 [+0.090, +0.280] | -0.230 [-0.298, -0.147] |
| SD-198 | P13-method cluster (pHash ∪ ImageNet-R50 cosine) | 6584 (6016) | latent | +0.467 [+0.455, +0.478] | -0.030 [-0.044, -0.017] | +0.435 [+0.417, +0.450] | met | +0.300 [+0.285, +0.313] | -0.234 [-0.247, -0.221] |
| SD-198 | P13-method cluster (pHash ∪ ImageNet-R50 cosine) | 6584 (6016) | backbone | +0.540 [+0.529, +0.551] | -0.162 [-0.176, -0.146] | +0.378 [+0.363, +0.394] | met | +0.462 [+0.449, +0.474] | -0.385 [-0.397, -0.372] |
| SCIN | case | 10406 (5033) | latent | +0.333 [+0.322, +0.343] | +0.021 [+0.010, +0.032] | +0.303 [+0.289, +0.314] | met | +0.208 [+0.196, +0.218] | -0.112 [-0.123, -0.101] |
| SCIN | case | 10406 (5033) | backbone | +0.317 [+0.305, +0.328] | -0.060 [-0.072, -0.048] | +0.257 [+0.245, +0.269] | met | +0.241 [+0.229, +0.253] | -0.277 [-0.288, -0.265] |
| BCN→HAM (HAM held-out) | lesion | 3040 (2241) | latent | +0.136 [+0.096, +0.175] | +0.092 [+0.049, +0.135] | +0.044 [-0.017, +0.108] | not met | +0.129 [+0.088, +0.168] | -0.041 [-0.079, -0.004] |
| HAM→BCN (BCN held-out) | lesion | 3664 (1073) | latent | +0.350 [+0.311, +0.391] | +0.305 [+0.257, +0.353] | +0.045 [-0.013, +0.104] | not met | +0.228 [+0.185, +0.271] | -0.142 [-0.191, -0.092] |

Partial ρ (controlling for the initial distance percentile P₀) is secondary and has no pass/fail rule. Negative partial ρ_orth (pad_heldout latent, pad_heldout backbone, Fitzpatrick17k latent, Fitzpatrick17k backbone, MED-NODE latent, MED-NODE backbone, SD-198 latent, SD-198 backbone, SCIN latent, SCIN backbone, BCN→HAM (HAM held-out) latent, HAM→BCN (BCN held-out) latent): reported as is; any reading of the sign was formed after the seed-42 smoke test and is labelled as such.

Criterion (ρ_dom > 0 and Δ CI entirely above 0) met in **5 of 7** sets (latent) → hypothesis supported at the per-image level (rule fixed in decisions: ≥ 4 of 7). Backbone: 5 of 5 lesion-level sets.

## Post-hoc: λ = 0 domain-classifier separation (labelled; does not change verdicts)

margin = mean(decision_function(OOD)) - mean(decision_function(ID_test)); bal_acc = balanced accuracy of the λ = 0 domain classifier on ID test vs that OOD set. Labelled post-hoc; does not change criterion verdicts.

BCN↔HAM / HAM→BCN criterion rows above stay **not met** (n = 1 seed pair each). This table only asks whether the domain direction itself is weaker there than for ISIC↔PAD / other OOD sets.

| set | rep | classifier | n seeds | bal acc | margin (OOD − ID) |
|---|---|---|---:|---|---|
| pad_heldout | latent | ISIC↔pad_adv | 5 | 0.914 ± 0.017 | +5.652 ± 1.453 |
| pad_heldout | backbone | ISIC↔pad_adv | 5 | 0.969 ± 0.003 | +32.236 ± 2.320 |
| Fitzpatrick17k | latent | ISIC↔pad_adv | 5 | 0.730 ± 0.020 | +3.314 ± 0.747 |
| Fitzpatrick17k | backbone | ISIC↔pad_adv | 5 | 0.845 ± 0.029 | +18.683 ± 1.435 |
| MED-NODE | latent | ISIC↔pad_adv | 5 | 0.600 ± 0.041 | +1.751 ± 0.848 |
| MED-NODE | backbone | ISIC↔pad_adv | 5 | 0.719 ± 0.034 | +12.249 ± 1.574 |
| SD-198 | latent | ISIC↔pad_adv | 5 | 0.734 ± 0.027 | +3.355 ± 0.891 |
| SD-198 | backbone | ISIC↔pad_adv | 5 | 0.855 ± 0.026 | +19.537 ± 1.359 |
| SCIN | latent | ISIC↔pad_adv | 5 | 0.656 ± 0.031 | +2.317 ± 0.124 |
| SCIN | backbone | ISIC↔pad_adv | 5 | 0.773 ± 0.042 | +14.021 ± 1.876 |
| BCN→HAM (HAM held-out) | latent | BCN→HAM | 1 | 0.955 | +38.528 |
| HAM→BCN (BCN held-out) | latent | HAM→BCN | 1 | 0.941 | +38.421 |

## Secondary pulls (points, mean over seeds)

| set | rep | ρ_dom pct | ρ_orth pct | ρ_dom raw | ρ_orth raw |
|---|---|---:|---:|---:|---:|
| pad_heldout | latent | +0.277 | -0.158 | +0.348 | -0.064 |
| pad_heldout | backbone | +0.420 | -0.211 | +0.441 | -0.165 |
| Fitzpatrick17k | latent | +0.376 | -0.143 | +0.421 | -0.063 |
| Fitzpatrick17k | backbone | +0.481 | -0.193 | +0.476 | -0.116 |
| MED-NODE | latent | +0.414 | -0.155 | +0.430 | -0.144 |
| MED-NODE | backbone | +0.430 | +0.170 | +0.441 | +0.194 |
| SD-198 | latent | +0.425 | -0.117 | +0.468 | -0.026 |
| SD-198 | backbone | +0.549 | -0.242 | +0.540 | -0.157 |
| SCIN | latent | +0.312 | -0.029 | +0.334 | +0.023 |
| SCIN | backbone | +0.320 | -0.109 | +0.317 | -0.054 |
| BCN→HAM (HAM held-out) | latent | +0.129 | +0.320 | +0.136 | +0.091 |
| HAM→BCN (BCN held-out) | latent | +0.034 | -0.431 | +0.351 | +0.309 |

## Across sets (descriptive, no test) with class mix (R6-R5)

| set | mean domain score latent | mean domain score backbone | Maha AUROC λ0 | λ2 | ΔAUROC | ρ_dom latent | TV (class mix) | majority ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pad_heldout | +2.30 | +20.73 | 0.808 | 0.403 | -0.405 | +0.344 | 0.566 | 0.24 |
| Fitzpatrick17k | -0.04 | +7.18 | 0.717 | 0.402 | -0.315 | +0.419 | 0.630 | 0.01 |
| MED-NODE | -1.60 | +0.74 | 0.536 | 0.306 | -0.230 | +0.433 | 0.152 | 0.79 |
| SD-198 | +0.00 | +8.03 | 0.713 | 0.375 | -0.339 | +0.467 | 0.274 | 0.81 |
| SCIN | -1.04 | +2.51 | 0.726 | 0.559 | -0.168 | +0.333 | 0.625 | 0.15 |
| BCN→HAM (HAM held-out) | +16.39 | — | 0.379 | 0.383 | +0.004 | +0.136 | 0.355 | 1.98 |
| HAM→BCN (BCN held-out) | +20.58 | — | 0.849 | 0.649 | -0.199 | +0.350 | 0.355 | 0.51 |

Spearman over sets (descriptive): mean latent domain score vs −ΔAUROC: +0.90 (5 lesion-level sets), -0.11 (all 7; BCN↔HAM scores come from a different classifier); mean backbone domain score vs −ΔAUROC (5 lesion sets): +0.90; TV vs −ΔAUROC (7 sets): +0.00; ρ_dom vs −ΔAUROC (7 sets): +0.61.

## LaTeX rows (set & ρ_dom & ρ_orth & Δ [CI] & ΔAUROC & TV)

```latex
pad_heldout & +0.34 & -0.07 & +0.27 [+0.20, +0.33] & -0.405 & 0.57 \\
Fitzpatrick17k & +0.42 & -0.07 & +0.35 [+0.33, +0.38] & -0.315 & 0.63 \\
MED-NODE & +0.43 & -0.14 & +0.29 [+0.18, +0.37] & -0.230 & 0.15 \\
SD-198 & +0.47 & -0.03 & +0.44 [+0.42, +0.45] & -0.339 & 0.27 \\
SCIN & +0.33 & +0.02 & +0.30 [+0.29, +0.31] & -0.168 & 0.63 \\
BCN$\to$HAM (HAM held-out) & +0.14 & +0.09 & +0.04 [-0.02, +0.11] & +0.004 & 0.36 \\
HAM$\to$BCN (BCN held-out) & +0.35 & +0.31 & +0.04 [-0.01, +0.10] & -0.199 & 0.36 \\
```

