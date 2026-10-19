# (1) Table 1 last.ckpt

Source: `r5/t1last/UNSEEN_ECE_REPORT.md`.

# Table 1 last.ckpt — unseen Maha CIs, paired ID bal Δ, ECE

Checkpoint: `last.ckpt` of the 35 Table 1 runs (`r2/item4`). Features under `results/paperB/r5/t1last/features_last/`. Commit `d3250a1`. PREREGISTER 2,000 draws, rng 20261005; ID = lesion; Mahalanobis class-conditional on `z_lesion_norm`.

Units: MED-NODE image (P7a group = image); SD-198 image and P13-style `cluster_union` (pHash Hamming ≤ 8 ∪ ImageNet-R50 cosine ≥ 0.95; 6016 clusters / 6584 images); SCIN image and case (P7a `group`).

## (1) Mahalanobis AUROC — image vs group/cluster

### mednode

| λ | n | unit | n_groups | mean | 95% CI | k CI<0.5 | rule A |
|---:|---:|---|---:|---:|---|---:|---|
| 0 | 5 | image | 170 | 0.559 | [0.532, 0.587] | 0/5 | not met |
| 0 | 5 | group(=image) | 170 | 0.559 | [0.530, 0.589] | 0/5 | not met |
| 0.25 | 5 | image | 170 | 0.308 | [0.281, 0.334] | 5/5 | met |
| 0.25 | 5 | group(=image) | 170 | 0.308 | [0.281, 0.336] | 5/5 | met |
| 0.5 | 5 | image | 170 | 0.309 | [0.283, 0.335] | 5/5 | met |
| 0.5 | 5 | group(=image) | 170 | 0.309 | [0.284, 0.335] | 5/5 | met |
| 1 | 5 | image | 170 | 0.319 | [0.291, 0.345] | 5/5 | met |
| 1 | 5 | group(=image) | 170 | 0.319 | [0.293, 0.347] | 5/5 | met |
| 2 | 5 | image | 170 | 0.288 | [0.262, 0.314] | 5/5 | met |
| 2 | 5 | group(=image) | 170 | 0.288 | [0.263, 0.315] | 5/5 | met |
| 4 | 5 | image | 170 | 0.300 | [0.274, 0.328] | 5/5 | met |
| 4 | 5 | group(=image) | 170 | 0.300 | [0.274, 0.327] | 5/5 | met |
| 8 | 5 | image | 170 | 0.294 | [0.270, 0.319] | 5/5 | met |
| 8 | 5 | group(=image) | 170 | 0.294 | [0.270, 0.320] | 5/5 | met |

### sd198

| λ | n | unit | n_groups | mean | 95% CI | k CI<0.5 | rule A |
|---:|---:|---|---:|---:|---|---:|---|
| 0 | 5 | image | 6584 | 0.708 | [0.694, 0.722] | 0/5 | not met |
| 0 | 5 | cluster_union | 6016 | 0.708 | [0.694, 0.722] | 0/5 | not met |
| 0.25 | 5 | image | 6584 | 0.383 | [0.371, 0.394] | 5/5 | met |
| 0.25 | 5 | cluster_union | 6016 | 0.383 | [0.369, 0.396] | 5/5 | met |
| 0.5 | 5 | image | 6584 | 0.351 | [0.339, 0.362] | 5/5 | met |
| 0.5 | 5 | cluster_union | 6016 | 0.351 | [0.338, 0.363] | 5/5 | met |
| 1 | 5 | image | 6584 | 0.368 | [0.356, 0.379] | 5/5 | met |
| 1 | 5 | cluster_union | 6016 | 0.368 | [0.356, 0.380] | 5/5 | met |
| 2 | 5 | image | 6584 | 0.376 | [0.364, 0.387] | 5/5 | met |
| 2 | 5 | cluster_union | 6016 | 0.376 | [0.362, 0.388] | 5/5 | met |
| 4 | 5 | image | 6584 | 0.330 | [0.319, 0.341] | 5/5 | met |
| 4 | 5 | cluster_union | 6016 | 0.330 | [0.318, 0.342] | 5/5 | met |
| 8 | 5 | image | 6584 | 0.347 | [0.335, 0.358] | 5/5 | met |
| 8 | 5 | cluster_union | 6016 | 0.347 | [0.335, 0.358] | 5/5 | met |

### scin

| λ | n | unit | n_groups | mean | 95% CI | k CI<0.5 | rule A |
|---:|---:|---|---:|---:|---|---:|---|
| 0 | 5 | image | 10406 | 0.723 | [0.710, 0.736] | 0/5 | not met |
| 0 | 5 | case | 5033 | 0.723 | [0.710, 0.736] | 0/5 | not met |
| 0.25 | 5 | image | 10406 | 0.547 | [0.533, 0.560] | 0/5 | not met |
| 0.25 | 5 | case | 5033 | 0.547 | [0.533, 0.560] | 0/5 | not met |
| 0.5 | 5 | image | 10406 | 0.529 | [0.516, 0.542] | 0/5 | not met |
| 0.5 | 5 | case | 5033 | 0.529 | [0.516, 0.542] | 0/5 | not met |
| 1 | 5 | image | 10406 | 0.528 | [0.515, 0.541] | 0/5 | not met |
| 1 | 5 | case | 5033 | 0.528 | [0.515, 0.541] | 0/5 | not met |
| 2 | 5 | image | 10406 | 0.551 | [0.538, 0.564] | 0/5 | not met |
| 2 | 5 | case | 5033 | 0.551 | [0.538, 0.564] | 0/5 | not met |
| 4 | 5 | image | 10406 | 0.499 | [0.486, 0.512] | 3/5 | not met |
| 4 | 5 | case | 5033 | 0.499 | [0.486, 0.512] | 3/5 | not met |
| 8 | 5 | image | 10406 | 0.536 | [0.524, 0.549] | 0/5 | not met |
| 8 | 5 | case | 5033 | 0.536 | [0.524, 0.549] | 0/5 | not met |

## (2) Paired ID bal acc: λ = 2 − λ = 0

Matched seeds: [42, 52, 62, 72, 82]. Mean Δ = -0.0091 ± 0.0207; seed-bootstrap 95% CI [-0.0260, 0.0064] (2,000 draws, rng 20261005).

| seed | ID bal λ=0 | ID bal λ=2 | Δ |
|---:|---:|---:|---:|
| 42 | 0.5080 | 0.4709 | -0.0371 |
| 52 | 0.4691 | 0.4439 | -0.0251 |
| 62 | 0.4702 | 0.4714 | +0.0012 |
| 72 | 0.4762 | 0.4851 | +0.0089 |
| 82 | 0.4820 | 0.4885 | +0.0065 |

## (3) ID ECE, OOD ECE (pad_heldout), X-dom bal acc by λ (last.ckpt)

| λ | n | ID bal acc | ID ECE | OOD ECE (pad_heldout) | X-dom bal acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 0.481 ± 0.016 | 0.207 ± 0.003 | 0.305 ± 0.042 | 0.277 ± 0.024 |
| 0.25 | 5 | 0.481 ± 0.010 | 0.207 ± 0.004 | 0.724 ± 0.011 | 0.246 ± 0.036 |
| 0.5 | 5 | 0.484 ± 0.005 | 0.203 ± 0.009 | 0.732 ± 0.007 | 0.270 ± 0.025 |
| 1 | 5 | 0.482 ± 0.017 | 0.210 ± 0.004 | 0.762 ± 0.008 | 0.229 ± 0.017 |
| 2 | 5 | 0.472 ± 0.018 | 0.206 ± 0.005 | 0.762 ± 0.015 | 0.244 ± 0.025 |
| 4 | 5 | 0.480 ± 0.016 | 0.202 ± 0.007 | 0.734 ± 0.024 | 0.253 ± 0.022 |
| 8 | 5 | 0.480 ± 0.018 | 0.203 ± 0.002 | 0.738 ± 0.004 | 0.259 ± 0.034 |


## SD-198 image vs cluster_union (sensitivity, Table-1 / P7a features)

See `results/paperB/r5/p13_sd198/REPORT.md`. Rule-A flips: - (a) code: `scripts/r6_r1_leakage_matched.py:244-247 cluster = recipe|lambda|shift_level`; - (b) code: `scripts/r6_r1_leakage_matched.py:283-296 seed bootstrap of point AUROCs`; - AUROC inputs: `maha_of(summary.json) ← metrics.ood.sd198.mahalanobis_classcond (point, no resample)`; - none.
