# Paper B R5 P5 — single-encoder GRL at λ = 4

Design: `decisions/paperB_r5_p5.md`. Commit `a7195ca`. Phase 15.1 single-encoder EfficientNet-B3 DANN, image-level ISIC split (the split these runs were trained on). Detectors fit on ISIC train. Bootstrap: PREREGISTER draws.

n (seeds) at λ = 4: 3 (42/52/62).
Pipeline check vs stored summary.json (pad_heldout): max |Δ| = 9.7e-04.

## ID resampled by image (pre-registered), OOD = pad_heldout

| Detector | mean ± s.d. | pooled 95% CI | per-seed points | seeds with CI < 0.5 | seeds with point < 0.5 |
|---|---|---|---|---|---|
| Mahalanobis | 0.512 ± 0.027 | [0.485, 0.538] | 0.482, 0.521, 0.534 | 0/3 | 1/3 |
| kNN k=50 | 0.514 ± 0.032 | [0.490, 0.539] | 0.539, 0.478, 0.526 | 0/3 | 1/3 |
| Cosine | 0.490 ± 0.018 | [0.468, 0.512] | 0.471, 0.491, 0.507 | 1/3 | 2/3 |
| MSP | 0.519 ± 0.017 | [0.497, 0.541] | 0.537, 0.504, 0.516 | 0/3 | 0/3 |
| Energy | 0.501 ± 0.034 | [0.480, 0.521] | 0.539, 0.472, 0.492 | 1/3 | 2/3 |

## ID resampled by image (pre-registered), OOD = fitzpatrick17k

| Detector | mean ± s.d. | pooled 95% CI | per-seed points | seeds with CI < 0.5 | seeds with point < 0.5 |
|---|---|---|---|---|---|
| Mahalanobis | 0.575 ± 0.067 | [0.564, 0.585] | 0.558, 0.648, 0.518 | 0/3 | 0/3 |
| kNN k=50 | 0.621 ± 0.075 | [0.612, 0.631] | 0.682, 0.646, 0.537 | 0/3 | 0/3 |
| Cosine | 0.566 ± 0.100 | [0.556, 0.575] | 0.647, 0.596, 0.454 | 1/3 | 1/3 |
| MSP | 0.469 ± 0.047 | [0.459, 0.478] | 0.521, 0.456, 0.430 | 2/3 | 2/3 |
| Energy | 0.532 ± 0.045 | [0.523, 0.542] | 0.578, 0.488, 0.532 | 1/3 | 1/3 |

## ID resampled by lesion group, OOD = pad_heldout

| Detector | mean ± s.d. | pooled 95% CI | per-seed points | seeds with CI < 0.5 | seeds with point < 0.5 |
|---|---|---|---|---|---|
| Mahalanobis | 0.512 ± 0.027 | [0.486, 0.539] | 0.482, 0.521, 0.534 | 0/3 | 1/3 |
| kNN k=50 | 0.514 ± 0.032 | [0.489, 0.538] | 0.539, 0.478, 0.526 | 0/3 | 1/3 |
| Cosine | 0.490 ± 0.018 | [0.467, 0.513] | 0.471, 0.491, 0.507 | 1/3 | 2/3 |
| MSP | 0.519 ± 0.017 | [0.496, 0.541] | 0.537, 0.504, 0.516 | 0/3 | 0/3 |
| Energy | 0.501 ± 0.034 | [0.481, 0.521] | 0.539, 0.472, 0.492 | 1/3 | 2/3 |

## ID resampled by lesion group, OOD = fitzpatrick17k

| Detector | mean ± s.d. | pooled 95% CI | per-seed points | seeds with CI < 0.5 | seeds with point < 0.5 |
|---|---|---|---|---|---|
| Mahalanobis | 0.575 ± 0.067 | [0.563, 0.586] | 0.558, 0.648, 0.518 | 0/3 | 0/3 |
| kNN k=50 | 0.621 ± 0.075 | [0.611, 0.632] | 0.682, 0.646, 0.537 | 0/3 | 0/3 |
| Cosine | 0.566 ± 0.100 | [0.556, 0.576] | 0.647, 0.596, 0.454 | 1/3 | 1/3 |
| MSP | 0.469 ± 0.047 | [0.458, 0.479] | 0.521, 0.456, 0.430 | 2/3 | 2/3 |
| Energy | 0.532 ± 0.045 | [0.522, 0.543] | 0.578, 0.488, 0.532 | 0/3 | 1/3 |

## All λ (context; point AUROC, mean ± s.d.)

| λ | n | leakage | Mahalanobis | kNN k=50 | Cosine | MSP | Energy |
|---:|---:|---|---|---|---|---|---|
| 4 | 3 | 0.798 ± 0.010 | 0.512 ± 0.027 | 0.514 ± 0.032 | 0.490 ± 0.018 | 0.519 ± 0.017 | 0.501 ± 0.034 |

## Plain reading

At λ = 4 (n = 3) the single-encoder Mahalanobis AUROC on pad_heldout is 0.512 ± 0.027, pooled 95% CI [0.485, 0.538]; 0 of 3 seeds have a CI entirely below 0.5 and 1 of 3 a point estimate below 0.5.

## LaTeX rows (detector & n & mean ± s.d. & pooled CI & seeds CI < 0.5), pad_heldout

```latex
Mahalanobis & 3 & 0.512\,$\pm$\,0.027 & [0.485, 0.538] & 0/3 \\
kNN k=50 & 3 & 0.514\,$\pm$\,0.032 & [0.490, 0.539] & 0/3 \\
Cosine & 3 & 0.490\,$\pm$\,0.018 & [0.468, 0.512] & 1/3 \\
MSP & 3 & 0.519\,$\pm$\,0.017 & [0.497, 0.541] & 0/3 \\
Energy & 3 & 0.501\,$\pm$\,0.034 & [0.480, 0.521] & 1/3 \\
```

