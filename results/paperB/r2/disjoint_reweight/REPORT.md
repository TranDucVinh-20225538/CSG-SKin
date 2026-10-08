# R2 — lesion-disjoint shift: leakage or class composition?

Commit: `876c313`. Representation: z_lesion^norm (16-d). Scores fit on ISIC train only.

Rule (stated before running): reweight the 2026-image lesion-disjoint ID set to the class mix of the full 5067-image test set. If AUROC stays closer to the disjoint value than to the full value → (a) leakage drives the shift; if it returns closer to the full value → (b) class composition. Converse check: reweight the full set to the disjoint class mix. Weights per ID image w_c = p_target(c) / p_source(c); OOD weight 1.

## 1. Class distribution

| Class | Full n | Full % | Disjoint n | Disjoint % |
|---|---:|---:|---:|---:|
| MEL | 904 | 17.8 | 184 | 9.1 |
| NV | 2575 | 50.8 | 1443 | 71.2 |
| BCC | 665 | 13.1 | 132 | 6.5 |
| AK | 173 | 3.4 | 25 | 1.2 |
| BKL | 525 | 10.4 | 181 | 8.9 |
| DF | 48 | 0.9 | 16 | 0.8 |
| VASC | 51 | 1.0 | 19 | 0.9 |
| SCC | 126 | 2.5 | 26 | 1.3 |
| total | 5067 | 100 | 2026 | 100 |

## 2–3. λ = 2 (5 seeds, mean): reweighting both ways

share = (disjoint − disjoint reweighted to full mix) / (disjoint − full): fraction of the full→disjoint shift attributable to class mix.

| Detector | OOD | Full | Disjoint | Disjoint → full mix | Full → disjoint mix | share (class mix) | Reading |
|---|---|---|---|---|---|---|---|
| Mahalanobis | pad_heldout | 0.427 | 0.515 | 0.447 | 0.480 | 0.77 | (b) class composition |
| Mahalanobis | fitzpatrick17k | 0.391 | 0.490 | 0.415 | 0.450 | 0.76 | (b) class composition |
| kNN k=50 | pad_heldout | 0.440 | 0.521 | 0.452 | 0.488 | 0.85 | (b) class composition |
| kNN k=50 | fitzpatrick17k | 0.423 | 0.509 | 0.436 | 0.473 | 0.85 | (b) class composition |

### Mahalanobis — pad_heldout, all λ (mean ± s.d.)

| λ | n | Full | Disjoint | Disjoint → full mix | Full → disjoint mix |
|---:|---:|---|---|---|---|
| 0 | 5 | 0.843 ± 0.023 | 0.890 ± 0.015 | 0.845 ± 0.022 | 0.879 ± 0.017 |
| 0.25 | 3 | 0.513 ± 0.018 | 0.597 ± 0.032 | 0.526 ± 0.025 | 0.569 ± 0.027 |
| 0.5 | 3 | 0.465 ± 0.015 | 0.553 ± 0.012 | 0.485 ± 0.010 | 0.520 ± 0.015 |
| 1 | 3 | 0.455 ± 0.036 | 0.535 ± 0.036 | 0.473 ± 0.036 | 0.504 ± 0.038 |
| 2 | 5 | 0.427 ± 0.025 | 0.515 ± 0.026 | 0.447 ± 0.025 | 0.480 ± 0.026 |
| 4 | 3 | 0.455 ± 0.039 | 0.545 ± 0.040 | 0.477 ± 0.037 | 0.510 ± 0.041 |
| 8 | 5 | 0.494 ± 0.027 | 0.584 ± 0.016 | 0.515 ± 0.020 | 0.548 ± 0.022 |

### Mahalanobis — fitzpatrick17k, all λ (mean ± s.d.)

| λ | n | Full | Disjoint | Disjoint → full mix | Full → disjoint mix |
|---:|---:|---|---|---|---|
| 0 | 5 | 0.764 ± 0.038 | 0.830 ± 0.027 | 0.769 ± 0.036 | 0.813 ± 0.031 |
| 0.25 | 3 | 0.439 ± 0.061 | 0.536 ± 0.052 | 0.457 ± 0.051 | 0.501 ± 0.059 |
| 0.5 | 3 | 0.449 ± 0.025 | 0.547 ± 0.029 | 0.470 ± 0.027 | 0.510 ± 0.027 |
| 1 | 3 | 0.457 ± 0.088 | 0.548 ± 0.086 | 0.477 ± 0.087 | 0.513 ± 0.089 |
| 2 | 5 | 0.391 ± 0.068 | 0.490 ± 0.072 | 0.415 ± 0.067 | 0.450 ± 0.071 |
| 4 | 3 | 0.431 ± 0.036 | 0.533 ± 0.036 | 0.456 ± 0.033 | 0.493 ± 0.039 |
| 8 | 5 | 0.474 ± 0.060 | 0.578 ± 0.046 | 0.498 ± 0.051 | 0.536 ± 0.056 |

### kNN k=50 — pad_heldout, all λ (mean ± s.d.)

| λ | n | Full | Disjoint | Disjoint → full mix | Full → disjoint mix |
|---:|---:|---|---|---|---|
| 0 | 5 | 0.918 ± 0.009 | 0.935 ± 0.008 | 0.911 ± 0.012 | 0.933 ± 0.007 |
| 0.25 | 3 | 0.541 ± 0.020 | 0.609 ± 0.036 | 0.542 ± 0.028 | 0.585 ± 0.031 |
| 0.5 | 3 | 0.498 ± 0.018 | 0.580 ± 0.016 | 0.509 ± 0.013 | 0.547 ± 0.020 |
| 1 | 3 | 0.471 ± 0.027 | 0.545 ± 0.022 | 0.481 ± 0.024 | 0.513 ± 0.028 |
| 2 | 5 | 0.440 ± 0.012 | 0.521 ± 0.012 | 0.452 ± 0.012 | 0.488 ± 0.013 |
| 4 | 3 | 0.483 ± 0.033 | 0.565 ± 0.031 | 0.496 ± 0.029 | 0.531 ± 0.031 |
| 8 | 5 | 0.507 ± 0.038 | 0.588 ± 0.034 | 0.519 ± 0.034 | 0.554 ± 0.036 |

### kNN k=50 — fitzpatrick17k, all λ (mean ± s.d.)

| λ | n | Full | Disjoint | Disjoint → full mix | Full → disjoint mix |
|---:|---:|---|---|---|---|
| 0 | 5 | 0.837 ± 0.030 | 0.872 ± 0.024 | 0.828 ± 0.031 | 0.866 ± 0.025 |
| 0.25 | 3 | 0.476 ± 0.054 | 0.556 ± 0.045 | 0.481 ± 0.046 | 0.525 ± 0.051 |
| 0.5 | 3 | 0.486 ± 0.034 | 0.574 ± 0.035 | 0.498 ± 0.035 | 0.539 ± 0.033 |
| 1 | 3 | 0.499 ± 0.063 | 0.575 ± 0.054 | 0.507 ± 0.057 | 0.543 ± 0.062 |
| 2 | 5 | 0.423 ± 0.059 | 0.509 ± 0.058 | 0.436 ± 0.056 | 0.473 ± 0.060 |
| 4 | 3 | 0.467 ± 0.033 | 0.555 ± 0.025 | 0.481 ± 0.026 | 0.518 ± 0.031 |
| 8 | 5 | 0.504 ± 0.073 | 0.593 ± 0.060 | 0.517 ± 0.065 | 0.556 ± 0.069 |

## Composition-free check: per-class AUROC (ID class c vs OOD), Mahalanobis, λ = 2, 5 seeds

Within a class, composition cannot differ. If full and disjoint agree within each class, the aggregate shift is composition; if disjoint is higher within classes, it is leakage.

**pad_heldout**

| Class | Full | Disjoint | Δ |
|---|---|---|---|
| MEL | 0.388 ± 0.025 | 0.376 ± 0.024 | -0.012 |
| NV | 0.559 ± 0.029 | 0.610 ± 0.028 | +0.051 |
| BCC | 0.257 ± 0.031 | 0.216 ± 0.028 | -0.041 |
| AK | 0.158 ± 0.018 | 0.150 ± 0.022 | -0.008 |
| BKL | 0.270 ± 0.024 | 0.288 ± 0.029 | +0.018 |
| DF | 0.115 ± 0.022 | — (n<20) | — |
| VASC | 0.157 ± 0.053 | — (n<20) | — |
| SCC | 0.142 ± 0.015 | 0.153 ± 0.015 | +0.010 |

**fitzpatrick17k**

| Class | Full | Disjoint | Δ |
|---|---|---|---|
| MEL | 0.343 ± 0.077 | 0.329 ± 0.078 | -0.014 |
| NV | 0.536 ± 0.075 | 0.594 ± 0.078 | +0.058 |
| BCC | 0.202 ± 0.059 | 0.162 ± 0.047 | -0.040 |
| AK | 0.109 ± 0.039 | 0.101 ± 0.035 | -0.008 |
| BKL | 0.218 ± 0.066 | 0.237 ± 0.071 | +0.019 |
| DF | 0.080 ± 0.030 | — (n<20) | — |
| VASC | 0.117 ± 0.052 | — (n<20) | — |
| SCC | 0.095 ± 0.032 | 0.105 ± 0.039 | +0.010 |

