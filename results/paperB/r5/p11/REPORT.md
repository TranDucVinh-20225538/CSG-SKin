# Paper B R5 P11: MK-MMD (DAN) and scaled CORAL — comparable invariance (held-out V1)

Recipe `current`. Dual encoder EfficientNet-B3, λ_adv = 0, last.ckpt primary. MK-MMD as Long et al. 2015 / Wu et al. 2023 (arXiv:2309.11705; DAN on rotated MNIST: Maha 32.9, ODIN 30.8). Comparable invariance: |held-out V1 − GRL λ=2 V1| ≤ 0.03 and |ID − GRL λ=0 ID| ≤ 0.05 (decisions/paperB_r5_p11.md amendment 2026-10-16).

GRL references: V1(λ=2) = 0.502 (n=5); ID(λ=0) = 0.496 (n=5).

## Scan (seed 42) and selection

| penalty | weight | epochs | selection probe | val bal acc | held-out V1 | ID bal acc | pad_heldout Maha | comparable? | flag |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| coral | 1 | 40 | 0.500 | 0.446 | 0.500 | 0.481 | 0.772 | yes |  |
| coral | 3 | 40 | 0.632 | 0.426 | 0.532 | 0.459 | 0.658 | yes |  |
| coral | 10 | 40 | 0.645 | 0.317 | 0.542 | 0.316 | 0.518 | no |  |
| coral | 30 | 40 | 0.591 | 0.193 | 0.495 | 0.183 | 0.477 | no |  |
| coral | 10 | 80 | 0.613 | 0.447 | 0.499 | 0.488 | 0.812 | yes |  |
| mkmmd | 1 | 40 | 0.541 | 0.448 | 0.501 | 0.447 | 0.378 | yes |  |
| mkmmd | 3 | 40 | 0.503 | 0.449 | 0.500 | 0.496 | 0.285 | yes |  |
| mkmmd | 10 | 40 | 0.623 | 0.433 | 0.500 | 0.437 | 0.365 | no |  |
| mkmmd | 30 | 40 | 0.522 | 0.446 | 0.500 | 0.503 | 0.460 | yes |  |
| mkmmd | 10 | 80 | 0.582 | 0.441 | 0.500 | 0.474 | 0.387 | yes |  |
| none | 0 | 40 | 0.954 | 0.432 | 0.879 | 0.470 | 0.821 | no |  |

Selected mkmmd: weight 10, 40 epochs (selection probe 0.623, val bal acc 0.433).

Selected coral: weight 10, 80 epochs (selection probe 0.613, val bal acc 0.447).

## Comparable invariance (all MK-MMD / CORAL settings)

| arm | n | held-out V1 | ΔV1 vs GRL λ2 | ID bal acc | ΔID vs GRL λ0 | meets |
|---|---:|---:|---:|---:|---:|---|
| coral|w10|ep40 | 1 | 0.542 | +0.040 | 0.316 | -0.180 | no |
| coral|w10|ep80 | 5 | 0.500 | -0.002 | 0.467 | -0.029 | yes |
| coral|w1|ep40 | 1 | 0.500 | -0.002 | 0.481 | -0.015 | yes |
| coral|w30|ep40 | 1 | 0.495 | -0.007 | 0.183 | -0.312 | no |
| coral|w3|ep40 | 1 | 0.532 | +0.030 | 0.459 | -0.037 | yes |
| mkmmd|w10|ep40 | 5 | 0.506 | +0.004 | 0.460 | -0.036 | yes |
| mkmmd|w10|ep80 | 1 | 0.500 | -0.002 | 0.474 | -0.022 | yes |
| mkmmd|w1|ep40 | 1 | 0.501 | -0.001 | 0.447 | -0.049 | yes |
| mkmmd|w30|ep40 | 1 | 0.500 | -0.002 | 0.503 | +0.007 | yes |
| mkmmd|w3|ep40 | 1 | 0.500 | -0.002 | 0.496 | +0.001 | yes |

**Outcome: see per-setting collapse/inversion below.**

## Settings that meet comparable invariance (5 detectors, rule A)

| arm | n | Mahalanobis | kNN k=50 | Cosine | MSP | Energy | Maha rule A pad_heldout | Maha rule A Fitz | collapse | inversion |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| coral|w10|ep80 | 5 | 0.820 ± 0.044 | 0.863 ± 0.052 | 0.801 ± 0.055 | 0.662 ± 0.057 | 0.676 ± 0.054 | above | above | no | no |
| coral|w10|ep80 (Fitz) | 5 | 0.788 ± 0.034 | 0.847 ± 0.045 | 0.811 ± 0.063 | 0.787 ± 0.036 | 0.850 ± 0.030 | — | above | — | — |
| coral|w1|ep40 | 1 | 0.772 | 0.805 | 0.733 | 0.620 | 0.639 | — | — | no | no |
| coral|w1|ep40 (Fitz) | 1 | 0.680 | 0.742 | 0.674 | 0.557 | 0.629 | — | — | — | — |
| coral|w3|ep40 | 1 | 0.658 | 0.836 | 0.762 | 0.683 | 0.728 | — | — | no | no |
| coral|w3|ep40 (Fitz) | 1 | 0.742 | 0.778 | 0.764 | 0.704 | 0.756 | — | — | — | — |
| mkmmd|w10|ep40 | 5 | 0.354 ± 0.022 | 0.366 ± 0.024 | 0.431 ± 0.017 | 0.342 ± 0.045 | 0.356 ± 0.033 | A | above | no | yes |
| mkmmd|w10|ep40 (Fitz) | 5 | 0.553 ± 0.042 | 0.599 ± 0.047 | 0.575 ± 0.051 | 0.555 ± 0.067 | 0.609 ± 0.056 | — | above | — | — |
| mkmmd|w10|ep80 | 1 | 0.387 | 0.401 | 0.473 | 0.417 | 0.481 | — | — | no | yes |
| mkmmd|w10|ep80 (Fitz) | 1 | 0.379 | 0.437 | 0.539 | 0.454 | 0.640 | — | — | — | — |
| mkmmd|w1|ep40 | 1 | 0.378 | 0.395 | 0.412 | 0.435 | 0.403 | — | — | no | yes |
| mkmmd|w1|ep40 (Fitz) | 1 | 0.341 | 0.403 | 0.434 | 0.464 | 0.564 | — | — | — | — |
| mkmmd|w30|ep40 | 1 | 0.460 | 0.462 | 0.509 | 0.448 | 0.482 | — | — | yes | yes |
| mkmmd|w30|ep40 (Fitz) | 1 | 0.660 | 0.592 | 0.785 | 0.476 | 0.737 | — | — | — | — |
| mkmmd|w3|ep40 | 1 | 0.285 | 0.339 | 0.366 | 0.339 | 0.335 | — | — | no | yes |
| mkmmd|w3|ep40 (Fitz) | 1 | 0.440 | 0.503 | 0.493 | 0.510 | 0.556 | — | — | — | — |

```latex
coral w10 ep80 & 5 & 0.820\,$\pm$\,0.044 & 0.863\,$\pm$\,0.052 & 0.801\,$\pm$\,0.055 & 0.662\,$\pm$\,0.057 & 0.676\,$\pm$\,0.054 \\
coral w1 ep40 & 1 & 0.772 & 0.805 & 0.733 & 0.620 & 0.639 \\
coral w3 ep40 & 1 & 0.658 & 0.836 & 0.762 & 0.683 & 0.728 \\
mkmmd w10 ep40 & 5 & 0.354\,$\pm$\,0.022 & 0.366\,$\pm$\,0.024 & 0.431\,$\pm$\,0.017 & 0.342\,$\pm$\,0.045 & 0.356\,$\pm$\,0.033 \\
mkmmd w10 ep80 & 1 & 0.387 & 0.401 & 0.473 & 0.417 & 0.481 \\
mkmmd w1 ep40 & 1 & 0.378 & 0.395 & 0.412 & 0.435 & 0.403 \\
mkmmd w30 ep40 & 1 & 0.460 & 0.462 & 0.509 & 0.448 & 0.482 \\
mkmmd w3 ep40 & 1 & 0.285 & 0.339 & 0.366 & 0.339 & 0.335 \\
```

## Results in plain words

coral|w10|ep80 (n=5): meets comparable invariance (V1=0.500, ID=0.467). Pad_heldout Mahalanobis 0.820 ± 0.044; rule A above (pooled CI 0.809-0.830, 0/5); collapse no; inversion no. Fitzpatrick Mahalanobis 0.788 ± 0.034.

coral|w1|ep40 (n=1): meets comparable invariance (V1=0.500, ID=0.481). Pad_heldout Mahalanobis 0.772; rule A —; collapse no; inversion no. Fitzpatrick Mahalanobis 0.680.

coral|w3|ep40 (n=1): meets comparable invariance (V1=0.532, ID=0.459). Pad_heldout Mahalanobis 0.658; rule A —; collapse no; inversion no. Fitzpatrick Mahalanobis 0.742.

mkmmd|w10|ep40 (n=5): meets comparable invariance (V1=0.506, ID=0.460). Pad_heldout Mahalanobis 0.354 ± 0.022; rule A A (pooled CI 0.338-0.370, 5/5); collapse no; inversion yes. Fitzpatrick Mahalanobis 0.553 ± 0.042.

mkmmd|w10|ep80 (n=1): meets comparable invariance (V1=0.500, ID=0.474). Pad_heldout Mahalanobis 0.387; rule A —; collapse no; inversion yes. Fitzpatrick Mahalanobis 0.379.

mkmmd|w1|ep40 (n=1): meets comparable invariance (V1=0.501, ID=0.447). Pad_heldout Mahalanobis 0.378; rule A —; collapse no; inversion yes. Fitzpatrick Mahalanobis 0.341.

mkmmd|w30|ep40 (n=1): meets comparable invariance (V1=0.500, ID=0.503). Pad_heldout Mahalanobis 0.460; rule A —; collapse yes; inversion yes. Fitzpatrick Mahalanobis 0.660.

mkmmd|w3|ep40 (n=1): meets comparable invariance (V1=0.500, ID=0.496). Pad_heldout Mahalanobis 0.285; rule A —; collapse no; inversion yes. Fitzpatrick Mahalanobis 0.440.

