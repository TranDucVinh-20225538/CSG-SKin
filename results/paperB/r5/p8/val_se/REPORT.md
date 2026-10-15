# Validation bal acc noise, A4 λ = 0 seed 42

| checkpoint | epoch | val bal acc | bootstrap SE | 95 % CI |
|---|---:|---:|---:|---|
| best-val | 1 | 0.539 | 0.023 | [0.494, 0.586] |
| last | 39 | 0.526 | 0.026 | [0.476, 0.577] |

Logged val bal acc: epoch 1 = 0.539; epochs 2–39 min 0.475, max 0.536, range 0.061, SD 0.017; |epoch-to-epoch change| median 0.010, max 0.031.

Methods line:

Validation balanced accuracy (A4, λ = 0, seed 42; lesion-level ISIC validation split, 4053 images / 2229 lesions) has a lesion-clustered bootstrap SE of 0.023 at the best-val checkpoint (epoch 1) and 0.026 at the last checkpoint (epoch 39) (2000 resamples); after epoch 1 it ranges 0.475–0.536 across epochs 2–39 (median epoch-to-epoch change 0.010).

LaTeX row (checkpoint & epoch & val bal acc & SE):

```
best-val & 1 & 0.539 & 0.023 \\
last & 39 & 0.526 & 0.026 \\
```

