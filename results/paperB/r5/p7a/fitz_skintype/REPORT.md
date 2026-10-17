# Paper B R5 P7(a): Fitzpatrick17k by skin type

Design: `decisions/paperB_r5_p7a.md` (addition of 2026-10-15). Commit `6e5713c`. Mahalanobis (class-conditional) on z_lesion_norm, fit on ISIC train; ID = lesion-level ISIC test. Mean ± s.d. over 5 seeds; 95% CI of the seed mean, PREREGISTER draws, Fitzpatrick side resampled by P13 `cluster_union` within the stratum (image-level CI in the last column). 153 of 3887 images have no skin-type label (−1) and are excluded.

| stratum | n images | n clusters | mapped to an ISIC class | λ | Mahalanobis | CI (clusters) | k seeds CI<0.5 | rule A | CI (images) |
|---|---:|---:|---:|---:|---|---|---:|---|---|
| I–II | 819 | 800 | 23.7% | 0 | 0.731 ± 0.035 | [0.716, 0.746] | 0/5 | no | [0.715, 0.746] |
| I–II | 819 | 800 | 23.7% | 2 | 0.427 ± 0.025 | [0.407, 0.447] | 5/5 | yes | [0.407, 0.446] |
| III–IV | 2078 | 2019 | 14.1% | 0 | 0.716 ± 0.037 | [0.702, 0.731] | 0/5 | no | [0.702, 0.730] |
| III–IV | 2078 | 2019 | 14.1% | 2 | 0.384 ± 0.018 | [0.369, 0.398] | 5/5 | yes | [0.370, 0.398] |
| V–VI | 837 | 818 | 8.8% | 0 | 0.706 ± 0.030 | [0.690, 0.722] | 0/5 | no | [0.691, 0.721] |
| V–VI | 837 | 818 | 8.8% | 2 | 0.403 ± 0.015 | [0.382, 0.424] | 5/5 | yes | [0.385, 0.420] |
| all labelled | 3734 | 3601 | 15.0% | 0 | 0.717 ± 0.034 | [0.704, 0.731] | 0/5 | no | [0.704, 0.731] |
| all labelled | 3734 | 3601 | 15.0% | 2 | 0.398 ± 0.018 | [0.384, 0.412] | 5/5 | yes | [0.385, 0.410] |

## LaTeX rows (stratum & n & λ & Mahalanobis & cluster CI)

```latex
I--II & 819 & 0 & 0.731\,$\pm$\,0.035 & [0.716, 0.746] \\
I--II & 819 & 2 & 0.427\,$\pm$\,0.025 & [0.407, 0.447] \\
III--IV & 2078 & 0 & 0.716\,$\pm$\,0.037 & [0.702, 0.731] \\
III--IV & 2078 & 2 & 0.384\,$\pm$\,0.018 & [0.369, 0.398] \\
V--VI & 837 & 0 & 0.706\,$\pm$\,0.030 & [0.690, 0.722] \\
V--VI & 837 & 2 & 0.403\,$\pm$\,0.015 & [0.382, 0.424] \\
all labelled & 3734 & 0 & 0.717\,$\pm$\,0.034 & [0.704, 0.731] \\
all labelled & 3734 & 2 & 0.398\,$\pm$\,0.018 & [0.384, 0.412] \\
```

